"""Compare new completed runs without historical results or model calls."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import statistics
import sys
import hashlib
import io
import tarfile
import tempfile
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.recompute_public_metrics import recompute
from research.progress import Journal,atomic_json


def load(path):
    return json.loads(path.read_text())


def collect(directory, include_trace=False):
    directory = Path(directory).resolve()
    if directory == ROOT or not directory.is_dir():
        raise ValueError('Pass a specific experiment output directory')
    runs = sorted({p.parent for name in ('summary.json', 'manifest.json') for p in directory.rglob(name)
                   if 'rollout-verification' not in p.relative_to(directory).parts})
    episodes, statuses, signatures, policy_signatures, initial_pools = [], [], {}, {}, {}
    for run in runs:
        summary = load(run/'summary.json') if (run/'summary.json').exists() else {}
        manifest = load(run/'manifest.json') if (run/'manifest.json').exists() else {}
        config = summary.get('config') or manifest.get('config') or {}
        if not config:
            continue
        stage = summary.get('stage', manifest.get('stage', 'unknown'))
        label = config.get('comparison_id', config.get('arm', 'unknown'))
        split = 'train' if stage == 'train' else 'test'
        provenance = summary.get('training_provenance') or {}
        if stage == 'eval' and label in ('B2', 'B3-no-support', 'B3', 'B4') and not provenance.get('episodes_completed'):
            split = 'untrained_control'
        if stage=='eval' and label in ('B2','B3-no-support','B3','B4') and provenance.get('optimizer_status') and not provenance['optimizer_status'].startswith('completed_'):
            split='untrained_control'
        if config.get('evaluation_role')=='rollout':continue
        version = config.get('experiment_version', 'unversioned')+':'+config.get('composition', 'unknown')
        found = {(e['seed'], e.get('episode_id', 0)): e for e in summary.get('episodes', [])}
        for path in sorted(run.glob('episode-????.json')):
            episode = load(path); key = (episode['seed'], episode.get('episode_id', 0))
            if key in found and found[key].get('joint_return') != episode.get('joint_return'):
                raise ValueError(f'Episode summary disagrees with its completed record: {path}')
            found[key] = episode
        completed = 0
        for episode in found.values():
            if episode.get('status') != 'completed' or episode.get('kind') != 'real_cage4':
                continue
            if config.get('steps') == 500 and (episode.get('joint_transitions'), episode.get('scored_transitions')) != (499, 498):
                raise ValueError('Invalid official 500-step episode evidence')
            identity = episode.get('model_identity') or summary.get('model_identity') or {}
            signature = json.dumps({
                'conditions': {k: config.get(k) for k in ('steps', 'composition', 'profile', 'scoring', 'communication', 'behavior_context', 'skip_busy_policy', 'action_contract')},
                'model': {k: config.get('model', {}).get(k) for k in ('model', 'revision', 'temperature', 'max_tokens')},
                'identity': {k: identity.get(k) for k in ('model', 'revision', 'dtype', 'chat_template_sha256')},
                'keep_weights': {k: v.get('sha256') for k, v in (episode.get('weights') or {}).items()},
                'implementation':manifest.get('implementation_sha256'),
                'protocol':config.get('protocol_digest'),
            }, sort_keys=True)
            group = (version, config['red'], split)
            if group in signatures and signatures[group] != signature:
                raise ValueError('Comparison conditions differ within the same version/red/split')
            signatures[group] = signature
            shared_initial=summary.get('comparison_initial_pool_sha256')
            if label!='B0' and shared_initial:
                if group in initial_pools and initial_pools[group]!=shared_initial:
                    raise ValueError('Comparison initial Pools differ')
                initial_pools[group]=shared_initial
            policy_key=group+(label,)
            policy_value=summary.get('pool_sha256')
            if policy_key in policy_signatures and policy_signatures[policy_key]!=policy_value:
                raise ValueError('Different frozen Pools cannot be merged within one arm')
            policy_signatures[policy_key]=policy_value
            item = {k: episode[k] for k in ('seed', 'status', 'kind', 'joint_return', 'execution_time_s')}
            item.update(version=version, label=label, red=config['red'], split=split,
                        reward_components=episode.get('reward_components'),
                        scored_step_rewards=episode.get('scored_step_rewards'), model_identity=identity,
                        source=str(run.relative_to(directory)), episode_id=episode.get('episode_id', 0))
            if include_trace and episode.get('trace'):
                path = Path(episode['trace'])
                if not path.is_absolute():
                    path = run/path
                counts, flags, busy = Counter(), Counter(), 0
                if path.is_file():
                    for line in path.open():
                        row = json.loads(line)
                        if row.get('policy') != 'llm':
                            continue
                        if row.get('policy_invoked') and not row.get('parse_error'):
                            counts[(row.get('requested_action') or 'unknown').split()[0]] += 1
                        elif not row.get('policy_invoked'):
                            busy += 1
                        for key in ('repeated_analysis_same_evidence', 'duplicate_successful_decoy', 'sleep_with_open_case'):
                            flags[key] += bool((row.get('behavior_outcome') or {}).get(key))
                    item['llm_behavior'] = {'free_actions': dict(counts), 'automatic_busy_ticks': busy, 'observable_flags': dict(flags)}
            episodes.append(item); completed += 1
        statuses.append({'directory': str(run.relative_to(directory)), 'label': label, 'red': config.get('red'),
                         'stage': stage, 'status': summary.get('status', 'not_committed'),
                         'completed_environment_episodes': completed, 'committed_iterations': len(summary.get('history', [])),
                         'optimizer_states':[h.get('optimizer_status','unknown') for h in summary.get('history',[])],
                         'pool_changes':sum(h.get('pool_changed',False) for h in summary.get('history',[])),
                         'failure':load(run/'failure.json') if (run/'failure.json').exists() else None})
    return episodes, statuses



COMPARISONS=(('B1','B0'),('B2','B1'),('B4','B2'),('B4','B0'),
             ('B3-no-support','B2'),('B3','B3-no-support'),('B3','B2'),('B4','B3'),
             ('B2-ASV','B2'),('B2-SSV','B2'),('B2-ASV-SSV','B2'),('B2-ASV-SSV-RV-screen','B2'))


def paired_comparisons(episodes):
    groups=defaultdict(dict)
    for e in episodes:groups[(e['version'],e['red'],e['split'])].setdefault(e['label'],{})[e['seed']]=e
    output=[]
    for (version,red,split),arms in sorted(groups.items()):
        for new,old in COMPARISONS:
            before=arms.get(old,{});after=arms.get(new,{})
            seeds=sorted(before.keys()&after.keys());delta=[after[s]['joint_return']-before[s]['joint_return'] for s in seeds]
            row={'comparison':new+'-'+old,'version':version,'red':red,'split':split,'paired_n':len(seeds),
                'seeds':seeds,'differences':delta,'mean_difference':statistics.mean(delta) if delta else None,
                'median_difference':statistics.median(delta) if delta else None,
                'win_tie_loss':[sum(d>0 for d in delta),sum(d==0 for d in delta),sum(d<0 for d in delta)] if delta else None,
                'paired_bootstrap_95ci':None,'reward_component_mean_differences':None,
                'training_repetitions':1 if delta else None,
                'interpretation':'Paired environment seeds do not identify individual-action causation or independent training-run uncertainty.'}
            if len(delta)>1:
                rng=random.Random(20261125);means=sorted(statistics.mean(rng.choices(delta,k=len(delta))) for _ in range(10000))
                row['paired_bootstrap_95ci']=[means[249],means[9749]]
            if delta and all(before[s].get('reward_components') is not None and after[s].get('reward_components') is not None for s in seeds):
                keys=set().union(*(before[s]['reward_components'].keys()|after[s]['reward_components'].keys() for s in seeds))
                row['reward_component_mean_differences']={k:statistics.mean(after[s]['reward_components'].get(k,0)-before[s]['reward_components'].get(k,0) for s in seeds) for k in keys}
            output.append(row)
    if not groups:
        output=[{'comparison':new+'-'+old,'paired_n':0,'mean_difference':None,'median_difference':None,
                 'win_tie_loss':None,'paired_bootstrap_95ci':None,'reason':'No completed paired real test episodes'} for new,old in COMPARISONS]
    return output


def optimizer_report(directory,output,statuses):
    states=[];changes=[];diagnoses=[];tasks=[];usage=None
    db=directory/'journal.sqlite3'
    if db.exists():
        journal=Journal(db);tasks=journal.tasks();usage=journal.budget_usage()
        keys=[r[0] for r in journal.db.execute("SELECT key FROM checkpoints WHERE key LIKE 'optimizer:%'").fetchall() if r[0].count(':')==1]
        for key in keys:
            state=journal.restore(key);result=state.get('result') or {}
            states.append({'loop_id':key,'arm':state.get('context',{}).get('arm'),'iteration':state.get('iteration'),
                           'status':state['status'],'pool_changed':result.get('pool_changed'),
                           'attempts':state['attempts'],'error':state.get('error')})
            for record in result.get('evolution',[]):
                for sg in record.get('semantic_gradients',[]):
                    for item in sg.get('evidence_audits',[]):diagnoses.append(item)
                for candidate in record.get('candidates',[]):
                    if candidate.get('accepted'):
                        changes.append({'loop_id':key,'arm':state.get('context',{}).get('arm'),'iteration':state.get('iteration'),
                            'parent':record['old_skill'],'after':candidate['skill'],
                            'semantic_gradients':record.get('semantic_gradients',[]),'gate':candidate.get('diagnostics'),
                            'rollout':candidate.get('rollout_validation'),'next_loop_used':None,
                            'next_loop_note':'Unknown until a later trace proves the new skill was invoked.'})
    for change in changes:
        matching=[r for r in statuses if r['stage']=='train' and r['label']==change['arm']]
        observed=[];complete=True
        for run in matching:
            path=directory/run['directory'];manifest=path/'manifest.json'
            if not manifest.exists() or change['iteration'] is None:complete=False;continue
            cfg=load(manifest)['config'];planned=cfg['train_episodes'];loops=cfg['iterations'];iteration=change['iteration']
            first=(iteration*planned+loops-1)//loops;last=((iteration+1)*planned+loops-1)//loops
            if first>=planned:complete=False;continue
            for episode in range(first,min(last,planned)):
                trace=path/f'episode-{episode:04d}.jsonl';record_path=trace.with_suffix('.json')
                if not trace.exists() or not record_path.exists():complete=False;continue
                observed.extend(json.loads(line) for line in trace.open())
        used=any(r.get('policy')=='llm' and r.get('skill_id')==change['after']['name'] for r in observed)
        change['next_loop_used']=True if used else False if matching and complete else None
        change['next_loop_note']='Based on completed next-batch raw traces; missing/unfinished future experience remains unknown.'
    atomic_json(output/'optimizer_progress.json',{'loops':states,'environment_runs':statuses,'tasks':tasks,
        'resource_phases':journal.restore('resource_phases') if db.exists() else None,
        'budget':usage,'unknown_is_not_zero':True,
        'completed_optimizer_loops':sum(s['status'].startswith('completed_') for s in states),
        'pool_change_loops':sum(s['pool_changed'] is True for s in states),
        'phase_costs':'See bottleneck_report; GPU intervals include startup/idle/cleanup, test episode seconds remain separate.'})
    atomic_json(output/'skill_changes.json',{'committed_changes':changes,'accepted_count':len(changes) if states and all(s['status'].startswith('completed_') for s in states) else None})
    path=output/'diagnosis_evaluation.json'
    if not path.exists():
        atomic_json(path,{'real_sg_audits':diagnoses,'labeled_real_cases':None,'false_accept_rate':None,'false_reject_rate':None,
            'reason':'No labeled real records available for rate estimation; offline contract tests are not real-Qwen diagnosis accuracy.',
            'limitation':'Restricted typed facts/proposal links are checked; general language truth, causation and defense improvement are not established.'})
    resume=output/'resume_verification.json'
    if not resume.exists():atomic_json(resume,{'actual_qwen_interruption_matrix':'not_run','offline_contract_verification':'See recorded test results',
        'reusable':'Completed exact requests, SG validation, candidates, per-sample scores, completed rollout seeds and committed Pool/baseline state',
        'replayed':'CPU control flow from the registered loop base; RNG/counter restored. Unknown requests may need an explicitly recorded retry.',
        'prior_training_recovery':'Unavailable without original full traces and complete training checkpoint; initial_pool is static.'})


def reproduction_bundle(directory,output):
    selected=[p for p in directory.rglob('*') if p.is_file() and output not in p.parents
              and p.suffix in ('.json','.jsonl','.log','.yaml','.sqlite3')]
    # WAL-backed SQLite requires a consistent backup, not a blind copy of its main file.
    selected += [p for p in output.rglob('*') if p.is_file() and p.suffix in ('.json','.md','.xml')
                 and p.name!='reproducibility_manifest.json']
    archive=output/'reproduction_bundle.tar.gz';manifest=[]
    with tempfile.TemporaryDirectory(prefix='lab-bundle-') as temp, tarfile.open(archive,'w:gz') as bundle:
        for path in sorted(selected):
            name=str(path.relative_to(directory));source=path
            if path.suffix=='.sqlite3':
                source=Path(temp)/(hashlib.sha256(name.encode()).hexdigest()+'.sqlite3')
                with sqlite3.connect(path) as db,sqlite3.connect(source) as backup:db.backup(backup)
            raw=source.read_bytes();manifest.append({'path':name,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),
                'kind':'full_trace' if path.suffix=='.jsonl' else 'durable_journal' if path.suffix=='.sqlite3' else 'checkpoint_or_metadata'})
            info=tarfile.TarInfo(name);info.size=len(raw)
            bundle.addfile(info,io.BytesIO(raw))
        source_paths=sorted(list((ROOT/'research').glob('*.py'))+list((ROOT/'acd').glob('*.py'))+
            list((ROOT/'scripts').glob('*.py'))+list((ROOT/'scripts').glob('*.sh'))+
            list((ROOT/'research_configs').glob('*.yaml'))+list(ROOT.glob('*lock*'))+
            [ROOT/'research.sh',ROOT/'artifacts/initial_pool.json'])
        for source in source_paths:
            if not source.is_file():continue
            name='checkout/'+str(source.relative_to(ROOT));raw=source.read_bytes()
            manifest.append({'path':name,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'kind':'reproduction_source_or_lock'})
            info=tarfile.TarInfo(name);info.size=len(raw)
            bundle.addfile(info,io.BytesIO(raw))
    atomic_json(output/'reproducibility_manifest.json',{'bundle':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'members':manifest,'compact_metrics':'episodes.json does not contain exact actor prompts, full raw observations, RNG or optimizer responses.',
        'source_model_locks':'Use protocol.json hashes and the pinned root checkout; weights are excluded. No external upload or Git action performed.'})

def report_experiment(directory, output=None, include_trace=False):
    directory = Path(directory).resolve()
    output = Path(output).resolve() if output else directory/'report'
    episodes, statuses = collect(directory, include_trace)
    metrics = recompute(episodes)
    for pair in metrics['paired_vs_B0']:
        delta = pair['differences']
        pair['median_difference'] = statistics.median(delta) if delta else None
        pair['paired_bootstrap_95ci'] = None
        if len(delta) > 1:
            rng = random.Random(20261125)
            means = [statistics.mean(rng.choices(delta, k=len(delta))) for _ in range(10000)]
            means.sort()
            pair['paired_bootstrap_95ci'] = [means[249], means[9749]]
    metrics['paired_comparisons']=paired_comparisons(episodes)
    for group in metrics['groups']:
        group['mean_execution_time_s']=group['total_execution_time_s']/group['n']
        items=[e for e in episodes if all(e[k]==group[k] for k in ('version','red','split','label'))]
        if all(e.get('reward_components') is not None for e in items):
            components=set().union(*(e['reward_components'] for e in items))
            group['mean_reward_components']={k:statistics.mean(e['reward_components'].get(k,0) for e in items) for k in components}
        else:group['mean_reward_components']=None
    decisions = []
    for path in sorted(directory.rglob('decision.json')):
        if 'rollout-verification' in path.relative_to(directory).parts:
            decisions.append({'source': str(path.relative_to(directory)),
                              **{k: v for k, v in load(path).items() if k != 'verified_pool'}})
    output.mkdir(parents=True, exist_ok=True)
    (output/'episodes.json').write_text(json.dumps({'schema': 'compact-episode-metrics-v1', 'episodes': episodes}, ensure_ascii=False, indent=2)+'\n')
    (output/'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2)+'\n')
    (output/'run_status.json').write_text(json.dumps(statuses, ensure_ascii=False, indent=2)+'\n')
    (output/'rollout_decisions.json').write_text(json.dumps(decisions, ensure_ascii=False, indent=2)+'\n')
    fmt = lambda value: 'N/A' if value is None else f'{value:.2f}' if isinstance(value, float) else str(value)
    lines = ['# Experiment comparison', '',
             'Only completed real CAGE4 episodes are counted. Train, test and untrained controls are separate. Candidate-validation episodes are excluded from final-test metrics.', '',
             '| Version | Red | Split | Arm | N | Mean reward | Sample SD | Total seconds | Paired gain vs B0 | Win rate | Median gain | 95% bootstrap CI |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    cases = sorted({tuple(g[k] for k in ('version', 'red', 'split')) for g in metrics['groups']})
    for version, red, split in cases:
        groups = {g['label']: g for g in metrics['groups'] if (g['version'], g['red'], g['split']) == (version, red, split)}
        paired = {g['label']: g for g in metrics['paired_vs_B0'] if (g['version'], g['red'], g['split']) == (version, red, split)}
        for label in sorted(set(groups)|set(('B0', 'B1', 'B2', 'B3', 'B4'))):
            group, pair = groups.get(label, {}), paired.get(label, {})
            row = [version, red, split, label, group.get('n', 0), group.get('mean'), group.get('sample_std'), group.get('total_execution_time_s'),
                   pair.get('mean_difference'), pair.get('win_rate'), pair.get('median_difference'), pair.get('paired_bootstrap_95ci')]
            lines.append('| '+' | '.join(fmt(v).replace('|', '/') for v in row)+' |')
    if not cases:
        for label in ('B0','B1','B2','B3-no-support','B3','B4'):
            lines.append('| registered | finite | test | '+label+' | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |')
        lines += ['', 'No completed real episodes. Mock or partial runs are not performance samples.']
    lines += ['', 'Small samples and bootstrap intervals do not establish a general performance gain. Environment completion and committed optimizer iterations are reported separately in run_status.json.', '']
    lines += ['', '| Comparison | Version | Red | Split | Paired N | Mean gain | Median | Win/tie/loss | 95% CI |',
        '|---|---|---|---|---|---|---|---|---|']
    for pair in metrics['paired_comparisons']:
        lines.append('| '+' | '.join(fmt(pair.get(k)).replace('|','/') for k in
            ('comparison','version','red','split','paired_n','mean_difference','median_difference','win_tie_loss','paired_bootstrap_95ci'))+' |')
    protocol=load(directory/'protocol.json') if (directory/'protocol.json').exists() else {}
    if protocol.get('protocol_kind')=='fixed-committed-update-verification-v1':
        lines+=['','## Fixed committed-update verification','',
            'B2 uses the exact historical completed update, evaluated on freshly registered test seeds. No new training loop is claimed. B3/B4 full training and 20-pair approval are N/A.',
            'The ASV/SSV ablations below screen one fixed historical whole update; they are not trained B3/B4 arms. A retained initial pool can reuse the matching B1 policy measurement, with zero additional episodes.','']
        for filename in ('action_support.json','selector_probe.json','fixed_update_rollout.json','fixed_update_ablations.json',
                         'scorer_parity.json','sg_boundary_profile.json','guard_policy_reuse.json','historical_forensics.json'):
            if (output/filename).exists():lines.append(f'- [{filename}]({filename})')
        screen=load(output/'fixed_update_rollout.json') if (output/'fixed_update_rollout.json').exists() else {}
        if screen:lines+=['',f"Reject-only validation: {screen.get('reason')}; paired seeds={screen.get('paired_seeds')}; differences={screen.get('differences')}. Partial validation never approves."]
    (output/'performance.md').write_text('\n'.join(lines))
    optimizer_report(directory,output,statuses)
    from scripts.audit_training_cost import audit
    audit(directory)
    reproduction_bundle(directory,output)
    return {'episodes': len(episodes), 'runs': len(statuses), 'rollout_decisions': len(decisions), 'output': str(output)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--reasoning', action='store_true', help='Also read available raw traces for Qwen-only behavior counts')
    args = parser.parse_args()
    print(json.dumps(report_experiment(args.directory, args.output, args.reasoning)))
