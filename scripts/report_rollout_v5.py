"""Audit real budgeted results and render the requested tables/eight figures."""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];FORK=ROOT/'revisions/rollout-v5-static-pool-fix';OUT=ROOT/'research_outputs/rollout-v5'
sys.path.insert(0,str(FORK))
from research.gate import clipped_terms
from research.behavior import ObservableCaseLedger
from behavior_metrics import episode_metrics

LABELS={'B0':'ACD + History','B1':'Static Skill','B2':'Original Skill-Pro','B3':'Rollout Verification','B4':'Proposed'}
KINDS=['action_cost','RIA','LWF','ASF']


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic_json_report(path,value):
    temporary=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)
def avg(values):return statistics.mean(values) if values else None
def sd(values):return statistics.stdev(values) if len(values)>1 else None
def fmt(x):return '미실행/자료 없음' if x is None else str(x) if type(x) is int else f'{x:.2f}' if isinstance(x,float) else str(x)


def paired(a,b):
    old={e['seed']:e['return'] for e in a};new={e['seed']:e['return'] for e in b};seeds=sorted(old.keys()&new.keys())
    values=[new[s]-old[s] for s in seeds];ci=None
    if len(values)>1:
        rng=np.random.default_rng(20261120)
        samples=rng.choice(np.asarray(values,dtype=float),size=(10000,len(values)),replace=True).mean(axis=1)
        ci=np.quantile(samples,[.025,.975]).tolist()
    return {'n':len(seeds),'seeds':seeds,'differences':values,'mean':avg(values),
        'median':statistics.median(values) if values else None,'wins':sum(v>0 for v in values),
        'ties':sum(v==0 for v in values),'win_rate':sum(v>0 for v in values)/len(values) if values else None,
        'difference_95ci':ci,'ci_method':'paired-seed percentile bootstrap, 10000 resamples; N<2 undefined; small-N inference limited'}


def audit_case(task,protocol,selection):
    path=Path(task['summary']);data=read(path);episode=data['episodes'][0];config=data['config']
    assert data['status']=='completed' and data['pool_frozen']
    assert data['model_identity']==episode['model_identity']==protocol['fixed_model_identity']
    assert episode['kind']=='real_cage4' and episode['joint_transitions']==499 and episode['scored_transitions']==498
    assert config['composition']=='hybrid' and config['steps']==500 and config['scoring']=='official'
    assert config['behavior_context']=='evidence_v1' and config['connection_leads'] is True and config['action_contract']=='explicit'
    assert task['started_unix']>=selection['frozen_unix'] and task['seed']==episode['seed']
    assert episode['reward_audit_passed'] and sum(episode['reward_components'].values())==episode['joint_return']
    assert data['pool_sha256']==data['initial_pool_sha256']
    if task['label']!='B0':
        origin=data['training_provenance'];expected=selection['selection'][task['label']]['sha256']
        assert origin['sha256']==sha(origin['path'])==expected
    manifest=read(path.parent/'manifest.json');assert manifest['implementation_sha256']==protocol['source_sha256']
    ticks=defaultdict(list);rows=[json.loads(s) for s in Path(episode['trace']).read_text().splitlines()]
    for row in rows:ticks[row['step']].append(row)
    assert len(ticks)==499 and len(rows)==2495
    for group in ticks.values():
        assert {r['agent'] for r in group}=={f'blue_agent_{i}' for i in range(5)}
        assert len({(r['joint_reward'],r['score_included']) for r in group})==1
        assert sum(r['policy']=='llm' for r in group)==1
    assert sum(group[0]['joint_reward'] for group in ticks.values() if group[0]['score_included'])==episode['joint_return']
    events=[json.loads(line) for line in Path(episode['reward_events']).read_text().splitlines()]
    components=Counter()
    for tick in events:
        group=ticks[tick['step']]
        assert tick['joint_reward']==group[0]['joint_reward'] and tick['score_included']==group[0]['score_included']
        assert sum(tick['components'].values())==tick['joint_reward']
        if tick['score_included']:components.update(tick['components'])
    assert dict(components)==episode['reward_components']
    behavior=episode_metrics(episode,rows,ledger_factory=lambda agent:
        ObservableCaseLedger(agent,connection_leads=config['connection_leads']))['blue_agent_4']
    initial={(s['name'],s.get('version',1)) for s in read(OUT/'initial_pool.json')['pool']['skills']}
    behavior['learned_skill_decisions']=sum(bool(r.get('policy_invoked')) and r.get('skill_id') is not None
        and (r['skill_id'],r.get('skill_version')) not in initial for r in rows if r['agent']=='blue_agent_4')
    own=[r for r in rows if r['agent']=='blue_agent_4'];first=next(r['state'] for r in own if r.get('policy_invoked'))
    return {'id':task['id'],'label':task['label'],'red':task['red'],'seed':episode['seed'],'split':task['split'],
        'return':episode['joint_return'],'execution_time_s':episode['execution_time_s'],
        **{k:episode['reward_components'].get(k,0) for k in KINDS},'behavior':behavior,
        'reason_examples':[{'step':r['step'],'action':r['requested_action'],'reason':r.get('reason'),'skill':r.get('skill_id')}
            for r in own if r.get('policy_invoked')][:3],
        'source_summary':str(path),'source_summary_sha256':sha(path),'source_trace_sha256':sha(episode['trace'])},first


def audit_evolution():
    rows=[];gradients=[];histories=[]
    for label in ['B2','B3','B4']:
        recovered=OUT/'recovery'/label/'train/summary.json'
        path=recovered if recovered.exists() else OUT/label/'train/summary.json'
        if not path.exists():continue
        data=read(path)
        retry_seconds=0.
        if path==recovered:
            completion=read(path.parent/'completed.json');provenance=read(path.parent/'provenance.json')
            retry_seconds=completion['completed_unix']-provenance['started_unix']
        for history in data.get('history',[]):
            histories.append({'label':label,'loop':history['iteration'],'training_reward':history['reward_mean'],
                'pool_size':history['pool_size'],'evolution_seconds_including_rollout':history['seconds']+retry_seconds,
                'original_update_seconds':history['seconds'],'metadata_retry_validation_seconds':retry_seconds})
            for parent in history.get('evolution',[]):
                for g in parent.get('semantic_gradients',[]):
                    for audit in g.get('evidence_audits',[]):
                        gradients.append({'label':label,'loop':history['iteration'],'parent':parent['parent'],
                            'passed':audit['passed'],'errors':audit['errors'],'references':audit['valid_references']})
                for c in parent['candidates']:
                    score=c.get('score');diagnostics=c.get('diagnostics',[])
                    if score is not None:
                        terms=[]
                        for d in diagnostics:
                            ratios,clipped=clipped_terms(d['old_logprobs'],d['new_logprobs'],d['advantages'],.2,log_ratio_bound=None)
                            assert np.allclose(ratios,d['importance_ratios'],atol=1e-12,rtol=0)
                            assert np.allclose(clipped,d['clipped_surrogates'],atol=1e-12,rtol=0)
                            terms.append(float(clipped.mean()))
                        assert abs(statistics.mean(terms)-score)<1e-10
                    validation=c.get('rollout_validation',{})
                    if c.get('accepted') and label in ('B3','B4'):
                        assert validation.get('accepted') and validation['live_pool_unchanged_during_validation']
                        assert c['semantic_support']['passed'] and score>0
                    rows.append({'label':label,'loop':history['iteration'],'parent':parent['parent'],
                        'candidate':c.get('skill',{}).get('name'),'score':score,'accepted':c['accepted'],
                        'ppo_selected':bool(c.get('ppo_accepted') or c['accepted']),
                        'semantic_support_passed':c.get('semantic_support',{}).get('passed'),
                        'valid_decisions':parent['valid_behavior_decisions'],'error':c.get('error') or parent.get('error'),
                        'rollout_reason':validation.get('reason'),'validation_mean_difference':validation.get('mean_difference'),
                        'validation_n':len(validation.get('seeds',[])),'validation_win_rate':validation.get('win_rate'),
                        'validation_checks':validation.get('checks'),'validation_component_changes':validation.get('green_component_differences'),
                        'deferred':False})
                if parent.get('deferred'):rows.append({'label':label,'loop':history['iteration'],'parent':parent['parent'],
                    'candidate':None,'score':None,'accepted':False,'valid_decisions':parent['valid_behavior_decisions'],
                    'error':None,'rollout_reason':parent['deferred'],'validation_mean_difference':None,'validation_n':0,'validation_win_rate':None,'deferred':True})
    return rows,gradients,histories


def measured_validation(protocol):
    """Count completed real validations even when the surrounding update is uncommitted."""
    records=[]
    initial={(s['name'],s.get('version',1)) for s in read(OUT/'initial_pool.json')['pool']['skills']}
    paths=list(OUT.glob('B*/train/rollout-verification/*/*-seed-*/summary.json'))
    paths+=list(OUT.glob('recovery/B*/train/rollout-verification/*/*-seed-*/summary.json'))
    original=read(OUT/'protocol.json')
    for path in sorted(paths):
        data=read(path)
        if data['status']!='completed':continue
        episode=data['episodes'][0]
        assert episode['kind']=='real_cage4' and episode['joint_transitions']==499 and episode['scored_transitions']==498
        assert data['pool_frozen'] and data['initial_pool_sha256']==data['pool_sha256']
        assert episode['model_identity']==protocol['fixed_model_identity']
        assert sum(episode['reward_components'].values())==episode['joint_return']
        assert sum(episode['scored_step_rewards'])==episode['joint_return']
        is_recovery=path.relative_to(OUT).parts[0]=='recovery'
        expected=read(OUT/'metadata-repair-protocol.json')['source_sha256'] if is_recovery else original['source_sha256']
        assert read(path.parent/'manifest.json')['implementation_sha256']==expected
        provenance=data['training_provenance'];assert sha(provenance['path'])==provenance['sha256']
        label=path.relative_to(OUT).parts[1 if is_recovery else 0];role=path.parent.name.split('-seed-')[0]
        usage=Counter();actions=Counter();completions=Counter();learned=0
        for line in Path(episode['trace']).read_text().splitlines():
            row=json.loads(line)
            if row['agent']!='blue_agent_4':continue
            if row.get('policy_invoked') and row.get('request_applied') and not row.get('parse_error'):
                usage[row.get('skill_id') or 'NONE']+=1
                actions[row['requested_action'].split()[0]]+=1
                learned+=bool(row.get('skill_id')) and (row['skill_id'],row.get('skill_version')) not in initial
            completed=(row.get('behavior_outcome') or {}).get('completed_action')
            if completed:completions[completed['name']+'_'+completed['status']]+=1
        records.append({'label':label,'comparison':path.parent.parent.name,'role':role,'seed':episode['seed'],
            'return':episode['joint_return'],**{k:episode['reward_components'].get(k,0) for k in KINDS},
            'execution_time_s':episode['execution_time_s'],'skill_valid_decisions':dict(usage),
            'learned_skill_valid_decisions':learned,'valid_model_action_choices':dict(actions),
            'completed_own_operations':dict(completions),'source':str(path),'source_sha256':sha(path)})
    return records


def completed_training_counts():
    counts={}
    for label in ['B2','B3','B4']:
        cases=[]
        for path in (OUT/label/'train').glob('episode-????.json'):
            data=read(path)
            if data.get('status')=='completed' and data.get('kind')=='real_cage4':
                assert data['joint_transitions']==499 and sum(data['reward_components'].values())==data['joint_return']
                cases.append({'seed':data['seed'],'return':data['joint_return'],'source':str(path)})
        counts[label]=cases
    return counts


def training_skill_statistics(evolution):
    records=[]
    for label in ['B2','B3','B4']:
        recovered=OUT/'recovery'/label/'train/pool-iteration-0001.json'
        source=recovered if recovered.exists() else OUT/label/'train/pool-iteration-0001.json'
        if not source.exists():continue
        state=read(source);uses=defaultdict(lambda:{'ticks':0,'valid_decisions':0,'phases':Counter(),
            'valid_decision_phases':Counter(),'advantages_by_seed':defaultdict(list),
            'red_policies':Counter(),'completed_operations':Counter(),'rtgs':[],'advantages':[]})
        task=state['experience']['fifo'][0]['env_name'];baseline=state['baselines'][task]
        for path in (OUT/label/'train').glob('episode-????.jsonl'):
            for line in path.read_text().splitlines():
                r=json.loads(line)
                if r['agent']!='blue_agent_4' or not r.get('skill_id'):continue
                u=uses[r['skill_id']];u['ticks']+=1;u['phases'][str(r['mission_phase'])]+=1
                u['red_policies']['finite']+=1
                if r.get('policy_invoked') and r.get('request_applied') and not r.get('parse_error'):
                    u['valid_decisions']+=1;u['rtgs'].append(r['return_to_go']);u['advantages'].append(r['return_to_go']-baseline)
                    u['valid_decision_phases'][str(r['mission_phase'])]+=1
                    u['advantages_by_seed'][r['seed']].append(r['return_to_go']-baseline)
                completed=(r.get('behavior_outcome') or {}).get('completed_action')
                if completed:u['completed_operations'][completed['name']+'_'+completed['status']]+=1
        for skill in state['pool']['skills']:
            name=skill['name'];u=uses[name];candidates=[e for e in evolution if e['label']==label and e['parent']==name and e['score'] is not None]
            records.append({'label':label,'skill':name,'version':skill['version'],
                'frequency':skill['frequency'],'avg_gain':skill['avg_gain'],'total_gain':skill['total_gain'],
                'maturity':skill['maturity'],'last_evolved_iter':skill['last_evolved_iter'],
                'parent_id':skill['parent_id'],'ticks':u['ticks'],'valid_decisions':u['valid_decisions'],
                'phases':dict(u['phases']),'valid_decision_phases':dict(u['valid_decision_phases']),
                'red_policies':dict(u['red_policies']),
                'completed_operations':dict(u['completed_operations']),'mean_rtg':avg(u['rtgs']),
                'mean_advantage':avg(u['advantages']),'scored_proposals':len(candidates),
                'trajectory_equal_mean_advantage':avg([avg(values) for values in u['advantages_by_seed'].values()]),
                'positive_gate_proposals':sum(c['score']>0 for c in candidates),
                'admitted_proposals':sum(c['accepted'] for c in candidates),
                'interpretation':'Training phase statistics only. TRUE completion is not proven cleanup; joint RTG exposure is not individual causal credit.'})
    return records


def decision_and_skill_diagnostics(episodes):
    traces={};usage={}
    for episode in episodes:
        data=read(episode['source_summary']);own=[]
        for line in Path(data['episodes'][0]['trace']).read_text().splitlines():
            r=json.loads(line)
            if r['agent']!='blue_agent_4':continue
            own.append({'step':r['step'],'action':r['requested_action'],'reason':r.get('reason'),
                'skill':r.get('skill_id'),'observation_sha256':hashlib.sha256(json.dumps(r['observation_before'],sort_keys=True).encode()).hexdigest()})
            if not r.get('skill_id'):continue
            key=(episode['label'],r['skill_id']);item=usage.setdefault(key,{'label':key[0],'skill':key[1],
                'ticks':0,'valid_decisions':0,'phases':Counter(),'red_policies':Counter(),
                'completed_operations':Counter(),'rtg_sum':0.,'rtg_n':0})
            item['ticks']+=1;item['phases'][str(r['mission_phase'])]+=1;item['red_policies'][episode['red']]+=1
            valid=r.get('policy_invoked') and r.get('request_applied') and not r.get('parse_error')
            item['valid_decisions']+=bool(valid)
            if valid:item['rtg_sum']+=r['return_to_go'];item['rtg_n']+=1
            completed=(r.get('behavior_outcome') or {}).get('completed_action')
            if completed:item['completed_operations'][completed['name']+'_'+completed['status']]+=1
        traces[(episode['red'],episode['seed'],episode['label'])]=own
    differences=[]
    for (red,seed,label),candidate in traces.items():
        if label=='B0' or (red,seed,'B0') not in traces:continue
        baseline=traces[(red,seed,'B0')];patterns=Counter();first=None
        for a,b in zip(baseline,candidate):
            if a['action']==b['action']:continue
            patterns[(a['action'] or 'None').split()[0]+' -> '+(b['action'] or 'None').split()[0]]+=1
            if first is None:first={'step':a['step'],'baseline':a,'candidate':b,
                'own_observation_equal':a['observation_sha256']==b['observation_sha256']}
        differences.append({'red':red,'seed':seed,'label':label,'first_request_difference':first,
            'later_request_patterns':dict(patterns),'causal_limitation':'After a branch, same seed does not imply identical simulator state.'})
    skills=[]
    for item in usage.values():
        item['mean_joint_rtg_during_valid_decisions']=item.pop('rtg_sum')/item['rtg_n'] if item['rtg_n'] else None
        for key in ['phases','red_policies','completed_operations']:item[key]=dict(item[key])
        item['interpretation']='TRUE means observed action completion, not verified cleanup; joint RTG is exposure, not individual causal credit.'
        skills.append(item)
    return differences,skills


def csvwrite(path,rows):
    if not rows:return
    temporary=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w',newline='') as stream:
        keys=list(dict.fromkeys(k for row in rows for k in row));writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows)
    temporary.replace(path)


def figures(report,metrics,episodes,pairs,evolution,histories):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    finite=[m for m in metrics if m['red']=='finite' and m['n']]
    def save(fig,name):
        temporary=report/f'.{name}.{os.getpid()}.png'
        fig.tight_layout();fig.savefig(temporary,dpi=170);plt.close(fig);temporary.replace(report/name)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for i,m in enumerate(finite):ax.bar(i,m['mean'],yerr=m['std'] or 0,capsize=4,label=f"{m['label']} N={m['n']}")
    ax.set_xticks(range(len(finite)),[f"{m['label']}\nN={m['n']}" for m in finite]);ax.set_ylabel('Official joint return (higher is better)');ax.set_title('FiniteState / frozen final pools / mean ± sample SD');save(fig,'01_reward.png')
    fig,ax=plt.subplots(figsize=(9,4.5))
    for p in pairs:
        if p['red']=='finite' and p['n']:ax.plot(p['seeds'],p['differences'],'o-',label=p['label'])
    ax.axhline(0,color='black',lw=1);ax.set_xlabel('Held-out scenario seed');ax.set_ylabel('Paired return difference versus B0');ax.legend();save(fig,'02_paired.png')
    fig,ax=plt.subplots(figsize=(8,4.5));bottom=np.zeros(len(finite))
    for kind in KINDS:
        values=np.array([-m[kind] for m in finite]);ax.bar(range(len(finite)),values,bottom=bottom,label=kind);bottom+=values
    ax.set_xticks(range(len(finite)),[m['label'] for m in finite]);ax.set_ylabel('Original penalty magnitude (lower is better)');ax.legend();save(fig,'03_reward_breakdown.png')
    fig,ax=plt.subplots(figsize=(8,4.5))
    for label in ['B2','B3','B4']:
        items=[h for h in histories if h['label']==label]
        if items:ax.plot([h['loop'] for h in items],[h['training_reward'] for h in items],'o-',label=label)
    ax.set_xlabel('Completed evolution loop');ax.set_ylabel('Pre-update training batch joint return');ax.set_title('Actual completed loops only; different training/test roles');ax.legend();save(fig,'04_evolution.png')
    fig,ax=plt.subplots(figsize=(8,4.5));shown=False
    for r in evolution:
        if r['score'] is not None and r['validation_mean_difference'] is not None:
            ax.scatter(r['score'],r['validation_mean_difference'],marker='o' if r['accepted'] else 'x',label=f"{r['label']} {r['candidate']}");shown=True
    ax.axhline(0,color='black',lw=1);ax.set_xlabel('Unchanged PPO gate J');ax.set_ylabel('Actual paired validation return difference')
    if shown:ax.legend(fontsize=7)
    else:ax.text(.5,.5,'No completed PPO/rollout pairs',ha='center',va='center',transform=ax.transAxes)
    save(fig,'05_gate_vs_rollout.png')
    fig,ax=plt.subplots(figsize=(9,4.5));actions=['Analyse','DeployDecoy','Remove','Restore','Sleep','BlockTrafficZone','AllowTrafficZone']
    for index,m in enumerate(finite):
        items=[e for e in episodes if e['label']==m['label'] and e['red']=='finite'];counts=Counter()
        for e in items:counts.update(e['behavior']['valid_generated_choice_actions'])
        total=sum(counts.values()) or 1;x=np.arange(len(actions))+(index-(len(finite)-1)/2)*.15
        ax.bar(x,[counts[a]/total for a in actions],width=.15,label=m['label'])
    ax.set_xticks(range(len(actions)),actions,rotation=20);ax.set_ylabel('Fraction of valid model-selected requests');ax.legend();save(fig,'06_actions.png')
    fig,ax=plt.subplots(figsize=(9,4.5));reds=['finite','aggressive','stealthy','impact','degrade']
    for label in ['B0','B2','B4']:
        values=[next((m['mean'] for m in metrics if m['label']==label and m['red']==red),None) for red in reds]
        ax.plot(reds,[np.nan if v is None else v for v in values],'o-',label=label)
    ax.set_ylabel('Official mean joint return');ax.set_title('Unrun policies are gaps, never zero scores');ax.legend();save(fig,'07_red_policies.png')
    fig,ax=plt.subplots(figsize=(8,4.5));labels=['B2','B3','B4'];categories=['PPO scored','Rollout compared','Admitted','Deferred parents']
    for index,category in enumerate(categories):
        values=[]
        for label in labels:
            items=[r for r in evolution if r['label']==label]
            values.append(sum((r['score'] is not None) if category=='PPO scored' else
                (r['validation_mean_difference'] is not None) if category=='Rollout compared' else
                r['accepted'] if category=='Admitted' else r.get('deferred',False) for r in items))
        ax.bar(np.arange(3)+(index-1.5)*.2,values,width=.2,label=category)
    ax.set_xticks(range(3),labels);ax.set_ylabel('Actual count');ax.legend(fontsize=8);save(fig,'08_admission_history.png')


def main(require_terminal=False):
    protocol=read(OUT/'protocol.json');repair=read(OUT/'metadata-repair-protocol.json');static=read(OUT/'static-pool-repair-protocol.json')
    for name,value in protocol['source_sha256'].items():assert sha(ROOT/'revisions/rollout-v5'/name)==value
    assert sha(ROOT/'scripts/rollout_v5_campaign.py')==protocol['driver_sha256']
    for name,value in repair['source_sha256'].items():assert sha(ROOT/'revisions/rollout-v5-metadata-fix'/name)==value
    assert sha(ROOT/'scripts/recover_rollout_v5.py')==repair['recovery_driver_sha256']
    assert sha(ROOT/'scripts/rollout_v5_repaired_campaign.py')==repair['campaign_driver_sha256']
    assert sha(OUT/'protocol.json')==repair['original_protocol_sha256']
    for name,value in static['source_sha256'].items():assert sha(FORK/name)==value
    assert sha(ROOT/'scripts/rollout_v5_final_campaign.py')==static['final_driver_sha256']
    assert sha(OUT/'metadata-repair-protocol.json')==static['metadata_repair_protocol_sha256']
    assert static['test_rewards_used'] is False
    for name,value in protocol['preserved_previous_results_sha256'].items():assert sha(ROOT/name)==value
    protocol={**protocol,'source_sha256':static['source_sha256']}
    queue=read(OUT/'queue.json') if (OUT/'queue.json').exists() else []
    selection=read(OUT/'selection.json') if (OUT/'selection.json').exists() else None
    episodes=[];prompts={};groups=defaultdict(list)
    for task in queue:
        if task['status']!='completed':continue
        episode,prompt=audit_case(task,protocol,selection);episodes.append(episode)
        groups[(task['red'],task['label'])].append(episode);prompts[(task['red'],task['seed'],task['label'])]=prompt
    metrics=[];pairs=[];common_prompts=0
    for red in ['finite','aggressive','stealthy','impact','degrade']:
        for label in LABELS:
            items=groups[(red,label)];values=[e['return'] for e in items]
            metrics.append({'red':red,'label':label,'n':len(items),'mean':avg(values),'std':sd(values),
                'execution_time_s':sum(e['execution_time_s'] for e in items) if items else None,
                **{k:avg([e[k] for e in items]) for k in KINDS}})
            if label!='B0':
                p={'red':red,'label':label,**paired(groups[(red,'B0')],items)};pairs.append(p)
                for seed in p['seeds']:
                    assert prompts[(red,seed,'B0')]==prompts[(red,seed,label)];common_prompts+=1
    evolution,gradients,histories=audit_evolution()
    decisions,skill_usage=decision_and_skill_diagnostics(episodes)
    validation=measured_validation(protocol);training=completed_training_counts()
    pool_statistics=training_skill_statistics(evolution)
    terminal=(OUT/'campaign_complete.json').exists()
    complete=read(OUT/'campaign_complete.json') if terminal else None
    if terminal:assert complete['elapsed_model_campaign_seconds']<=8*3600
    result={'status':'budgeted_campaign_complete' if terminal else 'running','episodes':episodes,'metrics':metrics,'paired':pairs,
        'evolution':evolution,'semantic_evidence_audits':gradients,'training_histories':histories,
        'actual_validation_episodes':validation,'actual_training_episodes':training,
        'decision_comparisons':decisions,'skill_usage_by_phase_and_red':skill_usage,'training_pool_statistics':pool_statistics,
        'queue_status_counts':dict(Counter(t['status'] for t in queue)),'selection':selection,
        'metadata_repair':repair,'static_pool_loading_repair':static,
        'audit':{'passed':True,'real_500_step_final_episodes':len(episodes),'common_initial_actor_prompt_pairs':common_prompts,
            'prior_files_preserved':len(protocol['preserved_previous_results_sha256']),'weights_frozen':True,
            'official_reward_unchanged':True,'source_unchanged_after_registration':True,'no_mock_performance':True},
        'limitation':'Eight-hour resource limit; planned100 seeds per finite arm is not inferred from partial actual N. Small-N bootstrap intervals do not certify general superiority.'}
    report=OUT/'report';report.mkdir(exist_ok=True)
    atomic_json_report(report/'results.json',result)
    for name,rows in [('performance.csv',metrics),('paired.csv',pairs),('evolution.csv',evolution),
        ('episodes.csv',[{k:v for k,v in e.items() if k not in ('behavior','reason_examples')} for e in episodes]),
        ('validation_episodes.csv',validation),('skill_usage.csv',skill_usage),('training_pool_statistics.csv',pool_statistics),('decision_comparisons.csv',decisions)]:csvwrite(report/name,rows)
    lines=['# Rollout 검증을 추가한 B0–B4 실제 비교','',
        f"상태: {result['status']}. 실제 최종 평가 {len(episodes)}회. 각 500-step, KEEP 4 + 고정 Qwen3-8B 1. 새 보상·모델 가중치 학습 없음.",
        '',f"별도 실제 훈련 {sum(map(len,training.values()))}회, 후보 검증 {len(validation)}회가 완료됐다. 훈련·검증·알려진 시드 회귀·40-step 점검은 최종 평가 N에 더하지 않는다.",
        '', '검증 Pool 메타데이터의 episodes_completed 누락으로 최초 worker들이 평가 시작 전에 실패했다. 원본 소스·실패·후보·훈련 결과를 보존하고 metadata-fix revision에서 필수 count 두 필드만 추가했다. 같은 후보 snapshot hash, 검증 시드, 기준과 최초 8시간 종료 시각을 유지한 재검증이다. 모델·정책·보상은 바꾸지 않았다.',
        '', '최종 테스트 전 B1의 지정 Pool 로드 누락도 수정했다. B1은 기본 6개가 아닌 등록한 공통 수동 7개 Skill을 실제로 읽는다. 별도 static-pool-fix revision의 변경은 이 E1 로드 조건 한 줄이며 E0/E2/E3 동작·진행 중 검증은 같다. 139개 회귀 검사를 통과했고 수정 시점에 최종 테스트는 없었다.',
        '', '## Table 1. FiniteState','',
        '| 지표 | B0 ACD+History | B1 Static | B2 Original Skill-Pro | B3 Rollout | B4 Proposed |','|---|---:|---:|---:|---:|---:|']
    fin={m['label']:m for m in metrics if m['red']=='finite'}
    for title,key in [('실제 N','n'),('Mean Joint Reward','mean'),('Reward sample SD','std'),('Execution time 합계(s)','execution_time_s')]:lines.append('| '+title+' | '+' | '.join(fmt(fin[l][key]) for l in LABELS)+' |')
    lines+=['','실행 시간은 병렬 서버 대기까지 포함한 실제 평가 시간이다. 모델 자체 추론 속도의 인과 비교가 아니다.',
        '', '| 비교: Bx−B0 | 대응 N | 평균 차이 | 개선 비율 | 중앙값 | 대응 차이 95% CI |','|---|---:|---:|---:|---:|---|']
    for p in pairs:
        if p['red']=='finite':lines.append(f"| {p['label']} | {p['n']} | {fmt(p['mean'])} | {fmt(p['win_rate'])} | {fmt(p['median'])} | {p['difference_95ci']} |")
    lines+=['','CI는 고정 난수의 대응 시드 percentile bootstrap 10,000회다. N<2에서는 정의하지 않으며 적은 시드의 CI를 확정적 통계 우위로 해석하지 않는다.',
        '', '## Table 2. Red Strategy','', '| Red | B0 N / 평균 | B2 N / 평균 | B4 N / 평균 | B4−B0 대응 N / 차이 |','|---|---|---|---|---|']
    for red in ['finite','aggressive','stealthy','impact','degrade']:
        ms={m['label']:m for m in metrics if m['red']==red};p=next(p for p in pairs if p['red']==red and p['label']=='B4')
        lines.append('| '+red+' | '+' | '.join(f"{ms[l]['n']} / {fmt(ms[l]['mean'])}" for l in ['B0','B2','B4'])+f" | {p['n']} / {fmt(p['mean'])} |")
    lines+=['','## Table 3. FiniteState Reward Breakdown','', '| 원본 항목 | B0 | B4 | 차이 |','|---|---:|---:|---:|']
    for key in KINDS+['mean']:
        a,b=fin['B0'][key],fin['B4'][key];lines.append(f"| {key} | {fmt(a)} | {fmt(b)} | {fmt(b-a if a is not None and b is not None else None)} |")
    lines+=['','구성요소는 중복 없는 원본 BlueRewardMachine 사건과 action_cost다. N이 다르면 이 표의 단순 평균 차이는 대응 효과가 아니며 시드별 CSV를 확인한다.',
        '', '## Table 4. Skill Evolution','',
        '| Version | PPO Gate | 실제 검증 | 자동 채택 수 | Test Reward |','|---|---|---|---:|---|',
        '| Initial | 미적용 / 수동 7개 | 이번 후보 검증의 previous 역할로 별도 기록 | 0 | 중간 Pool 테스트로 선택하지 않음 |']
    for label in ['B2','B3','B4']:
        items=[r for r in evolution if r['label']==label and not r.get('deferred')]
        validations=[r for r in items if r['validation_mean_difference'] is not None]
        info='; '.join(f"N={r['validation_n']}, Δ={fmt(r['validation_mean_difference'])}" for r in validations) or '완료된 대응 검증 없음'
        lines.append(f"| {label} Loop 1 | {sum(r['score'] is not None for r in items)} 후보 점수 계산 | {info} | {sum(r['accepted'] for r in items)} | 최종 선택 결과만 아래 기록 |")
    lines+=['| Loop 2 | 이번 예산에서 미실행 | 미실행 | — | 미실행 |']
    for label in ['B2','B3','B4']:
        chosen=(selection or {}).get('selection',{}).get(label,{})
        lines.append(f"| {label} Final | 테스트 전 동결 | {chosen.get('learned_skill_versions','미동결')}개 학습 Skill version | — | N={fin[label]['n']}, 평균 {fmt(fin[label]['mean'])} |")
    lines+=['','후보별 승인·거부와 보류는 다음과 같다.','', '| Arm | Loop | Parent / Candidate | PPO J | 검증 N / 평균 차이 | 채택 | 거부·보류 |','|---|---:|---|---:|---|---|---|']
    for r in evolution:lines.append(f"| {r['label']} | {r['loop']} | {r['parent']} / {r['candidate']} | {fmt(r['score'])} | {r['validation_n']} / {fmt(r['validation_mean_difference'])} | {r['accepted']} | {r['rollout_reason'] or r['error']} |")
    lines+=['','이번 8시간 계획은 자동 갱신 loop 1회/학습 2 episodes를 등록했다. 추가 loop 2와 100시드 전체를 실행했다고 주장하지 않는다. Test Reward는 selection.json에 먼저 고정한 최종 Pool만 측정하며 학습 중 후보 선택에 사용하지 않는다.',
        '', '## 행동·Reasoning 진단','',
        '최초로 다른 primitive 요청을 고른 step, 그때의 두 reason·Skill·직전 자기 관측 hash를 decision_comparisons.csv에 기록했다. 후속 Sleep→Restore·Remove→Analyse 등의 패턴은 상태가 갈라진 뒤의 기술 통계이며 통제된 인과 실험으로 해석하지 않는다.',
        '', 'skill_usage.csv는 Skill별 실제 결정·phase·Red 정책·완료/실패 상태·공동 RTG 노출을 기록한다. TRUE 완료를 제거 성공으로 바꾸거나 공동 보상을 개인 보상으로 재정의하지 않는다. 유효 모델 선택, 파싱 fallback, 자동 busy Sleep·Monitor는 results.json에서 구분한다.',
        '', '## 공정성·해석','',
        'B0–B4는 같은 v4 관측/사건 이력·Sleep·primitive 메뉴를 공유한다. B1/B2/B3의 selector는 원본 경로이며 B4만 관측 기반 domain initiation 검증을 추가한다. B3/B4는 SG 근거 검증과 실제 rollout admission을 추가한 Proposed 확장이다. 기존 B2의 PPO 수식과 용량 관리는 바꾸지 않았다.',
        '', '근거 없는 수정·미완료 환경 검증은 이전 Pool을 유지한다. 희귀 Skill 최소 경험 수의 최적값은 한 후보 비교만으로 식별되지 않으므로 임의로 고정하지 않는다. 검증하지 못한 부모는 별도 보류로 기록한다.',
        '', f"원시 상태: {result['queue_status_counts']}. 중단·실패는 0점이나 성공 표본으로 바꾸지 않는다. 수동 초기 Skill, B4의 domain 규칙, 자동으로 채택한 새 Skill의 기여를 구분해야 한다.",
        '', '[사건별 성능 저하 원인](../../../reports/reward_root_cause.md), [Skill-Pro 계산 감사](../../../reports/skillpro_algorithm_audit.md).']
    if episodes:
        figures(report,metrics,episodes,pairs,evolution,histories)
        lines+=['','## Figures','']+[f'![{name}]({name})' for name in ['01_reward.png','02_paired.png','03_reward_breakdown.png','04_evolution.png','05_gate_vs_rollout.png','06_actions.png','07_red_policies.png','08_admission_history.png']]
    (report/'performance.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':result['status'],'final_episodes':len(episodes),'queue':result['queue_status_counts'],'gate_candidates_checked':sum(r['score'] is not None for r in evolution)},indent=2))
    if require_terminal and not terminal:raise SystemExit('Campaign has not reached its registered budget end')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--require-terminal',action='store_true');main(parser.parse_args().require_terminal)
