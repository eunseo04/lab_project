"""Compare new completed runs without historical results or model calls."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.recompute_public_metrics import recompute


def load(path):
    return json.loads(path.read_text())


def collect(directory, include_trace=False):
    directory = Path(directory).resolve()
    if directory == ROOT or not directory.is_dir():
        raise ValueError('Pass a specific experiment output directory')
    runs = sorted({p.parent for name in ('summary.json', 'manifest.json') for p in directory.rglob(name)
                   if 'rollout-verification' not in p.relative_to(directory).parts})
    episodes, statuses, signatures = [], [], {}
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
        if stage == 'eval' and label in ('B2', 'B3', 'B4') and not provenance.get('episodes_completed'):
            split = 'untrained_control'
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
            }, sort_keys=True)
            group = (version, config['red'], split)
            if group in signatures and signatures[group] != signature:
                raise ValueError('Comparison conditions differ within the same version/red/split')
            signatures[group] = signature
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
                         'completed_environment_episodes': completed, 'committed_iterations': len(summary.get('history', []))})
    return episodes, statuses


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
            q = statistics.quantiles(means, n=40, method='inclusive')
            pair['paired_bootstrap_95ci'] = [q[0], q[-1]]
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
        lines += ['', 'No completed real episodes. Mock or partial runs are not performance samples.']
    lines += ['', 'Small samples and bootstrap intervals do not establish a general performance gain. Environment completion and committed optimizer iterations are reported separately in run_status.json.', '']
    (output/'performance.md').write_text('\n'.join(lines))
    return {'episodes': len(episodes), 'runs': len(statuses), 'rollout_decisions': len(decisions), 'output': str(output)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--reasoning', action='store_true', help='Also read available raw traces for Qwen-only behavior counts')
    args = parser.parse_args()
    print(json.dumps(report_experiment(args.directory, args.output, args.reasoning)))
