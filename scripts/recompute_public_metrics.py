"""Recompute separated summary statistics using only public compact episode data."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]


def recompute(episodes):
    grouped=defaultdict(list)
    for episode in episodes:
        if episode.get('status')!='completed' or episode.get('kind')!='real_cage4':continue
        if not math.isfinite(episode['joint_return']):raise ValueError('Nonfinite completed episode')
        grouped[tuple(episode[k] for k in ['version','red','split','label'])].append(episode)
    rows=[];paired=[]
    for key,items in sorted(grouped.items()):
        seeds=[e['seed'] for e in items]
        if len(seeds)!=len(set(seeds)):raise ValueError('Duplicate episode seed within version/red/split/label')
        rewards=[e['joint_return'] for e in items]
        for episode in items:
            series=episode.get('scored_step_rewards')
            if series is not None and sum(series)!=episode['joint_return']:raise ValueError('Step reward total mismatch')
            components=episode.get('reward_components')
            if components is not None and sum(components.values())!=episode['joint_return']:raise ValueError('Component total mismatch')
        rows.append(dict(zip(['version','red','split','label'],key),n=len(items),mean=statistics.mean(rewards),
            sample_std=statistics.stdev(rewards) if len(items)>1 else None,
            total_execution_time_s=sum(e['execution_time_s'] for e in items),
            rewards_by_seed={e['seed']:e['joint_return'] for e in items}))
        if key[-1]=='B0' or key[:3]+('B0',) not in grouped:continue
        baseline={e['seed']:e['joint_return'] for e in grouped[key[:3]+('B0',)]}
        current={e['seed']:e['joint_return'] for e in items}
        common=sorted(baseline.keys()&current.keys());delta=[current[s]-baseline[s] for s in common]
        paired.append(dict(zip(['version','red','split','label'],key),paired_n=len(common),seeds=common,
            differences=delta,mean_difference=statistics.mean(delta) if delta else None,
            win_rate=sum(d>0 for d in delta)/len(delta) if delta else None))
    return {'groups':rows,'paired_vs_B0':paired,'raw_trajectory_revalidation':False,
        'scope':'Summary arithmetic from published episode data, not an independent rerun of the model/environment.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes',type=Path,required=True,help='Compact episodes.json exported by report_experiment.py')
    args=parser.parse_args()
    value=json.loads(args.episodes.read_text())
    print(json.dumps(recompute(value['episodes']),ensure_ascii=False,indent=2))
