"""Recompute separated summary statistics using only public compact episode data."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]


def paired_episodes(before,after):
    """Pair within a registered comparison run; never overwrite an equal eval seed."""
    def index(items):
        result={}
        for e in items:
            key=(e.get('comparison_run_id','legacy'),e['seed'])
            if key in result:raise ValueError('Ambiguous repeated eval seed: distinct comparison_run_id required for training repetitions')
            result[key]=e
        return result
    old,new=index(before),index(after);keys=sorted(old.keys()&new.keys())
    return keys,[old[k] for k in keys],[new[k] for k in keys]


def recompute(episodes):
    grouped=defaultdict(list)
    for episode in episodes:
        if episode.get('status')!='completed' or episode.get('kind')!='real_cage4':continue
        if not math.isfinite(episode['joint_return']):raise ValueError('Nonfinite completed episode')
        grouped[tuple(episode[k] for k in ['version','red','split','label'])].append(episode)
    rows=[];paired=[]
    for key,items in sorted(grouped.items()):
        seeds=[e['seed'] for e in items]
        identities=[(e.get('comparison_run_id','legacy'),e.get('training_run_id'),e['seed']) for e in items]
        if len(identities)!=len(set(identities)):raise ValueError('Duplicate episode seed within one evaluation/training run')
        rewards=[e['joint_return'] for e in items]
        for episode in items:
            series=episode.get('scored_step_rewards')
            if series is not None and sum(series)!=episode['joint_return']:raise ValueError('Step reward total mismatch')
            components=episode.get('reward_components')
            if components is not None and sum(components.values())!=episode['joint_return']:raise ValueError('Component total mismatch')
        rows.append(dict(zip(['version','red','split','label'],key),n=len(items),mean=statistics.mean(rewards),
            sample_std=statistics.stdev(rewards) if len(items)>1 else None,
            total_execution_time_s=sum(e['execution_time_s'] for e in items),
            rewards_by_seed={e['seed']:e['joint_return'] for e in items} if len(seeds)==len(set(seeds)) else None,
            unique_eval_seed_count=len(set(seeds)),
            rewards_by_run_seed=[{'comparison_run_id':e.get('comparison_run_id','legacy'),
                'training_run_id':e.get('training_run_id'),'seed':e['seed'],'reward':e['joint_return']} for e in items]))
        if key[-1]=='B0' or key[:3]+('B0',) not in grouped:continue
        common,before,after=paired_episodes(grouped[key[:3]+('B0',)],items)
        delta=[b['joint_return']-a['joint_return'] for a,b in zip(before,after)]
        paired.append(dict(zip(['version','red','split','label'],key),paired_n=len(common),seeds=[k[1] for k in common],
            paired_run_seeds=[{'comparison_run_id':k[0],'seed':k[1]} for k in common],
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
