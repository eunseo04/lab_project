"""Recompute separated summary statistics using only public compact episode data."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
GROUP_FIELDS=('version','red','split','communication_encoder','action_contract','evaluation_scope')
CHECKPOINT_FIELDS=('base_arm','checkpoint_iteration','method_sha256','training_lineage_id','checkpoint_sha256','pool_sha256')


def episode_group(episode):
    # Old compact exports omitted the local encoder identifier.
    return (episode['version'],episode['red'],episode['split'],
            episode.get('communication_encoder','legacy_local_v1'),episode.get('action_contract','legacy'),
            episode.get('evaluation_scope',episode['split']))


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
        if episode.get('status')!='completed' or episode.get('kind')!='real_cage4' or episode.get('unit_test_fixture_only'):continue
        if not math.isfinite(episode['joint_return']):raise ValueError('Nonfinite completed episode')
        grouped[(*episode_group(episode),episode['label'])].append(episode)
    rows=[];paired=[]
    for key,items in sorted(grouped.items()):
        seeds=[e['seed'] for e in items]
        identities=[(e.get('comparison_run_id','legacy'),e.get('training_run_id'),e['seed']) for e in items]
        if len(identities)!=len(set(identities)):raise ValueError('Duplicate episode seed within one evaluation/training run')
        checkpoint_scope={}
        if any(e.get('checkpoint_iteration') is not None for e in items):
            scopes={(e.get('base_arm'),e.get('checkpoint_iteration'),e.get('method_sha256')) for e in items}
            if len(scopes)!=1:
                raise ValueError('Different checkpoint methods or iterations cannot share one evaluation group')
            checkpoint_scope=dict(zip(CHECKPOINT_FIELDS[:3],scopes.pop()))
            if key[-1]!=f"B2@{checkpoint_scope['checkpoint_iteration']}" or checkpoint_scope['base_arm']!='B2':
                raise ValueError('Checkpoint evaluation label does not match its iteration')
            by_run={}
            for episode in items:
                policy=tuple(episode.get(field) for field in CHECKPOINT_FIELDS)
                if None in policy or by_run.setdefault(episode.get('comparison_run_id','legacy'),policy)!=policy:
                    raise ValueError('Different or unidentified checkpoint Pools share one comparison run')
        rewards=[e['joint_return'] for e in items]
        for episode in items:
            series=episode.get('scored_step_rewards')
            if series is not None and sum(series)!=episode['joint_return']:raise ValueError('Step reward total mismatch')
            components=episode.get('reward_components')
            if components is not None and sum(components.values())!=episode['joint_return']:raise ValueError('Component total mismatch')
        rows.append(dict(zip((*GROUP_FIELDS,'label'),key),**checkpoint_scope,n=len(items),mean=statistics.mean(rewards),
            sample_std=statistics.stdev(rewards) if len(items)>1 else None,
            total_execution_time_s=sum(e['execution_time_s'] for e in items),
            rewards_by_seed={e['seed']:e['joint_return'] for e in items} if len(seeds)==len(set(seeds)) else None,
            unique_eval_seed_count=len(set(seeds)),
            rewards_by_run_seed=[{'comparison_run_id':e.get('comparison_run_id','legacy'),
                'training_run_id':e.get('training_run_id'),'seed':e['seed'],'reward':e['joint_return'],
                **{field:e[field] for field in CHECKPOINT_FIELDS if field in e}} for e in items]))
        incumbent=(*key[:-1],'B0')
        if key[-1]=='B0' or incumbent not in grouped:continue
        common,before,after=paired_episodes(grouped[incumbent],items)
        delta=[b['joint_return']-a['joint_return'] for a,b in zip(before,after)]
        paired.append(dict(zip((*GROUP_FIELDS,'label'),key),paired_n=len(common),seeds=[k[1] for k in common],
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
