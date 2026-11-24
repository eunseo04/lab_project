"""Proposed-only environment verification; the original PPO gate is unchanged."""
from copy import deepcopy
import hashlib
import json
import math
import statistics


def pool_digest(snapshot):
    return hashlib.sha256(json.dumps(snapshot,sort_keys=True).encode()).hexdigest()


def compare_rollouts(previous,candidate,rule):
    """Keep the previous pool unless a complete paired validation passes."""
    old={r['seed']:r for r in previous};new={r['seed']:r for r in candidate}
    expected=set(rule['seeds'])
    if not expected or len(expected)!=len(rule['seeds']):raise ValueError('Nonempty unique validation seeds required')
    if len(old)!=len(previous) or len(new)!=len(candidate):raise ValueError('Duplicate validation seed')
    if set(old)!=expected or set(new)!=expected:
        return {'accepted':False,'reason':'incomplete_validation','previous_n':len(old),'candidate_n':len(new)}
    for seed in expected:
        for episode in [old[seed],new[seed]]:
            if episode.get('status')!='completed':
                return {'accepted':False,'reason':'incomplete_episode_status','seed':seed}
            if episode.get('kind')!='real_cage4':raise ValueError('Only real CAGE4 validation can admit a skill')
            if episode['joint_transitions']!=499 or episode['scored_transitions']!=498:raise ValueError('Validation must use official 500-step arithmetic')
            if not math.isfinite(episode['joint_return']):raise ValueError('Nonfinite reward')
            if sum(episode['reward_components'].values())!=episode['joint_return']:raise ValueError('Reward decomposition does not match the official total')
    seeds=sorted(expected);delta=[new[s]['joint_return']-old[s]['joint_return'] for s in seeds]
    old_values=[old[s]['joint_return'] for s in seeds];new_values=[new[s]['joint_return'] for s in seeds]
    old_sd=statistics.stdev(old_values) if len(seeds)>1 else 0.
    new_sd=statistics.stdev(new_values) if len(seeds)>1 else 0.
    green_delta={kind:statistics.mean(new[s]['reward_components'].get(kind,0)-old[s]['reward_components'].get(kind,0) for s in seeds) for kind in ['LWF','ASF']}
    checks={'positive_mean':statistics.mean(delta)>rule['minimum_mean_gain'],
        'win_rate':sum(x>0 for x in delta)/len(delta)>=rule['minimum_win_rate'],
        'median':statistics.median(delta)>=rule['minimum_median_gain'],
        'worst_seed':min(delta)>=-rule['maximum_seed_loss'],
        'green_local_work':green_delta['LWF']>=-rule['maximum_mean_lwf_loss'],
        'green_access_service':green_delta['ASF']>=-rule['maximum_mean_asf_loss'],
        'spread':new_sd<=max(old_sd*rule['maximum_std_ratio'],old_sd+rule['maximum_std_increase'])}
    return {'accepted':all(checks.values()),'reason':'passed' if all(checks.values()) else 'failed_registered_criteria',
        'checks':checks,'seeds':seeds,'differences':delta,'mean_difference':statistics.mean(delta),
        'median_difference':statistics.median(delta),'win_rate':sum(x>0 for x in delta)/len(delta),
        'previous_mean':statistics.mean(old_values),'candidate_mean':statistics.mean(new_values),
        'previous_std':old_sd,'candidate_std':new_sd,'green_component_differences':green_delta}


def verify_transaction(pool,candidate,evaluate,rule):
    """Evaluate independent snapshots; never mutate the live pool on rejection."""
    before=deepcopy(pool.snapshot());original_hash=pool_digest(before)
    proposed=deepcopy(pool)
    proposed.add_skill(deepcopy(candidate))
    if rule.get('parent'):
        next(s for s in proposed.skills if s.name==rule['parent']).last_evolved_iter=rule['iteration']
    from .paper_algorithm import maintain_paper
    maintenance=maintain_paper(proposed,capacity_only=True)
    try:
        previous=evaluate(deepcopy(before),'previous')
        measured=evaluate(deepcopy(proposed.snapshot()),'candidate')
        result=compare_rollouts(previous,measured,rule)
    except Exception as exc:
        result={'accepted':False,'reason':'validation_error','error':f'{type(exc).__name__}: {exc}'}
    if pool_digest(pool.snapshot())!=original_hash:
        pool.restore(before)
        raise RuntimeError('Validator mutated the live pool; rolled back')
    result.update(previous_pool_sha256=original_hash,candidate_pool_sha256=pool_digest(proposed.snapshot()),
        live_pool_unchanged_during_validation=True,maintenance_before_validation=maintenance)
    if result['accepted']:result['verified_pool']=proposed.snapshot()
    return result
