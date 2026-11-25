"""Proposed-only environment verification; the original PPO gate is unchanged."""
from copy import deepcopy
from .progress import BudgetExceeded, IncompleteOptimization
import hashlib
import json
import math
import statistics


def pool_digest(snapshot):
    return hashlib.sha256(json.dumps(snapshot,sort_keys=True).encode()).hexdigest()


def _checked_episode(episode):
    if episode.get('status')!='completed':return False
    if episode.get('kind')!='real_cage4':raise ValueError('Only real CAGE4 validation can admit a skill')
    if episode['joint_transitions']!=499 or episode['scored_transitions']!=498:raise ValueError('Validation must use official 500-step arithmetic')
    if not math.isfinite(episode['joint_return']):raise ValueError('Nonfinite reward')
    if sum(episode['reward_components'].values())!=episode['joint_return']:raise ValueError('Reward decomposition does not match the official total')
    return True


def reject_screen(previous,candidate,rule):
    """A necessary worst-seed failure can reject early; partial data can NEVER approve."""
    old={r['seed']:r for r in previous};new={r['seed']:r for r in candidate}
    prefix=rule['seeds'][:rule.get('reject_screen_pairs',0)]
    if not prefix or len(old)!=len(previous) or len(new)!=len(candidate):raise ValueError('Invalid reject-screen records')
    if set(old)!=set(new) or not set(old)<=set(prefix):raise ValueError('Unpaired or unregistered reject-screen seed')
    if any(not _checked_episode(r) for r in [*old.values(),*new.values()]):raise IncompleteOptimization('Reject-screen episode incomplete')
    delta={s:new[s]['joint_return']-old[s]['joint_return'] for s in old}
    failed={s:d for s,d in delta.items() if d < -rule['maximum_seed_loss']}
    return {'accepted':False,'decision_type':'rejected' if failed else 'deferred',
        'reason':'failed_registered_worst_seed_early' if failed else 'reject_screen_inconclusive',
        'validation_complete':False,'paired_seeds':sorted(old),'differences':delta,
        'irreversible_registered_failures':failed,'required_approval_pairs':len(rule['seeds']),
        'acceptance_criteria_unchanged':True,'early_approval_allowed':False,
        'proof':'A completed seed below the registered worst-loss limit cannot be repaired by additional seeds.'}


def proposed_pool(pool,candidate,rule):
    """One exact transaction constructor shared by selection probes and real rollout."""
    proposed=deepcopy(pool);proposed.add_skill(deepcopy(candidate))
    if rule.get('parent'):next(s for s in proposed.skills if s.name==rule['parent']).last_evolved_iter=rule['iteration']
    from .paper_algorithm import maintain_paper
    maintenance=maintain_paper(proposed,capacity_only=True)
    return proposed,maintenance


def compare_rollouts(previous,candidate,rule):
    """Keep the previous pool unless a complete paired validation passes."""
    old={r['seed']:r for r in previous};new={r['seed']:r for r in candidate}
    expected=set(rule['seeds'])
    if not expected or len(expected)!=len(rule['seeds']):raise ValueError('Nonempty unique validation seeds required')
    if len(expected)<rule.get('minimum_pairs',len(expected)):raise ValueError('Registered minimum approval pair count was reduced')
    if len(old)!=len(previous) or len(new)!=len(candidate):raise ValueError('Duplicate validation seed')
    if set(old)!=expected or set(new)!=expected:
        return {'accepted':False,'reason':'incomplete_validation','previous_n':len(old),'candidate_n':len(new)}
    for seed in expected:
        for episode in [old[seed],new[seed]]:
            if not _checked_episode(episode):
                return {'accepted':False,'reason':'incomplete_episode_status','seed':seed}
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


def verify_transaction(pool,candidate,evaluate,rule,evaluate_pair=None,proposed_snapshot=None):
    """Evaluate independent snapshots; never mutate the live pool on rejection."""
    before=deepcopy(pool.snapshot());original_hash=pool_digest(before)
    if proposed_snapshot is None:
        proposed,maintenance=proposed_pool(pool,candidate,rule)
    else:
        if rule.get('scope')!='fixed_committed_update':raise ValueError('Exact snapshot override requires a separately registered fixed-update study')
        proposed=deepcopy(pool);proposed.restore(deepcopy(proposed_snapshot));maintenance='already_committed_in_original_training'
        if not any(s.__dict__==candidate.__dict__ for s in proposed.skills):raise ValueError('Registered candidate missing from exact committed snapshot')
    try:
        result=None
        if rule.get('reject_screen_pairs',0):
            if evaluate_pair is None:raise ValueError('Paired evaluator required for reject-only prefix')
            old_prefix=[];new_prefix=[]
            for seed in rule['seeds'][:rule['reject_screen_pairs']]:
                a,b=evaluate_pair(deepcopy(before),deepcopy(proposed.snapshot()),seed)
                old_prefix.append(a);new_prefix.append(b);screen=reject_screen(old_prefix,new_prefix,rule)
                if screen['decision_type']=='rejected':result=screen;break
            if result is None and rule.get('reject_screen_only',False):result=screen
        if result is None:
            previous=evaluate(deepcopy(before),'previous')
            measured=evaluate(deepcopy(proposed.snapshot()),'candidate')
            result=compare_rollouts(previous,measured,rule)
    except (BudgetExceeded, IncompleteOptimization):
        if pool_digest(pool.snapshot())!=original_hash:pool.restore(before)
        raise
    except Exception as exc:
        result={'accepted':False,'reason':'validation_error','error':f'{type(exc).__name__}: {exc}'}
    if pool_digest(pool.snapshot())!=original_hash:
        pool.restore(before)
        raise RuntimeError('Validator mutated the live pool; rolled back')
    result.update(previous_pool_sha256=original_hash,candidate_pool_sha256=pool_digest(proposed.snapshot()),
        live_pool_unchanged_during_validation=True,maintenance_before_validation=maintenance)
    if result['accepted']:result['verified_pool']=proposed.snapshot()
    return result
