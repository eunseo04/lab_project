"""Auxiliary, offline diagnostics; NEVER a reward or advantage replacement."""
from collections import Counter, defaultdict

def support_statistics(experiences):
    rows = [t for e in experiences for t in e.transitions]
    valid = [t for t in rows if t.get('policy_invoked') and t.get('request_applied') and not t.get('parse_error') and t.get('token_ids')]
    episodes = {(e.env_name, t.get('seed'), t.get('episode_id', 0)) for e in experiences for t in e.transitions if t.get('seed') is not None}
    phases = Counter(str(t.get('mission_phase', 'unknown')) for t in valid)
    outcomes = Counter()
    for row in rows:
        op = (row.get('behavior_outcome') or {}).get('completed_action')
        if op:
            outcomes[str(op.get('status', 'UNKNOWN'))] += 1
        elif row.get('action_status') == 'IN_PROGRESS':
            outcomes['IN_PROGRESS_ticks'] += 1
    return {'independent_episode_count': len(episodes), 'episodes': [list(k) for k in sorted(episodes, key=str)],
            'valid_policy_decisions': len(valid), 'red_policies': sorted({e.env_name.rsplit(':', 1)[-1] for e in experiences}),
            'mission_phases': dict(phases), 'skill_activations': sum(bool(event.get('activated')) for t in rows for event in (t.get('skill_controller') or {}).get('events', [])),
            'outcomes': dict(outcomes), 'independence_unit': 'episode, never individual decisions or agents in the same episode'}


def evolution_statistics(experiences, records):
    grouped = defaultdict(list)
    for exp in experiences:
        grouped[exp.skill].append(exp)
    result = {}
    by_name = {r['parent']: r for r in records}
    for name, exps in grouped.items():
        row = by_name.get(name, {})
        candidates = row.get('candidates', [])
        result[name] = {**support_statistics(exps),
            'semantic_gradient_count': len(row.get('semantic_gradients', [])),
            'ppo_gate_positive_candidates': sum(c.get('score') is not None and c['score'] > 0 for c in candidates),
            'rollout_approved': sum(c.get('rollout_validation', {}).get('accepted') is True for c in candidates),
            'rollout_rejected': sum(c.get('rollout_validation', {}).get('accepted') is False for c in candidates),
            'automatic_updates': sum(bool(c.get('accepted')) for c in candidates),
            'deferred_reason': row.get('deferred'), 'error': row.get('error')}
    return result


def auxiliary_attribution(rows, reward_ticks):
    """Join simultaneous observations and official events without causal claims."""
    by_step = defaultdict(list)
    for row in rows:
        by_step[row['step']].append(row)
    ticks = {t['step']: t for t in reward_ticks}
    diagnostics = []
    for row in rows:
        if row.get('policy') != 'llm' or not row.get('policy_invoked'):
            continue
        from .behavior import chosen_action
        action, host = chosen_action(row)
        before = (row.get('observation_before') or {}).get(host, {})
        after = (row.get('observation_after') or {}).get(host, {})
        events = ticks.get(row['step'], {}).get('events', [])
        related = [e for e in events if e.get('host') == host]
        zones = sorted({e['zone'] for e in related if e.get('zone')})
        peers = [{k: peer.get(k) for k in ('agent', 'requested_action', 'executed_actions', 'action_status')}
                 for peer in by_step[row['step']] if peer.get('policy') == 'rl']
        diagnostics.append({'seed': row['seed'], 'episode_id': row['episode_id'], 'step': row['step'],
            'blue_agent': row['agent'], 'skill': row.get('skill_id'), 'skill_version': row.get('skill_version'),
            'target_host': host, 'target_zones_observed_in_reward_events': zones, 'mission_phase': row.get('mission_phase'),
            'action': action, 'request_applied': row.get('request_applied'), 'action_status': row.get('action_status'),
            'observation_before': before, 'observation_after': after,
            'observed_service_change': before.get('Services') != after.get('Services') if isinstance(before, dict) and isinstance(after, dict) else None,
            'same_host_official_events': related, 'all_step_green_red_reward_events': events,
            'simultaneous_keep_actions': peers, 'joint_reward': row['joint_reward'],
            'causal_credit': 'unidentified', 'policy_input': False})
    return {'kind': 'auxiliary_joint_reward_diagnostics', 'causal_estimate': False,
            'reward_unchanged': True, 'records': diagnostics,
            'limitation': 'Same-host/time co-occurrence does not identify individual Skill effects. Observed service changes cannot be attributed to Red without independent evidence.'}


def verify_counterfactual_snapshot(before_state, after_state, before_rng, after_rng):
    """A prerequisite check, not a claim that seed pairing is causal evidence."""
    return {'state_equal': before_state == after_state, 'rng_equal': before_rng == after_rng,
            'eligible': before_state == after_state and before_rng == after_rng,
            'causal_effect_estimated': False}
