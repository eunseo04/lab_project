"""Auxiliary, offline diagnostics; NEVER a reward or advantage replacement."""
from collections import Counter, defaultdict


def selection_distribution(rows, families=None):
    """Different denominators for decisions, option activations and execution ticks."""
    from .diagnosis import operation
    from .behavior import chosen_action
    families=families or {};free=Counter();names=Counter();family=Counter();activations=Counter()
    executed=Counter();starts=Counter();completed=Counter();seen=set();busy=0;invalid=0;total=0
    accepted=set();confirmed=set();requests=Counter()
    for row in rows:
        if row.get('policy')!='llm':continue
        total+=1
        for action in row.get('executed_actions',[]):
            if isinstance(action,dict):executed[action.get('name','unknown')]+=1
        op=operation(row);key=(row.get('seed'),row.get('agent'),op.get('name'),op.get('target'),op.get('started_at'))
        if op.get('completed_at') is not None and op.get('status') in ('TRUE','FALSE') and key not in seen:
            completed[op.get('name','unknown')+':'+op['status']]+=1;seen.add(key);confirmed.add(key)
        if not row.get('policy_invoked'):busy+=1;continue
        name=row.get('skill_id') or 'NONE';names[name]+=1;family[families.get(name,name)]+=1
        for event in (row.get('skill_controller') or {}).get('events',[]):
            if event.get('activated'):activations[event['activated']]+=1
        if row.get('parse_error'):invalid+=1;continue
        action=(row.get('requested_action') or 'unknown').split(' ',1)[0];free[action]+=1
        if row.get('request_applied'):
            actual,target=chosen_action(row);key=(row.get('seed'),row.get('agent'),actual,target,row['step'])
            accepted.add(key);requests[actual]+=1
            if any(a.get('name')==actual for a in row.get('executed_actions',[]) if isinstance(a,dict)):confirmed.add(key)
    for key in accepted&confirmed:starts[key[2]]+=1
    n=sum(names.values())
    return {'llm_ticks':total,'free_decisions':n,'busy_automatic_wait_ticks':busy,'invalid_responses':invalid,
        'requested_free_actions':dict(free),'skill_free_decisions':dict(names),'family_free_decisions':dict(family),
        'skill_decision_fractions':{k:v/n for k,v in names.items()} if n else {},
        'family_decision_fractions':{k:v/n for k,v in family.items()} if n else {},
        'skill_activation_events':dict(activations),'activation_denominator':sum(activations.values()),
        'actual_execution_ticks':dict(executed),'execution_tick_denominator':sum(executed.values()),
        'actual_free_action_starts':dict(starts),'observed_completed_calls':dict(completed),
        'accepted_free_requests':dict(requests),'execution_linkage':'Immediate actual action or later own completion confirms a free request, including asynchronous busy completions.',
        'frequency_unit':'A free decision is not a fresh Skill activation or a completed action.'}


def selection_shift(previous, candidate, rule=None):
    """Paired choice-only replay; a shift requests validation, not rejection."""
    rule=rule or {}
    if not previous or len(previous)!=len(candidate):raise ValueError('Aligned nonempty selector probes required')
    if any(a['state_id']!=b['state_id'] or a['sampling_seed']!=b['sampling_seed'] for a,b in zip(previous,candidate)):
        raise ValueError('Selector probe states/seeds differ')
    old=Counter(r['choice'] or 'NONE' for r in previous);new=Counter(r['choice'] or 'NONE' for r in candidate)
    names=set(old)|set(new);n=len(previous);tv=sum(abs(old[k]-new[k]) for k in names)/(2*n)
    disagreement=sum(a['choice']!=b['choice'] for a,b in zip(previous,candidate))/n
    lost=sorted(k for k,v in old.items() if k!='NONE' and v>=rule.get('minimum_previous_choices',3) and new[k]==0)
    concentration=max((new[k]-old[k])/n for k in names)
    checks={'choice_disagreement':disagreement>rule.get('maximum_disagreement',.25),
        'total_variation':tv>rule.get('maximum_total_variation',.2),
        'concentration_increase':concentration>rule.get('maximum_share_increase',.2),
        'previously_used_skill_disappeared':bool(lost)}
    return {'schema':'selection-shift-v1','paired_probe_n':n,'previous_counts':dict(old),'candidate_counts':dict(new),
        'previous_fractions':{k:old[k]/n for k in sorted(names)},'candidate_fractions':{k:new[k]/n for k in sorted(names)},
        'disagreement_rate':disagreement,'total_variation_distance':tv,'largest_share_increase':concentration,
        'previously_used_skills_absent':lost,'risk_flags':checks,'requires_rollout':any(checks.values()),
        'bad_policy_proven':False,'causal_reward_effect_estimated':False,
        'scope':'Same recorded states and RNG seeds, actual upstream epsilon/LLM selector; no environment reward estimate.'}


def trace_divergence(previous, candidate, previous_rewards=(), candidate_rewards=()):
    """Locate observable divergence and joint reward events; never invent missing Red/Green actions."""
    from .progress import digest
    old={(r['step'],r['agent']):r for r in previous};new={(r['step'],r['agent']):r for r in candidate}
    if set(old)!=set(new):raise ValueError('Trace agent/step coverage differs')
    ot={r['step']:r for r in previous_rewards};nt={r['step']:r for r in candidate_rewards}
    first={'llm_skill':None,'llm_action':None,'keep_action':None,'visible_observation':None,'official_components':None};timeline=[]
    for step in sorted({k[0] for k in old}):
        differences=[]
        for key in sorted(k for k in old if k[0]==step):
            a,b=old[key],new[key];llm=a.get('policy')=='llm'
            for kind,changed in (('llm_skill',llm and a.get('skill_id')!=b.get('skill_id')),
                ('llm_action',llm and a.get('requested_action')!=b.get('requested_action')),
                ('keep_action',not llm and a.get('requested_action')!=b.get('requested_action')),
                ('visible_observation',digest(a.get('observation_before'))!=digest(b.get('observation_before')))):
                if changed and first[kind] is None:first[kind]={'step':step,'agent':key[1]}
            if a.get('skill_id')!=b.get('skill_id') or a.get('requested_action')!=b.get('requested_action'):
                differences.append({'agent':key[1],'policy':a.get('policy'),'previous':{k:a.get(k) for k in ('skill_id','requested_action','action_status','mission_phase')},'candidate':{k:b.get(k) for k in ('skill_id','requested_action','action_status','mission_phase')}})
        before,after=ot.get(step,{}),nt.get(step,{})
        if before.get('components')!=after.get('components') and first['official_components'] is None:first['official_components']={'step':step}
        if differences or before.get('events') or after.get('events'):
            timeline.append({'step':step,'blue_decision_differences':differences,
                'previous_reward_components':before.get('components'),'candidate_reward_components':after.get('components'),
                'previous_official_events':before.get('events'),'candidate_official_events':after.get('events'),
                'previous_red_green_actions':before.get('environment_actions'),
                'candidate_red_green_actions':after.get('environment_actions')})
    return {'first_divergence':first,'timeline':timeline,'full_state_and_rng_snapshot_available':False,
        'causal_effect_identified':False,'missing_red_green_commands':'None means not recorded, not no action.',
        'limitation':'After policy actions diverge, paired seeds do not ensure the same environment state or RNG.'}

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
            'rollout_rejected': sum(c.get('rollout_validation', {}).get('reason') in ('failed_registered_criteria','failed_registered_worst_seed_early') for c in candidates),
            'rollout_deferred':sum(c.get('rollout_validation',{}).get('reason')=='reject_screen_inconclusive' for c in candidates),
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
            'simultaneous_red_green_actions':ticks.get(row['step'],{}).get('environment_actions'),
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
