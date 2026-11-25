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


def response_timeline(rows,families=None,allowed_actions=None):
    """Visible cue-to-response links; delays are censored, never hidden-attack failures."""
    import json,re
    from .behavior import chosen_action
    from .diagnosis import operation,restore_boundary
    from .behavior_skills import initiation_matches
    families=families or {};allowed_actions=allowed_actions or {};groups=defaultdict(list)
    for row in rows:
        if row.get('policy')=='llm':groups[(row.get('seed'),row.get('episode_id'),row['agent'])].append(row)
    events=[];termination=Counter();selector=Counter();guidance=Counter();open_sleep=0;termination_logged=0;free_total=0
    for group,trace in groups.items():
        trace.sort(key=lambda r:r['step']);seen=set();free=[r for r in trace if r.get('policy_invoked')]
        def selection_at(record):
            audit=(record.get('skill_controller') or {}).get('selection_audit')
            return {'step':record['step'],**{k:audit.get(k) for k in ('source','selected','controller_contract','gain_filter_veto')}} if audit else None
        def termination_at(record):
            controller=record.get('skill_controller') or {};audit=controller.get('termination_audit')
            events=[e for e in controller.get('events',[]) if e.get('terminated') or e.get('error')]
            if audit:return {'step':record['step'],'status':audit.get('status'),'response':audit.get('response'),'events':events}
            return {'step':record['step'],'status':None,'events':events,'raw_status_not_recorded':True} if events else None
        for row in trace:
            controller=row.get('skill_controller') or {};ta=controller.get('termination_audit')
            if ta:termination[ta['status']]+=1;termination_logged+=1
            for e in controller.get('events',[]):
                if e.get('reason'):termination[e['reason']]+=1
                if e.get('error'):termination[e['error']]+=1
            sa=controller.get('selection_audit')
            if sa:
                selector['calls']+=1;selector['gain_filter_veto']+=bool(sa.get('gain_filter_veto'))
                selector['NONE']+=sa.get('selected') is None
                selector['llm_gain_audit_recorded']+='gain_filter_veto' in sa
            if row.get('policy_invoked'):
                free_total+=1;action,_=chosen_action(row);ctx=row.get('behavior_context') or {}
                open_sleep+=action=='Sleep' and bool(ctx.get('open_cases'))
                name=row.get('skill_id');family=families.get(name,re.sub(r'_v\d+$','',name or 'NONE'))
                if name and ctx and not initiation_matches(family,ctx,revision='v5'):guidance['observable_eligibility_mismatch']+=1
                permitted=allowed_actions.get(name)
                if permitted is not None and action not in permitted:guidance['outside_declared_guidance_actions']+=1
            for phase in ('before','after'):
                for host,body in (row.get('observation_'+phase) or {}).items():
                    if not isinstance(body,dict):continue
                    at=row['step']+int(phase=='after');epoch=restore_boundary(trace,row['agent'],host,at)
                    cues=[('pid',p['PID']) for p in body.get('Processes',[]) if isinstance(p,dict) and 'PID' in p and 'username' not in p]
                    cues += [('admin',f['File Name']) for f in body.get('Files',[]) if isinstance(f,dict) and f.get('File Name') in ('escalate.sh','escalate.exe')]
                    if isinstance(body.get('Services'),dict):cues += [('service',json.dumps([k,v],sort_keys=True)) for k,v in body['Services'].items()]
                    for kind,value in cues:
                        key=(host,epoch,kind,str(value))
                        if key in seen:continue
                        seen.add(key);later=[r for r in free if r['step']>=at];responses={}
                        for action in ('Analyse','Remove','Restore'):
                            request=next((r for r in later if chosen_action(r)==(action,host) and r.get('request_applied') and not r.get('parse_error')),None)
                            if request:
                                completion=next((operation(r) for r in trace if operation(r).get('name')==action and operation(r).get('target')==host and operation(r).get('started_at')==request['step'] and operation(r).get('completed_at') is not None),None)
                                responses[action]={'requested_step':request['step'],'elapsed_ticks':request['step']-at,
                                    'free_decisions_before_request':sum(r['step']<request['step'] for r in later),
                                    'busy_ticks_before_request':sum(at<=r['step']<request['step'] and not r.get('policy_invoked') for r in trace),
                                    'skill':request.get('skill_id'),
                                    'selection':next((selection_at(r) for r in reversed(trace) if r['step']<=request['step'] and selection_at(r)),None),
                                    'termination':termination_at(request),
                                    'completion':completion,'completion_linked':completion is not None}
                            else:responses[action]={'censored':True,'not_an_unobserved_attack_failure':True}
                        events.append({'seed':group[0],'episode_id':group[1],'agent':group[2],'observation_target':host,
                            'kind':kind,'value':value,'first_visible_at':at,'restore_epoch':epoch,'responses':responses,
                            'last_selection_before_cue':next((selection_at(r) for r in reversed(trace) if r['step']<at and selection_at(r)),None),
                            'first_selection_after_cue':next((selection_at(r) for r in trace if r['step']>=at and selection_at(r)),None),
                            'first_termination_after_cue':next((termination_at(r) for r in trace if r['step']>=at and termination_at(r)),None)})
    return {'scope':'First visible clue per host/epoch/value; correlation only. No hidden-state or causal failure labels.',
        'new_visible_cues':len(events),'events':events,'free_decisions':free_total,'unresolved_case_free_sleep':open_sleep,
        'termination_counts':dict(termination),'explicit_termination_audits':termination_logged,
        'CONTINUE_count':termination.get('CONTINUE',0) if termination_logged else None,
        'selector':dict(selector),'guidance_diagnostics':dict(guidance),
        'guidance_is_not_an_action_mask':True,'causal_effect_estimated':False}


def response_opportunity_audit(rows, families=None):
    """Use the recorded actor interface to count addressable defensive opportunities."""
    from bisect import bisect_left
    from statistics import median
    import json
    import re
    from .behavior import case_response_options, chosen_action

    families=families or {}
    groups=defaultdict(list)
    for row in rows:
        if row.get('policy')=='llm':
            groups[(row.get('seed'),row.get('episode_id'),row['agent'])].append(row)
    counts=Counter()
    cases=[]
    for (seed,episode,agent),trace in groups.items():
        trace.sort(key=lambda row:row['step'])
        completed={}
        for row in trace:
            op=(row.get('behavior_outcome') or {}).get('completed_action')
            if op and op.get('status') in ('TRUE','FALSE') and op.get('completed_at') is not None:
                completed[(op['started_at'],op['name'],op['target'])]=op
        free_steps=[row['step'] for row in trace if row.get('policy_invoked') and row.get('request_applied')]
        opportunities={}
        for row in trace:
            if not row.get('policy_invoked') or not row.get('request_applied'):
                counts['busy_or_not_invoked_ticks']+=1
                continue
            counts['free_decisions']+=1
            context=row.get('behavior_context')
            if not isinstance(context,dict) or 'open_cases' not in context:
                counts['case_context_missing']+=1
                continue
            if not context['open_cases']:
                continue
            counts['open_case_free_decisions']+=1
            if not row.get('parse_error') and chosen_action(row)[0]=='Sleep':
                counts['open_case_free_sleep']+=1
            hosts=row.get('assigned_hosts')
            if hosts is None:
                recorded=re.search(r'^Assigned host IDs: (\[[^\n]*\])$',row.get('user_prompt') or '',re.M)
                if recorded:
                    try:hosts=json.loads(recorded[1])
                    except json.JSONDecodeError:pass
            if not isinstance(hosts,list):
                counts['open_case_host_interface_unrecorded']+=1
                continue
            options=case_response_options(context,hosts)
            if not options:
                continue
            counts['addressable_case_free_opportunities']+=1
            top=max(option['priority'] for option in options)
            preferred=[option for option in options if option['priority']==top]
            urgent=top>=2
            counts['urgent_free_opportunities']+=urgent
            counts['admin_evidence_free_opportunities']+=top==3
            name=row.get('skill_id')
            family=families.get(name,re.sub(r'_v\d+$','',name or 'NONE'))
            aligned=[option for option in preferred if option['family']==family]
            counts['aligned_skill_free_opportunities']+=bool(aligned)
            counts['urgent_aligned_skill_free_opportunities']+=urgent and bool(aligned)
            controller=row.get('skill_controller')
            if urgent and name and not aligned:
                if isinstance(controller,dict) and 'selection_audit' in controller:
                    counts['urgent_different_family_held']+=controller['selection_audit'] is None
                else:
                    counts['urgent_selection_history_missing']+=1
            action,target=chosen_action(row) if not row.get('parse_error') else (None,None)
            counts['urgent_free_sleep']+=urgent and action=='Sleep'
            if aligned:
                matched=any((o['action'],o['host'])==(action,target) for o in aligned)
                counts['aligned_primitive_target_requests']+=matched
                counts['urgent_aligned_primitive_target_requests']+=urgent and matched
            for option in options:
                key=(option['host'],option['restore_epoch'],option['evidence_version'],option['action'])
                step=row['step']
                age=option['evidence_age']
                visible_at=step-age if type(age) is int and 0<=age<=step else None
                captured=opportunities.setdefault(key,{**option,'seed':seed,'episode_id':episode,'agent':agent,
                    'first_cue_step':visible_at,'first_addressable_step':step,'requests':[]})
                if (option['action'],option['host'])!=(action,target):
                    continue
                counts['addressable_case_requests']+=1
                if urgent and option in preferred:counts['urgent_priority_requests']+=1
                op=completed.get((step,action,target))
                counts['linked_completed_calls']+=op is not None
                if op:counts['completed_'+op['status']]+=1
                captured['requests'].append({'step':step,'active_skill':name,'selected_family':family,
                    'selection_audit_recorded':isinstance(controller,dict) and controller.get('selection_audit') is not None,
                    'termination_events':(controller or {}).get('events',[]),
                    'execution_observed_at_request':any(a.get('name')==action and a.get('target')==target for a in row.get('executed_actions',[])),
                    'completion':op})
        for case in opportunities.values():
            first=case['requests'][0]['step'] if case['requests'] else None
            cue=case['first_cue_step']
            case['right_censored_at_episode_end']=first is None
            case['first_request_delay_ticks']=first-cue if first is not None and cue is not None else None
            case['first_request_delay_free_decisions']=bisect_left(free_steps,first)-bisect_left(free_steps,cue) if first is not None and cue is not None else None
            cases.append(case)
    delays=[case['first_request_delay_free_decisions'] for case in cases if case['first_request_delay_free_decisions'] is not None]
    required=('open_case_free_decisions','addressable_case_free_opportunities','urgent_free_opportunities',
              'admin_evidence_free_opportunities','urgent_aligned_skill_free_opportunities','urgent_different_family_held',
              'aligned_skill_free_opportunities','aligned_primitive_target_requests','urgent_priority_requests',
              'urgent_aligned_primitive_target_requests',
              'addressable_case_requests','linked_completed_calls','completed_TRUE','completed_FALSE','urgent_free_sleep')
    for key in required:counts.setdefault(key,0)
    return {'scope':'Free decision ticks with a recorded case and parser-addressable host. Simulator execution is counted separately.',
        'counts':dict(counts),'cases':cases,'matched_case_delay_count':len(delays),
        'median_first_request_delay_free_decisions':median(delays) if delays else None,
        'censored_case_versions':sum(case['right_censored_at_episode_end'] for case in cases),
        'missing_completion_records':counts['addressable_case_requests']-counts['linked_completed_calls'],
        'observable_resolution_rate':None,
        'resolution_limitation':'Partial observations and call status do not establish disappearance or resolution of the original threat.',
        'opportunities_are_independent_samples':False}
