"""CAGE4 observation checks for proposed SGs; these do not prove causation."""
import re

COMPONENTS = {'initiation', 'execution', 'termination'}
FINDINGS = {'completed_success', 'completed_failure', 'pending', 'observed_pid',
            'observed_admin_indicator', 'observed_service_state', 'recorded_decision'}
TRAFFIC_ACTIONS = {'BlockTrafficZone', 'AllowTrafficZone'}
TRAFFIC_SCOPE_CONTRADICTION = 'ACD_traffic_control_has_one_fixed_source_and_one_destination'


def operation(row):
    completed = (row.get('behavior_outcome') or {}).get('completed_action')
    if completed:
        return completed
    pending = (row.get('behavior_context') or {}).get('pending_action')
    if pending and not row.get('policy_invoked'):
        return {**pending, 'status': row.get('action_status', 'UNKNOWN')}
    from .behavior import chosen_action
    action, target = chosen_action(row)
    status = row.get('action_status', 'UNKNOWN')
    return {'name': action, 'target': target, 'started_at': row['step'],
            'status': status,
            'completed_at': row['step'] + 1 if status in ('TRUE', 'FALSE') and row.get('request_applied') else None}


def restore_boundary(rows, agent, host, at):
    boundaries = []
    for row in rows:
        if row.get('agent') != agent:
            continue
        context = row.get('behavior_context') or {}
        ops = list(context.get('recent_completed_actions', []))
        completed = (row.get('behavior_outcome') or {}).get('completed_action')
        if completed:
            ops.append(completed)
        for item in context.get('open_cases', []):
            if item.get('host') == host and 0 <= item.get('last_successful_restore_at', -1) <= at:
                boundaries.append(item['last_successful_restore_at'])
        for item in ops:
            end = item.get('completed_at')
            if item.get('name') == 'Restore' and item.get('target') == host and item.get('status') == 'TRUE' and isinstance(end, int) and end <= at:
                boundaries.append(end)
    return max(boundaries, default=-1)


def canonical_operations(row, rows):
    """One definition for displayed and checked decision/operation targets and epochs."""
    from .behavior import chosen_action
    name,target=chosen_action(row) if row.get('requested_action') else (None,None)
    decision={'name':name,'target':target,'policy_invoked':bool(row.get('policy_invoked')),
        'request_applied':bool(row.get('request_applied')),'parse_error':row.get('parse_error'),
        'recorded_request':bool(row.get('requested_action')),'step':row['step']}
    if name in TRAFFIC_ACTIONS and row.get('requested_traffic_zone_pair'):
        decision['traffic_zone_pair'] = row['requested_traffic_zone_pair']
    result={}
    for key,op in (('requested_decision',decision),('completed_or_pending_operation',operation(row))):
        target=op.get('target')
        result[key]={**op,'as_of_before':row['step'],'as_of_after':row['step']+1,
            'restore_epoch_before':restore_boundary(rows,row['agent'],target,row['step']),
            'restore_epoch_after':restore_boundary(rows,row['agent'],target,row['step']+1)}
        if key=='completed_or_pending_operation':
            result[key]['recorded_operation']=bool((row.get('behavior_outcome') or {}).get('completed_action') or
                (row.get('behavior_context') or {}).get('pending_action') or
                (decision['policy_invoked'] and decision['request_applied'] and not decision['parse_error'] and decision['recorded_request']))
    return result


def validate_diagnosis(data, rows, provenance):
    """Require typed, falsifiable observation claims for each changed component.

    Free-form prose is not a formal proof. Checked claims establish consistency
    with a partial observation, never hidden compromise or individual reward credit.
    """
    errors, valid = [], []
    claims = data.get('claims')
    if data.get('diagnosis_status') != 'supported' or not isinstance(claims, list) or not claims:
        return {'passed': False, 'status': 'insufficient_evidence', 'errors': ['missing_supported_observation_claims'], 'valid_claims': []}
    catalog = {f'E{i}': row for i, row in enumerate(rows)}
    allowed = {(r.get('component'), r.get('evidence_id')) for r in provenance.get('valid_references', [])}
    for claim in claims:
        if not isinstance(claim, dict):
            errors.append('invalid_claim'); continue
        component = 'execution' if claim.get('component') == 'policy' else claim.get('component')
        row = catalog.get(claim.get('evidence_id'))
        finding = claim.get('finding')
        if row is None or component not in COMPONENTS or (component, claim.get('evidence_id')) not in allowed:
            errors.append('claim_without_matching_citation'); continue
        if finding not in FINDINGS or claim.get('causal_attribution') != 'unknown':
            errors.append('unproven_causal_or_unknown_claim'); continue
        operations=canonical_operations(row,rows)
        op=operations['requested_decision' if finding=='recorded_decision' else 'completed_or_pending_operation']
        observation_finding=finding.startswith('observed_')
        if finding == 'recorded_decision':
            # A decision fact is distinct from completion of an earlier action.
            if not row.get('policy_invoked') or not row.get('request_applied') or row.get('parse_error') or not row.get('requested_action'):
                errors.append('not_an_applied_policy_decision'); continue
        context_actions={op.get('name'),operations['requested_decision'].get('name'),'Monitor',None}
        if (not observation_finding and claim.get('action') != op.get('name') or
                observation_finding and claim.get('action') not in context_actions):
            errors.append('action_mismatch'); continue
        host = claim.get('observation_target',claim.get('target')) if observation_finding else claim.get('target')
        if observation_finding:
            if not isinstance(host,str) or not host or ('observation_target' in claim and
                    'target' in claim and claim['target']!=host):
                errors.append('target_mismatch'); continue
        elif ('target' not in claim or host != op.get('target') or
                (not host and not (host is None and op.get('name') in {'Sleep', 'Monitor'}))):
            errors.append('target_mismatch'); continue
        phase = claim.get('phase', 'after')
        if phase not in ('before', 'after'):
            errors.append('invalid_observation_phase'); continue
        recorded_at = row['step'] + int(phase == 'after')
        as_of = claim.get('as_of_step')
        if type(as_of) is not int or as_of != recorded_at:
            errors.append('observation_time_mismatch'); continue
        end, start = op.get('completed_at'), op.get('started_at')
        if finding in ('completed_success','completed_failure','pending') and not op['recorded_operation']:
            errors.append('own_operation_not_recorded');continue
        if finding == 'recorded_decision':
            if phase != 'after':
                errors.append('decision_not_recorded_before_action'); continue
        elif finding.startswith('completed_'):
            expected = 'TRUE' if finding == 'completed_success' else 'FALSE'
            if op.get('status') != expected or end is None or phase != 'after' or end > as_of or start is None or end < start:
                errors.append('not_a_completed_action'); continue
        elif finding == 'pending':
            if op.get('status') != 'IN_PROGRESS' or end is not None:
                errors.append('not_pending'); continue
        else:
            body = (row.get('observation_' + phase) or {}).get(host)
            if not isinstance(body, dict):
                errors.append('target_observation_missing'); continue
            if finding == 'observed_pid':
                pids = {str(p['PID']) for p in body.get('Processes', []) if isinstance(p, dict) and 'PID' in p and 'username' not in p}
                if str(claim.get('pid')) not in pids:
                    errors.append('pid_not_observed_at_claimed_time'); continue
                before=(row.get('observation_before') or {}).get(host,{})
                if claim.get('fresh') and (phase!='after' or any(str(p.get('PID'))==str(claim.get('pid'))
                        for p in before.get('Processes',[]) if isinstance(p,dict))):
                    errors.append('pid_not_fresh'); continue
            if finding == 'observed_admin_indicator':
                names = {f.get('File Name') for f in body.get('Files', []) if isinstance(f, dict)}
                if claim.get('indicator') not in names & {'escalate.sh', 'escalate.exe'}:
                    errors.append('admin_indicator_not_observed'); continue
            if finding == 'observed_service_state':
                services = body.get('Services', {})
                if not isinstance(services, dict) or claim.get('service') not in services or services[claim['service']] != claim.get('value'):
                    errors.append('service_state_not_observed'); continue
        boundary = restore_boundary(rows,row['agent'],host,as_of) if observation_finding else op['restore_epoch_'+phase]
        if 'restore_epoch' not in claim or claim['restore_epoch'] != boundary:
            errors.append('restore_epoch_mismatch'); continue
        # A newly observed PID is not evidence that an older Remove failed.
        if finding == 'completed_failure' and claim.get('pid') is not None:
            errors.append('call_status_is_not_pid_cleanup_evidence'); continue
        valid.append({**claim, 'component': component, 'status': 'observation_consistent',
            **({'observation_target':host,'target':host,'action':claim.get('action')} if observation_finding else {}),
            **traffic_fact_context(row, {} if observation_finding else op),
            'source_seed':row.get('seed'),'source_agent':row['agent'],'source_step':row['step'],
            'source_trace_sha256':provenance.get('trace_sha256')})
    supported = {c['component'] for c in valid}
    for component in provenance.get('changed_components', []):
        if component not in supported:
            errors.append('missing_diagnosis_for_' + component)
    # Forbid explicit direct joint-reward blame, including structured assertions.
    if data.get('reward_attribution', 'unknown') != 'unknown':
        errors.append('individual_joint_reward_attribution_unidentified')
    return {'passed': bool(valid) and not errors, 'status': 'supported_observation' if valid and not errors else 'insufficient_evidence',
            'errors': errors, 'valid_claims': valid,
            'limitation': 'Typed claims check observed facts and timing. They do not prove causal diagnosis or validate all natural-language advice.'}


PROMPT = '''
### CAGE4 diagnosis consistency schema (Proposed v9, separate from Skill-Pro)
Keep the original initiation/policy/termination semantic gradients. In addition,
return diagnosis_status="supported|unknown|insufficient_evidence", reward_attribution="unknown",
and claims=[{"component":"policy", "evidence_id":"E0", "action":"Remove",
"target":"actual host from that record", "finding":"completed_failure", "phase":"after",
"as_of_step":1, "restore_epoch":-1, "causal_attribution":"unknown"}].
Allowed findings: recorded_decision, completed_success, completed_failure, pending, observed_pid (include pid),
observed_admin_indicator (include indicator), observed_service_state (include service and value).
Use one matching evidence citation and observation-consistent claim for EVERY changed component.
Use requested_decision for recorded_decision, and completed_or_pending_operation
for completion/pending findings. Copy that object's name, target, as_of and target-specific
restore_epoch fields. They may describe different targets in the same row.
For observed_pid/admin/service use observation_target: the exact host in that Blue's
before/after observation, independently of the action target. Use that observation
host's Restore boundary. An optional action is context only, never evidence that
the action caused the observation. Never claim an observed PID is fresh if already
present before. Only actual operation claims establish completion/pending.
For recorded_decision only, use the actual requested action/target of an invoked,
applied, parsed policy decision, even if completed_or_pending_operation describes an
earlier completion. Sleep/Monitor have target JSON null, never the string "null"
or an invented host. Use phase="after". This finding verifies only a recorded
decision; UNKNOWN never establishes success, failure or an IN_PROGRESS operation.
as_of_step equals record step for before, step+1 for after. restore_epoch is the latest
successful same-host Restore completion visible at that time, or -1 if none is recorded.
IN_PROGRESS never supports completed_failure/success. FALSE is a call status, not proof
of its cause. Observed PIDs/files/services must exist in that exact host observation.
Retained old context is not a fresh observation. New PIDs or a post-Restore evidence
epoch never establish an earlier Remove failure. Joint reward cannot identify a Skill's
causal contribution. If evidence is insufficient, use empty gradients and is_related=false.
Do not claim that these checks prove causation. Never insert evidence IDs into Skills.
'''


PROPOSAL_FINDINGS = {
    'review_recorded_decision': {'recorded_decision'},
    'check_call_outcome': {'completed_success', 'completed_failure'},
    'wait_for_completion': {'pending'},
    'review_observed_pid': {'observed_pid'},
    'review_privilege_indicator': {'observed_admin_indicator'},
    'review_service_observation': {'observed_service_state'},
}


def traffic_fact_context(row, op):
    context = {}
    if isinstance(row.get('traffic_zone_interface'), dict):
        context['traffic_zone_interface'] = row['traffic_zone_interface']
    if op.get('name') in TRAFFIC_ACTIONS and isinstance(op.get('traffic_zone_pair'), dict):
        context['traffic_zone_pair'] = op['traffic_zone_pair']
    return context


def traffic_scope_contradictions(text, facts=()):
    """Recognize explicit scope contradictions. Unclassified prose stays unknown."""
    sources = set()
    completion_pairs = {}
    for fact in facts:
        interface = fact.get('traffic_zone_interface', {})
        pair = fact.get('traffic_zone_pair', {})
        agent = fact.get('agent', fact.get('source_agent', ''))
        if not isinstance(interface, dict) or not isinstance(pair, dict) or not (interface or pair):
            continue
        if not isinstance(agent, str) or not re.fullmatch(r'blue_agent_[0-4]', agent):
            continue
        from CybORG.Agents.LLMAgents.llm_adapter.utils.constants import CAGE4_SUBNETS
        fixed_source = CAGE4_SUBNETS[int(agent[-1])]
        if interface and interface.get('fixed_from_subnet') != fixed_source:
            continue
        if pair and (pair.get('from_subnet') != fixed_source or not isinstance(pair.get('to_subnet'), str)
                or pair['to_subnet'] != fact.get('target')):
            continue
        sources.add(fixed_source)
        action = fact.get('action')
        if pair and action in TRAFFIC_ACTIONS and fact.get('finding') in ('completed_success', 'completed_failure'):
            completion_pairs.setdefault(action.lower(), set()).add((fixed_source, pair['to_subnet']))

    # Separate independently asserted claims before checking direct negations.
    action = r'(?:BlockTrafficZone|AllowTrafficZone|block\w*|allow\w*|isolat\w*)'
    setter = r'(?:set|change|choose|select|specify|override|assign|supply|pick)'
    clause_break = r'[.!?;\n,]|\b(?:but|however|instead|whereas)\b|\b(?:and|then)\s+(?='+action+r'\b)'
    direct_negative = (r"\b(?:not|never|cannot|avoid|don['’]t|can['’]t|doesn['’]t|won['’]t)\s+"
        r"(?:(?:try to|attempt to|use|using|be used to|repeatedly)\s+)*(?:" + action + '|' + setter +
        r'|affect|cover|control)\b')
    broad_scope = (r'\b(?:all sources|any source subnet|any from_subnet|entire (?:subnet|zone)|'
        r'whole (?:subnet|zone)|all (?:incoming|outgoing) traffic|multiple (?:destinations?|source subnets?))\b')
    endpoint = r"""\s*(?:=|:|\bis\b)?\s*[`"']?([a-z0-9_]+_subnet)\b"""
    explicit_source = r'\b(?:from_subnet|(?:fixed\s+)?source(?:\s+subnet)?)'+endpoint
    explicit_target = r'\b(?:to_subnet|(?:selected\s+)?destination(?:\s+subnet)?)'+endpoint
    assigns_source = r'\b'+setter+r'\s+(?:the\s+)?(from_subnet|source (?:subnet|zone))\b'
    traffic_control = r'\b(?:BlockTrafficZone|AllowTrafficZone|block traffic|allow traffic|isolate)\b'
    completed_claim = r'\b(?:BlockTrafficZone|AllowTrafficZone)\s+(?:completed|succeeded|failed|executed)\b'
    for clause in re.split(clause_break, text, flags=re.I):
        assignment = re.search(assigns_source, clause, re.I)
        if (assignment and (assignment[1].lower() == 'from_subnet' or re.search(traffic_control, clause, re.I))
                and not re.search(direct_negative, clause[:assignment.end()], re.I)):
            return [TRAFFIC_SCOPE_CONTRADICTION]
        if not re.search(r'\b'+action+r'\b', clause, re.I):
            continue
        broad = re.search(broad_scope, clause, re.I)
        if broad and not (re.search(direct_negative, clause[:broad.start()], re.I) or
                          re.search(r'\b(?:not|never)\s+(?:the\s+)?$', clause[:broad.start()], re.I)):
            return [TRAFFIC_SCOPE_CONTRADICTION]
        claimed_source = {m[1].lower() for m in re.finditer(explicit_source, clause, re.I)
                          if not re.search(direct_negative, clause[:m.start()], re.I)}
        claimed_target = {m[1].lower() for m in re.finditer(explicit_target, clause, re.I)
                          if not re.search(direct_negative, clause[:m.start()], re.I)}
        if len(claimed_source) > 1 or len(claimed_target) > 1:
            return [TRAFFIC_SCOPE_CONTRADICTION]
        if claimed_source and sources and not claimed_source & sources:
            return [TRAFFIC_SCOPE_CONTRADICTION]
        actual = re.search(completed_claim, clause, re.I)
        if actual and claimed_target:
            pairs = completion_pairs.get(actual[0].split()[0].lower(), set())
            if pairs and not any((not claimed_source or source in claimed_source) and target in claimed_target
                                 for source, target in pairs):
                return [TRAFFIC_SCOPE_CONTRADICTION]
    return []


def canonical_proposals(data, validation):
    """Quarantine prose; only validated facts and model-selected typed hypotheses flow on.

    This is a deliberately limited proposal language. It checks a relationship,
    not whether the proposed procedure will improve reward or identify a cause.
    """
    from .progress import digest
    facts=[]
    fields=('component','evidence_id','action','target','observation_target','finding','phase','as_of_step','restore_epoch','pid','indicator','service','value',
        'source_seed','source_agent','source_step','source_trace_sha256','traffic_zone_interface','traffic_zone_pair')
    for claim in validation.get('valid_claims',[]):
        fact={k:claim[k] for k in fields if k in claim}
        fact['fact_id']='F'+digest(fact)[:16]
        facts.append(fact)
    by_evidence={(f['component'],f['evidence_id']):f for f in facts}
    proposals=[];errors=[]
    for index,proposal in enumerate(data.get('proposals',[])):
        if not isinstance(proposal,dict):
            errors.append('invalid_structured_proposal');continue
        component='execution' if proposal.get('component')=='policy' else proposal.get('component')
        refs=proposal.get('evidence_ids',[]);kind=proposal.get('operation')
        support=[by_evidence.get((component,r)) for r in refs] if isinstance(refs,list) else []
        if component not in COMPONENTS or not support or any(f is None for f in support):
            errors.append('proposal_without_matching_validated_fact');continue
        if kind not in PROPOSAL_FINDINGS or any(f['finding'] not in PROPOSAL_FINDINGS[kind] for f in support):
            errors.append('proposal_unrelated_to_validated_finding');continue
        scope = 'same_observed_decision' if kind == 'review_recorded_decision' else 'same_observed_host_and_epoch'
        if proposal.get('scope')!=scope or proposal.get('causal_status')!='untested_hypothesis':
            errors.append('unbounded_or_causal_proposal');continue
        if any(proposal.get('observed_action')!=f['action'] for f in support):
            errors.append('proposal_action_mismatch');continue
        proposals.append({'proposal_id':'P'+digest([component,kind,scope,[f['fact_id'] for f in support]])[:16],
            'component':component,'operation':kind,
            'scope':scope,'causal_status':'untested_hypothesis',
            'fact_ids':[f['fact_id'] for f in support],
            'observed_actions':sorted({f['action'] for f in support if f.get('action')})})
    gradients={key:'' for key in COMPONENTS}
    for proposal in proposals:
        selected=[f for f in facts if f['fact_id'] in proposal['fact_ids']]
        payload={'verified_observations':selected,'untested_modification_hypothesis':proposal,
                 'constraint':'Only the typed observation is verified. No causal or performance claim is established. Do not memorize instance IDs.'}
        import json
        gradients[proposal['component']]+=json.dumps(payload,ensure_ascii=False,sort_keys=True)+'\n'
    return {'passed':bool(proposals) and not errors,'facts':facts,'proposals':proposals,
            'gradients':gradients,'errors':errors,'raw_prose_forwarded':False,
            'limitation':'Only a restricted typed relationship is checked. Procedural effectiveness and general natural-language truth remain unverified.'}


PROPOSAL_PROMPT = '''
### Structured proposal boundary
Free diagnosis and semantic_gradient prose are kept only for audit, not supplied as verified facts.
Also return proposals, with one entry for each component you propose to change:
{"component":"policy","evidence_ids":["E0"],"observed_action":"Remove",
"operation":"check_call_outcome","scope":"same_observed_host_and_epoch",
"causal_status":"untested_hypothesis"}.
Allowed operation/finding pairs:
review_recorded_decision -> recorded_decision (use scope="same_observed_decision");
check_call_outcome -> completed_success or completed_failure;
wait_for_completion -> pending; review_observed_pid -> observed_pid;
review_privilege_indicator -> observed_admin_indicator;
review_service_observation -> observed_service_state.
Use the actually cited own action, matching component and validated evidence.
Observation-only proposals may use observed_action=null when no own action is
claimed. The observation_target remains the visible host, not an action effect.
A proposal is an untested procedure hypothesis, never proof of causation or reward gain.
Do not add unrelated remediation. Unchanged components need no proposal.
'''


def canonical_fact_catalog(rows):
    """Agent-visible facts with stable IDs; observation hosts are independent of actions."""
    from .progress import digest
    facts=[]
    for i,row in enumerate(rows):
        ops=canonical_operations(row,rows)
        def add(finding,phase,target,action=None,action_record=None,**details):
            at=row['step']+int(phase=='after')
            fact={'evidence_id':f'E{i}','agent':row['agent'],'step':row['step'],
                'finding':finding,'phase':phase,'as_of_step':at,'target':target,'action':action,
                'restore_epoch':restore_boundary(rows,row['agent'],target,at),
                'source_record_sha256':digest(row),**traffic_fact_context(row, action_record or {}),**details}
            fact['fact_id']='F'+digest(fact)[:16];facts.append(fact)
        decision=ops['requested_decision'];op=ops['completed_or_pending_operation']
        if decision['policy_invoked'] and decision['request_applied'] and not decision['parse_error'] and decision['recorded_request']:
            add('recorded_decision','after',decision['target'],decision['name'],action_record=decision)
        if op['recorded_operation'] and op.get('status') in ('TRUE','FALSE') and op.get('completed_at') is not None and op.get('started_at') is not None and op['started_at']<=op['completed_at']<=row['step']+1:
            add('completed_success' if op['status']=='TRUE' else 'completed_failure','after',op.get('target'),op.get('name'),
                action_record=op)
        if op['recorded_operation'] and op.get('status')=='IN_PROGRESS' and op.get('completed_at') is None:
            add('pending','after',op.get('target'),op.get('name'),action_record=op)
        for phase in ('before','after'):
            for host,body in (row.get('observation_'+phase) or {}).items():
                if not isinstance(body,dict):continue
                for process in body.get('Processes',[]):
                    if isinstance(process,dict) and 'PID' in process and 'username' not in process:
                        add('observed_pid',phase,host,pid=process['PID'],observation_target=host)
                for file in body.get('Files',[]):
                    if isinstance(file,dict) and file.get('File Name') in ('escalate.sh','escalate.exe'):
                        add('observed_admin_indicator',phase,host,indicator=file['File Name'],observation_target=host)
                services=body.get('Services',{})
                if isinstance(services,dict):
                    for service,value in services.items():
                        add('observed_service_state',phase,host,service=service,value=value,observation_target=host)
    return facts


def validate_interventions(data,rows):
    """Resolve selected IDs exactly, validate bounded hypotheses, never repair a claim."""
    import json
    from .progress import digest
    catalog={f['fact_id']:f for f in canonical_fact_catalog(rows)}
    hypotheses=data.get('interventions',[]);errors=[];facts={};proposals=[]
    required=('observed_problem','preconditions','proposed_procedure','success_observation','termination','failure_or_retry_rule')
    allowed=set(required)|{'supported_fact_ids','component_to_change','causal_status','target_binding'}
    if not isinstance(hypotheses,list):hypotheses=[];errors.append('invalid_interventions')
    if hypotheses and data.get('is_related') is not True:errors.append('inconsistent_related_status')
    if len(hypotheses)>32:errors.append('too_many_interventions');hypotheses=[]
    hosts={f['target'] for f in catalog.values() if isinstance(f.get('target'),str) and f.get('action') not in TRAFFIC_ACTIONS}
    for hypothesis in hypotheses:
        if not isinstance(hypothesis,dict) or set(hypothesis)-allowed:
            errors.append('invalid_intervention_fields');continue
        component='execution' if hypothesis.get('component_to_change')=='policy' else hypothesis.get('component_to_change')
        ids=hypothesis.get('supported_fact_ids')
        if component not in COMPONENTS or not isinstance(ids,list) or not ids or any(not isinstance(i,str) or i not in catalog for i in ids) or len(ids)!=len(set(ids)):
            errors.append('intervention_without_catalog_fact');continue
        if hypothesis.get('causal_status')!='untested_hypothesis' or any(not isinstance(hypothesis.get(k),str) or not hypothesis[k].strip() or len(hypothesis[k])>2048 for k in required):
            errors.append('invalid_or_causal_intervention');continue
        selected=[catalog[i] for i in ids]
        binding=hypothesis.get('target_binding')
        if binding not in ('observed_host','none') or (binding=='observed_host' and any(f.get('target') is None for f in selected)):
            errors.append('unsupported_runtime_binding');continue
        text='\n'.join(hypothesis[k] for k in required)
        contradictions=traffic_scope_contradictions(text, selected)
        if contradictions:
            errors.extend(contradictions);continue
        if re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b|\b[FP][a-f0-9]{16}\b|\bE\d+\b',text) or any(re.search(r'(?<!\w)'+re.escape(host)+r'(?!\w)',text) for host in hosts):
            errors.append('instance_identifier_in_intervention');continue
        # A selected fact proves only its exact visible content, not the stated problem/efficacy.
        for f in selected:facts[(component,f['fact_id'])]={**f,'component':component}
        proposal={**hypothesis,'component':component,'fact_ids':ids,'operation':'structured_intervention',
            'scope':'runtime_observation_binding','observed_actions':sorted({f['action'] for f in selected if f.get('action')})}
        proposal['proposal_id']='P'+digest(proposal)[:16];proposals.append(proposal)
    if data.get('reward_attribution','unknown')!='unknown':errors.append('individual_joint_reward_attribution_unidentified')
    gradients={key:'' for key in COMPONENTS}
    for p in proposals:
        gradients[p['component']]+=json.dumps({'verified_observations':[facts[(p['component'],i)] for i in p['fact_ids']],
            'untested_modification_hypothesis':p,'constraint':'Facts alone are verified. Problem, proposed procedure and efficacy remain untested. Bind targets from current observations; never copy instance IDs.'},ensure_ascii=False,sort_keys=True)+'\n'
    return {'passed':bool(proposals) and not errors,'schema':'fact-interventions-v1','facts':list(facts.values()),'proposals':proposals,
        'gradients':gradients,'errors':errors,'facts_verified':bool(facts) and not errors,'candidate_links_valid':None,
        'known_contradiction':TRAFFIC_SCOPE_CONTRADICTION in errors,'efficacy_verified':False,'raw_prose_forwarded':False,
        'limitation':'Selected catalog facts are exact; natural-language problem/procedure/efficacy are hypotheses, not verified semantics.'}


INTERVENTION_PROMPT = '''
### Proposed fact-interventions-v1 (separate from original B2)
The following catalogue contains ONLY this Blue's exact recorded observation facts.
Select supported_fact_ids exactly. Do not copy action/host/time/epoch fields: the code
resolves the chosen IDs without correcting a wrong selection. A fact about a visible
PID/file/service may concern a different host from your current action. No hidden
Red state or reward answer is provided. Treat a selected fact as observation only.
Return JSON with is_related, reward_attribution="unknown", and interventions:
[{"supported_fact_ids":["an actual catalogue ID"],"component_to_change":"policy",
"observed_problem":"hypothesized decision problem, not an established cause",
"preconditions":"visible evidence and free/busy prerequisites",
"proposed_procedure":"concrete procedure using the runtime bound target",
"success_observation":"observable follow-up to check",
"termination":"when to stop or reassess",
"failure_or_retry_rule":"bounded response to failure or insufficient evidence",
"target_binding":"observed_host","causal_status":"untested_hypothesis"}].
Use initiation/policy/termination components. target_binding is observed_host or none.
Use no hostname/IP/seed/fact IDs inside procedural text. No facts means is_related=false
and interventions=[]. Free diagnosis/semantic_gradient prose is audit-only. Facts are
verified separately from hypotheses, candidate links, PPO admission and real efficacy.
Block/Allow use the fixed source of this Blue and one selected destination, as shown
in the recorded traffic_zone_interface when present. Bind the destination to visible
evidence. One call cannot isolate a whole zone. Monitor is automatic; IN_PROGRESS is
waiting, Remove TRUE does not establish privileged cleanup; Restore has availability cost.
# Exact fact catalogue:
'''
