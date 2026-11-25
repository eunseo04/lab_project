"""Proposed extension: require observable trace references for semantic changes."""
import hashlib
import json
import re
from collections import Counter, defaultdict

COMPONENTS=('initiation','execution','termination')
FIELDS=('requested_action','observation_before','observation_after','action_status','behavior_context','behavior_outcome',
        'executed_actions','skill_terminated','policy_invoked','request_applied','joint_reward','parse_error','response','completed_action')


def action_support(parent, candidate, experiences, gate_experiences):
    """Describe the original Gate's coverage; absence is a rollout requirement, not an action ban.

    Literal primitive references are a limited intent detector, not a semantic
    proof. Previously named but unexecuted actions also need coverage when their
    procedure changes (the observed NetworkContainment case).
    """
    from .diagnosis import operation
    from .behavior import chosen_action
    from .behavior_skills import initiation_matches
    primitives=('Analyse','Remove','Restore','DeployDecoy','BlockTrafficZone','AllowTrafficZone','Sleep')
    def mentioned(text):
        found=set()
        for action in primitives:
            for match in re.finditer(r'\b'+action+r'\b',text,re.I):
                if not re.search(r"(?:do not|don't|never|avoid)\s+(?:repeated\s+|unnecessary\s+)?$",text[max(0,match.start()-32):match.start()],re.I):found.add(action)
        return found
    fields=('initiation','policy','termination');changed=[];old_actions=set();new_actions=set();referenced=set()
    for field in fields:
        before=getattr(parent,field);after=getattr(candidate,field)
        before=' '.join(before) if isinstance(before,list) else before
        after=' '.join(after) if isinstance(after,list) else after
        old_actions.update(mentioned(before));new_actions.update(mentioned(after))
        if before!=after:changed.append(field);referenced.update(mentioned(after))
    evaluated=Counter();started=Counter();completed=defaultdict(Counter);conditions=Counter();seen=set();context_rows=0
    accepted=set();executed=set();requests=Counter()
    for exp in gate_experiences:
        for row in exp.transitions:
            evaluated[(row.get('requested_action') or '').split(' ',1)[0]]+=1
    families={'Analyse':'RiskBasedInvestigation','Remove':'RapidProcessResponse','Restore':'PrivilegedRecovery',
              'DeployDecoy':'SelectiveDeception','BlockTrafficZone':'NetworkContainment','AllowTrafficZone':'TrafficRecovery'}
    for exp in experiences:
        for row in exp.transitions:
            context=row.get('behavior_context') or {}
            if row.get('policy_invoked') and context:
                context_rows+=1
                for action,family in families.items():
                    # Only the same observable diagnostic context, never true simulator state.
                    try:conditions[action]+=bool(initiation_matches(family,context))
                    except (KeyError,TypeError):pass
            if row.get('policy_invoked') and row.get('request_applied') and not row.get('parse_error'):
                requested,target=chosen_action(row)
                key=(row.get('seed'),row.get('agent'),requested,target,row['step'])
                accepted.add(key);requests[requested]+=1
                actual={a.get('name') for a in row.get('executed_actions',[]) if isinstance(a,dict)}
                if requested in actual:executed.add(key)
            op=operation(row)
            if op.get('status') not in ('TRUE','FALSE') or op.get('completed_at') is None:continue
            key=(row.get('seed'),row.get('agent'),op.get('name'),op.get('target'),op.get('started_at'))
            if key not in seen:completed[op.get('name')][op['status']]+=1;seen.add(key)
            executed.add(key)
    for key in accepted&executed:started[key[2]]+=1
    uncovered=sorted(a for a in referenced if not started[a] or not sum(completed[a].values()))
    unobserved_conditions=sorted(a for a in referenced if a in families and not conditions[a])
    return {'schema':'action-support-v1','changed_components':changed,
        'parent_named_actions':sorted(old_actions),'candidate_named_actions':sorted(new_actions),
        'newly_named_actions':sorted(new_actions-old_actions),'changed_guidance_actions':sorted(referenced),
        'gate_evaluated_requested_actions':dict(evaluated),'actual_started_actions':dict(started),
        'accepted_free_requests':dict(requests),
        'execution_linkage':'An accepted free request is confirmed by immediate actual execution or its later own completed_action; busy completions are included and deduplicated.',
        'observed_completed_actions':{a:dict(v) for a,v in completed.items()},
        'observable_condition_counts':dict(conditions),'observable_context_rows':context_rows,
        'changed_actions_without_observed_condition':unobserved_conditions,
        'actions_outside_empirical_execution_support':uncovered,
        'requires_rollout':bool(uncovered or unobserved_conditions),'forbidden_actions':[],
        'effect_verification':'Observed call outcomes and PPO likelihood do not establish defensive benefit; only a separate registered real-pool rollout can test it.',
        'intent_limit':'Explicit primitive mentions with simple direct negation handling; indirect natural-language intent is not fully classified.'}


def candidate_supported(parent,candidate,gradient_records):
    supported={component for item in gradient_records if item.get('related')
        for component,value in item['gradient'].items() if value.strip()}
    changed={component for component,field in [('initiation','initiation'),('execution','policy'),('termination','termination')]
        if getattr(parent,field)!=getattr(candidate,field)}
    text=' '.join([candidate.initiation,*candidate.policy,candidate.termination])
    instance=bool(re.search(r'_(?:user|server)_host_\d+|\b(?:\d{1,3}\.){3}\d{1,3}\b|\bF[a-f0-9]{16}\b|\bE\d+\b',text))
    return {'passed':bool(changed) and changed<=supported and not instance,
        'changed_components':sorted(changed),'supported_components':sorted(supported),
        'instance_specific_identifier':instance}


def audit_references(data,rows):
    # completed_action is displayed in the catalogue; resolve it to the exact
    # recorded outcome rather than rejecting a field that our own prompt exposes.
    resolved=[{**r,**({'completed_action':r['behavior_outcome']['completed_action']}
        if (r.get('behavior_outcome') or {}).get('completed_action') else {})} for r in rows]
    lookup={(r['step'],r['agent']):r for r in resolved};valid=[];errors=[]
    catalog={f'E{i}':r for i,r in enumerate(resolved)}
    gradients=data.get('semantic_gradient',{})
    references=data.get('evidence',[])
    if not isinstance(references,list):references=[]
    for reference in references:
        if not isinstance(reference,dict):errors.append('invalid_reference');continue
        field=reference.get('field');quote=reference.get('quote');component=reference.get('component')
        if component=='policy':component='execution'
        if 'evidence_id' in reference:
            row=catalog.get(reference['evidence_id'])
            if row is None or field not in FIELDS or field not in row or component not in COMPONENTS:
                errors.append('unknown_catalog_reference');continue
            if any(key in reference and reference[key]!=row[key] for key in ('step','agent')):
                errors.append('catalog_step_or_actor_mismatch');continue
            observed=row[field]
            quote=observed if isinstance(observed,str) else json.dumps(observed,ensure_ascii=False)
            valid.append({'component':component,'step':row['step'],'agent':row['agent'],'field':field,
                'quote':quote,'evidence_id':reference['evidence_id'],'selected_from_recorded_catalog':True})
            continue
        row=lookup.get((reference.get('step'),reference.get('agent')))
        if row is None or field not in FIELDS or component not in COMPONENTS:
            errors.append('reference_outside_actual_actor_trace');continue
        observed=row.get(field)
        rendered=observed if isinstance(observed,str) else json.dumps(observed,ensure_ascii=False)
        if not isinstance(quote,str) or not quote.strip() or quote not in rendered:
            errors.append('quote_not_in_recorded_field');continue
        valid.append({**reference,'component':component})
    changed=[key for key in COMPONENTS if str(gradients.get('policy' if key=='execution' else key,gradients.get(key,''))).strip()]
    supported={r['component'] for r in valid}
    for key in changed:
        if key not in supported:errors.append('missing_evidence_for_'+key)
    return {'passed':bool(changed) and not errors,'valid_references':valid,'errors':errors,
        'changed_components':changed,'trace_sha256':hashlib.sha256(json.dumps(rows,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
        'limitation':'A correct citation verifies trace provenance, not the semantic truth or causal interpretation of a suggestion.'}


class EvidenceModel:
    def __init__(self,model,rows,diagnosis_verification=False):
        self.model,self.rows,self.audits=model,rows,[]
        self.diagnosis_verification=diagnosis_verification
    def __call__(self,prompt):
        from .diagnosis import operation,restore_boundary
        catalog=[{'evidence_id':f'E{i}',**{key:r.get(key) for key in
            ('step','agent','requested_action','action_status','policy_invoked','skill_terminated')},
            'completed_action':(r.get('behavior_outcome') or {}).get('completed_action'),
            'own_operation':operation(r),'as_of_before':r['step'],'as_of_after':r['step']+1,
            'restore_epoch_before':restore_boundary(self.rows,r['agent'],operation(r).get('target'),r['step']),
            'restore_epoch_after':restore_boundary(self.rows,r['agent'],operation(r).get('target'),r['step']+1)}
            for i,r in enumerate(self.rows)]
        # Keep the original three semantic-gradient components, and put the
        # provenance field IN the concrete output schema rather than only prose.
        prompt=prompt.replace('"is_related": true,','"is_related": true,\n        "evidence": [{"component":"policy", "evidence_id":"E0", "field":"requested_action"}],',1)
        prompt+='''
### Required evidence audit (Proposed extension)
For EACH nonempty semantic_gradient component add an entry in a separate top-level
evidence list: {"component":"initiation|policy|termination", "evidence_id":"E0",
"field":"requested_action|observation_before|observation_after|action_status|completed_action|behavior_context|behavior_outcome|skill_terminated"}.
Use ONLY IDs from the actual-record catalog below. The verifier resolves each ID
to the original step, acting Blue and exact field value. Do not reconstruct long
JSON quotes. Cite a relevant record field for EVERY changed component.
These IDs index the same execution trace; they supply no additional hidden state.
Use the displayed own_operation, as_of_before/as_of_after and restore_epoch values
exactly. A completed_action citation refers to behavior_outcome.completed_action.
Sleep with UNKNOWN status has no completed-failure evidence. Omit unsupported components.
Do not invent hidden sessions/service health or treat busy waiting as a failed
decision. Negative JOINT reward does not establish which agent caused it.
An evidence ID proves trace provenance, not the truth of your causal interpretation.
Evidence IDs must not be inserted into the reusable Skill or policy text.
If no supported change is justified, return empty gradients and is_related=false.
\n# Actual recorded evidence IDs:
'''+json.dumps(catalog,ensure_ascii=False)+'''
\n# Final required response schema (JSON only)
{"diagnosis":"supported diagnosis or uncertainty","is_related":true,
"semantic_gradient":{"initiation":"supported update or empty string","policy":"supported update or empty string","termination":"supported update or empty string"},
"evidence":[{"component":"initiation","evidence_id":"ID of relevant actual row","field":"relevant recorded field"},
{"component":"policy","evidence_id":"ID of relevant actual row","field":"relevant recorded field"},
{"component":"termination","evidence_id":"ID of relevant actual row","field":"relevant recorded field"}]}
Use actual IDs, not the schema placeholders. Omit entries for unchanged components.
'''
        if self.diagnosis_verification:
            from .diagnosis import PROMPT,PROPOSAL_PROMPT
            prompt += PROMPT+PROPOSAL_PROMPT
        raw=self.model(prompt)
        from .upstream import OriginalEvolution
        data=OriginalEvolution._parse_json_safely(self,raw)
        data=data if isinstance(data,dict) else {}
        audited_data=data
        if self.diagnosis_verification and isinstance(data.get('proposals'),list):
            selected={'execution' if p.get('component')=='policy' else p.get('component')
                      for p in data['proposals'] if isinstance(p,dict)}
            audited_data={**data,'semantic_gradient':{
                ('policy' if key=='execution' else key):('structured proposal' if key in selected else '')
                for key in COMPONENTS}}
            for field in ('claims','evidence'):
                audited_data[field]=[r for r in data.get(field,[]) if isinstance(r,dict) and
                    ('execution' if r.get('component')=='policy' else r.get('component')) in selected]
        audit=audit_references(audited_data,self.rows)
        if self.diagnosis_verification:
            from .diagnosis import validate_diagnosis
            diagnosis=validate_diagnosis(audited_data,self.rows,audit)
            audit['diagnosis_validation']=diagnosis
            audit['passed']=audit['passed'] and diagnosis['passed']
            if audit['passed']:
                from .diagnosis import canonical_proposals
                proposal=canonical_proposals(data,diagnosis)
                audit['structured_proposals']=proposal
                audit['passed']=proposal['passed']
                if audit['passed']:
                    # No raw diagnosis or free-text gradient enters candidate generation.
                    data={'is_related':True,'diagnosis':'Validated observations with untested structured hypotheses.',
                          'semantic_gradient':{'initiation':proposal['gradients']['initiation'],
                                               'policy':proposal['gradients']['execution'],
                                               'termination':proposal['gradients']['termination']},
                          'verified_facts':proposal['facts'],'proposals':proposal['proposals']}

        self.audits.append({**audit,'raw_response':raw})
        if not audit['passed']:
            data.update(is_related=False,semantic_gradient={'initiation':'','policy':'','termination':''})
        return json.dumps(data,ensure_ascii=False)
