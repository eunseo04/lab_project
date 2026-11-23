"""Proposed extension: require observable trace references for semantic changes."""
import hashlib
import json
import re

COMPONENTS=('initiation','execution','termination')
FIELDS=('requested_action','observation_before','observation_after','action_status','behavior_context','behavior_outcome',
        'executed_actions','skill_terminated','policy_invoked','request_applied','joint_reward')


def candidate_supported(parent,candidate,gradient_records):
    supported={component for item in gradient_records if item.get('related')
        for component,value in item['gradient'].items() if value.strip()}
    changed={component for component,field in [('initiation','initiation'),('execution','policy'),('termination','termination')]
        if getattr(parent,field)!=getattr(candidate,field)}
    text=' '.join([candidate.initiation,*candidate.policy,candidate.termination])
    instance=bool(re.search(r'_(?:user|server)_host_\d+|\b(?:\d{1,3}\.){3}\d{1,3}\b',text))
    return {'passed':bool(changed) and changed<=supported and not instance,
        'changed_components':sorted(changed),'supported_components':sorted(supported),
        'instance_specific_identifier':instance}


def audit_references(data,rows):
    lookup={(r['step'],r['agent']):r for r in rows};valid=[];errors=[]
    catalog={f'E{i}':r for i,r in enumerate(rows)}
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
    def __init__(self,model,rows):self.model,self.rows,self.audits=model,rows,[]
    def __call__(self,prompt):
        catalog=[{'evidence_id':f'E{i}',**{key:r.get(key) for key in
            ('step','agent','requested_action','action_status','policy_invoked','skill_terminated')},
            'completed_action':(r.get('behavior_outcome') or {}).get('completed_action')}
            for i,r in enumerate(self.rows)]
        prompt+='''
### Required evidence audit (Proposed extension)
For EACH nonempty semantic_gradient component add an entry in a separate top-level
evidence list: {"component":"initiation|policy|termination", "evidence_id":"E0",
"field":"requested_action|observation_before|observation_after|action_status|behavior_context|behavior_outcome|skill_terminated"}.
Use ONLY IDs from the actual-record catalog below. The verifier resolves each ID
to the original step, acting Blue and exact field value. Do not reconstruct long
JSON quotes. Cite a relevant record field for EVERY changed component.
These IDs index the same execution trace; they supply no additional hidden state.
Do not invent hidden sessions/service health or treat busy waiting as a failed
decision. Negative JOINT reward does not establish which agent caused it.
An evidence ID proves trace provenance, not the truth of your causal interpretation.
Evidence IDs must not be inserted into the reusable Skill or policy text.
If no supported change is justified, return empty gradients and is_related=false.
\n# Actual recorded evidence IDs:
'''+json.dumps(catalog,ensure_ascii=False)
        raw=self.model(prompt)
        from .upstream import OriginalEvolution
        data=OriginalEvolution._parse_json_safely(self,raw)
        data=data if isinstance(data,dict) else {}
        audit=audit_references(data,self.rows);self.audits.append({**audit,'raw_response':raw})
        if not audit['passed']:
            data.update(is_related=False,semantic_gradient={'initiation':'','policy':'','termination':''})
        return json.dumps(data,ensure_ascii=False)
