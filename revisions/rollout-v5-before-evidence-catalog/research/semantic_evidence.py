"""Proposed extension: require observable trace references for semantic changes."""
import hashlib
import json
import re

COMPONENTS=('initiation','execution','termination')
FIELDS=('requested_action','observation_before','observation_after','action_status','behavior_context','behavior_outcome')


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
    gradients=data.get('semantic_gradient',{})
    references=data.get('evidence',[])
    if not isinstance(references,list):references=[]
    for reference in references:
        if not isinstance(reference,dict):errors.append('invalid_reference');continue
        row=lookup.get((reference.get('step'),reference.get('agent')))
        field=reference.get('field');quote=reference.get('quote');component=reference.get('component')
        if row is None or field not in FIELDS or component not in COMPONENTS:
            errors.append('reference_outside_actual_actor_trace');continue
        observed=row.get(field)
        rendered=observed if isinstance(observed,str) else json.dumps(observed,ensure_ascii=False,sort_keys=True)
        if not isinstance(quote,str) or not quote.strip() or quote not in rendered:
            errors.append('quote_not_in_recorded_field');continue
        valid.append(reference)
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
        prompt+='''
### Required evidence audit (Proposed extension)
For EACH nonempty semantic_gradient component, add at least one entry in a separate
"evidence" list: {"component":"initiation|execution|termination", "step":0,
"agent":"blue_agent_4", "field":"requested_action|observation_before|observation_after|action_status|behavior_context|behavior_outcome",
"quote":"an exact short substring of that field at that step"}.
Use actual step numbers, this acting agent and recorded outcomes. Do not cite other
agents, invent hidden sessions/service health, or treat busy waiting as a failed
decision. A negative joint reward does not establish which agent caused it.
Evidence is an audit field, not part of the reusable Skill or its policy text.
Add a top-level evidence list. Each entry must contain:
component: initiation, execution (for policy), or termination;
step: the integer recorded step; agent: the recorded acting Blue ID;
field: requested_action, observation_before, observation_after, action_status,
behavior_context, or behavior_outcome; quote: an exact nonempty excerpt from
that field in that row. Cite at least one entry for EVERY nonempty component.
If no supported change is justified, return empty gradients and is_related=false.
'''
        raw=self.model(prompt)
        from .upstream import OriginalEvolution
        data=OriginalEvolution._parse_json_safely(self,raw)
        data=data if isinstance(data,dict) else {}
        audit=audit_references(data,self.rows);self.audits.append({**audit,'raw_response':raw})
        if not audit['passed']:
            data.update(is_related=False,semantic_gradient={'initiation':'','policy':'','termination':''})
        return json.dumps(data,ensure_ascii=False)
