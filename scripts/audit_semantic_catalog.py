"""Rebuild SG chunks from saved training experiences; make no model calls."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_outputs/rollout-v5'
FORK=ROOT/'revisions/rollout-v5-metadata-fix';sys.path.insert(0,str(FORK))
from research.evolution import Evolution
from research.upstream import OriginalEvolution,Skill,Experience

def read(path):return json.loads(Path(path).read_text())

def main():
    result={'scope':'Read-only exact trace/citation reconstruction. This does not certify semantic truth or causal attribution.','arms':{}}
    original_method=OriginalEvolution._generate_semantic_gradient
    try:
        for label in ('B3','B4'):
            directory=OUT/label/'train';checkpoint=directory/'pool-iteration-0001.json'
            if not checkpoint.exists():continue
            state=read(checkpoint);cfg=read(OUT/f'{label}-config.json')
            chunks={};current_seed=None
            def capture(self,skill,experience):
                rows=json.loads(experience.trajectory)
                digest=hashlib.sha256(json.dumps(rows,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
                chunks[digest]={'seed':current_seed,'skill':skill.name,'rows':rows}
                return {'initiation':'','execution':'','termination':''},False
            OriginalEvolution._generate_semantic_gradient=capture
            ev=Evolution(lambda prompt: (_ for _ in ()).throw(AssertionError('No model calls permitted')),cfg['evolution'])
            initial={s['name']:s for s in read(OUT/'initial_pool.json')['pool']['skills']}
            for data in state['experience']['fifo']:
                current_seed=data['transitions'][0]['seed'];row=initial[data['skill']]
                skill=Skill(**{k:v for k,v in row.items() if k in Skill.__dataclass_fields__})
                ev._generate_semantic_gradient(skill,Experience(**data))
            raw={}
            for path in directory.glob('episode-????.jsonl'):
                for line in path.read_text().splitlines():
                    row=json.loads(line)
                    if row['agent']=='blue_agent_4':raw[(row['seed'],row['step'],row['agent'])]=row
            references=[];audits=0;passed=0;field_counts=Counter();examples=[]
            for history in state['history']:
                for parent in history['evolution']:
                    for gradient in parent.get('semantic_gradients',[]):
                        for audit in gradient.get('evidence_audits',[]):
                            audits+=1;passed+=bool(audit['passed'])
                            chunk=chunks[audit['trace_sha256']]
                            assert chunk['skill']==parent['parent']
                            assert [r['step'] for r in chunk['rows']]==gradient['steps']
                            for ref in audit['valid_references']:
                                row=chunk['rows'][int(ref['evidence_id'][1:])]
                                assert row['step']==ref['step'] and row['agent']==ref['agent']=='blue_agent_4'
                                field=ref['field'];value=row[field]
                                quote=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False)
                                assert quote==ref['quote']
                                source=raw[(chunk['seed'],ref['step'],ref['agent'])]
                                assert value==source[field]
                                references.append({'parent':parent['parent'],'seed':chunk['seed'],**ref})
                                field_counts[field]+=1
                            if len(examples)<3:
                                completed=[{'step':r['step'],**r['behavior_outcome']['completed_action']} for r in chunk['rows']
                                    if (r.get('behavior_outcome') or {}).get('completed_action')]
                                examples.append({'seed':chunk['seed'],'steps':gradient['steps'],
                                    'gradient':gradient['gradient'],'raw_sg_response':audit['raw_response'],
                                    'completed_actions_in_entire_chunk':completed,
                                    'citation_fields':[r['field'] for r in audit['valid_references']],
                                    'interpretation':'Exact citations can still support an incorrect diagnosis; outcome truth is not established by citing a requested action.'})
            result['arms'][label]={'all_references_exactly_match_saved_training':True,
                'sg_chunks':audits,'grounding_passed_chunks':passed,'reference_count':len(references),
                'reference_fields':dict(field_counts),'references':references,'examples':examples}
    finally:OriginalEvolution._generate_semantic_gradient=original_method
    path=OUT/'semantic_catalog_audit.json';path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({label:{k:v for k,v in data.items() if k not in ('references','examples')} for label,data in result['arms'].items()},indent=2))

if __name__=='__main__':main()
