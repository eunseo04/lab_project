"""Read-only uncached probability audit of real, active-skill training actions."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_outputs/rollout-v5'
FORK=ROOT/'revisions/rollout-v5-metadata-fix';sys.path.insert(0,str(FORK))
from research.client import ModelClient
from research.upstream import Skill

def main():
    dest=OUT/'active_skill_scoring_parity.json'
    assert not dest.exists(),'Do not overwrite measured audits'
    cfg=json.loads((OUT/'B4-config.json').read_text())
    cfg['model'].update(endpoint='http://127.0.0.1:8221',scoring_endpoints=[])
    initial=json.loads((OUT/'initial_pool.json').read_text())['pool']['skills']
    fields=set(Skill.__dataclass_fields__);skills={row['name']:Skill(**{k:v for k,v in row.items() if k in fields}) for row in initial}
    traces=list((OUT/'B4/train').glob('episode-????.jsonl'));records=[]
    for trace in sorted(traces):
        for line in trace.read_text().splitlines():
            r=json.loads(line)
            if r['agent']=='blue_agent_4' and r.get('skill_id') in skills and r.get('policy_invoked') and r.get('request_applied') and not r.get('parse_error') and r.get('generation_logprob') is not None:
                records.append((trace,r))
    selected=[]
    for name in skills:
        found=[item for item in records if item[1]['skill_id']==name]
        if found:selected.append(found[0])
    selected+= [item for item in records if item not in selected][:max(0,8-len(selected))]
    assert len(selected)>=6 and len({r['skill_id'] for _,r in selected})>=5
    samples=[]
    for _,r in selected:
        messages=json.loads(r['state']);messages[0]['content']+='\n# ACTIVE SECURITY SKILL\n'+skills[r['skill_id']].format_for_llm()
        samples.append({'messages':messages,'token_ids':r['token_ids']})
    client=ModelClient(cfg['model']);scores=client.score(samples,use_cache=False)
    rows=[{'trace':str(trace),'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),
        'seed':r['seed'],'step':r['step'],'agent':r['agent'],'skill':r['skill_id'],
        'generation_logprob':r['generation_logprob'],'independent_uncached_logprob':score,
        'absolute_difference':abs(r['generation_logprob']-score),
        'reconstructed_sample_sha256':hashlib.sha256(json.dumps(sample,sort_keys=True).encode()).hexdigest()}
        for (trace,r),score,sample in zip(selected,scores,samples)]
    maximum=max(row['absolute_difference'] for row in rows)
    result={'passed':maximum<1e-5,'n':len(rows),'active_skill_families':len({r['skill'] for r in rows}),
        'max_absolute_logprob_difference':maximum,'samples':rows,'model_identity':client.identity,
        'completed_unix':time.time(),'new_performance_episodes':0,
        'scope':'Original generated tokens are independently teacher-forced with the reconstructed active Skill prompt. No policy/candidate/reward changes.'}
    dest.write_text(json.dumps(result,indent=2)+'\n')
    assert result['passed']
    print(json.dumps({k:v for k,v in result.items() if k not in ('samples','model_identity')}))

if __name__=='__main__':main()
