from pathlib import Path
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.client import ModelClient
from research.config import load

c=load(ROOT/'research_configs/controlled.yaml')
client=ModelClient(c['model'])
rows=[]
for seed in [42,43]:
    messages=[{'role':'system','content':'Return only a JSON object with action "Sleep" and a brief reason.'},
              {'role':'user','content':'The previous defensive action is in progress. What is the next action?'}]
    generated=client.generate(messages,seed=seed,return_logprobs=True,max_tokens=96)
    scored=client.score([{'messages':messages,'token_ids':generated['token_ids']}],use_cache=False)[0]
    difference=abs(scored-generated['generation_logprob'])
    rows.append({'seed':seed,'response':generated['text'],'tokens':len(generated['token_ids']),
        'generation_logprob':generated['generation_logprob'],'teacher_forced_logprob':scored,
        'absolute_difference':difference,'pass':difference<0.0001,'generation_seconds':generated['seconds']})
result={'model':client.identity,'sampling':{'temperature':1.0,'top_p':1.0,'top_k':0,'enable_thinking':False},
        'rows':rows,'passed':all(r['pass'] for r in rows),
        'tolerance_note':'Same incremental cache path, exact observed token IDs; absolute sequence logprob tolerance 1e-4.'}
(ROOT/'research_outputs/scoring-validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
if not result['passed']: raise SystemExit('Exact scoring validation failed; do not claim a verified gate')
