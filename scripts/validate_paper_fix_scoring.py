"""Compare real generation and uncached exact scoring on identical GPU replicas."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.client import ModelClient
from research.config import load


def check(endpoint, config):
    client=ModelClient({**config,"endpoint":endpoint,"scoring_endpoints":[]})
    records=[]
    for seed, user in [(42,"The previous action is IN_PROGRESS. Choose Sleep to wait."),
                       (43,"Analyse host:admin_network_subnet_user_host_0 is the valid action for the observed host.")]:
        messages=[{"role":"system","content":"Return exactly one JSON object with string action and reason. Use the requested primitive action."},
                  {"role":"user","content":user}]
        generated=client.generate(messages,seed=seed,max_tokens=96,return_logprobs=True)
        sample={"messages":messages,"token_ids":generated["token_ids"]}
        actual=client.score([sample],use_cache=False)[0]
        assert abs(actual-generated["generation_logprob"])<1e-4
        records.append({"seed":seed,"response":generated["text"],"token_ids":generated["token_ids"],
            "generation_sum_logprob":generated["generation_logprob"],"uncached_teacher_forcing_sum_logprob":actual,
            "absolute_difference":abs(actual-generated["generation_logprob"])})
    return {"endpoint":endpoint,"identity":client.identity,"records":records}


if __name__=="__main__":
    config=load(ROOT/"research_configs/paper-cage4-fixed.yaml")["model"]
    endpoints=[f"http://127.0.0.1:{port}" for port in (8018,8019,8020,8021)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(lambda endpoint:check(endpoint,config),endpoints))
    for row in rows[1:]:
        assert row["identity"]==rows[0]["identity"]
        for got, expected in zip(row["records"],rows[0]["records"]):
            assert got["token_ids"]==expected["token_ids"]
            assert abs(got["uncached_teacher_forcing_sum_logprob"]-expected["uncached_teacher_forcing_sum_logprob"])<1e-4
    result={"passed":True,"cache_bypassed":True,"replicas":rows,
        "tolerance":1e-4,"weights_updated":False,"logprob_reduction":"sum of completion tokens, including EOS"}
    path=ROOT/"research_outputs/paper-fix/scoring-parity.json";path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"passed":True,"file":str(path),"replicas":4,"real_generation_checks":8,
        "maximum_absolute_difference":max(r["absolute_difference"] for row in rows for r in row["records"])}))
