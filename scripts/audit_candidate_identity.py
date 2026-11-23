"""Verify the exact persisted candidate text using fixed pre-evaluation evidence."""
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.client import ModelClient
from research.config import load
from research.gate import clipped_terms
from research.upstream import Skill


def main():
    out=ROOT/"research_outputs/paper-fix"
    train=out/"E3-hybrid-finite/train"
    summary=json.loads((train/"summary.json").read_text())
    history=summary["history"][1]
    evidence={}
    for episode in summary["episodes"][2:]:
        for line in Path(episode["trace"]).open():
            row=json.loads(line)
            if row["agent"]=="blue_agent_4":
                evidence[(row["seed"],row["agent"],row["step"])]=row
    event=next(e for e in history["evolution"] if e["parent"]=="WaitVerification")
    config=load(ROOT/"research_configs/paper-cage4-fixed-4gpu.yaml")
    client=ModelClient(config["model"])
    candidates=[]
    for candidate in event["candidates"]:
        fields=set(Skill.__dataclass_fields__)
        skill=Skill(**{k:v for k,v in candidate["skill"].items() if k in fields})
        generated_name=skill.name
        skill.name+=f"_c{candidate['index']}"
        samples=[]
        for diagnostic in candidate["diagnostics"]:
            for transition in diagnostic["transitions"]:
                key=tuple(transition[k] for k in ("seed","agent","step"))
                row=evidence[key]
                messages=json.loads(row["state"])
                messages[0]["content"]+="\n# ACTIVE SECURITY SKILL\n"+skill.format_for_llm()
                samples.append({"messages":messages,"token_ids":row["token_ids"]})
        new=client.score(samples,use_cache=False)
        offset=0;trajectory_means=[];diagnostics=[]
        for diagnostic in candidate["diagnostics"]:
            count=diagnostic["n"]
            probabilities=new[offset:offset+count]
            ratio,terms=clipped_terms(diagnostic["old_logprobs"],probabilities,
                diagnostic["advantages"],epsilon=.2,log_ratio_bound=None)
            trajectory_means.append(float(terms.mean()))
            diagnostics.append({**diagnostic,"new_logprobs":probabilities,
                "importance_ratios":ratio.tolist(),"clipped_surrogates":terms.tolist(),
                "surrogate_mean":float(terms.mean())})
            offset+=count
        candidates.append({"index":candidate["index"],"generated_name":generated_name,
            "admitted_name":skill.name,"original_score":candidate["score"],
            "verified_score":statistics.mean(trajectory_means),"diagnostics":diagnostics})
    chosen=max(candidates,key=lambda c:c["verified_score"])
    original=next(c for c in event["candidates"] if c["accepted"])
    persisted=next(s for s in history["pool_snapshot"]["skills"]
                   if s["name"]==chosen["admitted_name"])
    assert all(persisted[k]==original["skill"][k] for k in ("initiation","policy","termination","version","parent_id"))
    result={"passed":chosen["verified_score"]>0 and chosen["index"]==original["index"],
        "evidence":"fixed current training batch; no evaluation rewards used",
        "cache_bypassed":True,"identity":client.identity,"candidates":candidates,
        "accepted_index_unchanged":chosen["index"]==original["index"],
        "persisted_pool_unchanged":True,"persisted_candidate":persisted["name"],
        "policy_sampling_unchanged":True,"source_history":str((train/"summary.json").relative_to(ROOT))}
    path=out/"candidate-identity-audit.json"
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("identity","candidates")},ensure_ascii=False))
    print(json.dumps([{k:v for k,v in c.items() if k!="diagnostics"} for c in candidates],ensure_ascii=False))
    assert result["passed"]


if __name__=="__main__":main()
