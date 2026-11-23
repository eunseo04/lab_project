"""Audit real role formatting and uncached action probabilities before long runs."""
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.client import ModelClient
from research.config import load
from research.upstream import Skill


if __name__=="__main__":
    root=ROOT/"research_outputs/behavior-v2-smoke/E3-hybrid-finite/train"
    summary=json.loads((root/"summary.json").read_text())
    assert summary["status"]=="completed"
    pool=summary["history"][-1]["pool_snapshot"]
    skills={s["name"]:Skill(**{k:v for k,v in s.items() if k in Skill.__dataclass_fields__}) for s in pool["skills"]}
    episodes=summary["episodes"]+summary["history"][-1]["validation_episodes"]
    decisions=[];counts={"selector":0,"termination":0,"policy_errors":0}
    for episode in episodes:
        for call in episode["calls"]:
            if call["purpose"]=="skill_selection":
                counts["selector"]+=1
                assert re.search(r"<choice>.*?</choice>",call.get("response",""),re.I|re.S)
            if call["purpose"]=="termination":
                counts["termination"]+=1
                assert re.search(r"<status>\s*(DONE|CONTINUE)\s*</status>",call.get("response",""),re.I)
        for line in Path(episode["trace"]).open():
            row=json.loads(line)
            if row["agent"]!="blue_agent_4":continue
            assert "behavior_context" in row and "behavior_outcome" in row
            if row.get("policy_invoked"):
                assert not row.get("parse_error")
                decisions.append(row)
    chosen=[];seen=set()
    for row in sorted(decisions,key=lambda r:(not bool(r["behavior_context"]["open_cases"]),r["step"])):
        key=(row["requested_action"].split()[0],row.get("skill_id"))
        if key not in seen:chosen.append(row);seen.add(key)
        if len(chosen)==8:break
    client=ModelClient(load(ROOT/"research_configs/behavior-v2-4gpu.yaml")["model"])
    samples=[]
    for row in chosen:
        messages=json.loads(row["state"])
        if row.get("skill_id"):
            skill=skills[row["skill_id"]]
            assert skill.version==row["skill_version"]
            messages[0]["content"]+="\n# ACTIVE SECURITY SKILL\n"+skill.format_for_llm()
        samples.append({"messages":messages,"token_ids":row["token_ids"]})
    scores=client.score(samples,use_cache=False)
    checks=[{"seed":r["seed"],"step":r["step"],"action":r["requested_action"],"skill":r.get("skill_id"),
        "generation_logprob":r["generation_logprob"],"uncached_logprob":score,
        "absolute_difference":abs(r["generation_logprob"]-score)} for r,score in zip(chosen,scores)]
    assert checks and max(c["absolute_difference"] for c in checks)<1e-4
    assert not any(e["error"] for h in summary["history"] for e in h["evolution"])
    result={"passed":True,"real_episodes":len(episodes),"model_identity":client.identity,
        "role_counts":counts,"cache_bypassed":True,"probability_checks":checks,
        "maximum_absolute_difference":max(c["absolute_difference"] for c in checks),
        "validation_is_not_final_test":True}
    destination=ROOT/"research_outputs/behavior-v2-smoke/integrity.json"
    destination.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("model_identity","probability_checks")},indent=2))
