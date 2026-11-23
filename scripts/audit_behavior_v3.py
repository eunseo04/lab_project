"""Audit the development pair; never pool it with registered heldout scores."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
FORK=ROOT/"revisions/behavior-v3-dev-20261121"
sys.path.insert(0,str(FORK))
from research.behavior import ObservableCaseLedger
from behavior_metrics import episode_metrics

OUT=ROOT/"research_outputs/behavior-v3-development"


if __name__=="__main__":
    protocol=json.loads((OUT/"protocol.json").read_text())
    for name,digest in protocol["source_sha256"].items():
        assert hashlib.sha256((FORK/name).read_bytes()).hexdigest()==digest
    results={};prompts=[];identities=[]
    for arm in ("E0","E3"):
        summary=json.loads((OUT/arm/"summary.json").read_text())
        assert summary["status"]=="completed" and summary["result_kind"]=="real_cage4"
        assert summary["pool_frozen"] and summary["pool_sha256"]==summary["initial_pool_sha256"]
        config=summary["config"]
        assert config["connection_leads"] and config["case_presentation"]=="readable"
        assert config["selector"]["verification_history"]=="accepted_decisions"
        episode=summary["episodes"][0];assert episode["seed"]==23000
        rows=[json.loads(line) for line in Path(episode["trace"]).open()]
        ticks=defaultdict(list)
        for row in rows:ticks[row["step"]].append(row)
        total=0
        for step,items in ticks.items():
            assert len(items)==5 and len({(r["joint_reward"],r["score_included"]) for r in items})==1
            if items[0]["score_included"]:total+=items[0]["joint_reward"]
        assert total==episode["joint_return"] and len(ticks)==499 and episode["scored_transitions"]==498
        for call in episode["calls"]:
            if call["purpose"]=="skill_selection":assert re.search(r"<choice>.*?</choice>",call["response"],re.I|re.S)
            if call["purpose"]=="termination":assert re.search(r"<status>\s*(DONE|CONTINUE)\s*</status>",call["response"],re.I)
        metrics=episode_metrics(episode,rows,ledger_factory=lambda agent:ObservableCaseLedger(agent,connection_leads=True))
        own=[r for r in rows if r["agent"]=="blue_agent_4"]
        prompts.append(next(r["state"] for r in own if r.get("policy_invoked")))
        identities.append(episode["model_identity"])
        due=[r for r in own if r.get("policy_invoked") and any(c["investigation_due"] for c in r["behavior_context"]["investigation_leads"])]
        investigates=sum(r["requested_action"].split()[0]=="Analyse" and
            r["requested_action"].split()[-1] in {c["host"] for c in r["behavior_context"]["investigation_leads"] if c["investigation_due"]} for r in due)
        results[arm]={"return":total,"behavior":metrics["blue_agent_4"],"due_lead_opportunities":len(due),
            "targeted_lead_investigations":investigates,"source":str((OUT/arm/"summary.json").relative_to(ROOT))}
    assert prompts[0]==prompts[1] and identities[0]==identities[1]
    difference=results["E3"]["return"]-results["E0"]["return"]
    result={"status":"completed","audit_passed":True,"seed":23000,"development_only":True,
        "manual_initialization_not_learning":True,"same_initial_actor_state":True,"same_frozen_model":True,
        "arms":results,"E3_minus_E0":difference,"criterion_passed":difference>0,
        "recommendation":"Fresh prospective evaluation needed before any improvement claim" if difference>0 else "Not adopted: matched development criterion was not met"}
    (OUT/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="arms"},ensure_ascii=False,indent=2))
