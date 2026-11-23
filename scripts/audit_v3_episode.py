"""Shared raw-trace audit for the frozen manual v3 candidate."""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
FORK=Path(os.environ.get("BEHAVIOR_AUDIT_SOURCE_ROOT",str(ROOT/"revisions/behavior-v3-dev-20261121"))).resolve()
sys.path.insert(0,str(FORK))
from research.behavior import ObservableCaseLedger
from research.config import digest
from behavior_metrics import episode_metrics


def audit_episode(path,seed,source_hashes,fixed_identity=None):
    path=Path(path);summary=json.loads(path.read_text());config=summary["config"]
    manifest=json.loads((path.parent/"manifest.json").read_text())
    assert manifest["implementation_sha256"]==source_hashes
    assert manifest["config"]==config and manifest["config_digest"]==digest(config)
    assert summary["status"]=="completed" and summary["result_kind"]=="real_cage4"
    assert summary["pool_frozen"] and summary["pool_sha256"]==summary["initial_pool_sha256"]
    assert config["connection_leads"] and config["case_presentation"]=="readable"
    assert config["selector"]["verification_history"]=="accepted_decisions"
    assert config["steps"]==500 and config["scoring"]=="official" and config["composition"]=="hybrid"
    assert config["profile"]=="controlled-comparison" and config["action_contract"]=="explicit"
    assert config["communication"]=="paper_projected_keep" and config["skip_busy_policy"]
    assert len(summary["episodes"])==1
    episode=summary["episodes"][0];assert episode["seed"]==seed
    identity=episode["model_identity"]
    assert identity["weights_frozen"] and identity["trainable_parameters"]==0
    if fixed_identity is not None:assert identity==fixed_identity
    provenance=summary.get("training_provenance");available=set()
    if config["arm"]=="E3":
        assert provenance and hashlib.sha256(Path(provenance["path"]).read_bytes()).hexdigest()==provenance["sha256"]
        pool=json.loads(Path(provenance["path"]).read_text())
        assert pool["initialization_only"] and pool["episodes_completed"]==0
        assert not pool["training_seeds"] and len(pool["pool"]["skills"])==7
        available={s["name"].lower() for s in pool["pool"]["skills"]}
    rows=[json.loads(line) for line in Path(episode["trace"]).open()]
    assert len(rows)==2495
    ticks=defaultdict(list)
    for row in rows:ticks[row["step"]].append(row)
    raw=[];scored=[]
    for step,items in sorted(ticks.items()):
        assert {r["agent"] for r in items}=={f"blue_agent_{i}" for i in range(5)}
        assert len({(r["joint_reward"],r["score_included"],r["terminal"]) for r in items})==1
        assert all(r["policy"]==("llm" if r["agent"]=="blue_agent_4" else "rl") for r in items)
        raw.append(items[0]["joint_reward"])
        scored.append(items[0]["joint_reward"] if items[0]["score_included"] else 0.)
    total=sum(scored)
    assert total==episode["joint_return"] and len(ticks)==499 and episode["scored_transitions"]==498
    assert raw==episode["raw_joint_rewards"] and scored==episode["scored_step_rewards"]
    meta={"selection_xml_errors":0,"termination_xml_errors":0,"unknown_selection_names":0}
    for call in episode["calls"]:
        if call["purpose"]=="skill_selection":
            match=re.search(r"<choice>\s*(.*?)\s*</choice>",call.get("response",""),re.I|re.S)
            meta["selection_xml_errors"]+=not bool(match)
            if match and match[1].lower()!="none" and match[1].lower() not in available:meta["unknown_selection_names"]+=1
        if call["purpose"]=="termination":
            meta["termination_xml_errors"]+=not bool(re.search(r"<status>\s*(DONE|CONTINUE)\s*</status>",call.get("response",""),re.I))
    metrics=episode_metrics(episode,rows,ledger_factory=lambda agent:ObservableCaseLedger(agent,connection_leads=True))
    own=[r for r in rows if r["agent"]=="blue_agent_4"]
    assert metrics["blue_agent_4"]["learned_skill_decisions"]==0
    assert sum(c["purpose"]=="policy" for c in episode["calls"])==sum(bool(r.get("policy_invoked")) for r in own)
    due=[r for r in own if r.get("policy_invoked") and any(c["investigation_due"] for c in r["behavior_context"]["investigation_leads"])]
    investigates=sum(r["requested_action"].split()[0]=="Analyse" and
        r["requested_action"].split()[-1] in {c["host"] for c in r["behavior_context"]["investigation_leads"] if c["investigation_due"]} for r in due)
    result={"arm":config["arm"],"red":config["red"],"seed":seed,"return":total,
        "behavior":metrics["blue_agent_4"],"meta":meta,"due_lead_opportunities":len(due),
        "targeted_lead_investigations":investigates,"source":str(path.relative_to(ROOT))}
    prompt=next(r["state"] for r in own if r.get("policy_invoked"))
    return result,prompt,identity
