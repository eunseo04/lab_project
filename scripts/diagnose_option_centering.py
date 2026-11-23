"""Offline proposal only: compare a conditional baseline with the original Gate.

This does NOT change the registered run, approve a candidate, or claim reward
improvement. It uses stored exact probabilities to make the proposal reviewable.
"""
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.gate import clipped_terms


def centered_surrogate(diagnostics,epsilon=.2):
    offset=statistics.mean(statistics.mean(d["advantages"]) for d in diagnostics)
    scores=[];parent=[]
    for d in diagnostics:
        advantages=[a-offset for a in d["advantages"]]
        _,terms=clipped_terms(d["old_logprobs"],d["new_logprobs"],advantages,epsilon,log_ratio_bound=None)
        scores.append(float(terms.mean()))
        parent.append(statistics.mean(advantages))
    reference=statistics.mean(parent)
    assert abs(reference)<1e-9
    return statistics.mean(scores),offset,reference


if __name__=="__main__":
    source=ROOT/"research_outputs/behavior-v2/E3-hybrid-finite/train/pool-iteration-0001.json"
    history=json.loads(source.read_text())["history"][-1]
    rows=[]
    for event in history["evolution"]:
        for candidate in event["candidates"]:
            if candidate.get("score") is None:continue
            score,offset,reference=centered_surrogate(candidate["diagnostics"])
            rows.append({"parent":event["parent"],"candidate_index":candidate["index"],
                "original_score":candidate["score"],"original_accepted":candidate["accepted"],
                "parent_original_surrogate":offset,"conditional_baseline_diagnostic_score":score,
                "conditional_parent_reference":reference})
    result={"status":"offline_diagnostic_not_executed_as_training","source":str(source.relative_to(ROOT)),
        "original_reward_or_algorithm_modified":False,"new_model_or_environment_performance_measured":False,
        "proposal":"Condition the running RTG baseline on the invoked Skill; retain exact ratio, PPO clipping and J>0. This is a separate baseline-estimation variant, not an unmodified-paper reproduction.",
        "results":rows}
    out=ROOT/"research_outputs/behavior-v2/option-centering-proposal.json"
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps([{k:v for k,v in r.items() if k!="conditional_parent_reference"} for r in rows],indent=2))
