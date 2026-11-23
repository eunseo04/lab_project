"""Cross-check actual Gate old-policy scores against archived generation scores."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def verify(root,replay_missing=False,endpoint=None):
    root=Path(root)
    checked={}
    unavailable=0
    replayed=0
    clients={}
    for checkpoint in sorted(root.glob("*/train/checkpoint.json")):
        value=json.loads(checkpoint.read_text())
        episodes={}
        directories=[checkpoint.parent]
        for directory in directories:
            initial=directory/"initial_pool.json"
            if initial.exists():
                ancestor=Path(json.loads(initial.read_text())["training_provenance"]["path"]).parent
                if ancestor not in directories: directories.append(ancestor)
        for directory in directories:
            for path in sorted(directory.glob("episode-*.jsonl")):
                with path.open() as stream:
                    for line in stream:
                        t=json.loads(line)
                        key=(t["seed"],t["agent"],t["step"],t.get("skill_id"),t.get("skill_version"))
                        if key in episodes and episodes[key].get("token_ids")!=t.get("token_ids"):
                            raise ValueError("Ambiguous reused training seed; action reference must be disambiguated")
                        episodes[key]=t
        for history in value.get("history",[]):
            for evolution in history.get("evolution",[]):
                for candidate in evolution.get("candidates",[]):
                    for diagnostic in candidate.get("diagnostics",[]):
                        for ref,score in zip(diagnostic["transitions"],diagnostic["old_logprobs"]):
                            row=episodes[(ref["seed"],ref["agent"],ref["step"],ref["skill_id"],ref["skill_version"])]
                            key=(str(checkpoint),row["seed"],row["agent"],row["step"],evolution["parent"])
                            if key in checked:
                                if checked[key]["teacher_forcing"]!=score: raise ValueError("Old-policy scores changed across candidates")
                                continue
                            generation=row.get("generation_logprob")
                            origin="archived_generation"
                            if generation is None:
                                unavailable+=1
                                if not replay_missing:continue
                                sys.path.insert(0,str(ROOT))
                                from research.client import ModelClient
                                from research.upstream import Skill
                                config=json.loads((checkpoint.parent/"manifest.json").read_text())["config"]["model"].copy()
                                if endpoint:config["endpoint"]=endpoint
                                if config["endpoint"] not in clients:clients[config["endpoint"]]=ModelClient(config)
                                old=evolution["old_skill"]
                                skill=Skill(**{k:v for k,v in old.items() if k in Skill.__dataclass_fields__})
                                messages=json.loads(row["state"])
                                messages[0]["content"]+="\n# ACTIVE SECURITY SKILL\n"+skill.format_for_llm()
                                seed=int(hashlib.sha256(f"{row['seed']}:{row['step']}:{row['agent']}:policy".encode()).hexdigest()[:8],16)
                                generated=clients[config["endpoint"]].generate(messages,seed=seed,return_logprobs=True,purpose="gate_replay_validation")
                                if generated["token_ids"]!=row["token_ids"]:
                                    raise ValueError("Archived completion could not be reproduced; no parity claim")
                                generation=generated["generation_logprob"];origin="replayed_generation_identical_tokens";replayed+=1
                            if row.get("skill_id")!=evolution["parent"]:
                                raise ValueError("Gate parent differs from the behavior Skill")
                            checked[key]={"checkpoint":str(checkpoint),**ref,"generation":generation,"origin":origin,
                                "teacher_forcing":score,"absolute_difference":abs(generation-score)}
    results=list(checked.values())
    maximum=max((r["absolute_difference"] for r in results),default=None)
    result={"passed":bool(results) and maximum<=1e-4,"checked_unique_actions":len(results),
        "unavailable_older_records":unavailable,"replayed_unique_actions":replayed,
        "max_absolute_difference":maximum,"tolerance":1e-4,"actions":results}
    (root/"gate_policy_parity.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="actions"}))
    if not result["passed"]: raise SystemExit(1)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root");p.add_argument("--replay-missing",action="store_true");p.add_argument("--endpoint")
    args=p.parse_args()
    try:verify(args.root,args.replay_missing,args.endpoint)
    except Exception as exc:
        (Path(args.root)/"gate_policy_parity_failure.json").write_text(json.dumps({"passed":False,"error":str(exc)}))
        raise
