"""CPU-only exact replay diagnostics for completed registered model episodes."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get("BEHAVIOR_REPLAY_QUEUE_ROOT",str(ROOT/"research_outputs/behavior-v2"))).resolve()


def write_state(data):
    path=OUT/"reward_audit_queue_state.json"
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(data,indent=2)+"\n");temp.replace(path)


if __name__=="__main__":
    env={**os.environ,"PYTHONHASHSEED":"0","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
        "HF_HUB_OFFLINE":"1","WANDB_DISABLED":"true"}
    while True:
        queue=json.loads((OUT/"queue.json").read_text())
        pending=[];passed=[]
        for task in queue:
            if task["status"]!="completed":continue
            target=OUT/"reward-decomposition"/(task["id"]+".json")
            if target.exists() and json.loads(target.read_text()).get("passed"):
                passed.append(task["id"])
            else:pending.append(task)
        write_state({"status":"running","exact_replays_completed":passed,
            "ready_pending":[t["id"] for t in pending],
            "model_episodes_remaining":sum(t["status"] not in ("completed","skipped_by_prespecified_rule") for t in queue),
            "new_performance_samples":0,"pid":os.getpid()})
        if pending:
            task=pending[0]
            with (OUT/"reward-replay.log").open("a") as stream:
                result=subprocess.run([str(ROOT/".venv/bin/python"),str(ROOT/"scripts/replay_behavior_pipeline.py"),task["summary"]],
                    cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode:
                write_state({"status":"failed","task":task["id"],"returncode":result.returncode,
                    "new_performance_samples":0});raise SystemExit(result.returncode)
            continue
        if all(t["status"] in ("completed","skipped_by_prespecified_rule") for t in queue):
            write_state({"status":"completed","exact_replays_completed":passed,
                "model_episodes_remaining":0,"new_performance_samples":0,
                "skipped_by_prespecified_rule":[t["id"] for t in queue if t["status"]=="skipped_by_prespecified_rule"]});break
        if any(t["status"]=="failed" for t in queue):
            write_state({"status":"model_failure","new_performance_samples":0});break
        time.sleep(30)
