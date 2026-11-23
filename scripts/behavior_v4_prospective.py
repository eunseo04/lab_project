"""Prospective paired evaluation of the repaired, frozen manual v4 candidate.

No training, pool selection, reward change, or registered v2 task is performed.
Transfer is conditional on a predeclared FiniteState success rule.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
FORK=ROOT/"revisions/behavior-v4-observation-20261121"
OUT=ROOT/"research_outputs/behavior-v4-prospective"
sys.path.insert(0,str(FORK))
from research.config import validate
from research.experiment import run,atomic_json


def read(path):return json.loads(Path(path).read_text())


def register():
    if (OUT/"protocol.json").exists():raise RuntimeError("Prospective protocol already exists")
    development=read(ROOT/"research_outputs/behavior-v4-lifecycle-development/results.json")
    assert development["audit_passed"] and development["criterion_passed"]
    old=read(ROOT/"research_outputs/behavior-v4-lifecycle-development/protocol.json")
    for name,digest in old["source_sha256"].items():
        assert hashlib.sha256((FORK/name).read_bytes()).hexdigest()==digest
    OUT.mkdir(parents=True)
    pool_path=ROOT/"research_outputs/behavior-v4-lifecycle-development/initial_pool.json"
    pool=read(pool_path);assert pool["initialization_only"]
    atomic_json(OUT/"initial_pool.json",pool)
    protocol={"registered_at_utc":datetime.now(timezone.utc).isoformat(),
        "purpose":"Prospective test after repaired observation/lifecycle known-case validation; not learned Skill-Pro performance",
        "source_root":str(FORK),"source_sha256":old["source_sha256"],
        "driver_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pool_sha256":hashlib.sha256((OUT/"initial_pool.json").read_bytes()).hexdigest(),
        "finite_seeds":list(range(29000,29004)),"conditional_transfer_seeds":list(range(30000,30004)),
        "steps":500,"arms":["E0 same v3 evidence","E3 seven manual skills plus v3 evidence"],
        "same_initial_actor_state_required":True,"frozen_pool":True,
        "finite_success_rule":"mean(E3-E0)>0 and at least 3 of 4 paired seeds improve",
        "transfer_rule":"Run AggressiveFSM only if the finite rule passes; use the same frozen manual pool without selection or retraining",
        "transfer_success_rule":"mean(E3-E0)>0 and at least 3 of 4 paired seeds improve",
        "no_further_tuning_on_this_test":True,"development_seed":development["seed"],
        "main_v2_campaign_modified":False,"manual_initialization_not_learning":True,
        "fixed_model_identity":read(ROOT/"research_outputs/behavior-v4-lifecycle-development/E3/summary.json")["episodes"][0]["model_identity"]}
    queue=[{"id":f"{red}-{arm}-seed-{seed}","red":red,"arm":arm,"seed":seed,"status":"pending"}
        for red,seeds in (("finite",protocol["finite_seeds"]),("aggressive",protocol["conditional_transfer_seeds"]))
        for seed in seeds for arm in ("E0","E3")]
    atomic_json(OUT/"protocol.json",protocol);atomic_json(OUT/"queue.json",queue)
    env={**os.environ,"PYTHONHASHSEED":"0","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
        "HF_HUB_OFFLINE":"1","WANDB_DISABLED":"true"}
    jobs=[]
    for port in (8018,8119,8120,8121):
        with (OUT/f"worker-{port}.log").open("w") as stream:
            p=subprocess.Popen([str(FORK/".venv/bin/python"),str(Path(__file__).resolve()),"--worker",str(port)],cwd=FORK,
                env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        jobs.append({"port":port,"pid":p.pid})
    atomic_json(OUT/"jobs.json",jobs);print(json.dumps(jobs))


def finite_decision(queue):
    tasks=[t for t in queue if t["red"]=="finite"]
    if any(t["status"]!="completed" for t in tasks):return None
    values={(t["arm"],t["seed"]):read(t["summary"])["episodes"][0]["joint_return"] for t in tasks}
    seeds=sorted({t["seed"] for t in tasks})
    differences=[values[("E3",s)]-values[("E0",s)] for s in seeds]
    decision={"paired_differences":differences,"mean_difference":sum(differences)/len(differences),
        "wins":sum(d>0 for d in differences),"seeds":seeds}
    decision["passed"]=decision["mean_difference"]>0 and decision["wins"]>=3
    return decision


def claim(port):
    with (OUT/"queue.lock").open("w") as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        queue=read(OUT/"queue.json")
        if any(t["status"]=="failed" for t in queue):return None,True
        decision=finite_decision(queue)
        if decision is not None:
            atomic_json(OUT/"finite_decision.json",decision)
            if not decision["passed"]:
                for task in queue:
                    if task["red"]=="aggressive" and task["status"]=="pending":
                        task.update(status="skipped_by_prespecified_rule",reason="FiniteState criterion failed")
        ready=next((t for t in queue if t["status"]=="pending" and (t["red"]=="finite" or decision and decision["passed"])),None)
        if ready:
            ready.update(status="running",port=port,pid=os.getpid(),started_unix=time.time())
        atomic_json(OUT/"queue.json",queue)
        return ready,all(t["status"] in ("completed","skipped_by_prespecified_rule") for t in queue)


def finish(task,summary=None,error=None):
    with (OUT/"queue.lock").open("w") as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        queue=read(OUT/"queue.json")
        item=next(t for t in queue if t["id"]==task["id"])
        item.update(status="failed" if error else "completed",completed_unix=time.time())
        if error:item["error"]=error
        else:item["summary"]=str(summary)
        atomic_json(OUT/"queue.json",queue)


def worker(port):
    protocol=read(OUT/"protocol.json")
    for name,digest in protocol["source_sha256"].items():
        assert hashlib.sha256((FORK/name).read_bytes()).hexdigest()==digest
    assert hashlib.sha256((OUT/"initial_pool.json").read_bytes()).hexdigest()==protocol["pool_sha256"]
    while True:
        task,done=claim(port)
        if task is None:
            if done:return
            time.sleep(5);continue
        config=read(ROOT/"research_outputs/behavior-v4-lifecycle-development/E3-config.json")
        config.pop("development_validation",None);config.pop("known_debug_seed",None)
        config.update(arm=task["arm"],red=task["red"],eval_seed=task["seed"],episodes=1,validation_episodes=0,
            pool=str(OUT/"initial_pool.json") if task["arm"]=="E3" else None,output=str(OUT))
        config["model"].update(endpoint=f"http://127.0.0.1:{port}",scoring_endpoints=[])
        config=validate(config);directory=OUT/task["id"]
        try:
            run(config,output=directory)
            finish(task,directory/"summary.json")
        except Exception as exc:
            finish(task,error=f"{type(exc).__name__}: {exc}");raise


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--launch",action="store_true");parser.add_argument("--worker",type=int)
    args=parser.parse_args()
    if args.launch:register()
    elif args.worker:worker(args.worker)
    else:parser.error("Choose --launch or --worker PORT")
