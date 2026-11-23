"""Registered local campaign: frozen implementation, paired tests, external validation."""
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"research_outputs/behavior-v2"
sys.path.insert(0,str(ROOT))
from research.client import ModelClient
from research.config import load,validate
from research.experiment import atomic_json,run
from research.skills import SecuritySkillPool
from research.upstream import configure_encoder

SEEDS=list(range(22000,22004))
STAGES=[("baseline_raw","E0","finite",None),("baseline_context","E0","finite",None),
        ("static","E3","finite",0),("loop1","E3","finite",1),("loop2","E3","finite",2),
        ("transfer_baseline","E0","aggressive",None),("transfer_loop2","E3","aggressive",2)]


def pool_path(iteration):
    return OUT/"initial_pool.json" if iteration==0 else OUT/f"E3-hybrid-finite/train/pool-iteration-{iteration:04d}.json"


def locked_queue(callback):
    with (OUT/"queue.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        queue=json.loads((OUT/"queue.json").read_text())
        result=callback(queue)
        atomic_json(OUT/"queue.json",queue)
        return result


def worker(port):
    status_path=OUT/f"worker-{port}.json"
    atomic_json(status_path,{"status":"running","port":port})
    deadline=time.monotonic()+10*3600
    try:
        while True:
            def claim(queue):
                for task in queue:
                    if task["status"]=="pending" and (task["iteration"] is None or pool_path(task["iteration"]).is_file()):
                        task.update(status="running",port=port,pid=os.getpid(),started_unix=time.time())
                        return deepcopy(task)
                if all(t["status"]=="completed" for t in queue):return "finished"
                if any(t["status"]=="failed" for t in queue):raise RuntimeError("A registered evaluation failed")
                return None
            task=locked_queue(claim)
            if task=="finished":break
            if task is None:
                if (OUT/"E3-hybrid-finite/train/failure.json").exists():raise RuntimeError("Training failed")
                if time.monotonic()>deadline:raise TimeoutError("Registered pools or cases not completed")
                time.sleep(5);continue
            atomic_json(status_path,{"status":"running","port":port,"task":task["id"]})
            config=load(ROOT/"research_configs/behavior-v2.yaml")
            config.update(arm=task["arm"],red=task["red"],eval_seed=task["seed"],episodes=1,validation_episodes=0)
            if task["stage"]=="baseline_raw":
                config.update(behavior_context="legacy",skill_seed_set="legacy",case_presentation="json")
            config["model"].update(endpoint=f"http://127.0.0.1:{port}",scoring_endpoints=[])
            if task["iteration"] is not None:config["pool"]=str(pool_path(task["iteration"]))
            config=validate(config)
            destination=OUT/("validation" if task["split"]=="validation" else "paired")/task["id"]
            result=run(config,output=destination)
            assert result["status"]=="completed"
            def finish(queue):
                item=next(t for t in queue if t["id"]==task["id"])
                item.update(status="completed",summary=str(destination/"summary.json"),completed_unix=time.time())
            locked_queue(finish)
        atomic_json(status_path,{"status":"completed","port":port})
    except BaseException as exc:
        atomic_json(status_path,{"status":"failed","port":port,"error":f"{type(exc).__name__}: {exc}"})
        if 'task' in locals() and isinstance(task,dict):
            def fail(queue):
                next(t for t in queue if t["id"]==task["id"]).update(status="failed",error=str(exc))
            locked_queue(fail)
        raise


def train_then_worker():
    config=load(ROOT/"research_configs/behavior-v2-4gpu.yaml")
    config["validation_episodes"]=0  # Frozen-pool validation is scheduled separately.
    config=validate(config)
    result=run(config,"train")
    assert result["status"]=="completed"
    worker(8018)


def rl_controls():
    for red in ("finite","aggressive"):
        for seed in SEEDS:
            config=load(ROOT/"research_configs/behavior-v2.yaml")
            config.update(arm="E0",red=red,composition="all_rl",episodes=1,eval_seed=seed,validation_episodes=0)
            run(validate(config),output=OUT/"rl-controls"/f"{red}-seed-{seed}")


def launch():
    if (OUT/"protocol.json").exists():raise RuntimeError("Campaign already registered; do not overwrite or duplicate")
    smoke=json.loads((ROOT/"research_outputs/behavior-v2-smoke/E3-hybrid-finite/train/summary.json").read_text())
    assert smoke["status"]=="completed"
    assert json.loads((ROOT/"research_outputs/behavior-v2-smoke/integrity.json").read_text())["passed"]
    suite=ET.parse(ROOT/"research_outputs/behavior-v2-final-tests.xml").getroot().find("testsuite")
    assert int(suite.attrib["failures"])==int(suite.attrib["errors"])==0
    config=load(ROOT/"research_configs/behavior-v2.yaml")
    client=ModelClient(config["model"]);configure_encoder(client.embed)
    pool=SecuritySkillPool(config["pool_size"],seed_set=config["skill_seed_set"])
    atomic_json(pool_path(0),{"stage":"train","initialization_only":True,"episodes_completed":0,
        "total_training_episodes":0,"training_seeds":[],"pool":pool.snapshot(),"model_identity":client.identity})
    source=ROOT/"revisions/behavior-v2-registered-20261120";source.mkdir(parents=True,exist_ok=False)
    hashes={}
    for p in sorted((ROOT/"research").glob("*.py"))+[ROOT/"research.sh"]:
        relative=p.relative_to(ROOT);target=source/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target);hashes[str(relative)]=hashlib.sha256(p.read_bytes()).hexdigest()
    atomic_json(source/"manifest.json",hashes)
    stages=[("validation_baseline","E0","finite",None),("validation_initial","E3","finite",0)]
    tasks=[]
    def append(stage,arm,red,iteration,seed,split):
        tasks.append({"id":f"{stage}-seed-{seed}","stage":stage,"arm":arm,"red":red,
            "iteration":iteration,"seed":seed,"split":split,"status":"pending"})
    for stage,arm,red,iteration in stages:append(stage,arm,red,iteration,23000,"validation")
    for stage,arm,red,iteration in STAGES:
        for seed in SEEDS:append(stage,arm,red,iteration,seed,"heldout")
    for iteration in (1,2):append(f"validation_loop{iteration}","E3","finite",iteration,23000,"validation")
    atomic_json(OUT/"queue.json",tasks)
    preserved={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in (ROOT/"research_outputs/paper-fix").rglob("*") if p.is_file() and p.suffix in (".json",".jsonl")}
    atomic_json(OUT/"protocol.json",{"status":"registered_before_heldout_runs","heldout_seeds":SEEDS,
        "train_seeds":list(range(21000,21004)),"validation_seeds":[23000],"train_episodes":4,"iterations":2,
        "steps":500,"comparisons":STAGES,"heldout_episodes":28,"external_validation_episodes":4,
        "all_rl_control_episodes":8,"pool_size":24,"best_of_n":2,
        "fixed_model_identity":client.identity,"implementation_sha256":hashes,
        "smoke_integrity":"research_outputs/behavior-v2-smoke/integrity.json",
        "test_results":"research_outputs/behavior-v2-final-tests.xml",
        "source_snapshot":str(source.relative_to(ROOT)),"preserved_previous_results_sha256":preserved,
        "selection_rule":"No held-out reward is used to alter skills, gates, reward, hyperparameters or selected iteration. Report both loops and all seeds.",
        "fairness":"E0 and E3 retain identical Sleep/Monitor definitions and primitive action forms. New memory is given to both; raw E0 measures its contribution. KEEP Monitor/Sleep differences are reported separately.",
        "method":"Skill-Pro Algorithm 1, frozen Qwen, current-batch semantic gradients, RTG, exact clipped ratio, Best-of-N J>0, capacity maintenance unchanged."})
    env={**os.environ,"PYTHONHASHSEED":"0","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","WANDB_DISABLED":"true"}
    jobs=[]
    for label,arguments in [("train",["--train"]),*( (f"worker-{p}",["--worker",str(p)]) for p in (8119,8120,8121)),("rl-controls",["--rl-controls"])]:
        log=OUT/f"{label}.log"
        with log.open("w") as stream:
            proc=subprocess.Popen([str(ROOT/".venv/bin/python"),str(Path(__file__).resolve()),*arguments],cwd=ROOT,env=env,
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        jobs.append({"label":label,"pid":proc.pid,"log":str(log),"arguments":arguments})
    atomic_json(OUT/"jobs.json",jobs)
    print(json.dumps({"tasks":len(tasks),"heldout":28,"jobs":jobs},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--launch",action="store_true");parser.add_argument("--train",action="store_true")
    parser.add_argument("--worker",type=int);parser.add_argument("--rl-controls",action="store_true")
    args=parser.parse_args()
    if args.launch:launch()
    elif args.train:train_then_worker()
    elif args.worker:worker(args.worker)
    elif args.rl_controls:rl_controls()
    else:parser.error("Select one explicit campaign role")
