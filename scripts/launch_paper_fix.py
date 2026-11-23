"""Launch the registered experiment, reserving GPU 0 for sequential training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.client import ModelClient
from research.config import load
from research.experiment import atomic_json
from research.skills import SecuritySkillPool
from research.upstream import configure_encoder


if __name__=="__main__":
    out=ROOT/"research_outputs/paper-fix"
    if (out/"execution_jobs.json").exists():
        raise RuntimeError("Jobs were already launched; inspect/resume them instead of starting duplicates")
    config=load(ROOT/"research_configs/paper-cage4-fixed-4gpu.yaml")
    client=ModelClient(config["model"]);configure_encoder(client.embed)
    pool=SecuritySkillPool(config["pool_size"])
    atomic_json(out/"initial_pool.json",{"stage":"train","initialization_only":True,
        "description":"Untrained initial pool: zero environment episodes and zero optimization steps",
        "episodes_completed":0,"total_training_episodes":0,"training_seeds":[],
        "pool":pool.snapshot(),"model_identity":client.identity,"algorithm":config["evolution"]})
    smoke=json.loads((ROOT/"research_outputs/paper-fix-smoke/E3-hybrid-finite/train/summary.json").read_text())
    assert smoke["status"]=="completed"
    env=os.environ.copy();env.update(PYTHONHASHSEED="0",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",
        HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",WANDB_DISABLED="true")
    jobs=[]
    def start(name,command):
        path=ROOT/f"research_outputs/jobs/{name}.log"
        with path.open("a") as log:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record={"name":name,"pid":proc.pid,"command":command,"log":str(path),"started_unix":time.time()}
        atomic_json(ROOT/f"research_outputs/jobs/{name}.json",record)
        jobs.append(record)
    start("paper-fix-train-02",[str(ROOT/"research.sh"),"train","--config","research_configs/paper-cage4-fixed-4gpu.yaml"])
    for port,seed in ((8019,12001),(8020,12002),(8021,12003)):
        start(f"paper-fix-eval-02-{port}",[str(ROOT/".venv/bin/python"),"scripts/run_paper_fix_evaluations.py","--port",str(port),"--seeds",str(seed)])
    start("paper-fix-eval-02-deferred12000",[str(ROOT/".venv/bin/python"),"scripts/run_paper_fix_deferred_seed.py","--seed","12000"])
    atomic_json(out/"execution_jobs.json",jobs)
    print(json.dumps({"jobs":jobs,"initial_pool_size":len(pool.skills)},indent=2))
