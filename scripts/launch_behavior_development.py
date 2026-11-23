"""Development-only 500-step validation; these seeds are never final test seeds."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.client import ModelClient
from research.config import load,validate
from research.experiment import atomic_json
from research.skills import SecuritySkillPool
from research.upstream import configure_encoder


if __name__=="__main__":
    out=ROOT/"research_outputs/behavior-v1-development"
    if (out/"jobs.json").exists():raise RuntimeError("Development jobs already exist")
    config=load(ROOT/"research_configs/behavior-v1.yaml")
    client=ModelClient(config["model"]);configure_encoder(client.embed)
    pool=SecuritySkillPool(config["pool_size"],seed_set=config["skill_seed_set"])
    snapshot=out/"initial_pool.json"
    atomic_json(snapshot,{"stage":"train","initialization_only":True,"episodes_completed":0,
        "total_training_episodes":0,"training_seeds":[],"pool":pool.snapshot(),"model_identity":client.identity})
    jobs=[]
    for arm,port in (("E0",8119),("E3",8120)):
        case=load(ROOT/"research_configs/behavior-v1.yaml")
        case.update(arm=arm,episodes=1,eval_seed=23000,validation_episodes=0,development_validation=True)
        case["model"].update(endpoint=f"http://127.0.0.1:{port}",scoring_endpoints=[])
        if arm=="E3":case["pool"]=str(snapshot)
        case=validate(case)
        # Run in a fresh process, independent of terminal lifetime.
        script=out/f"config-{arm}.json";atomic_json(script,case)
        log=out/f"{arm}.log"
        code="import json,sys; from research.experiment import run; run(json.load(open(sys.argv[1])),output=sys.argv[2])"
        env={**os.environ,"PYTHONHASHSEED":"0","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","WANDB_DISABLED":"true","HF_HUB_OFFLINE":"1"}
        with log.open("w") as stream:
            proc=subprocess.Popen([str(ROOT/".venv/bin/python"),"-c",code,str(script),str(out/arm)],cwd=ROOT,
                env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        jobs.append({"arm":arm,"port":port,"pid":proc.pid,"log":str(log),"started_unix":time.time()})
    atomic_json(out/"jobs.json",jobs)
    print(json.dumps(jobs,indent=2))
