"""Predeclared per-seed evaluations; never choose a pool using held-out rewards."""
import argparse
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.config import load, validate
from research.experiment import run, atomic_json


def await_pool(iteration):
    train=ROOT/"research_outputs/paper-fix/E3-hybrid-finite/train"
    path=train/f"pool-iteration-{iteration:04d}.json"
    deadline=time.monotonic()+8*3600
    while not path.exists():
        if (train/"failure.json").exists():
            raise RuntimeError("Training failed; do not fabricate a trained-pool evaluation")
        if time.monotonic()>deadline:
            raise TimeoutError("The predeclared training snapshot was not committed")
        time.sleep(5)
    return path


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--port",type=int,required=True)
    parser.add_argument("--seeds",type=int,nargs="+",required=True)
    args=parser.parse_args()
    status_path=ROOT/f"research_outputs/paper-fix/eval-worker-{args.port}.json"
    status={"status":"running","port":args.port,"seeds":args.seeds,"completed":[]}
    stages=[("baseline","E0","finite",None),("static_verified","E3","finite",0),
            ("loop1","E3","finite",1),("loop2","E3","finite",2),
            ("transfer_baseline","E0","aggressive",None),("transfer_loop2","E3","aggressive",2)]
    try:
        for label,arm,red,iteration in stages:
            status["current_stage"]=label;atomic_json(status_path,status)
            pool=await_pool(iteration) if iteration else (
                ROOT/"research_outputs/paper-fix/initial_pool.json" if iteration==0 else None)
            for seed in args.seeds:
                config=load(ROOT/"research_configs/paper-cage4-fixed.yaml")
                config.update(arm=arm,red=red,eval_seed=seed,episodes=1)
                config["model"].update(endpoint=f"http://127.0.0.1:{args.port}",scoring_endpoints=[])
                if pool:config["pool"]=str(pool)
                config=validate(config)
                destination=ROOT/f"research_outputs/paper-fix/paired/{label}-seed-{seed}"
                status["current_seed"]=seed;atomic_json(status_path,status)
                result=run(config,output=destination,resume=(destination/"manifest.json").exists())
                if result["status"]!="completed":raise RuntimeError("Incomplete performance episode")
                status["completed"].append({"stage":label,"seed":seed,"summary":str(destination/"summary.json")})
                atomic_json(status_path,status)
        status["status"]="completed";atomic_json(status_path,status)
    except BaseException as exc:
        status.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        atomic_json(status_path,status)
        raise
