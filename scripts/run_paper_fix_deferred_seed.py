"""Move one evaluation off the training GPU without changing a predeclared case."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.config import load, validate
from research.experiment import atomic_json, run


def until(predicate):
    deadline=time.monotonic()+8*3600
    while not predicate():
        if (ROOT/"research_outputs/paper-fix/E3-hybrid-finite/train/failure.json").exists():
            raise RuntimeError("Training failed")
        if time.monotonic()>deadline:raise TimeoutError("Training snapshot not ready")
        time.sleep(5)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--seed",type=int,required=True)
    args=parser.parse_args();out=ROOT/"research_outputs/paper-fix"
    train=out/"E3-hybrid-finite/train";status_path=out/f"eval-worker-deferred-{args.seed}.json"
    status={"status":"running","seed":args.seed,"completed":[]}
    def spare_ready():
        p=out/"eval-worker-8019.json"
        return p.exists() and json.loads(p.read_text()).get("current_stage") in (
            "loop1","loop2","transfer_baseline","transfer_loop2")
    try:
        # The other worker first finishes its fixed initial-pool control.
        status["current_stage"]="waiting_for_control_gpu";atomic_json(status_path,status)
        until(spare_ready)
        stages=[("baseline","E0","finite",None), ("static_verified","E3","finite",0),
                ("loop1","E3","finite",1),("loop2","E3","finite",2),
                ("transfer_baseline","E0","aggressive",None),("transfer_loop2","E3","aggressive",2)]
        for label,arm,red,iteration in stages:
            destination=out/f"paired/{label}-seed-{args.seed}"
            summary=destination/"summary.json"
            if summary.exists() and json.loads(summary.read_text()).get("status")=="completed":
                status["completed"].append(label);continue
            port=8019
            if iteration==1:
                until(lambda:(train/"pool-iteration-0001.json").exists())
                until(lambda:json.loads((out/"eval-worker-8019.json").read_text()).get("current_stage") in (
                    "loop2","transfer_baseline","transfer_loop2"))
            if iteration==2 or label.startswith("transfer"):
                until(lambda:(train/"summary.json").exists() and json.loads((train/"summary.json").read_text()).get("status")=="completed")
                port=8018  # The training GPU is now free.
            config=load(ROOT/"research_configs/paper-cage4-fixed.yaml")
            config.update(arm=arm,red=red,eval_seed=args.seed,episodes=1)
            config["model"].update(endpoint=f"http://127.0.0.1:{port}",scoring_endpoints=[])
            if iteration is not None:
                config["pool"]=str(out/"initial_pool.json" if iteration==0 else train/f"pool-iteration-{iteration:04d}.json")
            config=validate(config)
            # The interrupted attempt is retained; no incomplete return is used.
            if destination.exists():
                archive=out/"interrupted-scheduling";archive.mkdir(exist_ok=True)
                shutil.move(str(destination),str(archive/(destination.name+f"-{time.time_ns()}")))
            status.update(current_stage=label,port=port);atomic_json(status_path,status)
            result=run(config,output=destination)
            assert result["status"]=="completed"
            status["completed"].append(label);atomic_json(status_path,status)
        status["status"]="completed";atomic_json(status_path,status)
    except BaseException as exc:
        status.update(status="failed",error=f"{type(exc).__name__}: {exc}");atomic_json(status_path,status)
        raise
