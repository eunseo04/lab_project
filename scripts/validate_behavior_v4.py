"""Known-case validation of partial observations and Restore evidence epochs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime,timezone
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
FORK=ROOT/"revisions/behavior-v4-observation-20261121"
OUT=ROOT/"research_outputs/behavior-v4-lifecycle-development"
sys.path.insert(0,str(FORK))
from research.config import validate
from research.experiment import run,atomic_json


def read(path):return json.loads(Path(path).read_text())


def launch():
    if (OUT/"protocol.json").exists():raise RuntimeError("Known-case v4 validation already registered")
    coverage_path=ROOT/"research_outputs/behavior-v4-development/input_coverage_audit.json"
    coverage=read(coverage_path);assert coverage["passed"]
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"input_coverage_audit.json").write_bytes(coverage_path.read_bytes())
    tests=ET.parse(FORK/"research_outputs/behavior-v4-tests.xml").getroot().find("testsuite")
    assert int(tests.attrib["failures"])==int(tests.attrib["errors"])==0
    files=list((FORK/"research").glob("*.py"))+[FORK/"research.sh"]
    source={str(p.relative_to(FORK)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    pool_path=ROOT/"research_outputs/behavior-v3-development/initial_pool.json"
    # Identical manual procedures; this revision repairs observation/outcome
    # bookkeeping, not a policy hand-tuned on the failed seed's host IDs.
    (OUT/"initial_pool.json").write_bytes(pool_path.read_bytes())
    protocol={"registered_at_utc":datetime.now(timezone.utc).isoformat(),"source_root":str(FORK),"source_sha256":source,
        "driver_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed":24001,"known_debug_seed_not_heldout":True,"steps":500,"episodes":2,
        "manual_skills_identical_to_v3":True,"pool_sha256":hashlib.sha256(pool_path.read_bytes()).hexdigest(),
        "test_count":int(tests.attrib["tests"]),"no_reward_or_gate_formula_change":True,
        "decision_rule":"E3 must exceed E0 with the same repaired input before a fresh prospective evaluation is recommended",
        "future_finite_seeds":list(range(29000,29004)),"future_conditional_transfer_seeds":list(range(30000,30004)),
        "future_rule":"If the known-case rule passes, frozen finite paired N4; transfer only if mean gain>0 and >=3/4 wins",
        "no_further_tuning_on_future_test":True,
        "fixed_model_identity":read(ROOT/"research_outputs/behavior-v3-development/E3/summary.json")["episodes"][0]["model_identity"]}
    atomic_json(OUT/"protocol.json",protocol)
    env={**os.environ,"PYTHONHASHSEED":"0","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
        "HF_HUB_OFFLINE":"1","WANDB_DISABLED":"true"}
    jobs=[]
    for arm,port in (("E0",8121),("E3",8018)):
        with (OUT/f"{arm}.log").open("w") as stream:
            p=subprocess.Popen([str(FORK/".venv/bin/python"),str(Path(__file__).resolve()),"--worker",arm,"--port",str(port)],
                cwd=FORK,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        jobs.append({"arm":arm,"port":port,"pid":p.pid})
    atomic_json(OUT/"jobs.json",jobs);print(json.dumps(jobs))


def worker(arm,port):
    protocol=read(OUT/"protocol.json")
    for name,digest in protocol["source_sha256"].items():
        assert hashlib.sha256((FORK/name).read_bytes()).hexdigest()==digest
    assert hashlib.sha256((OUT/"initial_pool.json").read_bytes()).hexdigest()==protocol["pool_sha256"]
    # Avoid adding a third generation client while a v3 prospective task uses
    # this server. The main registered campaign continues independently.
    while any(t["status"]=="running" and t.get("port")==port for t in read(ROOT/"research_outputs/behavior-v3-prospective/queue.json")):
        time.sleep(5)
    config=read(ROOT/"research_outputs/behavior-v3-development/E3-config.json")
    config.update(arm=arm,eval_seed=protocol["seed"],episodes=1,validation_episodes=0,
        pool=str(OUT/"initial_pool.json") if arm=="E3" else None,output=str(OUT),known_debug_seed=True)
    config["model"].update(endpoint=f"http://127.0.0.1:{port}",scoring_endpoints=[])
    config=validate(config);atomic_json(OUT/f"{arm}-config.json",config)
    atomic_json(OUT/f"{arm}-started.json",{"started_at_utc":datetime.now(timezone.utc).isoformat(),"port":port,"pid":os.getpid()})
    run(config,output=OUT/arm)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--launch",action="store_true");parser.add_argument("--worker",choices=("E0","E3"));parser.add_argument("--port",type=int)
    args=parser.parse_args()
    if args.launch:launch()
    elif args.worker and args.port:worker(args.worker,args.port)
    else:parser.error("Choose --launch or --worker ARM --port PORT")
