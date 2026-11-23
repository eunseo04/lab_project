"""Run a matched development pair from the isolated behavior-v3 source fork."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
FORK=ROOT/"revisions/behavior-v3-dev-20261121"
OUT=ROOT/"research_outputs/behavior-v3-development"


def write(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":
    if (OUT/"jobs.json").exists():
        raise RuntimeError("The development pair is already registered")
    # Frozen manual candidate from development data; not a learned pool.
    pool=json.loads((ROOT/"research_outputs/behavior-risk-development/initial_pool.json").read_text())
    skill=next(s for s in pool["pool"]["skills"] if s["name"]=="RiskBasedInvestigation")
    skill["initiation"]="An available agent has an investigation_due connection lead on an assigned host, without a higher-priority unresolved local case. Connection INFO is uncertain activity, not confirmed compromise."
    skill["policy"][0]="Read retained investigation_leads and current connection INFO. Choose one exact assigned host with investigation_due=true. Respect the recorded 16-step review cooldown and 32-step lead expiry; never sweep hosts without a lead."
    pool["design_evidence"]="Development seed 23000 only. Adds visible connection retention and fixes busy-aware repetition checks in an isolated source fork. No heldout reward or host ID is encoded."
    write(OUT/"initial_pool.json",pool)
    config=json.loads((ROOT/"research_outputs/behavior-risk-development/config.json").read_text())
    config.update(connection_leads=True,output=str(OUT),development_validation=True)
    config["selector"]["verification_history"]="accepted_decisions"
    files=list((FORK/"research").glob("*.py"))+[FORK/"research.sh"]
    source={str(p.relative_to(FORK)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    protocol={"registered_at_utc":datetime.now(timezone.utc).isoformat(),
        "purpose":"Matched development validation, not a heldout improvement claim",
        "source_root":str(FORK),"source_sha256":source,
        "seed":23000,"steps":500,"episodes":2,
        "arms":["E0 same connection memory","E3 same memory plus seven manual skills and busy-aware verifier"],
        "shared_actor_state":True,"manual_initialization_not_learning":True,
        "criterion":"E3 must exceed E0 on the same new state; a fresh prospective evaluation is required for generalization.",
        "main_registered_campaign_modified":False}
    write(OUT/"protocol.json",protocol)
    jobs=[]
    env={**os.environ,"PYTHONHASHSEED":"0","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
        "HF_HUB_OFFLINE":"1","WANDB_DISABLED":"true"}
    code="import json,sys; from research.experiment import run; run(json.load(open(sys.argv[1])),output=sys.argv[2])"
    for arm,port in (("E0",8119),("E3",8120)):
        c=json.loads(json.dumps(config))
        c.update(arm=arm,pool=str(OUT/"initial_pool.json") if arm=="E3" else None)
        c["model"].update(endpoint=f"http://127.0.0.1:{port}",scoring_endpoints=[])
        path=OUT/f"{arm}-config.json";write(path,c)
        with (OUT/f"{arm}.log").open("w") as stream:
            process=subprocess.Popen([str(FORK/".venv/bin/python"),"-c",code,str(path),str(OUT/arm)],
                cwd=FORK,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        jobs.append({"arm":arm,"port":port,"pid":process.pid,"output":str(OUT/arm)})
    write(OUT/"jobs.json",jobs)
    print(json.dumps(jobs))
