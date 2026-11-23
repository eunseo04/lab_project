"""One development-only initial-skill variant; no registered test is modified."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.client import ModelClient
from research.config import load,validate
from research.experiment import atomic_json
from research.skills import SecuritySkillPool
from research.upstream import configure_encoder,Skill


if __name__=="__main__":
    out=ROOT/"research_outputs/behavior-risk-development"
    if (out/"job.json").exists():raise RuntimeError("Development probe already launched")
    config=load(ROOT/"research_configs/behavior-v2.yaml")
    client=ModelClient(config["model"]);configure_encoder(client.embed)
    pool=SecuritySkillPool(config["pool_size"],seed_set=config["skill_seed_set"])
    skill=Skill(name="RiskBasedInvestigation",version=1,
        initiation="Available agent with a new observable network-connection lead on an assigned host not recently investigated. No PID or file alert does not establish that a host is safe. Prioritize any already confirmed urgent compromise first.",
        policy=["Read current connection INFO, mission phase and own previous action outcomes. Select one exact assigned host with an unreviewed connection lead or a recurrent lead whose last analysis is at least 16 steps old; do not blindly sweep the whole network.",
            "Analyse this host once to obtain information even when Monitor has not flagged a PID. Inspect the actual completed file result; do not keep waiting after the agent is FREE.",
            "For newly supported user compromise, Remove and check the outcome. For observed admin privilege indicators, Restore and inspect completion. An old cmd file alone is not a new attacker and does not justify Restore.",
            "Once the evidence has been reviewed and any indicated response dispatched, stop. Avoid repeated analysis of the same unchanged lead inside the cooldown and avoid discretionary decoys while a useful investigation or response is pending."],
        termination="One justified investigation and any indicated response have completed and their results were reviewed, or the lead no longer warrants intervention.")
    skill.family="RiskBasedInvestigation";skill.allowed_actions=["Analyse","Remove","Restore","Sleep"]
    pool.add_skill(skill)
    snapshot=out/"initial_pool.json"
    atomic_json(snapshot,{"stage":"train","initialization_only":True,"episodes_completed":0,"total_training_episodes":0,
        "training_seeds":[],"pool":pool.snapshot(),"model_identity":client.identity,
        "development_only":True,"expert_prior_variant":True,
        "design_evidence":"Validation seed 23000: connection-only observable leads were not investigated. No test reward or hidden host ID is encoded in this skill."})
    config.update(arm="E3",episodes=1,eval_seed=23000,validation_episodes=0,pool=str(snapshot),development_validation=True)
    config["model"].update(endpoint="http://127.0.0.1:8121",scoring_endpoints=[])
    config=validate(config);path=out/"config.json";atomic_json(path,config)
    code="import json,sys; from research.experiment import run; run(json.load(open(sys.argv[1])),output=sys.argv[2])"
    env={**os.environ,"PYTHONHASHSEED":"0","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","HF_HUB_OFFLINE":"1","WANDB_DISABLED":"true"}
    with (out/"validation.log").open("w") as stream:
        proc=subprocess.Popen([str(ROOT/".venv/bin/python"),"-c",code,str(path),str(out/"static")],cwd=ROOT,
            env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(out/"job.json",{"pid":proc.pid,"development_only":True,"seed":23000,"port":8121,"output":str(out/"static")})
    print(json.dumps({"pid":proc.pid,"seed":23000,"development_only":True,"initial_skills":len(pool.skills)}))
