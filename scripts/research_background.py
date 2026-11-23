"""Start an explicitly requested long local run independently of terminal timeouts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument("--name",required=True)
p.add_argument("arguments",nargs=argparse.REMAINDER)
args=p.parse_args()
if not args.name.replace("-","").replace("_","").isalnum():
    p.error("Use an alphanumeric job name")
command=args.arguments[1:] if args.arguments[:1]==["--"] else args.arguments
if not command or command[0] not in ("train","evaluate","compare","official","report"):
    p.error("Supply a research CLI command after --")
jobs=ROOT/"research_outputs/jobs";jobs.mkdir(parents=True,exist_ok=True)
pid_path=jobs/(args.name+".json")
if pid_path.exists():
    prior=json.loads(pid_path.read_text())
    try: os.kill(prior["pid"],0)
    except ProcessLookupError: pass
    else: raise RuntimeError(f"Job PID {prior['pid']} still exists; inspect it before starting a duplicate")
log_path=jobs/(args.name+".log")
with log_path.open("a",buffering=1) as log:
    log.write("\nStarting "+json.dumps(command)+"\n");log.flush()
    proc=subprocess.Popen([str(ROOT/"research.sh"),*command],cwd=ROOT,stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={"pid":proc.pid,"started_unix":time.time(),"command":command,"log":str(log_path)}
pid_path.write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps(record))
