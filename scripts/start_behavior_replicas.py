"""Start only task-owned replicas; never replace an existing local service."""
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
if __name__=="__main__":
    path=ROOT/"research_outputs/behavior-v1-servers.json"
    if path.exists():raise RuntimeError("Inspect the previously recorded replicas; do not launch duplicates")
    with urllib.request.urlopen("http://127.0.0.1:8018/health",timeout=10) as r:primary=json.load(r)
    assert primary["model"]=="Qwen/Qwen3-8B" and primary["weights_frozen"]
    records=[]
    for gpu,port in ((1,8119),(2,8120),(3,8121)):
        with socket.socket() as s:s.bind(("127.0.0.1",port))
        log=ROOT/f"research_outputs/behavior-v1-qwen-gpu{gpu}.log"
        env={**os.environ,"CUDA_VISIBLE_DEVICES":str(gpu)}
        with log.open("w") as stream:
            p=subprocess.Popen([str(ROOT/"scripts/start_qwen.sh"),"--port",str(port)],cwd=ROOT,env=env,
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        records.append({"gpu":gpu,"port":port,"pid":p.pid,"log":str(log),"started_unix":time.time()})
        path.write_text(json.dumps(records,indent=2)+"\n")
    print(json.dumps(records,indent=2))
