"""Start only task-owned replicas and record the user-approved eight-hour clock."""
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_outputs/rollout-v5'


def health(port):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=5) as stream:return json.load(stream)


def main():
    budget=OUT/'budget.json'
    if budget.exists():raise RuntimeError('Budget already started; do not duplicate model processes')
    identity=health(8018);now=time.time()
    record={'started_unix':now,'model_deadline_unix':now+8*3600,
        'validation_deadline_unix':now+5.25*3600,'no_new_episode_after_unix':now+8*3600-180,
        'limit_hours':8,'basis':'Total model campaign elapsed wall time with up to four GPUs; user approved eight-hour option.',
        'primary_health':identity,'primary_port':8018,'primary_pid':3761356}
    budget.write_text(json.dumps(record,indent=2)+'\n')
    replicas=[]
    for gpu,port in [(1,8219),(2,8220),(3,8221)]:
        probe=socket.socket();probe.settimeout(.5)
        in_use=probe.connect_ex(('127.0.0.1',port))==0;probe.close()
        if in_use:raise RuntimeError(f'Port {port} is already used; no process replaced')
        log=OUT/f'qwen-gpu{gpu}.log'
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
        with log.open('w') as stream:
            process=subprocess.Popen([str(ROOT/'.model-venv/bin/python'),'research/model_server.py',
                '--model',str(ROOT/'models/policy'),'--encoder',str(ROOT/'models/encoder'),'--port',str(port)],
                cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        replicas.append({'pid':process.pid,'port':port,'gpu':gpu,'log':str(log)})
        (OUT/'replicas.json').write_text(json.dumps(replicas,indent=2)+'\n')
    print(json.dumps({'budget':record,'replicas':replicas},indent=2))


if __name__=='__main__':main()
