"""Known-seed regression only, independent of prospective pool selection."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_outputs/rollout-v5';FORK=ROOT/'revisions/rollout-v5'


def main():
    budget=json.loads((OUT/'budget.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text())
    target=OUT/'debugging';target.mkdir(exist_ok=False)
    rule={'registered_unix':time.time(),'seeds':list(range(22000,22004)),'labels':['B0','B4'],
        'source_sha256':protocol['source_sha256'],'deadline_unix':min(time.time()+80*60,budget['validation_deadline_unix']-60),
        'scope':'Known historical seeds for regression only. No code/criteria/pool selection changes; never combined with fresh final tests.'}
    (target/'protocol.json').write_text(json.dumps(rule,indent=2)+'\n')
    def worker(port,seeds):
        results=[]
        for seed in seeds:
            for label in ['B0','B4']:
                if time.time()>=rule['deadline_unix']:return results
                for name,value in rule['source_sha256'].items():assert hashlib.sha256((FORK/name).read_bytes()).hexdigest()==value
                cfg=json.loads((OUT/f'{label}-config.json').read_text());cfg.update(eval_seed=seed,episodes=1,validation_episodes=0)
                cfg.pop('rollout_validation',None);cfg['evolution']['rollout_verification']=False
                cfg['model'].update(endpoint=f'http://127.0.0.1:{port}',scoring_endpoints=[])
                if label!='B0':cfg['pool']=str(OUT/'initial_pool.json')
                dest=target/f'{label}-seed-{seed}';dest.mkdir(exist_ok=False);config=dest/'config.json';config.write_text(json.dumps(cfg,indent=2)+'\n')
                env={**os.environ,'PYTHONHASHSEED':'0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
                with (dest/'worker.log').open('w') as stream:
                    process=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-m','research','evaluate','--config',str(config),'--output',str(dest)],
                        cwd=FORK,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    try:process.wait(timeout=max(.1,rule['deadline_unix']-time.time()))
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=10)
                        (dest/'budget_stop.json').write_text(json.dumps({'status':'not_completed_budget_limit','not_a_performance_sample':True})+'\n')
                        return results
                if process.returncode:raise RuntimeError(f'Debugging case failed: {dest}')
                results.append({'label':label,'seed':seed,'summary':str(dest/'summary.json'),'status':'completed'})
                print('debug-completed',label,seed,flush=True)
        return results
    results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(worker,8219,[22000,22002]),pool.submit(worker,8221,[22001,22003])]):results.extend(future.result())
    (target/'completed.json').write_text(json.dumps({'completed':results,'n':len(results),'scope':rule['scope']},indent=2)+'\n')


if __name__=='__main__':main()
