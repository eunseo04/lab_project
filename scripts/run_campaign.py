"""Bounded real-Qwen pilot. Never overwrites historical runs or relaxes admission."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
PORTS=[8418,8419,8420,8421]
WRITE_LOCK=threading.RLock()


def write(path,data):
    with WRITE_LOCK:
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'research_outputs/pilot')
    parser.add_argument('--gpu-hours',type=float,default=8.)
    args=parser.parse_args()
    if not 0<args.gpu_hours<=8:raise ValueError('Persistent user budget is at most eight GPU-hours')
    out=args.output.resolve()
    if out.exists():raise FileExistsError('Use a new output; completed and partial runs are never overwritten')
    for port in PORTS:
        with socket.socket() as probe:probe.bind(('127.0.0.1',port))
    sys.path.insert(0,str(ROOT))
    from research.config import validate
    from prepare_checkout import prepare
    prepare(ROOT,out/'configs')
    start=time.time();deadline=start+args.gpu_hours*3600/len(PORTS)
    training_end=deadline-35*60
    validation_end=training_end-90
    if training_end<=start:raise ValueError('Budget too small for the registered training/evaluation allocation')
    configs={}
    import yaml
    for label in ['B0','B1','B2','B3','B4']:
        cfg=yaml.safe_load((out/f'configs/rollout-{label}.yaml').read_text())
        cfg.update(train_episodes=2,iterations=1,episodes=4,experiment_version='cage4-skillpro-pilot')
        if label=='B4':
            cfg['rollout_validation'].update(ports=PORTS,deadline_unix=validation_end)
        cfg=validate(cfg)
        cfg.update(train_episodes=2,iterations=1,episodes=4,experiment_version='cage4-skillpro-pilot',output=str(out))
        cfg['model'].update(endpoint=f'http://127.0.0.1:{PORTS[0]}',scoring_endpoints=[])
        if label=='B4':cfg['rollout_validation'].update(ports=PORTS,deadline_unix=validation_end)
        configs[label]=validate(cfg)
        write(out/f'{label}-config.json',configs[label])
    source={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'research').glob('*.py'))+[ROOT/'research.sh']}
    protocol={'registered_unix':start,'experiment_version':'cage4-skillpro-pilot','source_sha256':source,
        'driver_sha256':sha(Path(__file__)),'configs':configs,'training_seeds':[91000,91001],
        'rollout_validation_seeds':configs['B4']['rollout_validation']['seeds'],
        'pilot_test_seeds':list(range(101000,101004)), 'reserved_full_test_seeds':list(range(101000,101100)),
        'generalization_seeds':[102000], 'budget_gpu_hours':args.gpu_hours,
        'maximum_wall_seconds':args.gpu_hours*3600/len(PORTS),'deadline_unix':deadline,
        'training_deadline_unix':training_end,'rollout_deadline_unix':validation_end,
        'budget_accounting':'Conservative sum of all four occupied GPUs, including model startup and idle time.',
        'selection':'Only a completed committed training checkpoint can be frozen; missing arms remain N/A. No test reward selects a pool.',
        'pilot_limitations':'Two training episodes/one iteration and four pilot test seeds. The full plan is six episodes/three iterations/100 tests. Never merge pilot and full.',
        'mapping':{k:v['arm'] for k,v in configs.items()},
        'initial_pool_file_sha256':sha(Path(configs['B1']['pool'])),
        'locks':{name:sha(ROOT/name) for name in ['research_sources.lock.json','research_models.lock.json','requirements.lock.txt','requirements-model.lock.txt']}}
    write(out/'protocol.json',protocol)
    owned=[];jobs=[]
    def stop(process):
        if process.poll() is not None:return
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait()
    def launch(command,log,env=None):
        with log.open('w') as stream:
            process=subprocess.Popen(command,cwd=ROOT,env=env or os.environ.copy(),
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        owned.append(process);return process
    def worker(label,stage,port,seed=None,red='finite',pool=None):
        for name,expected in source.items():
            if sha(ROOT/name)!=expected:raise RuntimeError('Registered source changed: '+name)
        tag=f'{label}-{red}-{stage}'+(f'-{seed}' if seed is not None else '')
        dest=out/'runs'/tag;dest.mkdir(parents=True)
        cfg=deepcopy(configs[label]);cfg['model']['endpoint']=f'http://127.0.0.1:{port}'
        if stage=='evaluate':
            cfg.update(episodes=1,eval_seed=seed,red=red,pool=str(pool) if pool else None)
            cfg['evolution']['rollout_verification']=False
            cfg.pop('rollout_validation',None)
        config_path=dest/'config.json';write(config_path,cfg)
        command=[str(ROOT/'.venv/bin/python'),'-m','research',stage,'--config',str(config_path),'--output',str(dest)]
        job={'label':label,'stage':stage,'seed':seed,'red':red,'directory':str(dest),'status':'running','started_unix':time.time()}
        jobs.append(job)
        process=launch(command,dest/'worker.log')
        end=training_end if stage=='train' else deadline-45
        while process.poll() is None and time.time()<end:time.sleep(.5)
        if process.poll() is None:
            stop(process);job['status']='budget_stopped'
        else:job['status']='completed' if process.returncode==0 else 'failed'
        job.update(returncode=process.returncode,finished_unix=time.time())
        summary=dest/'summary.json'
        if summary.exists():
            value=json.loads(summary.read_text());job['completed_episodes']=len(value.get('episodes',[]))
            job['summary_status']=value.get('status')
            if job['status']=='completed' and value.get('status')!='completed':job['status']='partial'
        write(out/'jobs.json',jobs)
        print(json.dumps({k:v for k,v in job.items() if k!='directory'}),flush=True)
        return job
    def evaluate_block(tasks,port):
        for task in tasks:
            if time.time()>=deadline-90:break
            label,stage,seed,red,pool=task
            worker(label,stage,port,seed,red,pool)
    def interrupted(signum,frame):raise InterruptedError('Campaign interrupted; terminate only owned workers')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        servers=[]
        for gpu,port in enumerate(PORTS):
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'2'}
            command=[str(ROOT/'.model-venv/bin/python'),str(ROOT/'research/model_server.py'),
                '--model',str(ROOT/'models/policy'),'--encoder',str(ROOT/'models/encoder'),'--port',str(port)]
            process=launch(command,out/f'qwen-{gpu}.log',env)
            servers.append({'gpu':gpu,'port':port,'pid':process.pid})
        write(out/'servers.json',servers)
        ready=set();startup_end=min(start+240,deadline)
        while len(ready)<len(PORTS) and time.time()<startup_end:
            for port in set(PORTS)-ready:
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=1) as stream:identity=json.load(stream)
                    if identity['model']!='Qwen/Qwen3-8B' or identity['revision']!=configs['B0']['model']['revision'] or identity['dtype']!='bfloat16' or not identity['weights_frozen']:
                        raise RuntimeError('Model identity mismatch')
                    ready.add(port);write(out/f'identity-{port}.json',identity)
                except OSError:pass
            if len(ready)<len(PORTS):time.sleep(1)
        if len(ready)!=len(PORTS):raise RuntimeError('Local Qwen startup failed; inspect task-owned server logs')
        print('All four pinned BF16 Qwen servers ready.',flush=True)
        # Reserve held-out seeds before any optimization. Controls can run while
        # training; their rewards are never read by any candidate generator.
        with ThreadPoolExecutor(max_workers=4) as executor:
            training=[executor.submit(worker,label,'train',PORTS[i]) for i,label in enumerate(['B2','B3','B4'])]
            baseline=executor.submit(evaluate_block,[('B0','evaluate',seed,'finite',None) for seed in protocol['pilot_test_seeds']],PORTS[3])
            for future in training:future.result()
            baseline.result()
        frozen=out/'frozen';selection={}
        for label in ['B1','B2','B3','B4']:
            source_pool=Path(configs[label]['pool']) if label=='B1' else out/f'runs/{label}-finite-train/pool-iteration-0001.json'
            summary=out/f'runs/{label}-finite-train/summary.json'
            complete=label=='B1' or summary.exists() and json.loads(summary.read_text()).get('status')=='completed'
            if not source_pool.exists() or not complete:
                selection[label]={'status':'N/A','reason':'training_not_committed_and_completed'};continue
            value=json.loads(source_pool.read_text());write(frozen/f'{label}.json',value)
            selection[label]={'status':'frozen','source':str(source_pool),'sha256':sha(frozen/f'{label}.json'),
                'automatic_iterations':0 if label=='B1' else 1}
        write(out/'selection.json',{'frozen_unix':time.time(),'selection':selection})
        tasks=[]
        for seed in protocol['pilot_test_seeds']:
            for label,item in selection.items():
                if item['status']=='frozen':tasks.append((label,'evaluate',seed,'finite',frozen/f'{label}.json'))
        for red in ['aggressive','stealthy','impact','degrade']:
            for label in ['B0','B2','B4']:
                if label=='B0' or selection[label]['status']=='frozen':tasks.append((label,'evaluate',102000,red,None if label=='B0' else frozen/f'{label}.json'))
        blocks=[tasks[i::len(PORTS)] for i in range(len(PORTS))]
        with ThreadPoolExecutor(max_workers=4) as executor:
            for future in [executor.submit(evaluate_block,block,port) for block,port in zip(blocks,PORTS)]:future.result()
    finally:
        for process in reversed(owned):stop(process)
        write(out/'jobs.json',jobs)
        write(out/'completion.json',{'finished_unix':time.time(),'elapsed_wall_seconds':time.time()-start,
            'conservative_gpu_hours':len(PORTS)*(time.time()-start)/3600,
            'completed_jobs':sum(j['status']=='completed' for j in jobs),'job_statuses':jobs,
            'all_owned_processes_stopped':all(p.poll() is not None for p in owned)})


if __name__=='__main__':main()
