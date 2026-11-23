"""Preregister and execute the eight-hour B0--B4 campaign; never invent missing runs."""
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1];FORK=ROOT/'revisions/rollout-v5';OUT=ROOT/'research_outputs/rollout-v5'
sys.path.insert(0,str(FORK))
from research.config import load,validate
from research.experiment import atomic_json,run

PORTS=[8018,8219,8220,8221]


def read(path):return json.loads(Path(path).read_text())


def assert_source():
    protocol=read(OUT/'protocol.json')
    for name,digest in protocol['source_sha256'].items():
        if hashlib.sha256((FORK/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Registered code changed: '+name)


def register():
    if (OUT/'protocol.json').exists():raise RuntimeError('Already registered; do not launch duplicate campaigns')
    budget=read(OUT/'budget.json')
    suite=ET.parse(FORK/'research_outputs/v5-final-tests.xml').getroot().find('testsuite')
    assert int(suite.attrib['tests'])>=129 and int(suite.attrib['failures'])==int(suite.attrib['errors'])==0
    smoke=read(OUT/'smoke-catalog/E3-hybrid-finite/train/summary.json')
    assert smoke['status']=='completed' and all(e['kind']=='real_cage4' for e in smoke['episodes'])
    base=load(FORK/'research_configs/behavior-v4.yaml')
    base.update(steps=500,episodes=100,train_seed=61000,train_episodes=2,iterations=1,
        validation_seed=63000,validation_episodes=0,eval_seed=64000,output=str(OUT))
    base['diagnostics']={'reward_events':True}
    initial=read(ROOT/'research_outputs/behavior-v4-lifecycle-development/initial_pool.json')
    atomic_json(OUT/'initial_pool.json',initial)
    rule={'seeds':list(range(62000,62020)),'ports':PORTS,'deadline_unix':budget['validation_deadline_unix'],
        'minimum_mean_gain':0.,'minimum_win_rate':.6,'minimum_median_gain':0.,'maximum_seed_loss':50.,
        'maximum_mean_lwf_loss':0.,'maximum_mean_asf_loss':0.,'maximum_std_ratio':1.25,'maximum_std_increase':25.,
        'support_threshold_grid':[0,4,8,16,32]}
    configurations={}
    for label,arm in [('B0','E0'),('B1','E1'),('B2','E2'),('B3','E2'),('B4','E3')]:
        cfg=deepcopy(base);cfg.update(arm=arm,comparison_id=label,pool=None if label=='B0' else str(OUT/'initial_pool.json'))
        proposed=label in ('B3','B4')
        cfg['selector']['domain_verification_revision']='v5' if label=='B4' else 'v4'
        cfg['evolution'].update(evidence_verification=proposed,rollout_verification=proposed,
            maximum_rollout_candidates_per_iteration=1)
        if proposed:cfg['rollout_validation']=deepcopy(rule)
        cfg=validate(cfg);configurations[label]=cfg;atomic_json(OUT/f'{label}-config.json',cfg)
    hashes={str(p.relative_to(FORK)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((FORK/'research').glob('*.py'))+[FORK/'research.sh']}
    preserved={}
    for directory in ['behavior-v2','behavior-v3-prospective','behavior-v4-prospective']:
        for p in (ROOT/'research_outputs'/directory).rglob('*'):
            if p.is_file() and p.suffix in ('.json','.jsonl') and 'reward-replay-work' not in p.parts:
                preserved[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    atomic_json(OUT/'protocol.json',{'registered_unix':time.time(),'source_sha256':hashes,
        'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'budget':budget,
        'configurations':configurations,'initial_pool_sha256':hashlib.sha256((OUT/'initial_pool.json').read_bytes()).hexdigest(),
        'train_seeds':[61000,61001],'rollout_validation_seeds':rule['seeds'],
        'final_finite_seeds':list(range(64000,64100)),'generalization_seeds':list(range(65000,65100)),
        'fixed_model_identity':budget['primary_health'],'regression_tests':int(suite.attrib['tests']),
        'preserved_previous_results_sha256':preserved,
        'resource_plan':'One two-episode automatic evolution loop per trained arm under the eight-hour user budget. The API supports further loops; no unrun second loop is claimed.',
        'selection_rule':'Freeze the committed first-loop pool before final tests. If training/validation is incomplete, keep the initial pool. No held-out reward selects a pool, threshold or iteration.',
        'support_rule':'Most-supported parent gets priority and at most one PPO-positive candidate receives a full 20-seed paired rollout check per arm/iteration. Others are deferred, not PPO-rejected. No fixed rare-skill cutoff is activated without identifying it in validation.',
        'training_priority':'B4 and B2 start first; B3 starts after B4 completes so the main Proposed twenty-seed check gets priority within the resource budget.',
        'finite_priority':'First four FiniteState test seeds for all five arms, then one seed per unseen Red policy for B0/B2/B4; extend balanced blocks if time remains.',
        'fairness':'Common v4 observer, readable evidence, Sleep, primitive menu, model, KEEP four, topology and official reward fixed. B4 alone adds domain initiation priority; B3/B4 evidence and rollout admission are declared Proposed extensions.'})
    atomic_json(OUT/'training_jobs.json',[{'label':label,'status':'pending','port':port} for label,port in [('B4',8018),('B3',8219),('B2',8220)]])


def train(label,port):
    assert_source();cfg=read(OUT/f'{label}-config.json')
    cfg['model'].update(endpoint=f'http://127.0.0.1:{port}',scoring_endpoints=[f'http://127.0.0.1:{p}' for p in PORTS if p!=port])
    run(validate(cfg),'train',output=OUT/label/'train')


def queue_update(callback):
    with (OUT/'queue.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        queue=read(OUT/'queue.json');value=callback(queue);atomic_json(OUT/'queue.json',queue);return value


def worker(port):
    budget=read(OUT/'budget.json')
    while time.time()<budget['no_new_episode_after_unix']:
        def claim(queue):
            task=next((t for t in queue if t['status']=='pending'),None)
            if task:task.update(status='running',pid=os.getpid(),port=port,started_unix=time.time())
            return deepcopy(task)
        task=queue_update(claim)
        if not task:return
        try:
            assert_source();cfg=read(OUT/f"{task['label']}-config.json")
            cfg.update(red=task['red'],episodes=1,eval_seed=task['seed'],validation_episodes=0)
            cfg.pop('rollout_validation',None);cfg['evolution']['rollout_verification']=False
            cfg['model'].update(endpoint=f'http://127.0.0.1:{port}',scoring_endpoints=[])
            if task['label']!='B0':cfg['pool']=str(OUT/'frozen'/f"{task['label']}.json")
            dest=OUT/'evaluation'/task['id'];result=run(validate(cfg),output=dest)
            assert result['status']=='completed'
            def finish(queue):
                item=next(t for t in queue if t['id']==task['id'])
                item.update(status='completed',summary=str(dest/'summary.json'),completed_unix=time.time())
            queue_update(finish)
        except BaseException as exc:
            def fail(queue):next(t for t in queue if t['id']==task['id']).update(status='failed',error=f'{type(exc).__name__}: {exc}')
            queue_update(fail)
            raise


def freeze_and_queue():
    frozen=OUT/'frozen';frozen.mkdir(exist_ok=False);selection={}
    for label in ['B1','B2','B3','B4']:
        committed=OUT/label/'train/pool-iteration-0001.json'
        source=committed if committed.exists() else OUT/'initial_pool.json'
        value=read(source);atomic_json(frozen/f'{label}.json',value)
        initial_names={(s['name'],s.get('version',1)) for s in read(OUT/'initial_pool.json')['pool']['skills']}
        selection[label]={'source':str(source),'sha256':hashlib.sha256((frozen/f'{label}.json').read_bytes()).hexdigest(),
            'committed_automatic_iterations':1 if source==committed else 0,
            'learned_skill_versions':sum((s['name'],s.get('version',1)) not in initial_names for s in value['pool']['skills'])}
    atomic_json(OUT/'selection.json',{'frozen_unix':time.time(),'selection':selection,'test_rewards_used':False})
    tasks=[]
    def append(seed,red,labels,split):
        for label in labels:tasks.append({'id':f'{red}-{label}-seed-{seed}','label':label,'red':red,'seed':seed,'split':split,'status':'pending'})
    reds=['aggressive','stealthy','impact','degrade']
    for seed in range(64000,64004):append(seed,'finite',['B0','B1','B2','B3','B4'],'final_finite')
    for red in reds:append(65000,red,['B0','B2','B4'],'generalization')
    for offset in range(1,100):
        if offset>=4:append(64000+offset,'finite',['B0','B1','B2','B3','B4'],'final_finite')
        for red in reds:append(65000+offset,red,['B0','B2','B4'],'generalization')
    # Include FiniteState offsets 4..99 exactly once; 0..3 were prioritized above.
    assert len([t for t in tasks if t['red']=='finite'])==500
    atomic_json(OUT/'queue.json',tasks)


def stop_group(process):
    if process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()


def child(args,log):
    stream=log.open('w')
    process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*args],cwd=ROOT,
        env={**os.environ,'PYTHONHASHSEED':'0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},
        stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    stream.close();return process


def cleanup():
    records=[]
    for item in read(OUT/'replicas.json'):
        path=Path(f"/proc/{item['pid']}/cmdline")
        args=path.read_bytes().decode().split('\0')[:-1] if path.exists() else []
        if args:
            assert 'research/model_server.py' in args and args[args.index('--port')+1]==str(item['port'])
            os.kill(item['pid'],signal.SIGTERM)
        records.append({**item,'stopped_by_task':bool(args)})
    with urllib.request.urlopen('http://127.0.0.1:8018/health',timeout=10) as stream:identity=json.load(stream)
    assert identity==read(OUT/'budget.json')['primary_health']
    atomic_json(OUT/'server_cleanup.json',{'replicas':records,'primary_8018_unchanged':True,'completed_unix':time.time()})


def manager():
    budget=read(OUT/'budget.json');processes=[]
    try:
        jobs=read(OUT/'training_jobs.json')
        active={}
        def start_job(job):
            process=child(['--train',job['label'],'--port',str(job['port'])],OUT/f"{job['label']}-train.log")
            job.update(status='running',pid=process.pid,started_unix=time.time())
            active[job['label']]=process;processes.append(process)
        for job in jobs:
            if job['label'] in ('B4','B2'):start_job(job)
        atomic_json(OUT/'training_jobs.json',jobs)
        while time.time()<budget['validation_deadline_unix']:
            if 'B3' not in active and active['B4'].poll() is not None:
                start_job(next(j for j in jobs if j['label']=='B3'));atomic_json(OUT/'training_jobs.json',jobs)
            if len(active)==3 and all(p.poll() is not None for p in active.values()):break
            time.sleep(1)
        for job in jobs:
            process=active.get(job['label'])
            if process is None:
                job.update(status='not_started_budget_limit');continue
            was_running=process.poll() is None
            if was_running:stop_group(process)
            job.update(status='stopped_budget_limit' if was_running else 'completed' if process.returncode==0 else 'failed',
                exit_code=process.returncode,completed_unix=time.time())
        atomic_json(OUT/'training_jobs.json',jobs)
        freeze_and_queue();processes=[]
        for port in PORTS:processes.append(child(['--worker','--port',str(port)],OUT/f'worker-{port}.log'))
        atomic_json(OUT/'evaluation_workers.json',[{'pid':p.pid,'port':port} for p,port in zip(processes,PORTS)])
        while any(p.poll() is None for p in processes) and time.time()<budget['model_deadline_unix']-120:time.sleep(1)
        for process in processes:stop_group(process)
        def close(queue):
            for task in queue:
                if task['status'] in ('pending','running'):task.update(status='not_completed_budget_limit',not_a_performance_sample=True)
        queue_update(close)
        atomic_json(OUT/'campaign_complete.json',{'completed_unix':time.time(),'budget_seconds':8*3600,
            'elapsed_model_campaign_seconds':time.time()-budget['started_unix'],'planned_100_per_arm_not_claimed_complete':True})
    finally:
        for process in processes:stop_group(process)
        cleanup()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');parser.add_argument('--manager',action='store_true')
    parser.add_argument('--train');parser.add_argument('--worker',action='store_true');parser.add_argument('--port',type=int)
    args=parser.parse_args()
    if args.launch:
        register();process=child(['--manager'],OUT/'manager.log');atomic_json(OUT/'manager.json',{'pid':process.pid,'started_unix':time.time()});print(json.dumps({'manager_pid':process.pid,'protocol':str(OUT/'protocol.json')}))
    elif args.manager:manager()
    elif args.train:train(args.train,args.port)
    elif args.worker:worker(args.port)
    else:parser.error('Choose a campaign action')
