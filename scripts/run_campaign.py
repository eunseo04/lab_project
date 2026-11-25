"""Bounded, resumable Qwen campaign. One B2 update precedes optional extensions."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.progress import Journal,atomic_json,BudgetExceeded,IncompleteOptimization,digest,implementation_identity
from research.config import load,validate,variant_config

PORTS=[8418,8419,8420,8421]
write=atomic_json


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity():
    paths=list((ROOT/'research').glob('*.py'))+list((ROOT/'acd').glob('*.py'))+list((ROOT/'scripts').glob('*.py'))
    paths+=list((ROOT/'scripts').glob('*.sh'))+list((ROOT/'research_configs').glob('*.yaml'))
    paths+=list(ROOT.glob('*lock*'))+[ROOT/'research.sh',ROOT/'artifacts/initial_pool.json']
    import research.adapter  # Hash the simulator imported by the official execution path.
    return {**implementation_identity(ROOT),**{str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}}


def available_gpus(minimum_free_mib=21000):
    command=['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits']
    result=subprocess.run(command,text=True,capture_output=True,check=True)
    rows=[{'gpu':int(a.strip()),'uuid':b.strip(),'free_mib':int(c.strip())}
          for a,b,c in (line.split(',') for line in result.stdout.splitlines())]
    # Free memory alone does not authorize sharing another researcher's GPU.
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],
        text=True,capture_output=True,check=True)
    occupied={}
    for line in apps.stdout.splitlines():
        if line.strip():
            uuid,pid=line.split(',');occupied.setdefault(uuid.strip(),[]).append(int(pid.strip()))
    for row in rows:row['compute_pids']=occupied.get(row['uuid'],[])
    return [r for r in rows if r['free_mib']>=minimum_free_mib and not r['compute_pids']],rows


def process_identity(pid,wait_seconds=0):
    path=Path('/proc')/str(pid)
    try:
        until=time.monotonic()+wait_seconds
        while True:
            command=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip()
            if command or time.monotonic()>=until:break
            time.sleep(.01)
        if not command:return None
        stat=(path/'stat').read_text().split(') ',1)[1].split()
        if stat[0]=='Z':return None
        return {'pid':pid,'uid':path.stat().st_uid,'start_ticks':stat[19],
                'command':command,'pgid':os.getpgid(pid)}
    except (OSError,IndexError):return None


def stop_owned(entry):
    """Only this recorded session leader and its children; reject reused/foreign PIDs."""
    current=process_identity(entry['pid'])
    if current is None:return {'pid':entry['pid'],'status':'already_stopped'}
    if any(current[k]!=entry[k] for k in ('uid','start_ticks','command','pgid')) or current['uid']!=os.getuid() or current['pgid']!=entry['pid']:
        raise RuntimeError('Process ownership/identity is ambiguous; not stopping PID '+str(entry['pid']))
    os.killpg(entry['pid'],signal.SIGTERM)
    until=time.monotonic()+8
    while process_identity(entry['pid']) and time.monotonic()<until:time.sleep(.1)
    if process_identity(entry['pid']):
        os.killpg(entry['pid'],signal.SIGKILL)
        until=time.monotonic()+3
        while process_identity(entry['pid']) and time.monotonic()<until:time.sleep(.1)
    return {'pid':entry['pid'],'status':'stopped' if process_identity(entry['pid']) is None else 'remaining'}


def cleanup_owned(entry):
    try:return stop_owned(entry)
    except (RuntimeError,OSError) as exc:
        return {'pid':entry['pid'],'status':'ownership_ambiguous','error':str(exc)}


def verify_release(entries):
    live=[process_identity(e['pid']) for e in entries if process_identity(e['pid'])]
    try:
        result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,capture_output=True,check=True)
        gpu=[{'pid':int(a.strip()),'memory_mib':b.strip()} for a,b in (r.split(',') for r in result.stdout.splitlines())]
    except (OSError,subprocess.SubprocessError,ValueError):gpu=None
    owned_pids={e['pid'] for e in entries}
    ports=[]
    for e in entries:
        if e.get('port'):
            with socket.socket() as probe:
                if probe.connect_ex(('127.0.0.1',e['port']))==0:ports.append(e['port'])
    remaining=[g for g in gpu if g['pid'] in owned_pids] if gpu is not None else None
    return {'live_owned_processes':live,'owned_gpu_processes':remaining,'listening_owned_ports':ports,
        'all_owned_resources_released':not live and remaining==[] and not ports}


def child_sessions(out):
    result=[]
    for path in out.glob('**/rollout-verification/episodes/*/worker.json'):
        entry=json.loads(path.read_text())
        if entry.get('owner_campaign')==str(out) and all(k in entry for k in ('pid','uid','start_ticks','command','pgid')):
            result.append(entry)
    return result


def make_protocol(args,out):
    if getattr(args,'fixed_update_from',None):return make_fixed_protocol(args,out)
    labels=['B0','B1','B2','B3-no-support','B3','B4'];configs={}
    for label in labels:
        cfg=load(ROOT/f'research_configs/rollout-{label}.yaml')
        cfg.update(train_seed=212000,eval_seed=211000,validation_seed=214000,
                   train_episodes=2,iterations=1,episodes=args.test_episodes,experiment_id=str(out),
                   experiment_version=f'cage4-skillpro-one-gpu-pilot-n{args.test_episodes}' if args.max_gpus==1 else 'cage4-skillpro-durable-pilot',
                   output=str(out),journal_path=str(out/'journal.sqlite3'))
        cfg['evolution']['preflight_required']=True
        if cfg['evolution'].get('rollout_verification'):cfg['rollout_validation']['seeds']=list(range(213000,213020))
        configs[label]=validate(cfg)
    requested=[label for label in ['B2','B3-no-support','B3','B4'] if label in set(args.arms)|{'B2'}]
    return {'registered_unix':time.time(),'schema':'durable-campaign-v1','experiment_id':str(out),
        'source_sha256':source_identity(),'configs':configs,'requested_training_arms':requested,
        'protocol_kind':'strict-input-preserving-pilot','training_seeds':[212000,212001],
        'pilot_test_seeds':list(range(211000,211000+args.test_episodes)),
        'reserved_full_test_seeds':list(range(211000,211100)),
        'known_development_seeds':list(range(101000,101100))+[102000,58000,201000,201001],
        'generalization_seeds':[215000],'generalization_requested':args.generalization,
        'budget_gpu_hours':args.gpu_hours,'maximum_wall_seconds':args.wall_hours*3600,
        'previous_campaign_gpu_hours':7.472,'previous_usage_is_not_new_budget':True,
        'maximum_servers':args.max_gpus,'minimum_free_mib':args.minimum_free_mib,
        'profile_full_sequence':getattr(args,'profile_full_sequence',False),
        'resource_policy':'One pinned BF16 model per selected GPU; serial model requests by default; KEEP, encoder and simulator stay on CPU. No quantization or input reduction.',
        'statistics':{'pilot_n':args.test_episodes,'bootstrap_repetitions':10000,'bootstrap_seed':20261125,
                      'interval':'paired percentile 95%; no performance-based stopping; training repetitions=1'},
        'initial_pool_file_sha256':sha(ROOT/'artifacts/initial_pool.json'),
        'locks':{n:sha(ROOT/n) for n in ('research_sources.lock.json','research_models.lock.json','sources.lock.json','requirements.lock.txt','requirements-model.lock.txt')},
        'legacy_allocation':'Previously four GPUs used deadline=start+gpu_hours*3600/4; eight GPU-hours meant 120 wall minutes, training 85 minutes and rollout 83.5 minutes, including startup.',
        'allocation':'Prioritize one B2 update; every later arm needs a fresh measured remaining-cost check. B4 reserves 20 old+20 candidate episodes. Generalization follows the main paired comparison.',
        'scorer':'incremental_reference; experimental full-sequence cannot be adopted without a real-Qwen parity certificate',
        'selection':'Freeze completed_changed/completed_unchanged/completed_no_eligible_skill only. An interrupted pool is untrained_control, never a trained policy.'}


def historical_batch(protocol):
    """Read old evidence with a read-only SQLite connection; never migrate the old DB."""
    import sqlite3
    from research.progress import _unpack
    source=protocol['fixed_update_source']
    for path,expected in source['files_sha256'].items():
        if sha(Path(path))!=expected:raise ValueError('Historical input changed: '+path)
    with sqlite3.connect('file:'+source['journal']+'?mode=ro',uri=True) as db:
        def get(key):
            record=db.execute('SELECT value FROM checkpoints WHERE key=?',(key,)).fetchone()
            if record is None:raise ValueError('Missing committed historical input: '+key)
            return _unpack(record[0])
        pending=get('pending_batch:1:B2');preparation=get('preparation:1:B2')
    return json.loads(Path(source['checkpoint']).read_text()),pending,preparation


def make_fixed_protocol(args,out):
    """A separate fixed-proposal verification study, not a new B3/B4 training claim."""
    source=args.fixed_update_from.resolve();campaign=source.parent.parent
    initial=ROOT/'artifacts/initial_pool.json';training=campaign/'runs/B2-finite-train'
    paths=[source,initial,campaign/'journal.sqlite3',*sorted(training.glob('episode-????.jsonl'))]
    if len(paths)!=5 or any(not p.is_file() for p in paths):raise ValueError('Exact committed checkpoint, journal and two training traces required')
    if args.max_gpus!=1:raise ValueError('Fixed-update study uses one unoccupied GPU')
    if args.prior_gpu_hours is None or not 0<=args.prior_gpu_hours<8 or args.gpu_hours>8-args.prior_gpu_hours+1e-9:
        raise ValueError('Carry-in plus the remaining cap must not exceed the prior eight GPU-hour authorization')
    labels=['B0','B1','B2','B3','B4','B2-ASV','B2-SSV','B2-ASV-SSV','B2-ASV-SSV-RV-screen'];configs={}
    for label in labels:
        cfg=variant_config(label)
        cfg.update(train_seed=91000,train_episodes=2,iterations=1,eval_seed=211000,
            validation_seed=214000,validation_episodes=0,episodes=args.test_episodes,
            experiment_id=str(out),experiment_version='cage4-support-shift-v1-fixed-update',
            output=str(out),journal_path=str(out/'journal.sqlite3'))
        cfg['evolution']['preflight_required']=False
        if cfg['evolution'].get('rollout_verification'):cfg['rollout_validation']['seeds']=list(range(213000,213020))
        configs[label]=validate(cfg)
    frozen=json.loads(source.read_text())
    if frozen.get('optimizer_status')!='completed_changed' or frozen.get('pending_experiences'):
        raise ValueError('Fixed proposal must come from a completed original training update')
    if frozen.get('training_seeds')!=[91000,91001]:raise ValueError('Unexpected historical training seeds')
    version='fixed-committed-update-verification-v1'
    return {'schema':'durable-campaign-v1','registered_unix':time.time(),'experiment_id':str(out),
        'source_sha256':source_identity(),'configs':configs,'protocol_kind':version,
        'fixed_update_source':{'checkpoint':str(source),'journal':str(campaign/'journal.sqlite3'),
            'training_traces':[str(p) for p in paths[3:]],'files_sha256':{str(p):sha(p) for p in paths}},
        'scope':'Verify the exact historical whole committed B2 update against the exact initial pool. No regenerated proposal or new training loop; full B3/B4 training remains separate.',
        'training_seeds':[91000,91001],'new_training_seeds_reserved':[212000,212001],
        'pilot_test_seeds':list(range(211000,211000+args.test_episodes)),
        'rollout_validation_seeds':list(range(213000,213020)),'reject_only_prefix':[213000,213001],
        'known_development_seeds':[201000,201001],'generalization_seeds':[215000],
        'generalization_red':'aggressive','generalization_requested':args.generalization,
        'budget_gpu_hours':args.gpu_hours,'maximum_wall_seconds':args.wall_hours*3600,
        'previous_campaign_gpu_hours':args.prior_gpu_hours,'combined_authorization_gpu_hours':8.,
        'previous_usage_is_not_new_budget':True,'maximum_servers':1,'minimum_free_mib':args.minimum_free_mib,
        'profile_full_sequence':args.profile_full_sequence,'maximum_profile_seconds':900.,
        'selector_probe':{'maximum_states':24,'include_controls':True,'epsilon':configs['B2']['selector']['epsilon'],
            'allowed_seeds':[91000,91001],
            'forbidden_seeds':list(range(211000,211100))+[201000,201001],
            'maximum_disagreement':.25,'maximum_total_variation':.2,'maximum_share_increase':.2},
        'statistics':{'pilot_n':args.test_episodes,'bootstrap_repetitions':10000,'bootstrap_seed':20261125,
            'interval':'paired percentile 95%; exploratory, one old training repetition; no seed selection by reward'},
        'initial_pool_file_sha256':sha(initial),'scorer':'incremental_reference; any incomplete/failed exact parity probe is not adopted',
        'resource_policy':'One BF16 server on a GPU with zero existing compute PIDs. Carry-in budget is preserved. All other jobs untouched.',
        'allocation':'Paired selector probes and bounded scorer/SG measurements; up to 2 reject-only validation pairs; fresh B0/B1/B2 tests; one aggressive-Red paired generalization only if its complete block fits.',
        'approval':'No partial screen can approve. Full approval still requires the same 20 paired seeds and every registered stability criterion.'}


def run_fixed_study(protocol,out,journal,servers,worker,remaining_seconds,switch_phase):
    from collections import defaultdict
    import shutil
    from research.client import ModelClient
    from research.upstream import Experience,configure_encoder
    from research.skills import SecuritySkillPool,probe_selection
    from research.semantic_evidence import action_support
    from research.attribution import selection_distribution,trace_divergence
    from research.rollout import pool_digest
    from research.rollout_execution import EnvironmentVerifier
    from scripts.audit_training_cost import profile_fixed_update
    trained,pending,preparation=historical_batch(protocol)
    previous=json.loads((ROOT/'artifacts/initial_pool.json').read_text())
    campaign=Path(protocol['fixed_update_source']['checkpoint']).parent.parent
    measured=[json.loads(p.read_text())['execution_time_s'] for p in campaign.glob('runs/*-evaluate-*/episode-0000.json')]
    episode_upper=max(measured)*1.1 if measured else 1100.
    required=(3*len(protocol['pilot_test_seeds'])+2*len(protocol['reject_only_prefix']))*episode_upper+900+90
    preflight={'episode_upper_seconds':episode_upper,'primary_and_reject_screen_seconds':required,
        'remaining_seconds':remaining_seconds(),'full_twenty_pair_validation_gpu_hours':40*episode_upper/3600,
        'full_B3_B4_new_training':'deferred; no new training loop is claimed in this fixed-update study',
        'status':'passed' if required<=remaining_seconds() else 'deferred'}
    write(out/'preflight.json',preflight)
    if preflight['status']!='passed':raise BudgetExceeded('Complete predeclared fixed-update pilot does not fit')
    runtime={**protocol['configs']['B2']['model'],'endpoint':f"http://127.0.0.1:{servers[0]['port']}",
        'journal_path':str(journal.path),'request_context':{'experiment':str(out),'arm':'fixed-update-audit','env_stage':'historical_training_probe'}}
    client=ModelClient(runtime);configure_encoder(client.embed)
    old_pool=SecuritySkillPool(previous['pool']['max_size'],previous['pool'].get('seed_set','legacy'));old_pool.restore(previous['pool'])
    new_pool=SecuritySkillPool(trained['pool']['max_size'],trained['pool'].get('seed_set','legacy'));new_pool.restore(trained['pool'])
    accepted=[(r,c) for r in trained['history'][0]['evolution'] for c in r['candidates'] if c.get('accepted')]
    if len(accepted)!=1 or accepted[0][0]['parent']!='NetworkContainment':raise ValueError('Study preregisters the single actual NetworkContainment update')
    parent_record,item=accepted[0];parent=next(s for s in old_pool.skills if s.name==parent_record['parent'])
    candidate=next(s for s in new_pool.skills if s.name==item['skill']['name'])
    experiences=[Experience(**e) for e in pending['experiences'] if e['skill']==parent.name];gate=[]
    for exp in experiences:
        rows=[t for t in exp.transitions if t.get('policy_invoked') and t.get('request_applied') and not t.get('parse_error') and t.get('token_ids')]
        if rows:gate.append(Experience(**{**exp.__dict__,'transitions':rows}))
    support=action_support(parent,candidate,experiences,gate)
    by_action=defaultdict(list)
    for exp,diag in zip(gate,item['diagnostics']):
        for row,advantage in zip(exp.transitions,diag['advantages']):by_action[row['requested_action'].split(' ',1)[0]].append(advantage)
    write(out/'report/action_support.json',{'scope':'fixed historical committed update','ppo_score':item['score'],
        'parent':parent.__dict__,'candidate':candidate.__dict__,'action_support':support,
        'advantages_by_evaluated_action':dict(by_action),'gate_formula_changed':False})
    training_rows=[]
    for path in protocol['fixed_update_source']['training_traces']:
        training_rows.extend(json.loads(line) for line in Path(path).open() if line.strip())
    # Old test outcomes are development-only forensic evidence, never admission inputs.
    forensic={}
    for seed in protocol['known_development_seeds']:
        pair=[];ticks=[]
        for label in ('B1','B2'):
            run=campaign/'runs'/f'{label}-finite-evaluate-{seed}'
            pair.append([json.loads(line) for line in (run/'episode-0000.jsonl').open()])
            tickfile=run/'episode-0000-reward-events.jsonl'
            ticks.append([json.loads(line) for line in tickfile.open()] if tickfile.exists() else [])
        forensic[str(seed)]={'data_role':'known_development_only','divergence':trace_divergence(*pair,*ticks),
            'selection_distributions':{label:selection_distribution(rows,{s.name:getattr(s,'family',s.name) for s in pool.skills})
                for label,rows,pool in zip(('B1','B2'),pair,(old_pool,new_pool))}}
    write(out/'report/historical_forensics.json',forensic)
    switch_phase('paired_selector_probe')
    key='fixed_selector:'+digest([previous['pool'],trained['pool'],protocol['selector_probe'],client.identity])
    shift=journal.restore(key)
    if shift is None:
        shift=probe_selection(previous['pool'],trained['pool'],training_rows,client,protocol['selector_probe'],'historical_training')
        journal.checkpoint(key,shift)
    write(out/'report/selector_probe.json',shift)
    switch_phase('exact_scorer_and_sg_profile')
    profile_fixed_update(out,protocol,client,trained,pending,preparation,remaining_seconds)
    switch_phase('reject_only_candidate_validation')
    cfg=deepcopy(protocol['configs']['B2-ASV-SSV-RV-screen'])
    cfg['rollout_validation'].update(ports=[s['port'] for s in servers],deadline_unix=time.time()+remaining_seconds()-90)
    cfg['verification_pool_provenance']={pool_digest(previous['pool']):{
        'episodes_completed':0,'total_training_episodes':0,'training_seeds':[]},pool_digest(trained['pool']):{
        'episodes_completed':trained['episodes_completed'],'total_training_episodes':trained['total_training_episodes'],
        'training_seeds':trained['training_seeds']}}
    verification_dir=out/'fixed-update';verification_dir.mkdir(exist_ok=True)
    verifier=EnvironmentVerifier(cfg,verification_dir,client.identity)
    result=verifier(old_pool,candidate,{'scope':'fixed_committed_update','parent':parent.name,'iteration':1,
        'valid_behavior_decisions':sum(len(e.transitions) for e in gate),'evidence_trajectories':len(gate)},proposed_snapshot=trained['pool'])
    write(out/'report/fixed_update_rollout.json',result)
    decisions={}
    for name,requires in [('B2-ASV',support['requires_rollout']),('B2-SSV',shift['requires_rollout']),
        ('B2-ASV-SSV',support['requires_rollout'] or shift['requires_rollout'])]:
        decisions[name]={'requires_rollout':requires,'automatic_update_admitted':not requires,
            'decision':'deferred_pending_rollout' if requires else 'original_gate_admitted',
            'scope':'fixed historical whole update, not a new trained arm',
            'pool_sha256':pool_digest(previous['pool'] if requires else trained['pool'])}
    decisions['B2-ASV-SSV-RV-screen']={'decision':result.get('decision_type','rejected' if not result['accepted'] else 'approved'),
        'automatic_update_admitted':result['accepted'],'reason':result['reason'],
        'pool_sha256':pool_digest(result['verified_pool'] if result['accepted'] else previous['pool'])}
    write(out/'report/fixed_update_ablations.json',decisions)
    frozen=out/'frozen/B2.json';frozen.parent.mkdir(exist_ok=True)
    if frozen.exists() and sha(frozen)!=sha(Path(protocol['fixed_update_source']['checkpoint'])):raise ValueError('Frozen committed update changed')
    if not frozen.exists():shutil.copyfile(protocol['fixed_update_source']['checkpoint'],frozen)
    switch_phase('fresh_frozen_test')
    primary=3*len(protocol['pilot_test_seeds'])*episode_upper+90
    if primary>remaining_seconds():raise BudgetExceeded('Fresh main paired block deferred; screen/probes remain separate evidence')
    labels=('B0','B1','B2')
    def block(seeds,red):
        for seed in seeds:
            for label in labels:
                pool=None if label=='B0' else ROOT/'artifacts/initial_pool.json' if label=='B1' else frozen
                _,job=worker(label,'evaluate',servers[0]['port'],seed,red,pool)
                if job['status']!='completed' or job.get('summary_status')!='completed':raise IncompleteOptimization('Fixed frozen test failed')
    block(protocol['pilot_test_seeds'],'finite')
    # Identical-pool reuse is explicit evidence reuse, never additional episodes or trained B3/B4.
    write(out/'report/guard_policy_reuse.json',{'decisions':decisions,'retained_initial_pool_sha256':pool_digest(previous['pool']),
        'retained_policy_measurement':'B1 fresh frozen-test episodes; only decisions with that exact Pool hash use this mapping',
        'additional_environment_episodes':0,'new_B3_B4_training_completed':False,
        'inference_equivalence':{k:protocol['configs']['B1'].get(k)==protocol['configs']['B2'].get(k) for k in
            ('composition','profile','behavior_context','action_contract','selector','skill_seed_set','model','scoring','communication','skip_busy_policy')},
        'inference_note':'E1 and E2 use the same SkillController with proposed=False during frozen evaluation; no evolution is invoked.'})
    if protocol['generalization_requested']:
        recent=[json.loads(p.read_text())['execution_time_s'] for p in out.glob('runs/*-evaluate-*/episode-0000.json')]
        upper=max([episode_upper,*recent])*1.1
        fits=len(labels)*len(protocol['generalization_seeds'])*upper+90<=remaining_seconds()
        write(out/'generalization_preflight.json',{'status':'passed' if fits else 'deferred_budget',
            'required_seconds':len(labels)*len(protocol['generalization_seeds'])*upper+90,'remaining_seconds':remaining_seconds()})
        if fits:
            switch_phase('aggressive_red_generalization');block(protocol['generalization_seeds'],protocol['generalization_red'])
    historical_batch(protocol)  # Also prove the old evidence/checkpoint files were not changed.
    write(out/'historical_inputs_unchanged.json',{'passed':True,'files_sha256':protocol['fixed_update_source']['files_sha256']})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu-hours',type=float,default=8.)
    parser.add_argument('--wall-hours',type=float,default=8.)
    parser.add_argument('--max-gpus',type=int,choices=range(1,5),default=1)
    parser.add_argument('--minimum-free-mib',type=int,default=21000)
    parser.add_argument('--profile-full-sequence',action='store_true',help='Opt in to the memory-intensive experimental scorer probe; reference scoring is unchanged')
    parser.add_argument('--fixed-update-from',type=Path,help='Verify an exact committed historical B2 update; no new training claim')
    parser.add_argument('--prior-gpu-hours',type=float,default=None,help='Carry-in from the existing eight GPU-hour authorization; required for --fixed-update-from')
    parser.add_argument('--test-episodes',type=int,default=4,choices=range(1,101))
    parser.add_argument('--arms',nargs='+',choices=['B2','B3-no-support','B3','B4'],default=['B2'])
    parser.add_argument('--generalization',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--cleanup-only',action='store_true')
    args=parser.parse_args()
    if not 0<args.gpu_hours<=8 or args.wall_hours<=0:raise ValueError('New authorized cap is at most eight GPU-hours; wall cap is independent')
    if args.minimum_free_mib<18000:raise ValueError('Do not lower the pinned BF16 model memory requirement to use a busy GPU')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    # Serialize campaign ownership, including resume and cleanup.
    import fcntl
    lease=(out/'.campaign-lock').open('w')
    try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Campaign already running')
    protocol_path=out/'protocol.json'
    if protocol_path.exists():
        if not (args.resume or args.cleanup_only):raise FileExistsError('Existing campaign requires --resume')
        protocol=json.loads(protocol_path.read_text())
        if protocol.get('schema')!='durable-campaign-v1':raise ValueError('Use a new directory for this campaign schema')
        if not args.cleanup_only:
            if source_identity()!=protocol['source_sha256']:raise ValueError('Source changed: use a new experiment ID')
            if (args.gpu_hours,args.wall_hours*3600)!=(protocol['budget_gpu_hours'],protocol['maximum_wall_seconds']):raise ValueError('Resume cannot reset budget caps')
    else:
        if args.cleanup_only:return
        protocol=make_protocol(args,out)
        common_digest=digest(protocol)
        protocol['comparison_protocol_digest']=common_digest
        for cfg in protocol['configs'].values():cfg['protocol_digest']=common_digest
        write(protocol_path,protocol)
    journal=Journal(out/'journal.sqlite3')
    registry=json.loads((out/'servers.json').read_text()) if (out/'servers.json').exists() else []
    known={e['pid'] for e in registry}
    registry.extend(e for e in child_sessions(out) if e['pid'] not in known)
    cleanup=[]
    # A prior crashed campaign may have left its own servers alive. Identity checks precede every signal.
    for entry in reversed(registry):
        stopped=cleanup_owned(entry);cleanup.append(stopped)
        if entry.get('budget_session') and stopped['status'] in ('stopped','already_stopped'):
            journal.budget_heartbeat(entry['budget_session'],stop=True)
    if any(e['status'] in ('ownership_ambiguous','remaining') for e in cleanup):
        write(out/'cleanup.json',{'stopped':cleanup,**verify_release(registry)})
        raise RuntimeError('Some recorded processes could not be safely identified/stopped; see cleanup.json')
    if args.cleanup_only:
        write(out/'cleanup.json',{'stopped':cleanup,**verify_release(registry)});return
    if args.plan_only:
        write(out/'preflight.json',{'status':'not_measured','reason':'No GPU calls in plan-only mode','budget':journal.budget_usage(),
            'planned_first_arm':'B2','later_arms':'deferred_until_measured_cost_fits'})
        print(json.dumps({'status':'planned','output':str(out)}));return
    available,all_gpus=available_gpus(protocol['minimum_free_mib'])
    write(out/'resource_check.json',{'checked_unix':time.time(),'gpus':all_gpus,'eligible':available})
    if not available:
        write(out/'completion.json',{'status':'blocked_resources','reason':'No GPU has enough free memory for pinned BF16 Qwen; foreign processes untouched',
            'new_campaign_usage':journal.budget_usage(),'all_owned_resources_released':True})
        print(json.dumps({'status':'blocked_resources','output':str(out)}));return
    # Validate assets on CPU before reserving GPU memory. No downloads/fallbacks.
    verification=out/f'assets-verification-{time.time_ns()}.json'
    subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/verify_research_assets.py'),
        '--output',str(verification)],cwd=ROOT,check=True)
    keep_lock=json.loads((ROOT/'research_sources.lock.json').read_text())['published_weights_sha256']
    keep={name:sha(ROOT/'third_party/cage-4-submission/weights'/name) for name in keep_lock}
    if keep!=keep_lock:raise ValueError('KEEP weights differ from the registered lock')
    write(out/f'keep-verification-{time.time_ns()}.json',keep)
    # Checksums can take time; do not reserve a GPU using stale free-memory data.
    available,all_gpus=available_gpus(protocol['minimum_free_mib'])
    write(out/'resource_check.json',{'checked_unix':time.time(),'gpus':all_gpus,'eligible':available})
    if not available:
        write(out/'completion.json',{'status':'blocked_resources','reason':'GPU capacity changed during CPU asset verification',
            'new_campaign_usage':journal.budget_usage(),'all_owned_resources_released':True})
        print(json.dumps({'status':'blocked_resources','output':str(out)}));return
    journal.budget_configure(protocol['budget_gpu_hours'],protocol['maximum_wall_seconds'])
    journal.check_budget()
    available=available[:protocol['maximum_servers']]
    servers=[];processes=[];jobs=json.loads((out/'jobs.json').read_text()) if (out/'jobs.json').exists() else [];failure=None
    abort=threading.Event()
    phases=journal.restore('resource_phases') or []
    phase={'name':'startup','gpu_hours':journal.budget_usage()['gpu_hours'],'unix':time.time()}
    def switch_phase(name):
        now=time.time();usage=journal.budget_usage()
        phases.append({'phase':phase['name'],'started_unix':phase['unix'],'ended_unix':now,
            'gpu_hours':usage['gpu_hours']-phase['gpu_hours']})
        journal.checkpoint('resource_phases',phases)
        phase.update(name=name,gpu_hours=usage['gpu_hours'],unix=now)
    def remaining_seconds():
        usage=journal.check_budget();count=max(1,len(servers))
        return min((usage['gpu_hour_cap']-usage['gpu_hours'])*3600/count,usage['wall_second_cap']-usage['wall_seconds'])
    def launch(command,log,env=None,**meta):
        with log.open('a') as stream:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        entry=process_identity(process.pid,wait_seconds=1)
        if entry is None:raise RuntimeError('Worker exited before ownership registration')
        entry.update(meta);processes.append((process,entry));registry.append(entry)
        write(out/'servers.json',registry)
        return process,entry
    def heartbeat():
        for server in servers:journal.budget_heartbeat(server['budget_session'])
    def worker(label,stage,port,seed=None,red='finite',pool=None):
        if source_identity()!=protocol['source_sha256']:raise RuntimeError('Registered source changed')
        tag=f'{label}-{red}-{stage}'+(f'-{seed}' if seed is not None else '')
        dest=out/'runs'/tag;dest.mkdir(parents=True,exist_ok=True)
        cfg=deepcopy(protocol['configs'][label]);cfg['model'].update(endpoint=f'http://127.0.0.1:{port}',
            generation_endpoints=[f"http://127.0.0.1:{s['port']}" for s in servers],
            scoring_endpoints=[f"http://127.0.0.1:{s['port']}" for s in servers],sg_workers=len(servers))
        if stage=='evaluate':
            cfg.update(episodes=1,eval_seed=seed,red=red,pool=str(pool) if pool else None,evaluation_role='frozen_test')
            cfg['evolution']['rollout_verification']=False;cfg.pop('rollout_validation',None)
        elif label=='B4':cfg['rollout_validation'].update(ports=[s['port'] for s in servers])
        config_path=dest/'config.json'
        if config_path.exists():cfg=json.loads(config_path.read_text())
        else:write(config_path,cfg)
        command=[str(ROOT/'.venv/bin/python'),'-m','research',stage,'--config',str(config_path),'--output',str(dest)]
        endpoints=[f"http://127.0.0.1:{s['port']}" for s in servers]
        command+=['--endpoint',f'http://127.0.0.1:{port}','--scoring-endpoints',*endpoints,
            '--generation-endpoints',*endpoints,'--sg-workers',str(len(servers))]
        if stage=='train' and label=='B4':command+=['--rollout-ports',*[str(s['port']) for s in servers]]
        if (dest/'manifest.json').exists():command.append('--resume')
        task_id=digest([protocol['experiment_id'],label,stage,seed,red,sha(pool) if pool else None])
        job={'task_id':task_id,'label':label,'stage':stage,'seed':seed,'red':red,'directory':str(dest),'started_unix':time.time()}
        job['status']='started';jobs.append(job);write(out/'jobs.json',jobs)
        process,entry=launch(command,dest/'worker.log',env={**os.environ,'PYTHONHASHSEED':'0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        try:
            while process.poll() is None:
                if abort.is_set():raise IncompleteOptimization('Campaign interrupted; stop this owned worker')
                if remaining_seconds()<45:raise BudgetExceeded('Checkpoint/shutdown reserve reached')
                heartbeat();time.sleep(.5)
        except BaseException as exc:
            job.update(status='interrupted_budget' if isinstance(exc,BudgetExceeded) else 'interrupted',error=str(exc),finished_unix=time.time())
            write(out/'jobs.json',jobs)
            raise
        finally:
            if process.poll() is None:stop_owned(entry);process.wait()
        job.update(status='completed' if process.returncode==0 else 'incomplete',returncode=process.returncode,finished_unix=time.time())
        if (dest/'summary.json').exists():
            summary=json.loads((dest/'summary.json').read_text());job.update(summary_status=summary['status'],completed_episodes=len(summary.get('episodes',[])))
        write(out/'jobs.json',jobs)
        return dest,job
    def interrupted(signum,frame):
        abort.set()
        raise InterruptedError('Campaign interrupted; finally cleans only its registered processes')
    previous_handlers={sig:signal.signal(sig,interrupted) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        for gpu,port in zip(available,PORTS):
            with socket.socket() as probe:probe.bind(('127.0.0.1',port))
            session=f"gpu-{gpu['gpu']}-{time.time_ns()}"
            journal.budget_start(session,1,{'gpu':gpu['gpu'],'port':port,'startup_included':True})
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu['gpu']),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'2'}
            try:
                server_command=[str(ROOT/'.model-venv/bin/python'),str(ROOT/'research/model_server.py'),
                    '--model',str(ROOT/'models/policy'),'--encoder',str(ROOT/'models/encoder'),'--port',str(port),
                    '--journal',str(journal.path)]
                if protocol.get('profile_full_sequence',True):server_command.append('--allow-full-sequence')
                process,entry=launch(server_command,out/f'qwen-{gpu["gpu"]}.log',env=env,
                    gpu=gpu['gpu'],port=port,budget_session=session)
            except BaseException:
                journal.budget_heartbeat(session,stop=True);raise
            servers.append(entry)
        for server in servers:
            ready=False;until=time.monotonic()+240
            while time.monotonic()<until:
                remaining_seconds();heartbeat()
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{server['port']}/health",timeout=1) as stream:identity=json.load(stream)
                    model=protocol['configs']['B0']['model']
                    if identity.get('model')!=model['model'] or identity.get('revision')!=model['revision'] or identity.get('dtype')!='bfloat16' or not identity.get('weights_frozen'):raise ValueError('Model identity mismatch')
                    write(out/f"identity-{server['port']}.json",identity);ready=True;break
                except OSError:time.sleep(.5)
            if not ready:raise RuntimeError('Pinned Qwen startup failed')
        if protocol['protocol_kind']=='fixed-committed-update-verification-v1':
            run_fixed_study(protocol,out,journal,servers,worker,remaining_seconds,switch_phase)
            return
        from scripts.audit_training_cost import audit,profile_optimizer,preflight
        switch_phase('training_sg_scoring_rollout')
        # A preflight guard saves the complete batch and exits BEFORE expensive optimizer requests.
        for label in protocol['requested_training_arms']:
            if label!='B2':
                first=journal.restore('profile:B2')
                estimate=preflight(protocol,first or {},journal.budget_usage(),len(servers),label,include_collection=True)
                journal.checkpoint('extension_plan:'+label,estimate)
                if estimate['status']!='passed':raise BudgetExceeded('Later arm deferred before collection: measured full path does not fit')
            dest,job=worker(label,'train',servers[0]['port'])
            checkpoint=dest/'checkpoint.json'
            if not checkpoint.exists() or not json.loads(checkpoint.read_text()).get('history'):
                pending=journal.restore('pending_batch:1:'+label) or journal.restore('pending_batch:1')
                if not pending:raise IncompleteOptimization('No complete pending training batch; inspect worker failure')
                measurements=profile_optimizer(out,label,pending,servers[0]['port'])
                estimate=preflight(protocol,measurements,journal.budget_usage(),len(servers),label)
                write(out/'preflight.json',estimate)
                if estimate['status']!='passed':
                    journal.checkpoint('preflight:'+label,estimate);raise BudgetExceeded('Measured remaining path exceeds the budget; durable pending batch retained')
                journal.checkpoint('preflight:'+label,estimate)
                dest,job=worker(label,'train',servers[0]['port'])
            if job['status']!='completed':raise IncompleteOptimization('Optimizer did not finish; later arms are deferred')
            trained=json.loads(checkpoint.read_text())
            if trained.get('pending_experiences') or not trained.get('optimizer_status','').startswith('completed_'):raise IncompleteOptimization('Training has no completed optimizer commit')
            frozen=out/'frozen'/f'{label}.json'
            if frozen.exists() and sha(frozen)!=sha(checkpoint):raise ValueError('Frozen policy changed')
            if not frozen.exists():write(frozen,trained)
        labels=['B0','B1']+protocol['requested_training_arms']
        episode_estimate=journal.restore('episode_cost_upper_seconds')
        required=len(labels)*len(protocol['pilot_test_seeds'])
        if episode_estimate is None or required*episode_estimate/max(1,len(servers))+60>remaining_seconds():
            raise BudgetExceeded('Completed optimizer retained; full predeclared paired pilot does not fit remaining budget')
        pending=queue.Queue()
        switch_phase('frozen_test')
        for seed in protocol['pilot_test_seeds']:
            for label in labels:pending.put((label,seed,'finite'))
        def evaluate_queue(port):
            while not abort.is_set():
                try:label,seed,red=pending.get_nowait()
                except queue.Empty:return
                try:
                    pool=None if label=='B0' else ROOT/'artifacts/initial_pool.json' if label=='B1' else out/'frozen'/f'{label}.json'
                    _,job=worker(label,'evaluate',port,seed,red,pool)
                    if job['status']!='completed' or job.get('summary_status')!='completed':
                        raise IncompleteOptimization('Required frozen-test episode did not complete')
                except BaseException:
                    abort.set();raise
                finally:pending.task_done()
        with ThreadPoolExecutor(max_workers=len(servers)) as executor:
            for future in [executor.submit(evaluate_queue,s['port']) for s in servers]:future.result()
        # Generalization requires the complete main comparison, and is never B0-only.
        if protocol['generalization_requested']:
            switch_phase('generalization_test')
            if len(labels)*4*episode_estimate/len(servers)+60<=remaining_seconds():
                for red in ('aggressive','stealthy','impact','degrade'):
                    for label in labels:pending.put((label,protocol['generalization_seeds'][0],red))
                with ThreadPoolExecutor(max_workers=len(servers)) as executor:
                    for future in [executor.submit(evaluate_queue,s['port']) for s in servers]:future.result()
        audit(out)
    except BaseException as exc:
        failure={'type':type(exc).__name__,'error':str(exc)}
        write(out/'failure.json',failure)
    finally:
        # Ignore a second interrupt while stopping owned children and persisting accounting.
        for sig in previous_handlers:signal.signal(sig,signal.SIG_IGN)
        switch_phase('cleanup')
        known={e['pid'] for e in registry}
        children=[e for e in child_sessions(out) if e['pid'] not in known]
        registry.extend(children)
        for entry in reversed(children):cleanup.append(cleanup_owned(entry))
        for process,entry in reversed(processes):
            stopped=cleanup_owned(entry)
            cleanup.append(stopped)
            try:process.wait(timeout=1)
            except subprocess.TimeoutExpired:pass
            if entry.get('budget_session') and stopped['status'] in ('stopped','already_stopped'):
                journal.budget_heartbeat(entry['budget_session'],stop=True)
        release=verify_release(registry)
        switch_phase('stopped')
        write(out/'cleanup.json',{'stopped':cleanup,**release})
        write(out/'completion.json',{'status':'partial' if failure else 'completed','failure':failure,
            'new_campaign_usage':journal.budget_usage(),'job_statuses':jobs,**release})
        for sig,handler in previous_handlers.items():signal.signal(sig,handler)
        from scripts.report_experiment import report_experiment
        report_experiment(out,include_trace=True)
    if failure:raise SystemExit(failure['type']+': '+failure['error'])


if __name__=='__main__':main()
