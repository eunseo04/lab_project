"""Bounded, resumable Qwen campaign with registered training and evaluation work."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.progress import (Journal,atomic_json,BudgetExceeded,IncompleteOptimization,digest,
    implementation_identity,LEGACY_GPU_AUTHORIZATION,validate_budget_limits,optimizer_batch_identity,file_sha256 as sha)
from research.config import (load,validate,variant_config,apply_seed_plan,validate_seed_plan,
    seed_exposure_history,split_seeds,VERIFICATION_VARIANTS,CONTROLLER_VARIANTS,COMPLETED_OPTIMIZER_STATUSES,REDS,
    execution_config,policy_interface,check_pool_interface,read_policy_manifest,read_training_manifest,
    checkpoint_method_digest,core_evaluation_labels,training_boundary,verify_completed_update,digest as config_digest)
from scripts.audit_training_cost import campaign_reservation, STARTUP_SHUTDOWN_SECONDS

PORTS=[8418,8419,8420,8421]
write=atomic_json


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
                'command':command,'pgid':os.getpgid(pid),'cwd':os.readlink(path/'cwd')}
    except (OSError,IndexError):return None


def stop_owned(entry):
    """Only this recorded session leader and its children; reject reused/foreign PIDs."""
    def owned_process():
        current=process_identity(entry['pid'])
        if current and (any(current[k]!=entry[k] for k in ('uid','start_ticks','command','pgid'))
                or current['uid']!=os.getuid() or current['pgid']!=entry['pid']
                or current['cwd']!=entry.get('cwd',str(ROOT))):
            raise RuntimeError('Process ownership/identity is ambiguous; not stopping PID '+str(entry['pid']))
        return current
    if not owned_process():return {'pid':entry['pid'],'status':'already_stopped'}
    os.killpg(entry['pid'],signal.SIGTERM)
    until=time.monotonic()+8
    while process_identity(entry['pid']) and time.monotonic()<until:time.sleep(.1)
    if owned_process():
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


def recorded_gpu_usage(paths):
    """Deduplicate cumulative ledgers by physical reservation, not output directory."""
    sessions={}
    for value in paths:
        path=Path(value);path=path/'completion.json' if path.is_dir() else path
        d=json.loads(path.read_text());usage=d.get('new_campaign_usage',{})
        if usage.get('gpu_hours',0) and not usage.get('sessions'):raise ValueError('GPU ledger lacks physical session identities')
        for s in usage.get('sessions',[]):
            if s.get('stop') is None:raise ValueError('Prior reservation is not closed; resolve its accounting first')
            start,stop,count=s['start'],s['stop'],s['gpu_count']
            if (not all(type(t) in (int,float) and math.isfinite(t) for t in (start,stop))
                    or stop<start or type(count) is not int or count<1):
                raise ValueError('Invalid physical GPU interval')
            interval=(start,stop,count)
            if s['id'] in sessions and sessions[s['id']]!=interval:raise ValueError('Conflicting physical GPU interval')
            sessions[s['id']]=interval
    return {'gpu_hours':sum((end-start)*count/3600 for start,end,count in sessions.values()),
        'unique_sessions':len(sessions),'sources':[str(Path(p).resolve()) for p in paths]}


def read_gpu_authorization(path):
    if path is None:
        return dict(LEGACY_GPU_AUTHORIZATION)
    source=Path(path).resolve()
    raw=source.read_bytes()
    grant=json.loads(raw)
    if not isinstance(grant,dict) or not isinstance(grant.get('approval_reference'),str) or not grant['approval_reference'].strip():
        raise ValueError('GPU authorization file needs an explicit approval reference')
    ledger_sources=grant.get('ledger_sources',[])
    if not isinstance(ledger_sources,list) or any(not isinstance(p,str) or not p.strip() for p in ledger_sources):
        raise ValueError('Authorization ledger_sources must be a list of completion paths')
    authorization={'scope':grant.get('scope'),'gpu_hour_cap':grant.get('gpu_hour_cap'),
        'approval_reference':grant['approval_reference'],'file':str(source),
        'sha256':hashlib.sha256(raw).hexdigest(),
        'ledger_sources':[str((source.parent/p).resolve()) for p in ledger_sources]}
    validate_budget_limits(0,1,authorization)
    return authorization


def campaign_budget(args, *, authorization=None, exclude=None):
    authorization=authorization if authorization is not None else read_gpu_authorization(getattr(args,'authorization',None))
    ledgers=getattr(args,'usage_ledger',None) or []
    def key(value):
        path=Path(value);return str((path/'completion.json' if path.is_dir() else path).resolve())
    omitted=key(exclude) if exclude is not None else None
    closure={key(p) for p in [*ledgers,*authorization.get('ledger_sources',[])]}-{omitted}
    protocols={};linked_registrations={}
    for path in (ROOT/'research_outputs').rglob('protocol.json'):
        if 'report' not in path.relative_to(ROOT/'research_outputs').parts:
            protocols[key(path.parent)]=json.loads(path.read_text())
    # New scope-tagged campaigns count even when their output was not supplied
    # explicitly. Plan-only directories have no physical reservations.
    for path,protocol in protocols.items():
        if path==omitted or (protocol.get('gpu_authorization') or {}).get('scope')!=authorization['scope']:
            continue
        database=Path(path).parent/'journal.sqlite3'
        if database.is_file():
            with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True) as db:
                rows=db.execute('SELECT id,start,stop,gpu_count FROM budget_sessions').fetchall()
            if rows:
                if not Path(path).is_file():
                    raise ValueError('A registered campaign has physical sessions without a completion ledger: '+path)
                recorded=json.loads(Path(path).read_text()).get('new_campaign_usage',{}).get('sessions',[])
                if {(s['id'],s['start'],s.get('stop'),s['gpu_count']) for s in recorded}!=set(rows):
                    raise ValueError('A registered campaign completion does not match its physical journal: '+path)
        if Path(path).is_file():
            closure.add(path)
    # A later child ledger may already have spent part of this same authorization.
    # Discover descendants so reusing an old carry-in command cannot reset usage.
    changed=True
    while changed:
        changed=False
        # A recent child completion contains its own occupied time, while its
        # protocol points to the cumulative ancestry. Follow both directions.
        for value in list(closure):
            protocol=Path(value).parent/'protocol.json'
            registration=protocols.get(value)
            if registration is None and protocol.is_file():registration=json.loads(protocol.read_text())
            if registration is None:continue
            linked_registrations[value]=registration
            recorded_scope=(registration.get('gpu_authorization') or {}).get('scope')
            if recorded_scope and recorded_scope!=authorization['scope']:
                raise ValueError('GPU usage ledger belongs to a different authorization scope: '+value)
            sources=(registration.get('prior_physical_usage') or {}).get('sources',[])
            for source in sources:
                ancestor=key(source)
                if ancestor==omitted:continue
                if not Path(ancestor).is_file():raise ValueError('Missing prior GPU accounting dependency: '+ancestor)
                if ancestor not in closure:closure.add(ancestor);changed=True
        for completion,registration in protocols.items():
            if completion==omitted or completion in closure or not Path(completion).is_file():continue
            parent_sources=(registration.get('prior_physical_usage') or {}).get('sources',[])
            if closure & {key(p) for p in parent_sources}:
                closure.add(completion);changed=True
    ledgers=sorted(closure)
    measured=recorded_gpu_usage(ledgers)
    stated=getattr(args,'prior_gpu_hours',None)
    if stated is not None:
        validate_budget_limits(0,1,authorization,stated)
    carry_floor=max(((p.get('prior_physical_usage') or {}).get('unitemized_gpu_hour_floor',0.)
                     for p in linked_registrations.values()),default=0.)
    prior=max(stated or 0,measured['gpu_hours']+carry_floor)
    validate_budget_limits(args.gpu_hours,getattr(args,'wall_hours',1)*3600,authorization,prior)
    if not (ledgers and (getattr(args,'usage_ledger',None) or authorization.get('ledger_sources'))) and not getattr(args,'plan_only',False):
        raise ValueError('GPU authorization needs a linked usage ledger before launch')
    return prior,{**measured,'unitemized_gpu_hour_floor':prior-measured['gpu_hours'],'authorization':authorization}


def registered_budget(prior,ledger):
    authorization=ledger.get('authorization',LEGACY_GPU_AUTHORIZATION)
    return {'previous_campaign_gpu_hours':prior,'prior_physical_usage':ledger,
        'gpu_authorization':authorization,'combined_authorization_gpu_hours':authorization['gpu_hour_cap'],
        'previous_usage_is_not_new_budget':True}


def campaign_seed_plan(args):
    if not getattr(args,'seed_plan',None):raise ValueError('Explicit --seed-plan JSON required; no automatic replacement holdout')
    history=seed_exposure_history(ROOT/'research_outputs')
    plan=validate_seed_plan(json.loads(Path(args.seed_plan).read_text()),history)
    if not getattr(args,'diagnosis_probes_only',False) and not plan['final_test']:
        raise ValueError('Core comparison requires registered final-test seeds')
    if getattr(args,'generalization',False) and not plan['transfer']:
        raise ValueError('Generalization requires registered transfer seeds')
    if getattr(args,'train_episodes',None) is not None and args.train_episodes!=len(plan['train']):
        raise ValueError('--train-episodes differs from the explicit seed plan')
    if getattr(args,'test_episodes',None) is not None and args.test_episodes!=len(plan['final_test']):
        raise ValueError('--test-episodes differs from the explicit seed plan')
    return plan,history


def core_cost_reservation(test_seeds):
    measured=[]
    for path in (ROOT/'research_outputs').glob('*/runs/*/episode-????.json'):
        e=json.loads(path.read_text());identity=e.get('model_identity') or {}
        if (e.get('kind')=='real_cage4' and e.get('status')=='completed' and
                e.get('joint_transitions')==499 and identity.get('model')=='Qwen/Qwen3-8B' and identity.get('dtype')=='bfloat16'):
            measured.append((e['execution_time_s'],str(path.resolve())))
    upper=max((v for v,_ in measured),default=None)
    return {'episodes':3*len(test_seeds),'episode_upper_seconds':upper*1.5 if upper is not None else None,
        'gpu_seconds_estimate':3*len(test_seeds)*upper*1.5 if upper is not None else None,
        'cost_sources':[p for _,p in measured],'estimate_only':True,
        'cost_guard':'Reserve core before SG/score and optional collection; first new measured preflight refreshes costs.'}


def historical_reference(args):
    path=getattr(args,'reference_report',None)
    if path is None:return None
    path=Path(path).resolve();commit=path.parent.parent/'run_commit.json'
    return {'path':str(path),'sha256':sha(path),'source_commit':json.loads(commit.read_text()).get('head') if commit.exists() else None,
        'role':'Historical context only; never merged into this study or counted as new loops/episodes.'}


def continuation_source(path, protocol, initial_pool_sha256):
    path = Path(path).resolve()
    if path.parent.name == 'frozen' and path.name == 'B2.json':
        original = path.parent.parent / 'runs/B2-finite-train/checkpoint.json'
        if sha(path) != sha(original):
            raise ValueError('Frozen B2 does not match its original training checkpoint')
        path = original
    manifest_path = path.parent / 'manifest.json'
    manifest_ref = {'path': str(manifest_path), 'sha256': sha(manifest_path)}
    checkpoint = json.loads(path.read_text())
    manifest = read_training_manifest(path, checkpoint, manifest_ref=manifest_ref)
    if checkpoint.get('training_completed') is not True:
        raise ValueError('Continuation requires a completed previous training segment')
    original_out = Path(manifest['config']['experiment_id'])
    original_protocol = original_out / 'protocol.json'
    original = json.loads(original_protocol.read_text())
    registered_digest = original['comparison_protocol_digest']
    protocol_input = deepcopy(original)
    protocol_input.pop('comparison_protocol_digest')
    for cfg in protocol_input['configs'].values():
        cfg.pop('protocol_digest')
    old_sources = original['source_sha256']
    orchestration = {'research/config.py', 'research/experiment.py', 'research/progress.py', 'research/reporting.py',
                     'scripts/run_campaign.py', 'scripts/audit_training_cost.py',
                     'scripts/report_experiment.py', 'scripts/recompute_public_metrics.py'}
    current_sources = protocol['source_sha256']
    if (registered_digest != digest(protocol_input) or registered_digest != manifest['config'].get('protocol_digest')
            or original['locks'] != protocol['locks']
            or original['keep_weights_sha256'] != protocol['keep_weights_sha256']
            or original['initial_pool_file_sha256'] != protocol['initial_pool_file_sha256']
            or not manifest.get('implementation_sha256')
            or any(old_sources.get(name) != value for name, value in manifest['implementation_sha256'].items())
            or any(old_sources.get(name) != current_sources.get(name)
                   for name in (old_sources.keys() | current_sources.keys()) - orchestration)):
        raise ValueError('Continuation source, policy implementation or asset locks differ')
    iteration = len(checkpoint.get('history', []))
    expected = {'iteration': iteration, 'total_training_episodes': checkpoint['total_training_episodes'],
                'training_seeds': checkpoint['training_seeds'], 'initial_pool_sha256': initial_pool_sha256}
    verified = verify_completed_update(checkpoint, expected, protocol['configs']['B2'], manifest)
    identity = checkpoint.get('model_identity') or {}
    configured = protocol['configs']['B2']['model']
    if (any(identity.get(key) != configured[key] for key in ('model', 'revision'))
            or identity.get('dtype') != 'bfloat16' or not identity.get('weights_frozen')):
        raise ValueError('Continuation requires the pinned frozen BF16 Qwen model')
    checkpoint_hash = sha(path)
    with sqlite3.connect((original_out / 'journal.sqlite3').as_uri() + '?mode=ro', uri=True) as db:
        row = db.execute('SELECT value FROM checkpoints WHERE key=?', (f'loop_commit:{iteration}:B2',)).fetchone()
    receipt = json.loads(zlib.decompress(row[0])) if row else {}
    if (receipt.get('checkpoint_sha256') != checkpoint_hash
            or receipt.get('optimizer_status') != checkpoint['optimizer_status']):
        raise ValueError('Continuation checkpoint does not match its durable commit receipt')
    files = {str(path): checkpoint_hash, str(manifest_path): manifest_ref['sha256'],
             str(original_protocol): sha(original_protocol)}
    for local_episode, seed in enumerate(split_seeds(manifest['config'], 'train')):
        record_path = path.parent / f'episode-{local_episode:04d}.json'
        result = json.loads(record_path.read_text())
        trace = record_path.with_suffix('.jsonl')
        if (result.get('kind') != 'real_cage4' or result.get('status') != 'completed'
                or result.get('seed') != seed or sha(trace) != result.get('trace_sha256')):
            raise ValueError('Continuation training trace does not match its completed episode')
        files.update({str(record_path): sha(record_path), str(trace): result['trace_sha256']})
    return {**expected, **verified, 'path': str(path), 'checkpoint_sha256': checkpoint_hash,
            'training_manifest': manifest_ref, 'model_identity': identity,
            'source_protocol': str(original_protocol), 'source_identity_sha256': digest(old_sources),
            'files_sha256': files}


def register_checkpoint_evaluations(args, out, protocol):
    iterations = getattr(args, 'checkpoint_iterations', None)
    previous = getattr(args, 'continue_from', None)
    if iterations is None and previous is None:
        return
    if protocol['requested_training_arms'] != ['B2'] or protocol['generalization_requested']:
        raise ValueError('Checkpoint evaluation covers the registered B0/B1/B2 FiniteState curve')
    cfg = protocol['configs']['B2']
    initial_hash = config_digest(json.loads((ROOT / 'artifacts/initial_pool.json').read_text())['pool'])
    inherited = continuation_source(previous, protocol, initial_hash) if previous else {}
    offset = inherited.get('iteration', 0)
    inherited_seeds = inherited.get('training_seeds', [])
    new_seeds = protocol['seed_plan']['train']
    if (set(inherited_seeds) & (set(new_seeds) | set(protocol['pilot_test_seeds']))
            or set(new_seeds) & set(protocol['seed_exposure_history']['exposed_seeds'])):
        raise ValueError('Checkpoint curve requires new training experience and a separate holdout')
    if inherited:
        protocol['continuation_source'] = inherited
        cfg.update(pool=inherited['path'], pool_manifest=inherited['training_manifest'],
                   continuation={key: inherited[key] for key in ('iteration', 'training_seeds',
                       'total_training_episodes', 'training_lineage_id', 'initial_pool_sha256', 'checkpoint_sha256')})
    total = offset + cfg['iterations']
    iterations = iterations if iterations is not None else list(range(max(offset, 1), total + 1))
    if iterations != sorted(set(iterations)) or any(not max(offset, 1) <= k <= total for k in iterations):
        raise ValueError('Checkpoint iterations must be ordered and available in the declared lineage')
    checkpoints = {}
    for iteration in iterations:
        local = training_boundary(len(new_seeds), cfg['iterations'], iteration - offset)
        seeds = sorted([*inherited_seeds, *new_seeds[:local]])
        checkpoints[f'B2@{iteration}'] = {'iteration': iteration, 'training_seeds': seeds,
            'total_training_episodes': len(seeds), 'initial_pool_sha256': initial_hash,
            'path': inherited['path'] if iteration == offset else
                str(out / 'runs' / 'B2-finite-train' / f'pool-iteration-{iteration:04d}.json')}
        if inherited:
            checkpoints[f'B2@{iteration}']['training_lineage_id'] = inherited['training_lineage_id']
    protocol['checkpoint_evaluations'] = checkpoints
    protocol['all_arms_required'] = True
    protocol['training_plan'].update(inherited_optimizer_iterations=offset,
        segment_optimizer_iterations=cfg['iterations'], total_optimizer_iterations=total,
        inherited_training_episodes=len(inherited_seeds), segment_training_episodes=len(new_seeds),
        total_training_episodes=len(inherited_seeds) + len(new_seeds))
    reservation = protocol['core_evaluation_reservation']
    count = len(core_evaluation_labels(protocol)) * len(protocol['pilot_test_seeds'])
    upper = reservation['episode_upper_seconds']
    reservation.update(episodes=count, gpu_seconds_estimate=count * upper if upper is not None else None)


def make_protocol(args,out):
    if getattr(args,'frozen_policy_pilot',False):
        plan,history=campaign_seed_plan(args);prior,ledger=campaign_budget(args)
        if len(plan['final_test'])!=1 or args.max_gpus!=1:raise ValueError('Frozen exploratory pair requires one registered seed and one GPU')
        variant=getattr(args,'pilot_controller','B1-selector-contract')
        baseline=variant=='B0'
        labels=['B0','B1'] if baseline else ['B1',variant]
        version='cage4-acd-parity-v1' if baseline else 'cage4-acd-static-selector-pilot-v1'
        configs={}
        for label in labels:
            c=variant_config(label)
            c.update(iterations=1,experiment_id=str(out),comparison_run_id=str(out),
                experiment_version=version,output=str(out),journal_path=str(out/'journal.sqlite3'))
            configs[label]=apply_seed_plan(c,plan,history)
        reservation=core_cost_reservation(plan['final_test'])
        if reservation['episode_upper_seconds'] is None:
            raise ValueError('A comparable measured episode cost is required before reserving both policies')
        reservation.update(episodes=2,gpu_seconds_estimate=2*reservation['episode_upper_seconds']+STARTUP_SHUTDOWN_SECONDS)
        return {'schema':'durable-campaign-v1','registered_unix':time.time(),'experiment_id':str(out),
            'protocol_kind':'exploratory-frozen-policy-pair-v1','source_sha256':source_identity(),'configs':configs,
            'seed_plan':plan,'seed_exposure_history':history,'pilot_test_seeds':plan['final_test'],
            'budget_gpu_hours':args.gpu_hours,'maximum_wall_seconds':args.wall_hours*3600,**registered_budget(prior,ledger),
            'maximum_servers':1,'minimum_free_mib':args.minimum_free_mib,'profile_full_sequence':False,
            'core_evaluation_reservation':reservation,'required_evaluation_labels':labels,
            'initial_pool_file_sha256':sha(ROOT/'artifacts/initial_pool.json'),
            'training_plan':{'episodes':0,'iterations':0},
            'scope':'Exploratory frozen policies. Both complete episodes required.',
            'changed_factor':('Add the common initial static Skill Pool to the ACD baseline.' if baseline else
                              'Reselect once on a new urgent visible case while another family is active.' if variant=='B1-urgent-preemption'
                              else 'Selector-only CAGE4 role and action semantics guidance.'),
            'allocation':'Reserve both full 500-nominal episodes plus startup/shutdown before GPU launch.',
            'historical_performance_reference':historical_reference(args)}
    if getattr(args,'fixed_update_from',None):return make_fixed_protocol(args,out)
    plan,history=campaign_seed_plan(args);prior,ledger=campaign_budget(args)
    requested=list(dict.fromkeys(['B2',*args.arms]))
    controller_baselines={arm:'B1'+arm[2:] for arm in requested if CONTROLLER_VARIANTS.get(arm)=='B3'}
    labels=['B0','B1',*requested,*controller_baselines.values()];configs={}
    for label in labels:
        cfg=variant_config(label)
        cfg.update(iterations=args.iterations,experiment_id=str(out),comparison_run_id=str(out),
                   experiment_version='cage4-acd-parity-v1',
                   output=str(out),journal_path=str(out/'journal.sqlite3'))
        cfg['evolution']['preflight_required']=True
        configs[label]=apply_seed_plan(cfg,plan,history)
    protocol={'registered_unix':time.time(),'schema':'durable-campaign-v1','experiment_id':str(out),
        'source_sha256':source_identity(),'configs':configs,'requested_training_arms':requested,
        'protocol_kind':'strict-input-preserving-pilot','seed_plan':plan,'seed_exposure_history':history,
        'historical_performance_reference':historical_reference(args),
        'training_seeds':plan['train'],'pilot_test_seeds':plan['final_test'],
        'reserved_full_test_seeds':plan['reserved_unseen_seeds'],
        'known_development_seeds':history['exposed_seeds'],
        'generalization_seeds':plan['transfer'],'generalization_requested':args.generalization,
        'core_evaluation_labels':['B0','B1','B2'],'optional_training_arms':[a for a in requested if a!='B2'],
        'controller_baselines':controller_baselines,
        'core_evaluation_reservation':core_cost_reservation(plan['final_test']),
        'all_arms_required':getattr(args,'require_all_arms',False),
        'budget_gpu_hours':args.gpu_hours,'maximum_wall_seconds':args.wall_hours*3600,**registered_budget(prior,ledger),
        'maximum_servers':args.max_gpus,'minimum_free_mib':args.minimum_free_mib,
        'profile_full_sequence':getattr(args,'profile_full_sequence',False),
        'resource_policy':'One pinned BF16 model per selected GPU; serial model requests by default; KEEP, encoder and simulator stay on CPU. No quantization or input reduction.',
        'training_plan':{'episodes':len(plan['train']),'iterations':args.iterations},
        'statistics':{'pilot_n':len(plan['final_test']),'bootstrap_repetitions':10000,'bootstrap_seed':20261125,
                      'interval':'paired percentile 95%; independent training counts require artifact provenance'},
        'initial_pool_file_sha256':sha(ROOT/'artifacts/initial_pool.json'),
        'locks':{n:sha(ROOT/n) for n in ('research_sources.lock.json','research_models.lock.json','sources.lock.json','requirements.lock.txt','requirements-model.lock.txt')},
        'keep_weights_sha256':json.loads((ROOT/'research_sources.lock.json').read_text())['published_weights_sha256'],
        'policy_interface_sha256':{label:digest(policy_interface(config)) for label,config in configs.items()},
        'legacy_allocation':'Previously four GPUs used deadline=start+gpu_hours*3600/4; eight GPU-hours meant 120 wall minutes, training 85 minutes and rollout 83.5 minutes, including startup.',
        'allocation':'Reserve mandatory collection, profiling, updates, paired validation, final/required transfer and shutdown before launch. Refresh with each pending batch.',
        'scorer':'incremental_reference; experimental full-sequence cannot be adopted without a real-Qwen parity certificate',
        'selection':'Freeze completed_changed/completed_unchanged/completed_no_eligible_skill only. An interrupted pool is untrained_control, never a trained policy.'}
    register_checkpoint_evaluations(args, out, protocol)
    protocol['required_work_reservations']=[campaign_reservation(protocol,count) for count in range(1,args.max_gpus+1)]
    return protocol


def historical_batch(protocol,root=None,*,require_manifest=False):
    """Read old evidence with a read-only SQLite connection; never migrate the old DB."""
    import sqlite3
    from research.progress import _unpack
    source=protocol['fixed_update_source']
    if require_manifest and not source.get('training_manifest'):
        raise ValueError('Unbound historical checkpoint is available for inspection only; register its original training manifest')
    def resolve(value):
        path=Path(value)
        if root is None:return path
        base=Path(root).resolve()
        if path.is_absolute() or '..' in path.parts:raise ValueError('Restored historical paths must be relative')
        path=(base/path).resolve()
        if not path.is_relative_to(base):raise ValueError('Historical path escapes restore directory')
        return path
    required=[source['checkpoint'],source['journal'],source['initial_pool'],*source['training_traces']]
    if any(path not in source['files_sha256'] for path in required):
        raise ValueError('Historical checkpoint, journal, initial Pool and traces need registered hashes')
    for path,expected in source['files_sha256'].items():
        if sha(resolve(path))!=expected:raise ValueError('Historical input changed: '+path)
    with sqlite3.connect('file:'+str(resolve(source['journal']))+'?mode=ro',uri=True) as db:
        def get(key):
            record=db.execute('SELECT value FROM checkpoints WHERE key=?',(key,)).fetchone()
            if record is None:raise ValueError('Missing committed historical input: '+key)
            return _unpack(record[0])
        pending=get('pending_batch:1:B2');preparation=get('preparation:1:B2')
    trained=json.loads(resolve(source['checkpoint']).read_text())
    if source.get('training_manifest'):
        if source['training_manifest'] not in source['files_sha256']:
            raise ValueError('Historical training manifest needs a registered hash')
        manifest=read_training_manifest(resolve(source['checkpoint']),trained,manifest_ref={
            'path':str(resolve(source['training_manifest']).resolve()),
            'sha256':source['files_sha256'][source['training_manifest']]})
        historical=policy_interface(manifest['config'])
        if historical!=source['policy_interface']:
            raise ValueError('Historical training interface differs from the registered study')
        check_pool_interface(trained,manifest['config'],manifest=manifest)
        if any(policy_interface(config)!=historical for config in protocol['configs'].values()):
            raise ValueError('Fixed update configurations must preserve the historical policy interface')
    return trained,pending,preparation


def make_fixed_protocol(args,out):
    """A separate fixed-proposal verification study, not a new B3/B4 training claim."""
    plan,history=campaign_seed_plan(args);prior,ledger=campaign_budget(args)
    source=args.fixed_update_from.resolve();campaign=source.parent.parent
    initial=ROOT/'artifacts/initial_pool.json';training=campaign/'runs/B2-finite-train'
    traces=sorted(training.glob('episode-????.jsonl'))
    manifest_path=training/'manifest.json';journal_path=campaign/'journal.sqlite3'
    paths=[source,initial,journal_path,manifest_path,*traces]
    if len(traces)!=2 or any(not p.is_file() for p in paths):
        raise ValueError('Exact committed checkpoint, training manifest, journal and two training traces required')
    manifest=read_policy_manifest(manifest_path)
    if manifest.get('stage')!='train':raise ValueError('Expected the historical training manifest')
    manifest_ref={'path':str(manifest_path),'sha256':sha(manifest_path)}
    old_config=manifest['config'];interface=policy_interface(old_config)
    if args.max_gpus!=1:raise ValueError('Fixed-update study uses one unoccupied GPU')
    probes_only=getattr(args,'diagnosis_probes_only',False)
    labels=['B0','B2','B3'] if probes_only else ['B0','B1','B2','B3','B4',*VERIFICATION_VARIANTS];configs={}
    for label in labels:
        cfg=variant_config(label)
        for key in ('communication','action_contract','profile','skip_busy_policy','behavior_context','case_presentation','connection_leads'):
            if key in old_config:cfg[key]=deepcopy(old_config[key])
        cfg['communication_encoder']=interface['communication_encoder']
        cfg['selector']=deepcopy(old_config['selector'])
        cfg.update(iterations=1,experiment_id=str(out),comparison_run_id=str(out),
            experiment_version='cage4-historical-facts-v1' if probes_only else 'cage4-support-shift-v3-fixed-update',
            output=str(out),journal_path=str(out/'journal.sqlite3'))
        cfg['evolution']['preflight_required']=False
        if label=='B2':cfg['pool_manifest']=manifest_ref
        configs[label]=apply_seed_plan(cfg,plan,history)
    frozen=json.loads(source.read_text())
    original_manifest=read_training_manifest(source,frozen,manifest_ref=manifest_ref)
    check_pool_interface(frozen,old_config,manifest=original_manifest)
    if frozen.get('optimizer_status')!='completed_changed' or frozen.get('pending_experiences'):
        raise ValueError('Fixed proposal must come from a completed original training update')
    if frozen.get('training_seeds')!=plan['train']:raise ValueError('Seed plan must name the exact historical training seeds')
    version='historical-diagnosis-probes-v1' if probes_only else 'fixed-committed-update-verification-v1'
    return {'schema':'durable-campaign-v1','registered_unix':time.time(),'experiment_id':str(out),
        'source_sha256':source_identity(),'configs':configs,'protocol_kind':version,
        'fixed_update_source':{'checkpoint':str(source),'journal':str(journal_path),'initial_pool':str(initial),
            'training_manifest':str(manifest_path),'policy_interface':interface,
            'training_traces':[str(p) for p in traces],'files_sha256':{str(p):sha(p) for p in paths}},
        'scope':'Verify the exact historical whole committed B2 update against the exact initial pool. No regenerated proposal or new training loop; full B3/B4 training remains separate.',
        'seed_plan':plan,'seed_exposure_history':history,
        'historical_performance_reference':historical_reference(args),
        'training_seeds':plan['train'],'new_training_seeds_reserved':[],
        'pilot_test_seeds':plan['final_test'],
        'rollout_validation_seeds':plan['validation'],'reject_only_prefix':plan['validation'][:2],
        'known_development_seeds':sorted({e['seed'] for e in history['events'] if e['role'] in ('final_test','historical_development_test')}),
        'generalization_seeds':plan['transfer'],
        'generalization_red':'aggressive','generalization_requested':args.generalization,
        'budget_gpu_hours':args.gpu_hours,'maximum_wall_seconds':args.wall_hours*3600,
        **registered_budget(prior,ledger),'maximum_servers':1,'minimum_free_mib':args.minimum_free_mib,
        'profile_full_sequence':args.profile_full_sequence,'maximum_profile_seconds':900.,
        'selector_probe':{'maximum_states':24,'include_controls':True,'epsilon':configs['B2']['selector']['epsilon'],
            'allowed_seeds':plan['train'],
            'forbidden_seeds':sorted(set(plan['final_test']+plan['transfer'])|{e['seed'] for e in history['events'] if e['role'] in ('final_test','historical_development_test')}),
            'maximum_disagreement':.25,'maximum_total_variation':.2,'maximum_share_increase':.2},
        'diagnosis_probe_count':getattr(args,'diagnosis_probe_count',6),
        'statistics':{'pilot_n':len(plan['final_test']),'bootstrap_repetitions':10000,'bootstrap_seed':20261125,
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
    trained,pending,preparation=historical_batch(protocol,require_manifest=True)
    if protocol['protocol_kind']=='historical-diagnosis-probes-v1':
        runtime={**protocol['configs']['B3']['model'],'endpoint':f"http://127.0.0.1:{servers[0]['port']}",
            'journal_path':str(journal.path),'request_context':{'experiment':str(out),'arm':'B3','env_stage':'historical_training_probe'}}
        client=ModelClient(runtime)
        if client.identity!=trained.get('model_identity'):
            raise ValueError('Historical diagnosis model identity differs from the recorded training model')
        configure_encoder(client.embed);switch_phase('historical_diagnosis_probes')
        profile_fixed_update(out,protocol,client,trained,pending,preparation,remaining_seconds,diagnosis_only=True)
        historical_batch(protocol)
        write(out/'historical_inputs_unchanged.json',{'passed':True,'files_sha256':protocol['fixed_update_source']['files_sha256']})
        return
    previous=json.loads((ROOT/'artifacts/initial_pool.json').read_text())
    campaign=Path(protocol['fixed_update_source']['checkpoint']).parent.parent
    measured=[json.loads(p.read_text())['execution_time_s'] for p in campaign.glob('runs/*-evaluate-*/episode-0000.json')]
    episode_upper=max(measured)*1.1 if measured else 1100.
    required=(3*len(protocol['pilot_test_seeds'])+2*len(protocol['reject_only_prefix']))*episode_upper+900+90
    preflight={'episode_upper_seconds':episode_upper,'primary_and_reject_screen_seconds':required,
        'remaining_seconds':remaining_seconds(),
        'full_registered_validation_gpu_hours':2*len(protocol['rollout_validation_seeds'])*episode_upper/3600,
        'full_B3_B4_new_training':'deferred; no new training loop is claimed in this fixed-update study',
        'status':'passed' if required<=remaining_seconds() else 'deferred'}
    write(out/'preflight.json',preflight)
    if preflight['status']!='passed':raise BudgetExceeded('Complete predeclared fixed-update pilot does not fit')
    runtime={**protocol['configs']['B2']['model'],'endpoint':f"http://127.0.0.1:{servers[0]['port']}",
        'journal_path':str(journal.path),'request_context':{'experiment':str(out),'arm':'fixed-update-audit','env_stage':'historical_training_probe'}}
    client=ModelClient(runtime)
    if client.identity!=trained.get('model_identity'):
        raise ValueError('Historical fixed-update model identity differs from the recorded training model')
    configure_encoder(client.embed)
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
        paths=[campaign/'runs'/f'{label}-finite-evaluate-{seed}'/'episode-0000.jsonl' for label in ('B1','B2')]
        if any(not path.exists() for path in paths):continue
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
            ('composition','profile','behavior_context','action_contract','selector','skill_seed_set','model','scoring','communication','communication_encoder','skip_busy_policy')},
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


def read_cost_progress(journal):
    progress=journal.restore('campaign_cost_progress') or {}
    curve_receipts=progress.get('curve_evaluation_receipts',{})
    completed_curve={value['summary_path']:value for value in curve_receipts.values()}
    for receipt in [*progress.get('evaluation_receipts',{}).values(),completed_curve]:
        for path, value in receipt.items():
            summary_hash = value['summary_sha256'] if isinstance(value, dict) else value
            if not Path(path).is_file() or sha(path) != summary_hash:
                raise ValueError('A completed evaluation record changed after it was reserved as complete')
    return progress


def bind_checkpoints(protocol, out, journal):
    """Bind each predeclared update to its unique committed checkpoint and Pool."""
    bindings = journal.restore('checkpoint_evaluation_bindings') or {}
    original = protocol.get('continuation_source') or {}
    cfg = protocol['configs']['B2']
    for label, expected in protocol.get('checkpoint_evaluations', {}).items():
        path = Path(expected['path'])
        inherited = path == Path(original.get('path', ''))
        manifest_path = path.parent / 'manifest.json'
        manifest_ref = original['training_manifest'] if inherited else {
            'path': str(manifest_path), 'sha256': sha(manifest_path)}
        checkpoint = json.loads(path.read_text())
        manifest = read_training_manifest(path, checkpoint, manifest_ref=manifest_ref)
        if not inherited and manifest['config'].get('protocol_digest') != protocol['comparison_protocol_digest']:
            raise ValueError('Checkpoint training did not use the registered campaign')
        proof = verify_completed_update(checkpoint, expected, cfg, manifest)
        checkpoint_hash = sha(path)
        receipt = original if inherited else journal.restore(f"loop_commit:{expected['iteration']}:B2")
        if not receipt or receipt['checkpoint_sha256'] != checkpoint_hash:
            raise ValueError('Checkpoint differs from the committed optimizer receipt')
        binding = {**expected, **proof, 'checkpoint_sha256': checkpoint_hash,
                   'training_manifest': manifest_ref, 'training_run_id': checkpoint['training_run_id']}
        if label in bindings and bindings[label] != binding:
            raise ValueError('Registered iteration has a different checkpoint hash or identity')
        bindings[label] = binding
    if bindings:
        journal.checkpoint('checkpoint_evaluation_bindings', bindings)
        write(out / 'frozen' / 'checkpoints.json', bindings)
    return bindings


def run_training_study(protocol,out,journal,servers,worker,remaining_seconds,switch_phase,abort):
    """Refresh the complete outstanding reservation before every pending update."""
    from scripts.audit_training_cost import audit,profile_optimizer,optimizer_forecast,SHUTDOWN_SECONDS
    core=core_evaluation_labels(protocol);extensions={};trained_labels=[]
    progress=read_cost_progress(journal)
    for key in ('updates','collected','profiles','evaluation_receipts'):
        progress.setdefault(key,{})
    progress['evaluated_blocks']=list(progress['evaluation_receipts'])
    all_required=protocol.get('all_arms_required',False)
    def evaluation_block(label):
        if label=='B2':return core
        static=protocol.get('controller_baselines',{}).get(label)
        return [static,label] if static else [label]
    def save_progress():
        journal.checkpoint('campaign_cost_progress',progress)
    def reserve(label,phase):
        scope=protocol
        if not all_required and label!='B2':
            scope={**protocol,'all_arms_required':True,'requested_training_arms':[label],
                   'core_evaluation_labels':evaluation_block(label),'generalization_requested':False}
            state={**progress,'evaluated_blocks':['core'] if label in progress['evaluated_blocks'] else []}
        else:
            state=progress
        result=campaign_reservation(scope,len(servers),progress=state,usage=journal.budget_usage(),starting=False)
        journal.checkpoint('remaining_required_work',result)
        write(out/f'remaining-work-{time.time_ns()}.json',{'phase':phase,'arm':label,**result})
        if result['status']!='passed':
            raise BudgetExceeded(f'Outstanding required work does not fit during {phase}: {label}')
        return result
    def train(label):
        switch_phase('training_'+label)
        reserve(label,'collection')
        cfg=protocol['configs'][label]
        offset=(cfg.get('continuation') or {}).get('iteration',0)
        while True:
            dest,job=worker(label,'train',servers[0]['port'])
            checkpoint=dest/'checkpoint.json'
            trained=json.loads(checkpoint.read_text()) if checkpoint.is_file() else {}
            completed=len(trained.get('history',[]))-offset if trained else 0
            if not 0<=completed<=cfg['iterations'] or trained.get('segment_completed_iterations',completed)!=completed:
                raise IncompleteOptimization('Committed updates disagree with this training segment: '+label)
            progress['updates'][label]=completed
            if job['status']=='completed':
                break
            iteration=offset+completed+1
            if job.get('failure_status')!='waiting_for_cost_preflight' or completed>=cfg['iterations']:
                raise IncompleteOptimization('Optimizer did not finish: '+label)
            pending=journal.restore(f'pending_batch:{iteration}:{label}')
            if not pending:
                raise IncompleteOptimization('No complete pending training batch for '+label)
            batch=optimizer_batch_identity(pending)
            key=f'preflight:{label}:{iteration}'
            previous=journal.restore(key)
            if previous and previous.get('status')=='passed':
                raise IncompleteOptimization('Worker is waiting on a previously approved optimizer batch: '+label)
            progress['collected'][label]=pending['environment_episodes_completed']
            if pending.get('segment_completed_iterations',completed)!=completed:
                raise IncompleteOptimization('Pending batch update count differs from the committed segment')
            save_progress()
            reserve(label,'before_cost_profile')
            deadline=time.time()+remaining_seconds()-SHUTDOWN_SECONDS
            measurements=profile_optimizer(out,label,pending,servers[0]['port'],iteration=iteration,deadline=deadline)
            forecast=optimizer_forecast(measurements)
            if forecast['status']!='estimated':
                write(out/f'preflight-{label}-{iteration:04d}.json',{'batch_identity':batch,**forecast})
                raise BudgetExceeded('An actual optimizer cost forecast is unavailable: '+label)
            progress['episode_upper_seconds']=max(progress.get('episode_upper_seconds',0),
                measurements['episode_upper_seconds'])
            progress['profiles'][label]={'iteration':iteration,'batch_identity':batch,**forecast}
            save_progress()
            try:
                estimate=reserve(label,'after_cost_profile')
            except BudgetExceeded:
                estimate=journal.restore('remaining_required_work')
                journal.checkpoint(key,{'batch_identity':batch,**estimate})
                write(out/f'preflight-{label}-{iteration:04d}.json',{'batch_identity':batch,**estimate})
                raise
            approval={'batch_identity':batch,'profile':forecast,**estimate}
            journal.checkpoint(key,approval)
            write(out/f'preflight-{label}-{iteration:04d}.json',approval)
        if (trained.get('pending_experiences') or trained.get('pending_results') or trained.get('optimizer_status') not in COMPLETED_OPTIMIZER_STATUSES or
                trained.get('training_completed') is False or trained.get('episodes_completed',0)<cfg['train_episodes'] or
                completed!=cfg['iterations']):
            raise IncompleteOptimization('No completed registered training plan: '+label)
        progress['collected'][label]=trained['episodes_completed']
        save_progress()
        if label=='B2' and protocol.get('checkpoint_evaluations'):
            return
        frozen=out/'frozen'/f'{label}.json'
        if frozen.exists() and sha(frozen)!=sha(checkpoint):raise ValueError('Frozen policy changed')
        if not frozen.exists():write(frozen,trained)
    def evaluate(block,labels,seeds,reds=('finite',),*,checkpoints=None):
        if block in progress['evaluated_blocks']:
            return
        if block!='transfer' or all_required:
            reserve('B2' if block in ('core','transfer') else block,'before_frozen_evaluation')
        estimate=max((protocol['core_evaluation_reservation']['episode_upper_seconds'] or 0),
                     (progress.get('episode_upper_seconds') or 0))
        required=len(labels)*len(seeds)*len(reds)
        pending=queue.Queue();records={};records_lock=threading.Lock()
        curve_receipts=progress.setdefault('curve_evaluation_receipts',{}) if checkpoints else {}
        for red in reds:
            for seed in seeds:
                for label in labels:
                    receipt=curve_receipts.get(f'{label}:{red}:{seed}')
                    if receipt:records[receipt['summary_path']]=receipt
                    else:pending.put((label,seed,red))
        if not estimate or math.ceil(pending.qsize()/len(servers))*estimate+60>remaining_seconds():
            raise BudgetExceeded('Complete predeclared evaluation block does not fit')
        def queue_worker(port):
            while not abort.is_set():
                try:label,seed,red=pending.get_nowait()
                except queue.Empty:return
                try:
                    binding = (checkpoints or {}).get(label)
                    if binding:
                        dest,job=worker(label,'evaluate',port,seed,red,Path(binding['path']),checkpoint=binding)
                    else:
                        static=label=='B1' or CONTROLLER_VARIANTS.get(label)=='B1'
                        pool=None if label=='B0' else ROOT/'artifacts/initial_pool.json' if static else out/'frozen'/f'{label}.json'
                        dest,job=worker(label,'evaluate',port,seed,red,pool)
                    if job['status']!='completed' or job.get('summary_status')!='completed':raise IncompleteOptimization('Required evaluation incomplete')
                    summary=dest/'summary.json'
                    receipt=sha(summary)
                    if binding:
                        receipt={'summary_sha256':receipt,'checkpoint_iteration':binding['iteration'],
                                 'checkpoint_sha256':binding['checkpoint_sha256'],'pool_sha256':binding['pool_sha256'],
                                 'seed':seed,'task_id':job['task_id']}
                    with records_lock:
                        if checkpoints:
                            if not binding:receipt={'summary_sha256':receipt,'seed':seed,'task_id':job['task_id']}
                            receipt['summary_path']=str(summary)
                            curve_receipts[f'{label}:{red}:{seed}']=receipt
                            save_progress()
                        records[str(summary)]=receipt
                except BaseException:abort.set();raise
                finally:pending.task_done()
        with ThreadPoolExecutor(max_workers=len(servers)) as executor:
            for future in [executor.submit(queue_worker,s['port']) for s in servers]:future.result()
        if len(records)!=required:
            raise IncompleteOptimization('Evaluation block has unverified episode receipts')
        progress['evaluation_receipts'][block]=records
        progress['evaluated_blocks']=list(progress['evaluation_receipts'])
        save_progress()
    train('B2')
    checkpoints=bind_checkpoints(protocol,out,journal) if protocol.get('checkpoint_evaluations') else None
    switch_phase('core_frozen_test')
    evaluate('core',core,protocol['pilot_test_seeds'],checkpoints=checkpoints)
    write(out/'core_comparison.json',{'status':'completed','labels':core,'seeds':protocol['pilot_test_seeds'],
        'actual_episodes':len(core)*len(protocol['pilot_test_seeds']),'optional_arms_not_selected_by_reward':True})
    for label in protocol['requested_training_arms']:
        if label=='B2':continue
        try:
            train(label);switch_phase('extension_frozen_test_'+label)
            evaluate(label,evaluation_block(label),protocol['pilot_test_seeds'])
        except BudgetExceeded as exc:
            extensions[label]={'status':'deferred_budget','reason':str(exc)};write(out/'extension_status.json',extensions)
            if all_required:raise
            continue
        trained_labels.extend(evaluation_block(label));extensions[label]={'status':'completed'};write(out/'extension_status.json',extensions)
    if protocol.get('generalization_requested'):
        switch_phase('generalization_test')
        try:evaluate('transfer',core+trained_labels,protocol['generalization_seeds'],REDS[1:])
        except BudgetExceeded as exc:
            write(out/'generalization_preflight.json',{'status':'deferred_budget','reason':str(exc)})
            if all_required:raise
    audit(out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu-hours',type=float,default=LEGACY_GPU_AUTHORIZATION['gpu_hour_cap'])
    parser.add_argument('--wall-hours',type=float,default=8.)
    parser.add_argument('--authorization',type=Path,help='Explicit approved cumulative GPU cap and usage-ledger scope')
    parser.add_argument('--max-gpus',type=int,choices=range(1,5),default=1)
    parser.add_argument('--minimum-free-mib',type=int,default=21000)
    parser.add_argument('--profile-full-sequence',action='store_true',help='Opt in to the memory-intensive experimental scorer probe; reference scoring is unchanged')
    parser.add_argument('--fixed-update-from',type=Path,help='Verify an exact committed historical B2 update; no new training claim')
    parser.add_argument('--continue-from',type=Path,help='Start a new B2 training segment from a committed campaign checkpoint')
    parser.add_argument('--checkpoint-iterations',nargs='+',type=int,help='Cumulative B2 updates to evaluate on the shared registered holdout')
    parser.add_argument('--seed-plan',type=Path,help='Explicit train/validation/final_test/transfer seed lists and purpose')
    parser.add_argument('--reference-report',type=Path,help='Immutable historical metrics JSON for separate context, not new evidence')
    parser.add_argument('--usage-ledger',type=Path,nargs='+',help='Prior cumulative completion ledgers; sessions are deduplicated')
    parser.add_argument('--diagnosis-probes-only',action='store_true',help='Historical training SG probes only; no training or final evaluation')
    parser.add_argument('--frozen-policy-pilot',action='store_true',help='One preregistered frozen-policy pair')
    parser.add_argument('--pilot-controller',choices=['B0',*[name for name,base in CONTROLLER_VARIANTS.items() if base=='B1']],
                        default='B1-selector-contract',help='Comparison with static B1 for --frozen-policy-pilot')
    parser.add_argument('--diagnosis-probe-count',type=int,default=6,choices=range(1,13))
    parser.add_argument('--train-episodes',type=int,help='Must equal the explicit training seed count')
    parser.add_argument('--iterations',type=int,default=1,help='Actual training optimizer iterations; full six/three requires explicit plan')
    parser.add_argument('--require-all-arms',action='store_true',help='Reserve all registered training/evaluation arms up front')
    parser.add_argument('--prior-gpu-hours',type=float,default=None,help='Recorded GPU carry-in floor for the same authorization')
    parser.add_argument('--test-episodes',type=int,default=None,choices=range(1,101))
    parser.add_argument('--arms',nargs='+',choices=['B2','B3-no-support','B3','B4',*VERIFICATION_VARIANTS,
        *[name for name,base in CONTROLLER_VARIANTS.items() if base=='B3']],default=['B2'])
    parser.add_argument('--generalization',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--cleanup-only',action='store_true')
    args=parser.parse_args()
    if args.frozen_policy_pilot and (args.fixed_update_from or args.diagnosis_probes_only):raise ValueError('Frozen pilot and historical diagnosis are separate protocols')
    if args.diagnosis_probes_only and not args.fixed_update_from:raise ValueError('Diagnosis probes require the exact historical training source')
    if (args.continue_from or args.checkpoint_iterations) and (args.fixed_update_from or args.frozen_policy_pilot):
        raise ValueError('Checkpoint curves require a B2 training campaign')
    if not math.isfinite(args.gpu_hours) or args.gpu_hours<=0 or not math.isfinite(args.wall_hours) or args.wall_hours<=0:
        raise ValueError('Positive finite GPU-hour and wall-hour caps are required')
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
            registered=protocol.get('gpu_authorization',LEGACY_GPU_AUTHORIZATION)
            source=args.authorization or registered.get('file')
            if read_gpu_authorization(source)!=registered:
                raise ValueError('Resume must retain the registered GPU authorization')
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
    reservation=protocol.get('core_evaluation_reservation',{})
    required=reservation.get('gpu_seconds_estimate')
    planned_work=protocol.get('required_work_reservations')
    if args.plan_only:
        labels=protocol.get('required_evaluation_labels',[])
        write(out/'preflight.json',{'status':'not_measured','reason':'No GPU calls in plan-only mode','budget':journal.budget_usage(),
            'planned_first_arm':labels[0] if labels else 'B2',
            'reserved_evaluation_gpu_seconds':required,
            'reservation_fits_gpu_cap':required<=protocol['budget_gpu_hours']*3600 if required is not None else None,
            'required_work_reservations':planned_work,
            'required_work_fits_caps':any(r['status']=='passed' for r in planned_work) if planned_work is not None else None,
            'later_arms':'deferred_until_measured_cost_fits'})
        print(json.dumps({'status':'planned','output':str(out)}));return
    completed=out/'completion.json'
    if completed.exists() and json.loads(completed.read_text()).get('status')=='completed':
        print(json.dumps({'status':'reused_completed_campaign','new_gpu_sessions':0,'output':str(out)}));return
    authorization=protocol.get('gpu_authorization',LEGACY_GPU_AUTHORIZATION)
    def defer_launch(reason, estimates=None):
        write(out/'completion.json',{'status':'deferred_budget','reason':reason,'required_work_reservations':estimates,
            'core_reservation':reservation,'new_campaign_usage':journal.budget_usage(),'all_owned_resources_released':True})
        print(json.dumps({'status':'deferred_budget','new_gpu_sessions':0,'output':str(out)}))
    if planned_work is None and required is not None and (
            required>protocol['budget_gpu_hours']*3600 or required>protocol['maximum_wall_seconds']):
        defer_launch('Reserved mandatory evaluation exceeds the GPU-hour or wall cap')
        return
    # Reconcile under a shared lease immediately before a real launch/resume.
    # A new output and an older carry-in record cannot independently spend the
    # same remaining scope.
    budget_root=ROOT/'research_outputs'
    budget_root.mkdir(parents=True,exist_ok=True)
    scope_lease=(budget_root/('.gpu-budget-'+digest(authorization['scope'])[:16]+'.lock')).open('a')
    try:fcntl.flock(scope_lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('A campaign is already using this GPU authorization scope')
    carry=SimpleNamespace(gpu_hours=protocol['budget_gpu_hours'],wall_hours=protocol['maximum_wall_seconds']/3600,
        usage_ledger=(protocol.get('prior_physical_usage') or {}).get('sources',[]),
        prior_gpu_hours=protocol.get('previous_campaign_gpu_hours',0))
    _,live_ledger=campaign_budget(carry,authorization=authorization,exclude=out)
    write(out/f'authorization-check-{time.time_ns()}.json',live_ledger)
    def fitting_replicas(limit):
        if planned_work is None:
            return limit,None
        progress=read_cost_progress(journal)
        usage=journal.budget_usage()
        estimates=[campaign_reservation(protocol,count,progress=progress,usage=usage) for count in range(1,limit+1)]
        fit=max((r['replicas'] for r in estimates if r['status']=='passed'),default=0)
        write(out/f'required-work-preflight-{time.time_ns()}.json',{'selected_replicas':fit,'estimates':estimates})
        return fit,estimates
    replica_count,estimates=fitting_replicas(protocol['maximum_servers'])
    if not replica_count:
        defer_launch('Complete required work cannot be reserved within both caps',estimates)
        return
    for path, expected in protocol.get('continuation_source', {}).get('files_sha256', {}).items():
        if sha(path) != expected:
            raise ValueError('Registered continuation input changed: '+path)
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
    replica_count,estimates=fitting_replicas(min(len(available),protocol['maximum_servers']))
    if not replica_count:
        defer_launch('Complete required work cannot be reserved with the available GPU count',estimates)
        return
    journal.budget_configure(protocol['budget_gpu_hours'],protocol['maximum_wall_seconds'],
        authorization=authorization,prior_gpu_hours=protocol.get('previous_campaign_gpu_hours',0))
    journal.check_budget()
    available=available[:replica_count]
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
    def worker(label,stage,port,seed=None,red='finite',pool=None,*,checkpoint=None):
        if source_identity()!=protocol['source_sha256']:raise RuntimeError('Registered source changed')
        policy_tag=label
        if checkpoint:
            if stage!='evaluate' or checkpoint!=json.loads((out/'frozen/checkpoints.json').read_text()).get(label):
                raise ValueError('Worker checkpoint is absent from the locked evaluation list')
            policy_tag=f"{label}-{checkpoint['checkpoint_sha256'][:16]}"
        tag=f'{policy_tag}-{red}-{stage}'+(f'-{seed}' if seed is not None else '')
        dest=out/'runs'/tag;dest.mkdir(parents=True,exist_ok=True)
        cfg=deepcopy(protocol['configs']['B2' if checkpoint else label]);cfg['model'].update(endpoint=f'http://127.0.0.1:{port}',
            generation_endpoints=[f"http://127.0.0.1:{s['port']}" for s in servers],
            scoring_endpoints=[f"http://127.0.0.1:{s['port']}" for s in servers],sg_workers=len(servers))
        if stage=='evaluate':
            cfg.update(episodes=1,eval_seed=seed,eval_seeds=[seed],evaluation_disabled=False,
                red=red,pool=str(pool) if pool else None,evaluation_role='exploratory_frozen_pilot' if protocol['protocol_kind']=='exploratory-frozen-policy-pair-v1' else 'frozen_test' if red=='finite' else 'transfer_test')
            if checkpoint:
                cfg.update(comparison_id=label,base_comparison_id='B2',checkpoint_evaluation=checkpoint,
                           pool_manifest=checkpoint['training_manifest'])
        elif cfg['evolution'].get('rollout_verification'):cfg['rollout_validation'].update(ports=[s['port'] for s in servers])
        cfg=execution_config(cfg,'eval' if stage=='evaluate' else 'train')
        config_path=dest/'config.json'
        if config_path.exists():
            saved=json.loads(config_path.read_text())
            if checkpoint and any(saved.get(key)!=cfg[key] for key in (
                    'checkpoint_evaluation','pool','protocol_digest','eval_seeds','comparison_id')):
                raise ValueError('Existing evaluation config differs from the registered checkpoint')
            cfg=saved
        else:write(config_path,cfg)
        command=[str(ROOT/'.venv/bin/python'),'-m','research',stage,'--config',str(config_path),'--output',str(dest)]
        endpoints=[f"http://127.0.0.1:{s['port']}" for s in servers]
        command+=['--endpoint',f'http://127.0.0.1:{port}','--scoring-endpoints',*endpoints,
            '--generation-endpoints',*endpoints,'--sg-workers',str(len(servers))]
        if stage=='train' and cfg['evolution'].get('rollout_verification'):command+=['--rollout-ports',*[str(s['port']) for s in servers]]
        if (dest/'manifest.json').exists():command.append('--resume')
        pool_hash=checkpoint['checkpoint_sha256'] if checkpoint else sha(pool) if pool else None
        task_id=digest([protocol['experiment_id'],label,stage,seed,red,pool_hash])
        job={'task_id':task_id,'label':label,'stage':stage,'seed':seed,'red':red,'directory':str(dest),'started_unix':time.time()}
        if checkpoint:job.update(checkpoint_iteration=checkpoint['iteration'],checkpoint_sha256=pool_hash)
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
        if process.returncode and (dest/'failure.json').is_file():
            job['failure_status']=json.loads((dest/'failure.json').read_text()).get('status')
        if (dest/'summary.json').exists():
            summary=json.loads((dest/'summary.json').read_text());job.update(summary_status=summary['status'],completed_episodes=len(summary.get('episodes',[])))
            if stage=='evaluate' and job['status']=='completed':
                episodes=summary.get('episodes',[])
                if (summary.get('result_kind')!='real_cage4' or len(episodes)!=1 or episodes[0].get('seed')!=seed
                        or episodes[0].get('status')!='completed' or episodes[0].get('kind')!='real_cage4'):
                    job['status']='incomplete'
                if checkpoint and ((summary.get('training_provenance') or {}).get('checkpoint_evaluation')!=checkpoint
                        or summary.get('pool_sha256')!=checkpoint['pool_sha256']
                        or not episodes or episodes[0].get('input_pool_sha256')!=checkpoint['pool_sha256']):
                    job['status']='incomplete'
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
                    model=next(iter(protocol['configs'].values()))['model']
                    if identity.get('model')!=model['model'] or identity.get('revision')!=model['revision'] or identity.get('dtype')!='bfloat16' or not identity.get('weights_frozen'):raise ValueError('Model identity mismatch')
                    write(out/f"identity-{server['port']}.json",identity);ready=True;break
                except OSError:time.sleep(.5)
            if not ready:raise RuntimeError('Pinned Qwen startup failed')
        if protocol['protocol_kind']=='exploratory-frozen-policy-pair-v1':
            switch_phase('exploratory_frozen_pilot')
            for label in protocol['required_evaluation_labels']:
                pool=None if label=='B0' else ROOT/'artifacts/initial_pool.json'
                for seed in protocol['pilot_test_seeds']:
                    _,job=worker(label,'evaluate',servers[0]['port'],seed,pool=pool)
                    if job.get('status')!='completed' or job.get('summary_status')!='completed':
                        raise IncompleteOptimization('Required exploratory episode incomplete; no paired claim')
            return
        if protocol['protocol_kind'] in ('fixed-committed-update-verification-v1','historical-diagnosis-probes-v1'):
            run_fixed_study(protocol,out,journal,servers,worker,remaining_seconds,switch_phase)
            return
        run_training_study(protocol,out,journal,servers,worker,remaining_seconds,switch_phase,abort)
    except BaseException as exc:
        failure={'type':type(exc).__name__,'error':str(exc)}
        previous=out/'failure.json'
        if previous.exists():
            previous.replace(out/f'failure-{time.time_ns()}.json')
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
        status='deferred_budget' if failure and failure['type']=='BudgetExceeded' else 'partial' if failure else 'completed'
        write(out/'completion.json',{'status':status,'failure':failure,
            'new_campaign_usage':journal.budget_usage(),'job_statuses':jobs,**release})
        for sig,handler in previous_handlers.items():signal.signal(sig,handler)
        from scripts.report_experiment import report_experiment
        report_experiment(out,include_trace=True)
    if failure:raise SystemExit(failure['type']+': '+failure['error'])


if __name__=='__main__':main()
