"""Durable, queued paired CAGE4 validation; completed seeds are never overwritten."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
from . import ROOT
from .rollout import pool_digest,verify_transaction
from .config import split_seeds, execution_config, policy_interface, check_pool_interface
from .progress import Journal,atomic_json,digest,BudgetExceeded,IncompleteOptimization,implementation_identity


class EnvironmentVerifier:
    def __init__(self,config,output,identity):
        self.config=deepcopy(config);self.directory=Path(output)/'rollout-verification'
        self.directory.mkdir(exist_ok=True);self.identity=identity
        self.rule=deepcopy(config['rollout_validation'])
        used=set(split_seeds(config,'train'))|set(split_seeds(config,'final_test'))
        if used&set(self.rule['seeds']):raise ValueError('Rollout validation seeds overlap training/final evaluation')
        self.deadline=self.rule.get('deadline_unix',float('inf'))
        self.journal=Journal(config.get('journal_path',Path(output)/'journal.sqlite3'))
        self.source=implementation_identity(ROOT)
        self.completed_training_episodes=config['train_episodes']
        inherited_path=Path(config['pool']) if config.get('pool') else None
        inherited=json.loads(inherited_path.read_text()) if inherited_path and inherited_path.is_file() else {}
        if 'pool' in inherited:check_pool_interface(inherited,config,static_initial=True)
        self.inherited_episodes=inherited.get('total_training_episodes',inherited.get('episodes_completed',0))
        self.inherited_seeds=inherited.get('training_seeds',[])
        self.inherited_iterations=len(inherited.get('history',[]))

    def __call__(self,pool,candidate,support,proposed_snapshot=None):
        planned=self.config['train_episodes'];iterations=self.config['iterations']
        local_iteration=support['iteration']-self.inherited_iterations
        self.completed_training_episodes=min(planned,(local_iteration*planned+iterations-1)//iterations)
        decision_rule={k:v for k,v in self.rule.items() if k not in ('ports','deadline_unix')}
        case_key=digest([support,self._reuse_identity(pool.snapshot(),None),candidate.__dict__,proposed_snapshot,decision_rule])
        case=self.directory/f"iteration-{support['iteration']:04d}-{case_key[:20]}"
        case.mkdir(exist_ok=True)
        with (case/'.lock').open('w') as stream:
            try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise IncompleteOptimization('Validation case has a live worker')
            if (case/'decision.json').exists():return json.loads((case/'decision.json').read_text())
            rule={**self.rule,'parent':support['parent'],'iteration':support['iteration'],'scope':support.get('scope','live_candidate')}
            options={}
            if rule.get('reject_screen_pairs',0):
                options['evaluate_pair']=lambda a,b,seed:(self.evaluate(a,'previous',case,[seed])[0],self.evaluate(b,'candidate',case,[seed])[0])
            if proposed_snapshot is not None:options['proposed_snapshot']=proposed_snapshot
            result=verify_transaction(pool,candidate,lambda snapshot,role:self.evaluate(snapshot,role,case),rule,**options)
            if result.get('reason') in ('incomplete_validation','incomplete_episode_status','validation_error'):
                raise IncompleteOptimization('Required paired validation remains incomplete: '+result['reason'])
            result['support']=support
            result['support_threshold_sensitivity']=[{'minimum_decisions':n,
                'candidate_eligible':support['valid_behavior_decisions']>=n,
                'would_accept_this_candidate':result['accepted'] and support['valid_behavior_decisions']>=n}
                for n in rule.get('support_threshold_grid',[0,4,8,16,32])]
            result['support_threshold_identified']=False
            result['support_note']='One candidate cannot identify a rare-skill threshold; deferred parents remain unchanged.'
            atomic_json(case/'decision.json',result)
            return result

    def _reuse_identity(self,snapshot,seed):
        cfg=deepcopy(self.config)
        # Placement and deadlines affect scheduling, never policy/environment identity.
        for key in ('output','journal_path','rollout_validation','experiment_id','execution_stage'):cfg.pop(key,None)
        for key in ('endpoint','scoring_endpoints','generation_endpoints','sg_workers','score_batch_size','timeout_seconds','journal_path','request_context','deadline_unix'):
            cfg['model'].pop(key,None)
        cfg.pop('pool',None)
        cfg.update(policy_interface(self.config))
        return digest({'pool':snapshot,'seed':seed,'model':self.identity,'config':cfg,'source':self.source})

    def evaluate(self,snapshot,role,case,seeds=None):
        requested=self.rule['seeds'] if seeds is None else list(seeds)
        if not requested or len(set(requested))!=len(requested) or not set(requested)<=set(self.rule['seeds']):
            raise ValueError('Validation tasks must belong to the registered seed set')
        key=pool_digest(snapshot);pool_file=case/f'{role}-pool.json'
        interface=policy_interface(self.config)
        payload={'stage':'train','pool':snapshot,'episodes_completed':self.completed_training_episodes,
            'total_training_episodes':self.inherited_episodes+self.completed_training_episodes,
            'training_seeds':sorted(set(self.inherited_seeds)|set(split_seeds(self.config,'train')[:self.completed_training_episodes])),
            'model_identity':self.identity,'validation_only':True,
            'communication_encoder':interface['communication_encoder'],'policy_interface':interface}
        provenance=self.config.get('verification_pool_provenance',{}).get(key)
        if self.config.get('verification_pool_provenance') and provenance is None:
            raise ValueError('Exact validation snapshot lacks registered training provenance')
        if provenance is not None:
            for field in ('episodes_completed','total_training_episodes','training_seeds'):payload[field]=deepcopy(provenance[field])
        if pool_file.exists() and json.loads(pool_file.read_text())!=payload:raise ValueError('Registered validation Pool changed')
        if not pool_file.exists():atomic_json(pool_file,payload)
        pending=queue.Queue()
        for seed in requested:pending.put(seed)
        results={};result_lock=threading.Lock();stop=threading.Event()
        def work(port):
            while not stop.is_set():
                try:seed=pending.get_nowait()
                except queue.Empty:return
                process=None
                try:
                    if time.time()>=self.deadline:raise BudgetExceeded('Paired rollout deadline reached')
                    self.journal.check_budget()
                    task=self._reuse_identity(snapshot,seed)
                    dest=self.directory/'episodes'/task;dest.mkdir(parents=True,exist_ok=True)
                    with (dest/'.case-lock').open('w') as lease:
                        try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:raise IncompleteOptimization('Validation seed is already running: '+task)
                        summary_path=dest/'summary.json'
                        summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
                        if summary.get('status')!='completed':
                            cfg={k:deepcopy(v) for k,v in self.config.items() if k!='execution_stage'}
                            cfg.update(pool=str(pool_file),eval_seed=seed,eval_seeds=[seed],episodes=1,
                                evaluation_disabled=False,evaluation_role='rollout',journal_path=str(self.journal.path))
                            cfg['model'].update(endpoint=f'http://127.0.0.1:{port}',scoring_endpoints=[],generation_endpoints=[])
                            cfg['diagnostics']={'reward_events':True}
                            cfg=execution_config(cfg,'eval')
                            config_path=dest/'case-config.json'
                            # A resumed run retains its original config/placement. The task ID is independent of the endpoint.
                            if config_path.exists():cfg=json.loads(config_path.read_text())
                            else:atomic_json(config_path,cfg)
                            cmd=[sys.executable,'-m','research','evaluate','--config',str(config_path),'--output',str(dest)]
                            endpoint=f'http://127.0.0.1:{port}'
                            cmd+=['--endpoint',endpoint,'--scoring-endpoints',endpoint,'--generation-endpoints',endpoint,'--sg-workers','1']
                            if (dest/'manifest.json').exists():cmd.append('--resume')
                            context={'seed':seed,'pool_hash':key,'role':role,'stage':'rollout','case':str(case)}
                            self.journal.start(task,'rollout_seed_task',context,{'config':cfg,'source':self.source,'model':self.identity})
                            env={**os.environ,'PYTHONHASHSEED':'0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
                            with (dest/'worker.log').open('a') as log:
                                process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                                proc=Path('/proc')/str(process.pid)
                                # Persist child session identity so a crashed parent campaign
                                # can clean up these separately isolated rollout workers too.
                                entry={'pid':process.pid,'uid':os.getuid(),'task':task,
                                    'owner_campaign':self.config.get('experiment_id'),
                                    'argv':cmd}
                                try:
                                    entry.update(start_ticks=(proc/'stat').read_text().split(') ',1)[1].split()[19],
                                        command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip(),pgid=os.getpgid(process.pid))
                                except OSError:
                                    if process.poll() is None:raise
                                atomic_json(dest/'worker.json',entry)
                                while process.poll() is None:
                                    if stop.is_set():raise IncompleteOptimization('Another validation worker interrupted')
                                    if time.time()>=self.deadline:raise BudgetExceeded('Paired rollout deadline reached')
                                    self.journal.check_budget();time.sleep(.25)
                            if process.returncode:raise IncompleteOptimization(f'Validation worker failed: {dest}')
                            summary=json.loads(summary_path.read_text())
                        if summary.get('status')!='completed':raise IncompleteOptimization('Incomplete CAGE4 validation')
                        if summary['model_identity']!=self.identity:raise ValueError('Validation model identity changed')
                        if not summary['pool_frozen']:raise ValueError('Validation altered the pool')
                        frozen=json.loads((dest/'frozen_pool.json').read_text())
                        check_pool_interface(frozen,self.config)
                        if pool_digest(frozen['pool'])!=key:raise ValueError('Reused validation Pool differs')
                        episode={**summary['episodes'][0],'summary':str(summary_path),'validated_pool_sha256':key}
                        check_pool_interface(episode,self.config)
                        if episode['seed']!=seed:raise ValueError('Validation seed differs')
                        record=self.journal.task(task)
                        if record and record['status']!='completed':self.journal.finish(task,episode,{'reused':False})
                        elif record:self.journal.event(task,'rollout_seed_task','completed',record['context'],{'cache_hit':True})
                        with result_lock:results[seed]=episode
                except BaseException as exc:
                    stop.set()
                    if 'task' in locals():
                        record=self.journal.task(task)
                        if record and record['status']!='completed':self.journal.mark(task,'interrupted' if isinstance(exc,BudgetExceeded) else 'unknown',{'error':str(exc)})
                    raise
                finally:
                    if process is not None and process.poll() is None:
                        os.killpg(process.pid,signal.SIGTERM)
                        try:process.wait(timeout=8)
                        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                    pending.task_done()
        with ThreadPoolExecutor(max_workers=len(self.rule['ports'])) as executor:
            futures=[executor.submit(work,p) for p in self.rule['ports']]
            try:
                for future in futures:future.result()
            except BaseException:
                stop.set()
                raise
        if set(results)!=set(requested):raise IncompleteOptimization('All requested validation seeds must finish')
        return [results[seed] for seed in requested]
