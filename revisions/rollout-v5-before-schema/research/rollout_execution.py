"""Run isolated, real paired CAGE4 validations using the pinned local model."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from . import ROOT
from .rollout import pool_digest,verify_transaction


class EnvironmentVerifier:
    def __init__(self,config,output,identity):
        self.config=deepcopy(config);self.directory=Path(output)/'rollout-verification'
        self.directory.mkdir(exist_ok=True);self.identity=identity;self.count=0
        self.rule=deepcopy(config['rollout_validation'])
        used=set(range(config['train_seed'],config['train_seed']+config['train_episodes']))|set(range(config['eval_seed'],config['eval_seed']+config['episodes']))
        if used&set(self.rule['seeds']):raise ValueError('Rollout validation seeds overlap training/final evaluation')
        self.deadline=self.rule.get('deadline_unix',float('inf'))

    def __call__(self,pool,candidate,support):
        self.count+=1;case=self.directory/f"iteration-{support['iteration']:04d}-{self.count:03d}-{candidate.name}"
        case.mkdir(exist_ok=False)
        rule={**self.rule,'parent':support['parent'],'iteration':support['iteration']}
        def evaluate(snapshot,role):return self.evaluate(snapshot,role,case)
        result=verify_transaction(pool,candidate,evaluate,rule)
        result['support']=support
        # A declared validation grid exposes whether support thresholds are identified.
        # No arbitrary rare-skill cutoff is silently inserted into Algorithm 1.
        result['support_threshold_sensitivity']=[{'minimum_decisions':n,
            'candidate_eligible':support['valid_behavior_decisions']>=n,
            'would_accept_this_candidate':result['accepted'] and support['valid_behavior_decisions']>=n}
            for n in rule.get('support_threshold_grid',[0,4,8,16,32])]
        result['support_threshold_identified']=False
        result['support_note']='This one-candidate comparison cannot identify the best rare-skill threshold. Other parents remain deferred under the registered rollout budget.'
        (case/'decision.json').write_text(json.dumps(result,indent=2)+'\n')
        return result

    def evaluate(self,snapshot,role,case):
        key=pool_digest(snapshot);pool_file=case/f'{role}-pool.json'
        pool_file.write_text(json.dumps({'stage':'train','pool':snapshot,
            'training_seeds':list(range(self.config['train_seed'],self.config['train_seed']+self.config['train_episodes'])),
            'model_identity':self.identity,'validation_only':True},indent=2)+'\n')
        ports=self.rule['ports'];assignments=[[] for _ in ports]
        results=[]
        for index,seed in enumerate(self.rule['seeds']):assignments[index%len(ports)].append(seed)
        def work(port,seeds):
            found=[]
            for seed in seeds:
                if time.time()>=self.deadline:break
                dest=case/f'{role}-seed-{seed}';dest.mkdir(exist_ok=False)
                cfg=deepcopy(self.config);cfg.update(pool=str(pool_file),eval_seed=seed,episodes=1,validation_episodes=0)
                cfg.pop('rollout_validation',None)
                cfg['evolution']['rollout_verification']=False
                cfg['model'].update(endpoint=f'http://127.0.0.1:{port}',scoring_endpoints=[])
                cfg['diagnostics']={'reward_events':True}
                # The parent split remains reserved, but this child's one episode is
                # itself validation, not a final test. Keep it out of optimizers.
                config_path=dest/'case-config.json';config_path.write_text(json.dumps(cfg,indent=2)+'\n')
                cmd=[sys.executable,'-m','research','evaluate','--config',str(config_path),'--output',str(dest)]
                env={**os.environ,'PYTHONHASHSEED':'0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
                with (dest/'worker.log').open('w') as stream:
                    process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                    while process.poll() is None:
                        if time.time()>=self.deadline:
                            process.terminate()
                            try:process.wait(timeout=10)
                            except subprocess.TimeoutExpired:process.kill();process.wait()
                            (dest/'budget_stop.json').write_text(json.dumps({'status':'not_completed_budget_limit','not_a_performance_sample':True})+'\n')
                            break
                        time.sleep(.5)
                if process.returncode!=0:
                    if time.time()>=self.deadline:break
                    raise RuntimeError(f'Validation worker failed: {dest}')
                summary=json.loads((dest/'summary.json').read_text())
                if summary['status']!='completed':raise RuntimeError('Incomplete CAGE4 validation')
                if summary['model_identity']!=self.identity:raise RuntimeError('Validation model identity changed')
                if not summary['pool_frozen']:raise RuntimeError('Validation altered the pool')
                episode=summary['episodes'][0]
                episode={**episode,'summary':str(dest/'summary.json'),'validated_pool_sha256':key}
                found.append(episode)
            return found
        with ThreadPoolExecutor(max_workers=len(ports)) as workers:
            for future in as_completed([workers.submit(work,p,s) for p,s in zip(ports,assignments)]):results.extend(future.result())
        return results
