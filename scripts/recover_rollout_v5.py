"""Metadata-only validation repair; preserve trained candidates and original clock."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
FORK=ROOT/'revisions/rollout-v5-metadata-fix'
OLD=ROOT/'revisions/rollout-v5'
OUT=ROOT/'research_outputs/rollout-v5'
sys.path.insert(0,str(FORK))
from research.client import ModelClient
from research.config import validate
from research.experiment import atomic_json
from research.rollout import pool_digest
from research.rollout_execution import EnvironmentVerifier
from research.skills import SecuritySkillPool
from research.upstream import Skill,configure_encoder

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def assert_source():
    protocol=read(OUT/'metadata-repair-protocol.json')
    for name,digest in protocol['source_sha256'].items():assert sha(FORK/name)==digest,name
    original=read(OUT/'protocol.json')
    for name,digest in original['source_sha256'].items():assert sha(OLD/name)==digest,name
    assert sha(ROOT/'scripts/rollout_v5_campaign.py')==original['driver_sha256']
    assert sha(__file__)==protocol['recovery_driver_sha256']

def retry(label):
    assert_source();source=OUT/label/'train'
    base=read(source/'pool-iteration-0001.json');summary=read(source/'summary.json')
    assert summary['status']=='completed' and base['episodes_completed']==2
    targets=[(parent,c) for parent in base['history'][0]['evolution']
        for c in parent['candidates'] if c.get('ppo_accepted')]
    assert len(targets)==1,targets
    parent,item=targets[0];failure=item['rollout_validation']
    assert failure['reason']=='validation_error' and failure['live_pool_unchanged_during_validation']
    failed_dir=next((source/'rollout-verification').glob('iteration-0001-001-*'))
    assert "KeyError: 'episodes_completed'" in (failed_dir/'previous-seed-62000/worker.log').read_text()
    previous=read(failed_dir/'previous-pool.json')['pool']
    assert pool_digest(previous)==failure['previous_pool_sha256']==pool_digest(base['pool'])
    cfg=read(OUT/f'{label}-config.json');cfg['model']['scoring_endpoints']=[]
    client=ModelClient(cfg['model']);configure_encoder(client.embed)
    assert client.identity==read(OUT/'protocol.json')['fixed_model_identity']
    pool=SecuritySkillPool(previous['max_size'],previous.get('seed_set','legacy'));pool.restore(previous)
    fields=set(Skill.__dataclass_fields__);row=item['skill']
    candidate=Skill(**{k:v for k,v in row.items() if k in fields})
    candidate.family=row['family'];candidate.allowed_actions=row['allowed_actions'];candidate.maturity=0
    proposed=deepcopy(pool);proposed.add_skill(deepcopy(candidate))
    next(s for s in proposed.skills if s.name==parent['parent']).last_evolved_iter=1
    from research.paper_algorithm import maintain_paper
    maintain_paper(proposed,capacity_only=True)
    assert pool_digest(proposed.snapshot())==failure['candidate_pool_sha256']
    dest=OUT/'recovery'/label/'train';dest.mkdir(parents=True,exist_ok=False)
    atomic_json(dest/'provenance.json',{'original_snapshot':str(source/'pool-iteration-0001.json'),
        'original_snapshot_sha256':sha(source/'pool-iteration-0001.json'),
        'original_candidate_pool_sha256':failure['candidate_pool_sha256'],
        'original_previous_pool_sha256':failure['previous_pool_sha256'],
        'candidate_regenerated':False,'same_seeds_criteria_budget':True,'started_unix':time.time()})
    verifier=EnvironmentVerifier(cfg,dest,client.identity)
    result=verifier(pool,candidate,failure['support'])
    if result['accepted']:
        pool.restore(result['verified_pool'])
        base.get('experience',{}).get('buffers',{})[parent['parent']]=[]
    item['validation_retry_previous_error']=deepcopy(failure)
    item['rollout_validation']=result;item['accepted']=result['accepted']
    parent['accepted']=any(c['accepted'] for c in parent['candidates'])
    base['pool']=pool.snapshot();base['history'][0]['pool_snapshot']=pool.snapshot()
    base['recovery_provenance']=read(dest/'provenance.json')
    base['history'][0]['metadata_repair']=True
    for name in ('pool-iteration-0001.json','checkpoint.json'):atomic_json(dest/name,base)
    summary['history']=base['history'];summary['pool_sha256']=pool_digest(base['pool'])
    summary['metadata_repair_provenance']=base['recovery_provenance']
    atomic_json(dest/'summary.json',summary)
    atomic_json(dest/'completed.json',{'completed_unix':time.time(),'accepted':result['accepted'],
        'reason':result['reason'],'new_training_episodes':0,'source_root':str(FORK)})

def register():
    assert not (OUT/'queue.json').exists(), 'Final tests already began'
    original=read(OUT/'protocol.json')
    for name,digest in original['source_sha256'].items():assert sha(OLD/name)==digest
    changed=[n for n in original['source_sha256'] if sha(FORK/n)!=sha(OLD/n)]
    assert changed==['research/rollout_execution.py'],changed
    suite=ET.parse(FORK/'research_outputs/metadata-final-tests.xml').getroot().find('testsuite')
    assert int(suite.attrib['tests'])==138 and int(suite.attrib['failures'])==int(suite.attrib['errors'])==0
    atomic_json(OUT/'metadata-repair-protocol.json',{'registered_unix':time.time(),
        'source_root':str(FORK),'source_sha256':{n:sha(FORK/n) for n in original['source_sha256']},
        'original_protocol_sha256':sha(OUT/'protocol.json'),'recovery_driver_sha256':sha(__file__),
        'campaign_driver_sha256':sha(ROOT/'scripts/rollout_v5_repaired_campaign.py'),
        'change':'Add episodes_completed and total_training_episodes to validation snapshot metadata only.',
        'unchanged':'All actor, observation, model, SG, PPO, pool, reward, seeds, admission criteria and original deadlines.',
        'failed_validation_not_performance':True,'reuse_existing_candidate_and_training':True,
        'selection_rule':original['selection_rule'],'budget':original['budget'],
        'tests':138,'integration_before':'KeyError episodes_completed reproduced; after actual normal evaluation loader passes',
        'validation_priority':'Retry B4 first. B3 keeps running its original training; retry its unchanged candidate only after B4 completes and before the original validation deadline.'})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--register',action='store_true');parser.add_argument('--retry',choices=['B3','B4'])
    args=parser.parse_args()
    if args.register:register()
    elif args.retry:retry(args.retry)
    else:parser.error('Choose --register or --retry')
