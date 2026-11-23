"""Validation provenance must reflect completed episodes and inherited experience."""
import json
from research.config import validate
from research.rollout_execution import EnvironmentVerifier
from research.skills import SecuritySkillPool
from research.upstream import Skill
from test_research import offline


def snapshot_at_first_boundary(tmp_path,offline,monkeypatch,warm=None):
    config=validate({'arm':'E2','composition':'all_llm','steps':6,
        'episodes':1,'eval_seed':44000,'train_seed':42000,'train_episodes':5,
        'iterations':2,'validation_episodes':0,'pool':str(warm) if warm else None})
    config['rollout_validation']={'seeds':[43000],'ports':[8018],'deadline_unix':0}
    pool=SecuritySkillPool()
    def incomplete(pool,candidate,evaluate,rule):
        assert evaluate(pool.snapshot(),'previous')==[]
        return {'accepted':False,'reason':'incomplete_validation'}
    monkeypatch.setattr('research.rollout_execution.verify_transaction',incomplete)
    verifier=EnvironmentVerifier(config,tmp_path,offline.identity)
    prior_iterations=len(json.loads(warm.read_text()).get('history',[])) if warm else 0
    verifier(pool,Skill('candidate','i',['p'],'t'),{'parent':pool.skills[0].name,
        'iteration':prior_iterations+1,'valid_behavior_decisions':5,'evidence_trajectories':2})
    return json.loads(next(tmp_path.glob('rollout-verification/*/previous-pool.json')).read_text())


def test_uneven_two_loop_boundary_does_not_claim_future_training_episodes(tmp_path,offline,monkeypatch):
    saved=snapshot_at_first_boundary(tmp_path,offline,monkeypatch)
    assert saved['episodes_completed']==3  # Actual ceil(5/2) boundary; two not run yet.
    assert saved['total_training_episodes']==3
    assert saved['training_seeds']==[42000,42001,42002]


def test_warm_start_validation_keeps_inherited_training_provenance(tmp_path,offline,monkeypatch):
    warm=tmp_path/'warm.json'
    warm.write_text(json.dumps({'episodes_completed':4,'total_training_episodes':7,'history':[{'iteration':1},{'iteration':2}],
        'training_seeds':list(range(41000,41007))}))
    saved=snapshot_at_first_boundary(tmp_path,offline,monkeypatch,warm)
    assert saved['episodes_completed']==3 and saved['total_training_episodes']==10
    assert saved['training_seeds']==list(range(41000,41007))+[42000,42001,42002]
