"""Exercise validation serialization through the actual evaluation loader."""
from copy import deepcopy
import json
import pytest
from research.progress import BudgetExceeded
from research.config import validate
from research.experiment import run
from research.rollout_execution import EnvironmentVerifier
from research.skills import SecuritySkillPool
from test_research import offline


def test_validation_snapshot_loads_in_normal_evaluation(offline,tmp_path):
    config=validate({'arm':'E2','composition':'all_llm','steps':6,
        'episodes':1,'eval_seed':44000,'train_seed':42000,'train_episodes':2,'iterations':1,
        'validation_episodes':0})
    # An expired deadline skips worker/model calls but exercises the real writer.
    config['rollout_validation']={'seeds':[43000],'ports':[8018],'deadline_unix':0}
    verifier=EnvironmentVerifier(config,tmp_path,offline.identity)
    case=tmp_path/'case';case.mkdir()
    snapshot=SecuritySkillPool().snapshot()
    with pytest.raises(BudgetExceeded):verifier.evaluate(snapshot,'previous',case)
    saved=json.loads((case/'previous-pool.json').read_text())
    assert saved['pool']==snapshot and saved['episodes_completed']==2
    assert saved['training_seeds']==[42000,42001]
    evaluation=deepcopy(config);evaluation.pop('rollout_validation')
    evaluation['pool']=str(case/'previous-pool.json')
    result=run(validate(evaluation),output=tmp_path/'evaluation',client=offline)
    assert result['status']=='completed' and result['pool_frozen']
    assert result['training_provenance']['episodes_completed']==2
    assert result['result_kind']=='mock_test'  # No fabricated model performance.


def test_partial_rollout_resume_reuses_completed_seeds_and_waits_for_all(offline,tmp_path,monkeypatch):
    from research.progress import IncompleteOptimization
    from test_rollout_verification import episodes
    config=validate({'arm':'E2','composition':'all_llm','steps':500,'episodes':1,
        'eval_seed':44000,'train_seed':42000,'train_episodes':2,'iterations':1,'validation_episodes':0})
    seeds=list(range(43000,43020))
    config['rollout_validation']={'seeds':seeds,'ports':[8018]}
    verifier=EnvironmentVerifier(config,tmp_path,offline.identity)
    case=tmp_path/'case';case.mkdir();snapshot=SecuritySkillPool().snapshot()
    calls=[];fault={'armed':True}
    class Worker:
        def __init__(self,cmd,**kwargs):
            self.pid=99999999;self.returncode=0
            cfg=json.loads(__import__('pathlib').Path(cmd[cmd.index('--config')+1]).read_text())
            dest=__import__('pathlib').Path(cmd[cmd.index('--output')+1]);seed=cfg['eval_seed']
            if seed==seeds[1] and fault['armed']:
                fault['armed']=False;raise BudgetExceeded('Injected after one completed rollout')
            calls.append(seed)
            frozen=json.loads(__import__('pathlib').Path(cfg['pool']).read_text())
            (dest/'frozen_pool.json').write_text(json.dumps(frozen))
            ep={**episodes()[0],'seed':seed}
            (dest/'summary.json').write_text(json.dumps({'status':'completed','model_identity':offline.identity,'pool_frozen':True,'episodes':[ep]}))
        def poll(self):return 0
    monkeypatch.setattr('research.rollout_execution.subprocess.Popen',Worker)
    with pytest.raises(BudgetExceeded):verifier.evaluate(snapshot,'previous',case)
    assert calls==[seeds[0]] and not (case/'decision.json').exists()
    actual=EnvironmentVerifier(config,tmp_path,offline.identity).evaluate(snapshot,'previous',case)
    assert [e['seed'] for e in actual]==seeds and calls==seeds
    moved=deepcopy(config);moved['rollout_validation']['ports']=[8019,8020];moved['model'].update(endpoint='http://new',sg_workers=2)
    again=EnvironmentVerifier(moved,tmp_path,offline.identity).evaluate(snapshot,'previous',case)
    assert again==actual and calls==seeds
