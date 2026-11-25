"""Exercise validation serialization through the actual evaluation loader."""
from copy import deepcopy
import json
import pytest
from research.progress import BudgetExceeded
from research.config import execution_config, variant_config
from research.experiment import run
from research.rollout_execution import EnvironmentVerifier
from research.skills import SecuritySkillPool
from test_research import offline


def validation_config(steps=500):
    config=variant_config('B4')
    config.update(composition='all_llm',steps=steps,episodes=1,eval_seed=44000,
                  train_seed=42000,train_episodes=2,iterations=1,validation_episodes=0)
    config['rollout_validation'].update(seeds=list(range(43000,43020)),ports=[8018])
    return execution_config(config,'train')


def test_b4_snapshot_loads_in_registered_evaluation_without_optimizer(offline,tmp_path,monkeypatch):
    config=validation_config(steps=6)
    # An expired deadline skips worker/model calls but exercises the real writer.
    config['rollout_validation']['deadline_unix']=0
    verifier=EnvironmentVerifier(config,tmp_path,offline.identity)
    case=tmp_path/'case';case.mkdir()
    snapshot=SecuritySkillPool(24,'response_v1').snapshot()
    with pytest.raises(BudgetExceeded):verifier.evaluate(snapshot,'previous',case)
    saved=json.loads((case/'previous-pool.json').read_text())
    assert saved['pool']==snapshot and saved['episodes_completed']==2
    assert saved['training_seeds']==[42000,42001]
    assert saved['communication_encoder']=='acd_original_v1'
    evaluation=execution_config({**config,'execution_stage':'eval','evaluation_role':'rollout',
        'pool':str(case/'previous-pool.json'),'eval_seed':43000},'eval')
    with pytest.raises(ValueError,match='Execution stage'):
        run(evaluation,stage='train',output=tmp_path/'wrong-stage',client=offline)
    with pytest.raises(ValueError,match='registered rollout validation set'):
        execution_config({**evaluation,'eval_seed':45000},'eval')
    with pytest.raises(ValueError,match='overlap'):
        execution_config({**evaluation,'evaluation_role':'frozen_test'},'eval')
    monkeypatch.setattr('research.experiment.Evolution',lambda *args:pytest.fail('optimizer instantiated during eval'))
    result=run(evaluation,output=tmp_path/'evaluation',client=offline)
    assert result['status']=='completed' and result['pool_frozen']
    assert result['config']['evolution']['rollout_verification']
    assert result['training_provenance']['episodes_completed']==2
    assert result['result_kind']=='mock_test'  # No fabricated model performance.


def test_partial_rollout_resume_reuses_completed_seeds_and_waits_for_all(offline,tmp_path,monkeypatch):
    from research.progress import IncompleteOptimization
    from test_rollout_verification import episodes
    config=validation_config()
    seeds=config['rollout_validation']['seeds']
    verifier=EnvironmentVerifier(config,tmp_path,offline.identity)
    case=tmp_path/'case';case.mkdir();snapshot=SecuritySkillPool(24,'response_v1').snapshot()
    calls=[];fault={'armed':True}
    class Worker:
        def __init__(self,cmd,**kwargs):
            self.pid=99999999;self.returncode=0
            cfg=json.loads(__import__('pathlib').Path(cmd[cmd.index('--config')+1]).read_text())
            dest=__import__('pathlib').Path(cmd[cmd.index('--output')+1]);seed=cfg['eval_seed']
            assert cfg['execution_stage']=='eval' and cfg['evaluation_role']=='rollout'
            assert cfg['evolution']['admission_mode']=='rollout' and cfg['evolution']['rollout_verification']
            assert cfg['rollout_validation']['seeds']==seeds
            if seed==seeds[1] and fault['armed']:
                fault['armed']=False;raise BudgetExceeded('Injected after one completed rollout')
            calls.append(seed)
            frozen=json.loads(__import__('pathlib').Path(cfg['pool']).read_text())
            (dest/'frozen_pool.json').write_text(json.dumps(frozen))
            ep={**episodes()[0],'seed':seed,'communication_encoder':cfg['communication_encoder']}
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
    different={**config,'communication_encoder':'legacy_local_v1'}
    assert EnvironmentVerifier(different,tmp_path,offline.identity)._reuse_identity(snapshot,seeds[0])!=verifier._reuse_identity(snapshot,seeds[0])


def test_original_communication_cannot_warm_start_or_evaluate_a_legacy_learned_pool(offline,tmp_path):
    config=variant_config('B3');config.update(steps=6,episodes=1,iterations=1,train_episodes=2,composition='all_llm')
    initial=json.loads(__import__('pathlib').Path(config['pool']).read_text())
    initial.update(initialization_only=False,episodes_completed=2,training_seeds=[1,2],model_identity=offline.identity)
    legacy=tmp_path/'legacy.json';legacy.write_text(json.dumps(initial));config['pool']=str(legacy)
    for stage in ('train','eval'):
        with pytest.raises(ValueError,match='communication encoder differs'):
            run(config,stage=stage,output=tmp_path/stage,client=offline)
