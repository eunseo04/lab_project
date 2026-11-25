"""Exercise validation serialization through the actual evaluation loader."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pytest
from research.progress import BudgetExceeded
from research.config import (check_pool_interface, digest, execution_config, policy_interface,
                             read_training_manifest, variant_config)
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
            ep={**episodes()[0],'seed':seed,'communication_encoder':cfg['communication_encoder'],
                'policy_interface':policy_interface(cfg)}
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


def test_legacy_pool_checks_original_manifest_before_warm_start_or_evaluation(offline,tmp_path,monkeypatch):
    config=variant_config('B2')
    config.update(communication_encoder='legacy_local_v1',action_contract='explicit',
                  steps=6,episodes=1,iterations=1,train_episodes=2,composition='all_llm')
    source=tmp_path/'source'
    source.mkdir()
    pool_path=source/'checkpoint.json'
    manifest_path=source/'manifest.json'
    snapshot=json.loads(Path(config['pool']).read_text())['pool']
    checkpoint={'stage':'train','pool':snapshot,'episodes_completed':2,'training_seeds':[1,2],
                'model_identity':offline.identity,'controller_contract':'upstream','config_digest':digest(config)}
    pool_path.write_text(json.dumps(checkpoint))
    manifest={'stage':'train','config':config,'config_digest':digest(config)}
    manifest_path.write_text(json.dumps(manifest))
    ref={'path':str(manifest_path),'sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    assert read_training_manifest(pool_path,checkpoint,manifest_ref=ref)==manifest
    campaign=tmp_path/'historical-campaign'
    frozen=campaign/'frozen'/'B2.json'
    frozen.parent.mkdir(parents=True)
    original=campaign/'runs'/'B2-finite-train'
    original.mkdir(parents=True)
    frozen.write_bytes(pool_path.read_bytes())
    (original/'checkpoint.json').write_bytes(pool_path.read_bytes())
    (original/'manifest.json').write_bytes(manifest_path.read_bytes())
    assert read_training_manifest(frozen,checkpoint)==manifest
    (original/'checkpoint.json').write_text('{}')
    assert read_training_manifest(frozen,checkpoint) is None
    with pytest.raises(ValueError,match='Missing Skill Pool policy interface'):
        check_pool_interface(checkpoint,config)
    with pytest.raises(ValueError,match='controller contract'):
        check_pool_interface({**checkpoint,'controller_contract':'cage4_role_v1',
                              'policy_interface':policy_interface(config)},config)
    with pytest.raises(ValueError,match='controller contract'):
        check_pool_interface({**checkpoint,'controller_contract':'cage4_role_v1'},config,manifest=manifest)
    other_contract={**config,'action_contract':'explicit_zone_pair_v1'}
    with pytest.raises(ValueError,match='disagrees with its original manifest'):
        check_pool_interface({**checkpoint,'policy_interface':policy_interface(other_contract)},
                             other_contract,manifest=manifest)
    original_restore=SecuritySkillPool.restore
    restores=[]
    def observe_restore(self,value):
        restores.append(value)
        return original_restore(self,value)
    monkeypatch.setattr(SecuritySkillPool,'restore',observe_restore)
    for stage in ('train','eval'):
        exact={**config,'pool':str(pool_path)}
        different={**exact,'action_contract':'explicit_zone_pair_v1'}
        before=len(restores)
        with pytest.raises(ValueError,match='policy interface differs'):
            run(different,stage,output=tmp_path/(stage+'-different'),client=offline,limit=0)
        assert len(restores)==before
        measured=run(exact,stage,output=tmp_path/stage,client=offline,limit=0)
        assert measured['status']=='partial' and len(restores)==before+1
        mismatched={**checkpoint,'model_identity':{**offline.identity,'dtype':'float16'}}
        pool_path.write_text(json.dumps(mismatched))
        with pytest.raises(ValueError,match='policy model identity'):
            run(exact,stage,output=tmp_path/(stage+'-different-model'),client=offline,limit=0)
        assert len(restores)==before+1
        pool_path.write_text(json.dumps(checkpoint))
    manifest_path.write_text(json.dumps({**manifest,'extra_fixture_field':True}))
    with pytest.raises(ValueError,match='file hash'):
        read_training_manifest(pool_path,checkpoint,manifest_ref=ref)
    manifest_path.write_text(json.dumps({**manifest,'config':{**config,'action_contract':'explicit_zone_pair_v1'}}))
    with pytest.raises(ValueError,match='configuration digest'):
        read_training_manifest(pool_path,checkpoint)


def test_resume_checks_explicit_checkpoint_against_its_run_manifest(offline,tmp_path,monkeypatch):
    config=variant_config('B2')
    config.update(steps=6,iterations=1,train_episodes=2,composition='all_llm')
    run(config,'train',output=tmp_path,client=offline,limit=0)
    manifest=json.loads((tmp_path/'manifest.json').read_text())
    interface=policy_interface(manifest['config'])
    checkpoint={'stage':'train','pool':{'fixture_only':True},'config_digest':manifest['config_digest'],
                'training_run_id':manifest['training_run_id'],'model_identity':offline.identity,
                'policy_interface':interface}
    def forbid_restore(*_):
        pytest.fail('Mismatched checkpoint was restored')
    monkeypatch.setattr(SecuritySkillPool,'restore',forbid_restore)
    for overrides,reason in [
        ({'policy_interface':{**interface,'action_contract':'explicit'}},'disagrees with its original manifest'),
        ({'controller_contract':'cage4_role_v1'},'controller contract'),
        ({'training_run_id':'different-run'},'does not belong to the original training manifest'),
    ]:
        (tmp_path/'checkpoint.json').write_text(json.dumps({**checkpoint,**overrides}))
        with pytest.raises(ValueError,match=reason):
            run(config,'train',output=tmp_path,resume=True,client=offline,limit=0)
