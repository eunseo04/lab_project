"""Exercise validation serialization through the actual evaluation loader."""
from copy import deepcopy
import json
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
    assert verifier.evaluate(snapshot,'previous',case)==[]
    saved=json.loads((case/'previous-pool.json').read_text())
    assert saved['pool']==snapshot and saved['episodes_completed']==2
    assert saved['training_seeds']==[42000,42001]
    evaluation=deepcopy(config);evaluation.pop('rollout_validation')
    evaluation['pool']=str(case/'previous-pool.json')
    result=run(validate(evaluation),output=tmp_path/'evaluation',client=offline)
    assert result['status']=='completed' and result['pool_frozen']
    assert result['training_provenance']['episodes_completed']==2
    assert result['result_kind']=='mock_test'  # No fabricated model performance.
