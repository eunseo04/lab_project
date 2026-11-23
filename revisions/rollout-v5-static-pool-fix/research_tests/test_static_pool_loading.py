"""A static control must load the same registered initial skills as trained arms."""
import json
from research.config import validate
from research.experiment import run
from research.skills import SecuritySkillPool
from research.upstream import Skill
from test_research import offline


def test_static_control_loads_explicit_common_pool_not_default_six_seeds(tmp_path,offline):
    pool=SecuritySkillPool(24,'response_v1')
    added=Skill('RiskBasedInvestigation','When uncertain observed activity needs review',['Review actual outcomes'],'When reviewed',version=1)
    added.family=added.name;added.allowed_actions=['Analyse','Sleep'];pool.add_skill(added)
    assert len(pool.skills)==7
    source=tmp_path/'initial.json';source.write_text(json.dumps({'stage':'train','pool':pool.snapshot(),
        'episodes_completed':0,'total_training_episodes':0,'training_seeds':[],'model_identity':offline.identity}))
    config=validate({'arm':'E1','composition':'all_llm','steps':6,'episodes':1,'iterations':1,
        'skill_seed_set':'response_v1','behavior_context':'evidence_v1','pool':str(source)})
    result=run(config,output=tmp_path/'eval',client=offline)
    frozen=json.loads((tmp_path/'eval/frozen_pool.json').read_text())
    assert frozen['pool']==pool.snapshot()
    assert result['status']=='completed' and result['pool_frozen']
    assert result['training_provenance']['path']==str(source.resolve())
    assert result['result_kind']=='mock_test'
