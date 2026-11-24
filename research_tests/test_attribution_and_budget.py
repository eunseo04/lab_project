from copy import deepcopy

from research.attribution import auxiliary_attribution, verify_counterfactual_snapshot
from research.evolution import Evolution
from research.skills import SecuritySkillPool
from research.upstream import Skill
from test_research import offline, experience


def test_auxiliary_attribution_preserves_inputs_and_shared_reward():
    rows=[{'step':2,'seed':17,'episode_id':0,'policy':'llm','agent':'blue_agent_4',
        'policy_invoked':True,'request_applied':True,'requested_action':'Remove h1',
        'action_status':'TRUE','joint_reward':-3,'mission_phase':0,
        'observation_before':{'h1':{'Processes':[{'PID':41}]}},'observation_after':{'h1':{}}},
        {'step':2,'policy':'rl','agent':'blue_agent_0','requested_action':'Restore h2'}]
    ticks=[{'step':2,'events':[{'kind':'LWF','host':'h2','zone':'other_zone','reward':-3}]}]
    before=deepcopy(rows);result=auxiliary_attribution(rows,ticks)
    assert rows==before and result['reward_unchanged'] and not result['causal_estimate']
    own=result['records'][0]
    assert not own['same_host_official_events'] and own['all_step_green_red_reward_events']
    assert own['joint_reward']==-3 and own['causal_credit']=='unidentified' and not own['policy_input']


def test_counterfactual_eligibility_rejects_same_state_with_different_rng():
    check=verify_counterfactual_snapshot(b'full-state',b'full-state',b'rng-a',b'rng-b')
    assert check['state_equal'] and not check['eligible'] and not check['causal_effect_estimated']


def test_single_rollout_budget_rotates_between_supported_skill_families(offline,monkeypatch):
    pool=SecuritySkillPool(24)
    parents=pool.skills[:2];calls=[]
    exps=[]
    for parent in parents:
        exp=experience();exp.skill=parent.name
        for row in exp.transitions:row.update(seed=17,episode_id=0)
        exps.append(exp)
    ev=Evolution(offline,{'schedule':'paper_batch','gate_mode':'paper_rtg',
        'maintenance_mode':'paper_algorithm1','best_of_n':1,'rollout_verification':True,
        'maximum_rollout_candidates_per_iteration':1},
        rollout_validator=lambda pool,candidate,support:(calls.append(support['parent']) or {'accepted':False,'reason':'fixture_negative_rollout'}))
    monkeypatch.setattr(ev,'evolve',lambda parent,exps:(Skill(parent.name+'_candidate','i',['p'],'t'),'REFINE'))
    monkeypatch.setattr(ev,'probabilities',lambda parent,exps:[0.]*sum(len(exp.transitions) for exp in exps))
    first=ev.update(pool,exps,{'test':0},1)
    second=ev.update(pool,exps,{'test':0},2)
    assert calls==[parents[0].name,parents[1].name]
    assert any(row.get('deferred') for row in first['evolution'])
    assert any(row.get('deferred') for row in second['evolution'])
