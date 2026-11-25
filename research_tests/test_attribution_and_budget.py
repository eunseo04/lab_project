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


def test_campaign_cost_preflight_counts_40_validation_rollouts_and_independent_caps():
    from scripts.audit_training_cost import preflight
    protocol={'statistics':{'pilot_n':4},'requested_training_arms':['B2','B4']}
    measured={'status':'measured','episode_upper_seconds':100,'prompt_input_tokens':{'maximum':100},
        'planned_sg_chunks':1,'planned_parents':1,'best_of_n':2,'planned_valid_decisions':2,'uncertainty':'fixture',
        'samples':[{'stage':'sg_chunk','seconds':1,'input_tokens':100,'output_tokens':768},
                   {'stage':'aggregation_candidate','seconds':1,'input_tokens':24576,'output_tokens':768},
                   {'stage':'old_score','seconds':1,'samples':1}]}
    usage={'gpu_hour_cap':8,'gpu_hours':1,'wall_second_cap':10000,'wall_seconds':0}
    result=preflight(protocol,measured,usage,2,'B4')
    assert result['parts_seconds']['paired_candidate_validation']==40*100/2
    assert result['b4_pairs']==20 and result['status']=='passed'
    # A wall-hour cap cannot authorize more GPU-hours.
    exhausted={**usage,'gpu_hours':7.99,'wall_second_cap':1000000}
    assert preflight(protocol,measured,exhausted,2,'B4')['status']=='deferred'


def test_campaign_cleanup_checks_process_identity_before_signaling(tmp_path):
    import subprocess,sys
    from scripts.run_campaign import process_identity,stop_owned
    import pytest
    worker=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
    try:
        entry=process_identity(worker.pid,wait_seconds=1)
        with pytest.raises(RuntimeError,match='ambiguous'):stop_owned({**entry,'start_ticks':'wrong'})
        assert worker.poll() is None
        assert stop_owned(entry)['status']=='stopped'
    finally:
        if worker.poll() is None:worker.terminate()
        worker.wait()


def test_resume_can_move_identical_policy_to_new_endpoints_without_changing_science():
    from research.config import validate,compatible_digest
    original=validate({'model':{'endpoint':'http://old','model':'Qwen/Qwen3-8B','revision':'fixture','temperature':1.,'max_tokens':192}})
    moved=deepcopy(original)
    moved['model'].update(endpoint='http://new',generation_endpoints=['http://new','http://second'],sg_workers=2)
    assert compatible_digest(original)==compatible_digest(moved)
    moved['model']['temperature']=.5
    assert compatible_digest(original)!=compatible_digest(moved)


def test_cost_forecast_separates_prefill_and_decode_instead_of_multiplying_both():
    from scripts.audit_training_cost import generation_cost_upper
    import pytest
    sample={'seconds':12.,'input_tokens':1000,'output_tokens':100,
        'timing':{'prefill_gpu_seconds':2.,'decode_gpu_seconds':9.}}
    # A doubled context: quadratic prefill, linear decode, one second overhead;
    # retain the registered 50% margin and full 768-token reservation.
    expected=(8+9/99*767*((2000+768)/(1000+100))+768/100)*1.5
    assert generation_cost_upper([sample],2000)==pytest.approx(expected)
    assert generation_cost_upper([{'seconds':2.,'input_tokens':100,'output_tokens':768}],200)==6.


def test_generation_forecast_uses_measured_long_context_instead_of_short_probe():
    from scripts.audit_training_cost import generation_cost_upper
    short={'seconds':2.,'input_tokens':100,'output_tokens':100,'timing':{'prefill_gpu_seconds':.5,'decode_gpu_seconds':1.}}
    long={'seconds':20.,'input_tokens':10000,'output_tokens':100,'timing':{'prefill_gpu_seconds':4.,'decode_gpu_seconds':15.}}
    assert generation_cost_upper([short,long],10000)==generation_cost_upper([long],10000)
    assert generation_cost_upper([short,long],10000)<generation_cost_upper([short],10000)


def test_selector_probe_uses_upstream_prompts_and_restores_live_rng_and_counter(offline):
    import random
    from research.skills import probe_selection
    from research.client import TextModel
    from research.progress import rng_snapshot
    pool=SecuritySkillPool(24);before=pool.snapshot();pool.add_skill(Skill('next','new initiation',['Remove'],'end'))
    rows=[{'seed':17,'agent':'blue_agent_4','step':2,'policy_invoked':True,
        'user_prompt':'actual selector state\n\n# EXECUTABLE ACTION CONTRACT\nactor only'}]
    model=TextModel(offline);model.counter=71;rng=rng_snapshot()
    result=probe_selection(before,pool.snapshot(),rows,offline,{'maximum_states':1,'epsilon':0.,'include_controls':True})
    assert rng_snapshot()==rng and model.counter==71 and result['paired_probe_n']==1
    for probes in result['probes'].values():
        prompt=probes[0]['model_calls'][0]['prompt']
        assert 'actual selector state' in prompt and 'actor only' not in prompt
        assert probes[0]['selector_source']=='llm'


def test_selector_probe_rejects_test_states_and_forbidden_seeds(offline):
    import pytest
    from research.skills import probe_selection
    p=SecuritySkillPool(24).snapshot();rows=[{'seed':17,'step':0,'policy_invoked':True,'user_prompt':'state'}]
    with pytest.raises(ValueError,match='Final-test'):probe_selection(p,p,rows,offline,data_role='frozen_test')
    with pytest.raises(ValueError,match='leaked'):probe_selection(p,p,rows,offline,{'forbidden_seeds':[17]})


def test_selection_shift_records_lost_skills_and_is_not_a_reward_rejection():
    from research.attribution import selection_shift
    old=[{'state_id':i,'sampling_seed':i,'choice':'RapidProcessResponse'} for i in range(4)]
    new=[{**r,'choice':'NetworkContainment_v2'} for r in old]
    r=selection_shift(old,new)
    assert r['disagreement_rate']==r['total_variation_distance']==1
    assert r['requires_rollout'] and not r['bad_policy_proven'] and not r['causal_reward_effect_estimated']
    assert r['previously_used_skills_absent']==['RapidProcessResponse']


def test_selection_distribution_separates_free_busy_starts_and_completions():
    from research.attribution import selection_distribution
    rows=[{'step':0,'agent':'a','policy':'llm','policy_invoked':True,'request_applied':True,
        'skill_id':'s','requested_action':'Remove h1','executed_actions':[{'name':'Remove'}],'action_status':'IN_PROGRESS'},
        {'step':1,'agent':'a','policy':'llm','policy_invoked':False,'requested_action':'Sleep',
         'executed_actions':[{'name':'Sleep'}],'action_status':'TRUE','behavior_outcome':{
         'completed_action':{'name':'Remove','target':'h1','started_at':0,'completed_at':2,'status':'TRUE'}}}]
    result=selection_distribution(rows,{'s':'family'})
    assert result['free_decisions']==1 and result['busy_automatic_wait_ticks']==1
    assert result['requested_free_actions']=={'Remove':1} and result['actual_free_action_starts']=={'Remove':1}
    assert result['observed_completed_calls']=={'Remove:TRUE':1} and result['family_decision_fractions']=={'family':1.}


def test_gpu_selection_excludes_foreign_compute_even_with_enough_memory(monkeypatch):
    from scripts.run_campaign import available_gpus
    from types import SimpleNamespace
    def run(command,**kwargs):
        return SimpleNamespace(stdout='0, GPU-a, 24000\n1, GPU-b, 24560\n' if 'index,uuid,memory.free' in command[1] else 'GPU-a, 99\n')
    monkeypatch.setattr('scripts.run_campaign.subprocess.run',run)
    eligible,all_rows=available_gpus(23000)
    assert [r['gpu'] for r in eligible]==[1] and all_rows[0]['compute_pids']==[99]


def test_live_selector_instrumentation_preserves_original_prompt_choice_and_rng(offline):
    import random
    from research.skills import SkillController
    pool=SecuritySkillPool(24);state={'user_prompt':'same observed state'}
    for seed in range(20):
        observed=[];direct=[]
        def audited_model(prompt,**kwargs):observed.append((prompt,kwargs));return '<choice>DeceptionDeployment</choice>'
        def original_model(prompt,**kwargs):direct.append((prompt,kwargs));return '<choice>DeceptionDeployment</choice>'
        controller=SkillController(pool,audited_model,{'epsilon':.1})
        random.seed(seed);actual=controller.choose(state,0);after=random.getstate()
        random.seed(seed);expected=pool.select_skill(state['user_prompt'],llm_agent=original_model,select_type='llm_model',epsilon=.1)
        assert (actual.name if actual else None)==(expected.name if expected else None)
        assert random.getstate()==after and observed==direct
        audit=controller.last_event['selection_audit']
        assert audit['source']==('llm' if direct else 'epsilon_exploration')


def test_scorer_profile_fills_missing_per_token_generation_cache_with_reference(tmp_path,monkeypatch):
    import json,sqlite3
    from contextlib import nullcontext
    from scripts.audit_training_cost import profile_fixed_update
    original=Skill('NetworkContainment','i',['Sleep'],'t')
    children=[Skill('candidate0','i',['Sleep'],'t'),Skill('candidate1','i',['Analyse'],'t')]
    exp=experience();exp.skill='NetworkContainment'
    source=tmp_path/'historical.sqlite3'
    with sqlite3.connect(source) as db:db.execute('CREATE TABLE tasks(id,kind,status,request,response)')
    class Client:
        identity={'model':'unit-test-fixture'};base_context={}
        def _fingerprint(self):return self.identity
        def _validate_scorer(self,identity):assert identity==self.identity
        def _request(self,route,body):
            assert route=='score'
            return {'identity':self.identity,'scores':[{'sum':-1.,'token_logprobs':[-1.],
                'tokens':1,'timing':{}} for row in body['samples']]}
    protocol={'fixed_update_source':{'journal':str(source)},'profile_full_sequence':True,
        'configs':{'B2':{'model':{'temperature':1.}},'B3':{}}}
    trained={'model_identity':Client.identity,'history':[{'evolution':[{'parent':'NetworkContainment',
        'old_skill':original.__dict__,'candidates':[{'skill':s.__dict__,'score':1.5} for s in children]}]}]}
    monkeypatch.setattr('scripts.audit_training_cost.render_sg',lambda *args:([],{}))
    result=profile_fixed_update(tmp_path,protocol,Client(),trained,{'experiences':[exp.__dict__]},
        {'ready':{'baselines':{'test':0},'pool':{}}},lambda:1000.)
    assert result['historical_exact_reference_samples']==0 and result['fresh_missing_per_token_reference_samples']==6
    assert result['reference_gate_reproduced'] and not result['backend_adopted']
    assert all(row['ppo_decisions_match'] for row in result['microbatches'].values())
