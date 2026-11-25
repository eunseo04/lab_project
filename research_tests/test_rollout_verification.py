from research.behavior_skills import initiation_matches


def case(**updates):
    value={'host':'h1','observed_level':2,'flagged_pids':[41],
        'response_due':True,'verification_due':False,'analysis_due':True,
        'evidence_version':2,'last_remove':None}
    value.update(updates);return value


def test_confirmed_admin_recovery_takes_priority_over_followup_analysis():
    context={'open_cases':[case(observed_level=3,verification_due=True)]}
    assert not initiation_matches('EscalationDiagnosis',context,revision='v5')
    assert initiation_matches('PrivilegedRecovery',context,revision='v5')


def test_new_flagged_pid_receives_response_before_redundant_diagnosis():
    context={'open_cases':[case()]}
    assert initiation_matches('RapidProcessResponse',context,revision='v5')
    assert not initiation_matches('EscalationDiagnosis',context,revision='v5')


from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pytest
from research.config import validate
from research.evolution import Evolution
from research.rollout import compare_rollouts,verify_transaction
from research.semantic_evidence import audit_references,EvidenceModel,candidate_supported
from research.skills import SecuritySkillPool
from research.upstream import Skill
from test_research import offline,experience


def rule():
    return {'seeds':list(range(41000,41020)),'minimum_pairs':20,'minimum_mean_gain':0.,'minimum_win_rate':.6,
        'minimum_median_gain':0.,'maximum_seed_loss':50.,'maximum_mean_lwf_loss':0.,
        'maximum_mean_asf_loss':0.,'maximum_std_ratio':1.25,'maximum_std_increase':25.,'ports':[8018]}


def episodes(gain=0):
    # Synthetic arithmetic fixtures only; never written as model performance.
    return [{'seed':s,'status':'completed','kind':'real_cage4','joint_transitions':499,'scored_transitions':498,
        'joint_return':-100+gain,'reward_components':{'action_cost':-20,'RIA':-30,'LWF':-40+gain,'ASF':-10},
        'unit_test_fixture_only':True} for s in rule()['seeds']]


def test_failed_remove_requests_diagnosis_but_new_pid_allows_new_response():
    c=case(verification_due=True,last_remove={'status':'FALSE','evidence_version':2})
    assert not initiation_matches('RapidProcessResponse',{'open_cases':[c]},revision='v5')
    assert initiation_matches('EscalationDiagnosis',{'open_cases':[c]},revision='v5')
    c['evidence_version']=3
    assert initiation_matches('RapidProcessResponse',{'open_cases':[c]},revision='v5')


def test_failure_on_one_host_does_not_hide_a_new_pid_on_another():
    c=case(last_remove={'status':'FALSE','evidence_version':2})
    assert initiation_matches('RapidProcessResponse',{'open_cases':[c,case(host='h2')]},revision='v5')


def test_default_domain_rules_remain_legacy():
    context={'open_cases':[case(observed_level=3,verification_due=True)]}
    assert initiation_matches('EscalationDiagnosis',context)


def test_reanalysis_after_failed_remove_can_justify_reassessment_and_retry():
    c=case(verification_due=False,last_remove={'status':'FALSE','evidence_version':2,'completed_at':10},
        last_analysis={'status':'TRUE','completed_at':13})
    assert initiation_matches('RapidProcessResponse',{'open_cases':[c]},revision='v5')
    assert not initiation_matches('EscalationDiagnosis',{'open_cases':[c]},revision='v5')


def test_consistent_complete_environment_gain_passes():
    result=compare_rollouts(episodes(),episodes(10),rule())
    assert result['accepted'] and result['win_rate']==1 and result['mean_difference']==10


@pytest.mark.parametrize('minimum,count',[(None,2),(None,20),(2,2),(19,19),(20,19)])
def test_all_approval_entry_points_require_explicit_twenty_pairs(minimum,count,offline):
    bad=rule();bad['seeds']=bad['seeds'][:count]
    if minimum is None:bad.pop('minimum_pairs')
    else:bad['minimum_pairs']=minimum
    with pytest.raises(ValueError):compare_rollouts(episodes()[:count],episodes(10)[:count],bad)
    cfg={'train_seed':100,'eval_seed':10000,'rollout_validation':bad,
        'evolution':{'rollout_verification':True,'schedule':'paper_batch',
                     'gate_mode':'paper_rtg','maintenance_mode':'paper_algorithm1'}}
    with pytest.raises(ValueError):validate(cfg)
    pool=SecuritySkillPool(24);before=pool.snapshot()
    with pytest.raises(ValueError):
        verify_transaction(pool,Skill('c','i',['p'],'t'),lambda *a:pytest.fail('no rollout before valid registration'),bad)
    assert pool.snapshot()==before


def test_outlier_average_alone_is_not_admitted():
    new=episodes(-1);new[-1]['joint_return']=0;new[-1]['reward_components']['LWF']=60
    result=compare_rollouts(episodes(),new,rule())
    assert result['mean_difference']>0 and not result['accepted']


def test_severe_single_seed_loss_rejects_positive_average():
    new=episodes(20);new[-1]['joint_return']=-160;new[-1]['reward_components']['LWF']=-100
    result=compare_rollouts(episodes(),new,rule());assert not result['checks']['worst_seed']


def test_green_local_work_regression_cannot_hide_behind_ria_gain():
    new=episodes(10)
    for r in new:r['reward_components'].update(LWF=-45,RIA=-15)
    result=compare_rollouts(episodes(),new,rule())
    assert result['mean_difference']>0 and not result['checks']['green_local_work'] and not result['accepted']


def test_incomplete_validation_cannot_admit():
    assert compare_rollouts(episodes(),episodes(10)[:-1],rule())['reason']=='incomplete_validation'


@pytest.mark.parametrize('mutation',[lambda e:e.update(kind='mock_test'),lambda e:e.update(scored_transitions=499),
    lambda e:e.update(joint_return=float('nan')),lambda e:e.update(reward_components={'LWF':-1})])
def test_invalid_validation_evidence_is_rejected(mutation):
    new=episodes(10);mutation(new[0])
    with pytest.raises(ValueError):compare_rollouts(episodes(),new,rule())


def test_duplicate_seed_is_rejected():
    new=episodes(10);new[-1]['seed']=new[0]['seed']
    with pytest.raises(ValueError):compare_rollouts(episodes(),new,rule())


def test_failed_rollout_leaves_live_pool_byte_identical(offline):
    pool=SecuritySkillPool(24);before=deepcopy(pool.snapshot())
    candidate=Skill('candidate','When evidence exists',['Inspect result'],'When reviewed')
    result=verify_transaction(pool,candidate,lambda snapshot,role:episodes(0 if role=='previous' else -10),rule())
    assert not result['accepted'] and pool.snapshot()==before and 'verified_pool' not in result


def test_successful_rollout_returns_exact_verified_snapshot_without_early_commit(offline):
    pool=SecuritySkillPool(24);before=deepcopy(pool.snapshot())
    candidate=Skill('candidate','When evidence exists',['Inspect result'],'When reviewed')
    result=verify_transaction(pool,candidate,lambda snapshot,role:episodes(0 if role=='previous' else 10),rule())
    assert result['accepted'] and pool.snapshot()==before
    assert any(s['name']=='candidate' for s in result['verified_pool']['skills'])


def test_worker_failure_keeps_old_pool(offline):
    pool=SecuritySkillPool(24);before=deepcopy(pool.snapshot())
    def failure(*args):raise RuntimeError('real server unavailable')
    result=verify_transaction(pool,Skill('candidate','i',['p'],'t'),failure,rule())
    assert result['reason']=='validation_error' and pool.snapshot()==before


def test_positive_ppo_score_can_still_be_rejected_by_rollout(offline,monkeypatch):
    pool=SecuritySkillPool(24);before=deepcopy(pool.snapshot());candidate=Skill('candidate','i',['p'],'t')
    config={'schedule':'paper_batch','gate_mode':'paper_rtg','maintenance_mode':'paper_algorithm1',
        'best_of_n':1,'rollout_verification':True}
    ev=Evolution(offline,config,rollout_validator=lambda *args:{'accepted':False,'reason':'negative_real_rollout'})
    monkeypatch.setattr(ev,'evolve',lambda *args:(candidate,'REFINE'))
    monkeypatch.setattr(ev,'probabilities',lambda skill,exps:[0.]*sum(len(e.transitions) for e in exps))
    log=ev.update(pool,[experience()],{'test':0},1)['evolution'][0]
    assert log['candidates'][0]['score']>0 and log['candidates'][0]['ppo_accepted']
    assert not log['accepted'] and pool.snapshot()==before


def refs():
    rows=[{'step':5,'agent':'blue_agent_4','requested_action':'Remove h1','action_status':'FALSE'}]
    data={'is_related':True,'semantic_gradient':{'policy':'Inspect the failed Remove outcome'},
        'evidence':[{'component':'execution','step':5,'agent':'blue_agent_4','field':'action_status','quote':'FALSE'}]}
    return data,rows


def test_semantic_gradient_reference_points_to_real_actor_outcome():
    data,rows=refs();assert audit_references(data,rows)['passed']


@pytest.mark.parametrize('update',[{'step':6},{'agent':'blue_agent_3'},{'quote':'TRUE'},{'field':'hidden_red_sessions'}])
def test_invented_or_other_agent_references_do_not_support_update(update):
    data,rows=refs();data['evidence'][0].update(update)
    assert not audit_references(data,rows)['passed']


def test_every_changed_component_requires_its_own_support():
    data,rows=refs();data['semantic_gradient']['termination']='Stop earlier'
    assert not audit_references(data,rows)['passed']


def test_unsupported_semantic_update_does_not_fall_back_to_new_skill(offline):
    ev=Evolution(offline,{'evidence_verification':True})
    assert ev._run_semantic_update(Skill('s','i',['p'],'t'),[],'NEW')=={}
    assert not offline.calls


def test_rollout_seeds_cannot_overlap_training():
    c={'train_seed':41000,'eval_seed':50000,'episodes':100,'rollout_validation':rule(),
        'evolution':{'rollout_verification':True,'schedule':'paper_batch','gate_mode':'paper_rtg','maintenance_mode':'paper_algorithm1'}}
    with pytest.raises(ValueError,match='overlap'):validate(c)


def test_original_gate_and_common_history_are_unchanged():
    root=Path(__file__).resolve().parents[1]
    reference=json.loads((root/'tests/fixtures/reference_integrity.json').read_text())
    for name in ['gate.py','paper_algorithm.py','behavior.py','adapter.py']:
        assert hashlib.sha256((root/'research'/name).read_bytes()).hexdigest()==reference['core_sha256'][name]


def test_aggregated_update_cannot_change_an_unsupported_component():
    parent=Skill('p','original',['old policy'],'original end')
    candidate=Skill('c','changed without evidence',['new policy'],'original end')
    records=[{'related':True,'gradient':{'execution':'Change policy'}}]
    assert not candidate_supported(parent,candidate,records)['passed']
    candidate.initiation=parent.initiation
    result=candidate_supported(parent,candidate,records)
    assert not result['passed'] and 'unlinked_candidate_changes' in result['linkage_errors']


def supported_candidate_fixture():
    rows=[{'step':5,'seed':7,'agent':'blue_agent_4','requested_action':'Remove h1',
        'policy_invoked':True,'request_applied':True,'action_status':'TRUE'}]
    data={'is_related':True,'diagnosis_status':'supported','reward_attribution':'unknown',
        'semantic_gradient':{'policy':'Review actual call status; its cause remains unknown.'},
        'evidence':[{'component':'policy','evidence_id':'E0','field':'action_status'}],
        'claims':[{'component':'policy','evidence_id':'E0','action':'Remove','target':'h1',
            'finding':'completed_success','phase':'after','as_of_step':6,'restore_epoch':-1,'causal_attribution':'unknown'}],
        'proposals':[{'component':'policy','evidence_ids':['E0'],'observed_action':'Remove',
            'operation':'check_call_outcome','scope':'same_observed_host_and_epoch','causal_status':'untested_hypothesis'}]}
    model=EvidenceModel(lambda _:json.dumps(data),rows,diagnosis_verification=True)
    canonical=json.loads(model('controlled boundary'))
    parent=Skill('parent','same initiation',['Check status'],'same termination')
    candidate=Skill('child',parent.initiation,['Review the observed Remove call outcome before retrying.'],parent.termination)
    records=[{'related':True,'gradient':{'execution':canonical['semantic_gradient']['policy']},'evidence_audits':model.audits}]
    links=[{'component':'execution','element':0,'candidate_text':candidate.policy[0],
        'proposal_ids':[canonical['proposals'][0]['proposal_id']]}]
    return parent,candidate,records,links


def test_linked_normal_candidate_has_bounded_support_without_claiming_semantic_truth():
    parent,candidate,records,links=supported_candidate_fixture()
    result=candidate_supported(parent,candidate,records,links)
    assert result['passed'] and result['change_links'][0]['fact_ids']
    assert not result['semantics_verified'] and result['requires_rollout']
    links[0]['candidate_text']='Different unlinked text'
    assert not candidate_supported(parent,candidate,records,links)['passed']


def test_normal_grounded_candidate_can_pass_original_gate_and_full_pool_validation(offline,monkeypatch):
    parent,candidate,records,links=supported_candidate_fixture()
    pool=SecuritySkillPool(24);snapshot=pool.snapshot();snapshot['skills']=[parent.__dict__];pool.restore(snapshot)
    exp=experience();exp.skill=parent.name
    def validator(p,c,s):
        return verify_transaction(p,c,lambda _,role:episodes(0 if role=='previous' else 10),rule())
    ev=Evolution(offline,{'schedule':'paper_batch','gate_mode':'paper_rtg','maintenance_mode':'paper_algorithm1',
        'best_of_n':1,'evidence_verification':True,'diagnosis_verification':True,'rollout_verification':True},rollout_validator=validator)
    def evolve(*args):
        ev.gradients.extend(deepcopy(records));ev._candidate_links=deepcopy(links)
        return deepcopy(candidate),'REFINE'
    monkeypatch.setattr(ev,'evolve',evolve)
    monkeypatch.setattr(ev,'probabilities',lambda skill,exps:[0.]*sum(len(e.transitions) for e in exps))
    result=ev.update(pool,[exp],{'test':0},1)['evolution'][0]['candidates'][0]
    assert result['semantic_support']['passed'] and result['ppo_accepted'] and result['accepted']
    assert result['rollout_validation']['accepted'] and len(result['rollout_validation']['seeds'])==20
    assert any(s.name==candidate.name for s in pool.skills)


def test_linked_remove_success_cannot_prove_all_privileged_persistence_eliminated():
    parent,candidate,records,links=supported_candidate_fixture()
    candidate.policy=['Remove TRUE proves all privileged persistence is eliminated.'];links[0]['candidate_text']=candidate.policy[0]
    result=candidate_supported(parent,candidate,records,links)
    assert not result['passed'] and result['known_rule_contradictions']


def test_indirect_unclassified_policy_and_termination_changes_require_rollout():
    from research.semantic_evidence import action_support
    parent=Skill('p','i',['Observe'],'when reviewed')
    candidate=Skill('c','i',['Immediately isolate the affected network zone and keep that zone disconnected.'],'retain containment indefinitely')
    result=action_support(parent,candidate,[],[])
    assert result['requires_rollout'] and result['semantic_status']=='unknown'
    assert set(result['unclassified_changes'])=={'policy','termination'} and result['forbidden_actions']==[]


def test_reusable_candidate_cannot_memorize_training_host_or_ip():
    parent=Skill('p','i',['p'],'t')
    records=[{'related':True,'gradient':{'execution':'Investigate observable lead'}}]
    for text in ['Analyse public_access_zone_subnet_user_host_1','Investigate 10.0.0.4']:
        result=candidate_supported(parent,Skill('c','i',[text],'t'),records)
        assert not result['passed'] and result['instance_specific_identifier']


def test_catalog_reference_resolves_exact_step_actor_and_outcome():
    data,rows=refs();data['evidence']=[{'component':'policy','evidence_id':'E0','field':'action_status'}]
    result=audit_references(data,rows)
    assert result['passed'] and result['valid_references'][0]['quote']=='FALSE'
    assert result['valid_references'][0]['step']==5 and result['valid_references'][0]['agent']=='blue_agent_4'


@pytest.mark.parametrize('update',[{'evidence_id':'E999'},{'agent':'blue_agent_0'},{'step':99},{'field':'hidden_red_sessions'}])
def test_catalog_cannot_invent_records_agents_or_hidden_fields(update):
    data,rows=refs();ref={'component':'policy','evidence_id':'E0','field':'action_status'};ref.update(update)
    data['evidence']=[ref];assert not audit_references(data,rows)['passed']


def test_asv_tracks_gate_scope_and_requires_rollout_for_previously_named_unexecuted_action():
    from research.semantic_evidence import action_support
    parent=Skill('NetworkContainment','Observe connections',['Reassess before BlockTrafficZone'],'After review')
    candidate=Skill('NetworkContainment_v2','Observe connections',['Immediately BlockTrafficZone when justified'],'After review')
    exp=experience();exp.transitions[0].update(requested_action='Sleep',executed_actions=[{'name':'Sleep'}],action_status='TRUE')
    exp.transitions[1].update(requested_action='Analyse h1',executed_actions=[{'name':'Analyse'}],action_status='TRUE')
    support=action_support(parent,candidate,[exp],[exp])
    assert support['gate_evaluated_requested_actions']=={'Sleep':1,'Analyse':1}
    assert support['newly_named_actions']==[]
    assert support['actions_outside_empirical_execution_support']==['BlockTrafficZone']
    assert support['requires_rollout'] and support['forbidden_actions']==[]
    assert support['observable_context_rows']==0


def test_asv_does_not_confuse_requested_primitive_with_executed_action():
    from research.semantic_evidence import action_support
    parent=Skill('p','i',['Sleep'],'t');candidate=Skill('c','i',['Remove'],'t')
    exp=experience(count=1);exp.transitions[0].update(requested_action='Remove h1',request_applied=False,
        executed_actions=[{'name':'Sleep'}],action_status='IN_PROGRESS')
    support=action_support(parent,candidate,[exp],[])
    assert support['actual_started_actions']=={} and support['requires_rollout']


def test_asv_includes_later_busy_completion_of_an_accepted_async_request():
    from research.semantic_evidence import action_support
    parent=Skill('p','i',['Analyse'],'t');candidate=Skill('c','i',['Remove'],'t')
    exp=experience(count=2)
    exp.transitions[0].update(requested_action='Remove h1',executed_actions=[{'name':'Sleep'}],action_status='IN_PROGRESS')
    exp.transitions[1].update(policy_invoked=False,request_applied=False,requested_action='Sleep',
        action_status='TRUE',behavior_outcome={'completed_action':{'name':'Remove','target':'h1','started_at':0,'completed_at':2,'status':'TRUE'}})
    report=action_support(parent,candidate,[exp],[])
    assert report['actual_started_actions']=={'Remove':1} and report['observed_completed_actions']=={'Remove':{'TRUE':1}}


def test_asv_is_separate_from_original_gate_and_defers_without_forbidding_action(offline,monkeypatch):
    pool=SecuritySkillPool(24);before=deepcopy(pool.snapshot());candidate=Skill('new','i',['BlockTrafficZone'],'t')
    ev=Evolution(offline,{'schedule':'paper_batch','gate_mode':'paper_rtg',
        'maintenance_mode':'paper_algorithm1','best_of_n':1,'action_support_verification':True})
    monkeypatch.setattr(ev,'evolve',lambda *args:(candidate,'REFINE'))
    monkeypatch.setattr(ev,'probabilities',lambda skill,exps:[0.]*sum(len(e.transitions) for e in exps))
    item=ev.update(pool,[experience()],{'test':0},1)['evolution'][0]['candidates'][0]
    assert item['score']>0 and item['ppo_accepted'] and item['deferred']=='additional_real_rollout_required'
    assert not item['post_gate_verification']['action_banned'] and pool.snapshot()==before


def test_new_verification_requires_a_named_version_and_original_B2_remains_disabled():
    from research.config import variant_config
    assert not variant_config('B2')['evolution'].get('action_support_verification',False)
    assert variant_config('B2-ASV')['comparison_id']=='B2-ASV'
    for comparison in ('B2',None):
        with pytest.raises(ValueError,match='distinct ablation'):
            validate({'comparison_id':comparison,'experiment_version':'fixture','evolution':{'action_support_verification':True}})


def test_reject_only_screen_never_approves_and_boundary_matches_full_rule():
    from research.rollout import reject_screen
    r={**rule(),'reject_screen_pairs':2,'reject_screen_only':True}
    assert reject_screen(episodes()[:2],episodes(20)[:2],r)['decision_type']=='deferred'
    assert not reject_screen(episodes()[:2],episodes(20)[:2],r)['accepted']
    assert reject_screen(episodes()[:1],episodes(-50)[:1],r)['decision_type']=='deferred'
    assert reject_screen(episodes()[:1],episodes(-51)[:1],r)['decision_type']=='rejected'


def test_early_rejection_preserves_pool_and_stops_remaining_pairs(offline):
    pool=SecuritySkillPool(24);before=pool.snapshot();calls=[]
    r={**rule(),'reject_screen_pairs':2,'reject_screen_only':True}
    def pair(a,b,seed):
        calls.append(seed);return episodes()[0],episodes(-51)[0]
    result=verify_transaction(pool,Skill('c','i',['p'],'t'),lambda *args:pytest.fail('no full approval'),r,pair)
    assert result['reason']=='failed_registered_worst_seed_early' and len(calls)==1
    assert pool.snapshot()==before and 'verified_pool' not in result


def test_screen_prefix_followed_by_full_twenty_pairs_is_required_for_approval(offline):
    pool=SecuritySkillPool(24);calls=[];r={**rule(),'reject_screen_pairs':2}
    def pair(a,b,seed):
        i=r['seeds'].index(seed);return episodes()[i],episodes(10)[i]
    def full(snapshot,role):
        calls.append(role);return episodes(0 if role=='previous' else 10)
    result=verify_transaction(pool,Skill('c','i',['p'],'t'),full,r,pair)
    assert result['accepted'] and calls==['previous','candidate'] and len(result['seeds'])==20


def test_catalog_completed_action_alias_resolves_only_an_actual_completed_record():
    data,rows=refs();rows[0]['behavior_outcome']={'completed_action':{'name':'Remove','target':'h1','status':'FALSE','started_at':3,'completed_at':6}}
    data['evidence']=[{'component':'policy','evidence_id':'E0','field':'completed_action'}]
    result=audit_references(data,rows)
    assert result['passed'] and 'h1' in result['valid_references'][0]['quote']
    rows[0]['behavior_outcome']={}
    assert not audit_references(data,rows)['passed']


def test_reward_diagnostics_handle_inactive_red_and_green_without_observations(tmp_path):
    from types import SimpleNamespace
    from research.reward_audit import RewardAudit
    from CybORG.Simulator.Actions import Sleep
    called=[]
    calc=SimpleNamespace(calculate_reward=lambda *args:(called.append(args) or 0),get_phase_rewards=lambda phase:{})
    controller=SimpleNamespace(team_reward_calculators={'Blue':{'BlueRewardMachine':calc}},reward={'Blue':{'action_cost':0}})
    audit=RewardAudit(SimpleNamespace(env=SimpleNamespace(environment_controller=controller)),tmp_path/'events.jsonl')
    state=SimpleNamespace(mission_phase=0,ip_addresses={},hostname_subnet_map={},
        sessions={'green_0':{0:SimpleNamespace(active=False)},'red_0':{0:SimpleNamespace(active=False)}})
    actions={'green_0':[Sleep()],'red_0':[Sleep()]};obs={a:SimpleNamespace(observations=[]) for a in actions}
    assert calc.calculate_reward(None,actions,obs,False,state)==0 and called[0][1] is actions
    audit.capture(0,0,True);audit.finish(0)
    assert all(r['status']=='NOT_OBSERVED' and not r['active_session'] and not r['policy_input'] for r in audit.environment_actions)


def test_factual_ppo_change_can_commit_reload_and_reach_the_next_actor(offline,monkeypatch,tmp_path):
    from research.adapter import DefensePolicy
    from research.config import variant_config
    parent,candidate,records,links=supported_candidate_fixture()
    pool=SecuritySkillPool(24);snapshot=pool.snapshot();snapshot['skills']=[parent.__dict__];pool.restore(snapshot)
    exp=experience();exp.skill=parent.name
    ev=Evolution(offline,{'schedule':'paper_batch','gate_mode':'paper_rtg','maintenance_mode':'paper_algorithm1',
        'best_of_n':1,'evidence_verification':True,'admission_mode':'factual_ppo'})
    def evolve(*args):
        ev.gradients.extend(deepcopy(records));ev._candidate_links=deepcopy(links)
        return deepcopy(candidate),'REFINE'
    monkeypatch.setattr(ev,'evolve',evolve);monkeypatch.setattr(ev,'probabilities',lambda s,es:[0.]*sum(len(e.transitions) for e in es))
    result=ev.update(pool,[exp],{'test':0},1)['evolution'][0]['candidates'][0]
    assert result['accepted'] and not result['efficacy_verified'] and result['admission_mode']=='factual_ppo'
    path=tmp_path/'frozen.json';path.write_text(json.dumps(pool.snapshot()))
    restored=SecuritySkillPool(24);restored.restore(json.loads(path.read_text()))
    cfg=variant_config('B3');cfg['selector']['epsilon']=0
    seen=[]
    def generated(messages,**kw):
        seen.extend(messages);return {'text':'{"action":"Sleep","reason":"CPU fixture"}','token_ids':[1]}
    monkeypatch.setattr(offline,'generate',generated)
    actor=DefensePolicy('blue_agent_4',offline,lambda *a,**k:'<choice>child</choice>',cfg,restored)
    state={'agent':'blue_agent_4','observation':{'success':'UNKNOWN'},'user_prompt':'visible',
        'hosts':['h1'],'subnets':[],'allowed_actions':['Sleep'],'behavior_context':{}}
    _,record=actor.decide(state,episode=1,step=0,seed=1)
    assert record['skill_id']=='child' and candidate.policy[0] in seen[0]['content']


def test_asv_only_is_not_a_blanket_changed_text_veto():
    from research.semantic_evidence import action_support
    parent=Skill('p','i',['Remove the affected host'],'t');candidate=Skill('c','i',['Remove the affected host and review the call result'],'t')
    exp=experience();exp.transitions=[{'seed':1,'step':0,'agent':'blue_agent_4','requested_action':'Remove h1',
        'policy_invoked':True,'request_applied':True,'action_status':'TRUE','executed_actions':[{'name':'Remove'}],
        'behavior_context':{'open_cases':[{'response_due':True,'flagged_pids':[77],'observed_level':2}]}}]
    check=action_support(parent,candidate,[exp],[exp])
    assert not check['requires_rollout'] and not check['efficacy_verified']


def test_new_fact_schema_runs_original_sg_candidate_gate_and_commit_path(offline):
    import re
    from research.config import variant_config
    from research.diagnosis import canonical_fact_catalog
    from test_diagnosis_validation import fact_intervention_fixture
    parent=Skill('parent','same initiation',['Check process evidence'],'same termination',version=1)
    pool=SecuritySkillPool(24);snapshot=pool.snapshot();snapshot['skills']=[parent.__dict__];pool.restore(snapshot)
    procedure='Remove the runtime bound host, wait for completion, then review the visible process evidence.'
    class Model:
        def __init__(self):self.rows=[];self.sg=0;self.candidates=0;self.score_samples=0
        def capture_evidence_rows(self,rows):self.rows=rows
        def __call__(self,prompt):
            if 'Exact fact catalogue' in prompt:
                self.sg+=1;_,data=fact_intervention_fixture()
                fact=next(f for f in canonical_fact_catalog(self.rows) if f['finding']=='observed_pid')
                data['interventions'][0]['supported_fact_ids']=[fact['fact_id']]
                return json.dumps(data)
            self.candidates+=1
            ids=list(dict.fromkeys(re.findall(r'\bP[a-f0-9]{16}\b',prompt)))
            assert ids
            return json.dumps({'skill_name':'child','initiation':parent.initiation,'policy':[procedure],
                'termination':parent.termination,'change_links':[{'component':'execution','element':0,
                    'candidate_text':procedure,'proposal_ids':ids}]})
        def score(self,samples,**kw):self.score_samples+=len(samples);return [0.]*len(samples)
    model=Model();exps=[]
    for seed in (1,2):
        exp=experience();exp.skill=parent.name
        for row in exp.transitions:
            row.update(seed=seed,episode_id=0,agent='blue_agent_4',requested_action='Sleep',action_status='UNKNOWN',
                observation_before={},observation_after={'h1':{'Processes':[{'PID':77}]}})
        exps.append(exp)
    config=variant_config('B3')['evolution'];ev=Evolution(model,config)
    result=ev.update(pool,exps,{'test':0},1)
    assert result['evolution'][0]['accepted']
    assert model.sg==2 and model.candidates==config['best_of_n'] and model.score_samples>0
    assert any(s.name=='parent_v2' and s.policy==[procedure] for s in pool.skills)
