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
    return {'seeds':list(range(41000,41020)),'minimum_mean_gain':0.,'minimum_win_rate':.6,
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
    assert candidate_supported(parent,candidate,records)['passed']


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
