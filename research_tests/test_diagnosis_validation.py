from copy import deepcopy
import json
import pytest

from research.diagnosis import validate_diagnosis
from research.semantic_evidence import audit_references, EvidenceModel
from research.evolution import Evolution
from research.skills import SecuritySkillPool
from research.upstream import Skill
from research.attribution import support_statistics
from research.rollout import compare_rollouts
from test_research import offline, experience
from test_rollout_verification import episodes, rule


def evidence(action='Remove', status='FALSE'):
    row={'step':5,'seed':11,'episode_id':0,'agent':'blue_agent_4','requested_action':action+' h1',
         'policy_invoked':True,'request_applied':True,'action_status':status,
         'observation_before':{'h1':{}},'observation_after':{'h1':{}},'behavior_context':{}}
    data={'diagnosis_status':'supported','reward_attribution':'unknown',
          'semantic_gradient':{'policy':'Review the recorded call outcome before retrying.'},
          'evidence':[{'component':'policy','evidence_id':'E0','field':'action_status'}],
          'claims':[{'component':'policy','evidence_id':'E0','action':action,'target':'h1',
              'finding':'completed_failure','phase':'after','as_of_step':6,
              'restore_epoch':-1,'causal_attribution':'unknown'}]}
    return data,[row]


def audit(data,rows):
    return validate_diagnosis(data,rows,audit_references(data,rows))


def test_analyse_cannot_be_cited_as_remove_failure():
    data,rows=evidence(action='Analyse')
    data['claims'][0]['action']='Remove'
    assert 'action_mismatch' in audit(data,rows)['errors']


def test_busy_wait_cannot_be_declared_completed_failure():
    data,rows=evidence(status='IN_PROGRESS')
    result=audit(data,rows)
    assert not result['passed'] and 'not_a_completed_action' in result['errors']


def test_another_hosts_observation_cannot_support_the_diagnosis():
    data,rows=evidence()
    data['claims'][0]['target']='h2'
    assert 'target_mismatch' in audit(data,rows)['errors']


def test_pre_restore_evidence_cannot_be_called_current_epoch():
    data,rows=evidence()
    rows[0]['behavior_context']={'recent_completed_actions':[{'name':'Restore','target':'h1','status':'TRUE','completed_at':4}]}
    assert 'restore_epoch_mismatch' in audit(data,rows)['errors']


def test_new_pid_after_restore_is_new_evidence_not_old_remove_failure():
    data,rows=evidence('Analyse','TRUE')
    rows[0]['behavior_context']={'recent_completed_actions':[{'name':'Restore','target':'h1','status':'TRUE','completed_at':4}]}
    rows[0]['observation_after']['h1']={'Processes':[{'PID':42}]}
    claim=data['claims'][0]
    claim.update(finding='observed_pid',pid=41,restore_epoch=4)
    assert 'pid_not_observed_at_claimed_time' in audit(data,rows)['errors']
    claim['pid']=42
    assert audit(data,rows)['passed']
    claim.update(action='Remove',finding='completed_failure')
    assert not audit(data,rows)['passed']


def test_correct_recorded_failure_supports_a_procedural_update():
    data,rows=evidence()
    result=audit(data,rows)
    assert result['passed'] and result['status']=='supported_observation'


def test_nonexistent_admin_and_service_evidence_are_rejected():
    data,rows=evidence()
    for finding,extras in [('observed_admin_indicator',{'indicator':'escalate.sh'}),
                           ('observed_service_state',{'service':'SSH','value':'stopped'})]:
        data['claims'][0].update(finding=finding,**extras)
        assert not audit(data,rows)['passed']


def test_joint_reward_blame_is_unidentified():
    data,rows=evidence();data['reward_attribution']='skill_caused_loss'
    assert 'individual_joint_reward_attribution_unidentified' in audit(data,rows)['errors']


def test_insufficient_diagnosis_suppresses_gradient_and_retains_pool(offline,monkeypatch):
    data,rows=evidence();data['diagnosis_status']='insufficient_evidence'
    model=EvidenceModel(lambda prompt:json.dumps(data),rows,diagnosis_verification=True)
    response=json.loads(model('original SG prompt'))
    assert not response['is_related'] and not any(response['semantic_gradient'].values())
    pool=SecuritySkillPool(24);before=deepcopy(pool.snapshot())
    ev=Evolution(offline,{'schedule':'paper_batch','gate_mode':'paper_rtg',
        'maintenance_mode':'paper_algorithm1','evidence_verification':True,'diagnosis_verification':True,'best_of_n':1})
    monkeypatch.setattr(ev,'_generate_semantic_gradient',lambda *args:({'initiation':'','execution':'','termination':''},False))
    result=ev.update(pool,[experience()],{'test':0},1)
    assert not any(r['accepted'] for r in result['evolution']) and pool.snapshot()==before


def test_two_hundred_decisions_are_one_independent_episode():
    exp=experience(count=200)
    for row in exp.transitions:row.update(seed=8,episode_id=0)
    result=support_statistics([exp])
    assert result['independent_episode_count']==1 and result['valid_policy_decisions']==200


def test_rare_skill_is_deferred_before_any_model_call(offline):
    pool=SecuritySkillPool(24);before=deepcopy(pool.snapshot());exp=experience()
    for row in exp.transitions:row.update(seed=8,episode_id=0)
    ev=Evolution(offline,{'schedule':'paper_batch','gate_mode':'paper_rtg',
        'maintenance_mode':'paper_algorithm1','minimum_independent_episodes':2})
    row=ev.update(pool,[exp],{'test':0},1)['evolution'][0]
    assert row['deferred']=='insufficient_independent_training_episodes'
    assert not offline.calls and not offline.score_calls and pool.snapshot()==before


def test_rollout_attempts_persist_across_resume(offline):
    old=Evolution(offline,{});old.rollout_attempts['RapidProcessResponse']=2
    new=Evolution(offline,{});new.restore(old.snapshot())
    assert new.rollout_attempts['RapidProcessResponse']==2


def test_partial_status_with_complete_seed_list_still_rejects():
    old,new=episodes(),episodes(20);new[0]['status']='partial'
    result=compare_rollouts(old,new,rule())
    assert not result['accepted'] and result['reason']=='incomplete_episode_status'
