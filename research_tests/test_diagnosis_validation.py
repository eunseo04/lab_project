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


def test_displayed_completed_action_supports_a_typed_normal_case_and_still_checks_target():
    data,rows=evidence()
    rows[0]['behavior_outcome']={'completed_action':{'name':'Remove','target':'h1',
        'status':'FALSE','started_at':3,'completed_at':6}}
    data['evidence'][0]['field']='completed_action'
    assert audit(data,rows)['passed']
    data['claims'][0]['target']='h2'
    assert not audit(data,rows)['passed']


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


def add_proposal(data,operation='check_call_outcome'):
    data['proposals']=[{'component':'policy','evidence_ids':['E0'],'observed_action':data['claims'][0]['action'],
        'operation':operation,'scope':'same_observed_host_and_epoch','causal_status':'untested_hypothesis'}]
    data['is_related']=True
    return data


@pytest.mark.parametrize('status',['TRUE','FALSE'])
def test_structured_positive_cases_pass_and_raw_contradiction_is_quarantined(status):
    data,rows=evidence(status=status)
    data['claims'][0]['finding']='completed_success' if status=='TRUE' else 'completed_failure'
    add_proposal(data)
    data['diagnosis']='Contradictory raw prose: Remove succeeded AND failed; blame other agents.'
    data['semantic_gradient']['policy']='UNSAFE_RAW_INSTRUCTION: Restore every host without evidence.'
    model=EvidenceModel(lambda prompt:json.dumps(data),rows,diagnosis_verification=True)
    result=json.loads(model('original prompt'))
    assert result['is_related'] and model.audits[-1]['passed']
    assert 'UNSAFE_RAW_INSTRUCTION' not in json.dumps(result)
    assert 'Contradictory raw prose' not in json.dumps(result)
    assert 'UNSAFE_RAW_INSTRUCTION' in model.audits[-1]['raw_response']
    payload=json.loads(result['semantic_gradient']['policy'])
    assert payload['verified_observations'][0]['finding']==data['claims'][0]['finding']
    assert payload['untested_modification_hypothesis']['causal_status']=='untested_hypothesis'


def test_correct_claim_with_unrelated_proposal_is_deferred():
    data,rows=evidence();add_proposal(data,operation='review_privilege_indicator')
    model=EvidenceModel(lambda prompt:json.dumps(data),rows,diagnosis_verification=True)
    result=json.loads(model('prompt'))
    assert not result['is_related'] and not model.audits[-1]['passed']
    assert 'proposal_unrelated_to_validated_finding' in model.audits[-1]['structured_proposals']['errors']


def sleep_decision_evidence():
    data,rows=evidence('Sleep','UNKNOWN')
    rows[0]['requested_action']='Sleep'
    data['evidence'][0]['field']='requested_action'
    data['claims'][0].update(target=None,finding='recorded_decision')
    add_proposal(data,'review_recorded_decision')
    data['proposals'][0]['scope']='same_observed_decision'
    return data,rows


def test_unknown_targetless_sleep_supports_only_a_recorded_decision():
    data,rows=sleep_decision_evidence()
    data['diagnosis']='Unverified prose: Sleep failed and caused the joint reward loss.'
    data['semantic_gradient']['policy']='UNVERIFIED_SLEEP_FAILURE: restore every host.'
    model=EvidenceModel(lambda _:json.dumps(data),rows,diagnosis_verification=True)
    result=json.loads(model('unchanged source of observed facts'))
    assert result['is_related'] and model.audits[-1]['passed']
    assert result['verified_facts'][0]['finding']=='recorded_decision'
    assert result['verified_facts'][0]['target'] is None
    assert result['proposals'][0]['scope']=='same_observed_decision'
    assert 'UNVERIFIED_SLEEP_FAILURE' not in json.dumps(result)
    assert 'UNVERIFIED_SLEEP_FAILURE' in model.audits[-1]['raw_response']


@pytest.mark.parametrize('invalid',[
    'busy','not_applied','parse_error','missing_request','missing_target','invented_target',
    'string_null','empty_target','wrong_time','before_decision','invented_failure','invented_pending','unrelated_proposal','wrong_scope'])
def test_targetless_decision_does_not_admit_unobserved_outcomes_or_requests(invalid):
    data,rows=sleep_decision_evidence();claim=data['claims'][0]
    if invalid=='busy':rows[0]['policy_invoked']=False
    elif invalid=='not_applied':rows[0]['request_applied']=False
    elif invalid=='parse_error':rows[0]['parse_error']='invalid response'
    elif invalid=='missing_request':rows[0]['requested_action']=None
    elif invalid=='missing_target':del claim['target']
    elif invalid=='invented_target':claim['target']='h1'
    elif invalid=='string_null':claim['target']='null'
    elif invalid=='empty_target':claim['target']=''
    elif invalid=='wrong_time':claim['as_of_step']=7
    elif invalid=='before_decision':claim.update(phase='before',as_of_step=5)
    elif invalid=='invented_failure':claim['finding']='completed_failure'
    elif invalid=='invented_pending':claim['finding']='pending'
    elif invalid=='unrelated_proposal':data['proposals'][0]['operation']='check_call_outcome'
    elif invalid=='wrong_scope':data['proposals'][0]['scope']='same_observed_host_and_epoch'
    model=EvidenceModel(lambda _:json.dumps(data),rows,diagnosis_verification=True)
    result=json.loads(model('audit only'))
    assert not result['is_related'] and not model.audits[-1]['passed']
    assert not any(result['semantic_gradient'].values())


def test_current_sleep_decision_is_not_the_previous_completed_remove():
    data,rows=sleep_decision_evidence()
    rows[0]['behavior_outcome']={'completed_action':{'name':'Remove','target':'h1',
        'status':'FALSE','started_at':3,'completed_at':6}}
    model=EvidenceModel(lambda _:json.dumps(data),rows,diagnosis_verification=True)
    result=json.loads(model('the prior completion and current decision are separate'))
    assert result['is_related'] and result['verified_facts'][0]['action']=='Sleep'
    assert result['verified_facts'][0]['target'] is None


def controlled_native_diagnosis_cases():
    """Real partial CAGE4 records under scripted actions, not an LLM performance run."""
    from research.adapter import make_environment,plain,BLUE_AGENTS
    from research.behavior import ObservableCaseLedger
    from research.diagnosis import operation,restore_boundary
    from research.config import validate
    from acd.runner import seed_everything
    from CybORG.Simulator.Actions import Sleep
    seed=58000;config=validate({'steps':500});seed_everything(seed,1)
    env=make_environment(config,seed);env.reset(seed=seed)
    agent='blue_agent_4';ledger=ObservableCaseLedger(agent);rows=[]
    host_actions=[a for a in env.actions(agent) if type(a).__name__=='Analyse']
    host=next(a.hostname for a in host_actions if 'server_host' in a.hostname)
    sequence=['Analyse','Remove','Restore','Analyse','Remove_invalid_session'];next_action=0
    for step in range(30):
        raw=plain(env.raw_observations()[agent]);busy='IN_PROGRESS' in str(raw.get('success'))
        ledger.context({'observation':raw,'hosts':[host],'user_prompt':'controlled native diagnostic'},step)
        if not busy and next_action<len(sequence):
            kind=sequence[next_action];next_action+=1
            action=deepcopy(next(a for a in env.actions(agent) if type(a).__name__==kind.split('_')[0] and getattr(a,'hostname',None)==host))
            if kind.endswith('_invalid_session'):action.session=999999
        else:action=Sleep()
        chosen={a:Sleep() for a in BLUE_AGENTS};chosen[agent]=action
        env.step(chosen)
        after=plain(env.raw_observations()[agent])
        row={'step':step,'seed':seed,'episode_id':-1,'agent':agent,'policy':'scripted_diagnostic',
            'requested_action':str(action),'policy_invoked':not busy,'request_applied':not busy,
            'action_status':str(after['success']),'observation_before':raw,'observation_after':after}
        ledger.transition(row);rows.append(row)
        if next_action==len(sequence) and ledger.pending is None:break
    selected={}
    for row in rows:
        op=operation(row)
        if not op.get('target'):continue
        finding=('completed_success' if op['status']=='TRUE' else 'completed_failure' if op['status']=='FALSE' else 'pending' if op['status']=='IN_PROGRESS' else None)
        if finding:selected.setdefault((op['name'],finding),row)
    cases=[]
    def data_for(row):
        op=operation(row);finding='pending' if op['status']=='IN_PROGRESS' else 'completed_success' if op['status']=='TRUE' else 'completed_failure'
        data={'diagnosis_status':'supported','reward_attribution':'unknown','is_related':True,
            'semantic_gradient':{'policy':'This prose is not trusted as a fact.'},
            'evidence':[{'component':'policy','evidence_id':'E0','field':'action_status'}],
            'claims':[{'component':'policy','evidence_id':'E0','action':op['name'],'target':op['target'],
                'finding':finding,'phase':'after','as_of_step':row['step']+1,
                'restore_epoch':restore_boundary([row],agent,op['target'],row['step']+1),'causal_attribution':'unknown'}]}
        return add_proposal(data,'wait_for_completion' if finding=='pending' else 'check_call_outcome')
    for (name,finding),row in selected.items():cases.append((name+'_'+finding,True,data_for(row),[row]))
    analyse=next(r for r in selected.values() if operation(r)['name']=='Analyse' and operation(r)['status']=='TRUE')
    busy=next(r for r in selected.values() if operation(r)['status']=='IN_PROGRESS')
    data=data_for(busy);data['claims'][0]['finding']='completed_failure';data['proposals'][0]['operation']='check_call_outcome'
    cases.append(('busy_as_failure',False,data,[busy]))
    data=data_for(analyse);data['claims'][0]['action']='Remove';data['proposals'][0]['observed_action']='Remove'
    cases.append(('analyse_as_remove',False,data,[analyse]))
    data=data_for(analyse);data['claims'][0]['target']='other_unobserved_host'
    cases.append(('other_host',False,data,[analyse]))
    data=data_for(analyse);data['claims'][0].update(finding='observed_pid',pid=99999999);data['proposals'][0]['operation']='review_observed_pid'
    cases.append(('other_pid',False,data,[analyse]))
    restored=next(r for r in selected.values() if operation(r)['name']=='Restore' and operation(r)['status']=='TRUE')
    data=data_for(restored);data['claims'][0]['restore_epoch']=-1
    cases.append(('old_restore_epoch',False,data,[restored]))
    data=data_for(analyse);data['semantic_gradient']['policy']='UNTRUSTED_MARKER: Analyse was Remove and failed.'
    data['diagnosis']='UNTRUSTED_MARKER: blame the skill for all joint reward loss'
    cases.append(('valid_claim_contradictory_prose_quarantined',True,data,[analyse]))
    data=data_for(analyse);data['proposals'][0]['operation']='review_privilege_indicator'
    cases.append(('unrelated_proposal',False,data,[analyse]))
    return rows,cases


def evaluate_controlled_native_cases():
    rows,cases=controlled_native_diagnosis_cases();out=[]
    for name,expected,data,source in cases:
        model=EvidenceModel(lambda prompt:json.dumps(data),source,diagnosis_verification=True)
        response=json.loads(model('Controlled validator input; not Qwen-generated'))
        passed=bool(response['is_related']);audit=model.audits[-1]
        out.append({'case':name,'expected_safe_canonical_path':expected,'actual_safe_canonical_path':passed,
            'raw_prose_quarantined':'UNTRUSTED_MARKER' not in json.dumps(response),'audit':audit})
    return rows,out


def test_native_cage4_observation_cases_have_positive_and_negative_paths():
    rows,result=evaluate_controlled_native_cases()
    assert any(r['case']=='Remove_completed_failure' for r in result)
    assert rows and any(r['expected_safe_canonical_path'] for r in result) and any(not r['expected_safe_canonical_path'] for r in result)
    assert all(r['expected_safe_canonical_path']==r['actual_safe_canonical_path'] and r['raw_prose_quarantined'] for r in result)
