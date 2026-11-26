from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket

import numpy as np
import pytest

from research.config import validate
from research.gate import gate_score, returns_to_go
from research.metrics import JointRewardLedger, aggregate_episodes
from research.upstream import Skill, Experience, configure_encoder, OriginalEvolution
from research.skills import SecuritySkillPool, SkillController, CAGEVerifier
from research.evolution import Evolution
from research.experiment import run
from research.adapter import make_environment, make_state, DefensePolicy, messages_for
from acd.llm_agent import LLMDefender, parse_response


class TestModel:
    __test__ = False
    is_mock = True
    identity = {"model":"Qwen/Qwen3-8B","revision":"test-only-not-a-real-model"}
    def __init__(self): self.calls=[]; self.score_calls=0
    def embed(self,text):
        values=np.frombuffer(hashlib.sha256(text.encode()).digest(),dtype=np.uint8).astype(float)
        return (values/np.linalg.norm(values)).tolist()
    def generate(self,messages,seed=42,purpose="policy",**kwargs):
        text=messages[-1]["content"]
        if "<status>" in text: answer="<status>CONTINUE</status>"
        elif "AVAILABLE skills" in text: answer="<choice>DeceptionDeployment</choice>"
        elif "Skill Doctor" in text:
            answer=json.dumps({"is_related":True,"semantic_gradient":{"initiation":"Only observable evidence","policy":"Verify action result","termination":"Wait for completion"}})
        elif "Skill Evolver" in text:
            answer=json.dumps({"skill_name":"Updated","initiation":"When available","policy":["Analyse observed host"],"termination":"When analysis completes"})
        else: answer=json.dumps({"action":"Sleep","reason":"Deterministic TEST FIXTURE, no real model"})
        self.calls.append({"purpose":purpose})
        return {"text":answer,"token_ids":[1,2],"identity":self.identity,"prompt_tokens":10,"output_tokens":2,"seconds":0}
    def score(self,samples):
        self.score_calls+=1
        return [-1.0]*len(samples)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(socket.socket,"connect",lambda *a,**k: (_ for _ in ()).throw(AssertionError("Unexpected network call in mock test")))
    model=TestModel(); configure_encoder(model.embed)
    return model


def experience(reward=4, count=2, **kw):
    transitions=[{"step":i,"return_to_go":float(count-i),"state":json.dumps([{"role":"system","content":"base"},{"role":"user","content":"visible"}]),
        "token_ids":[1],"action":"test_action","policy_invoked":True,"request_applied":True,"parse_error":None} for i in range(count)]
    return Experience(reward=reward,skill="DeceptionDeployment",trajectory="full step history",transitions=transitions,env_name="test")


def test_official_gate_matches_manual_clipped_formula():
    exps=[experience(4,2),experience(0,1)]
    score,_=gate_score(exps,[0,0,0],np.log([1.5,1.5,.5]),{"test":2})
    assert score==pytest.approx((1.2+1.2-1.6)/3)


def test_paper_rtg_is_distinct_and_exact():
    assert returns_to_go([1,2,3],.5)==[2.75,3.5,3]
    exp=experience(4,2)
    score,_=gate_score([exp],[0,0],[0,0],{"test":.5},mode="paper_rtg")
    assert score==pytest.approx(((2-.5)+(1-.5))/2)


def test_official_dispatcher_parity(monkeypatch):
    pool=SecuritySkillPool(); parent=pool.skills[3]
    evolver=OriginalEvolution(None,threshold=1)
    candidate=Skill(name="next",initiation="valid",policy=["one"],termination="done")
    monkeypatch.setattr(evolver,"evolve",lambda *a:(candidate,"REFINE"))
    monkeypatch.setattr(evolver,"_compute_skill_logprobs",lambda s,*a:np.zeros(2) if s is parent else np.log([1.5,1.5]))
    exp=experience(); exp.skill=parent.name
    log=evolver.run_skill_evolution_with_verification(pool,[exp],None,{"test":2},best_of_n=1,acceptance_margin=0)
    ours,_=gate_score([exp],[0,0],np.log([1.5,1.5]),{"test":2})
    assert log[0]["j_score"]==pytest.approx(ours)


def test_bad_probability_fails_closed(offline,monkeypatch):
    pool=SecuritySkillPool(); original=deepcopy(pool.snapshot())
    ev=Evolution(offline,{"threshold":1,"best_of_n":1})
    monkeypatch.setattr(ev,"probabilities",lambda *a:(_ for _ in ()).throw(ValueError("missing logprob")))
    result=ev.update(pool,[experience()],{"test":0},1)
    assert result["evolution"][0]["accepted"] is False
    assert "missing logprob" in result["evolution"][0]["error"]
    assert pool.snapshot()==original


def test_no_gate_ablation_never_computes_logprob(offline,monkeypatch):
    pool=SecuritySkillPool(); ev=Evolution(offline,{"threshold":1,"best_of_n":1,"ppo_gate":False})
    candidate=Skill(name="candidate",initiation="valid",policy=["one"],termination="done")
    monkeypatch.setattr(ev,"evolve",lambda *a:(candidate,"REFINE"))
    monkeypatch.setattr(ev,"probabilities",lambda *a:(_ for _ in ()).throw(AssertionError("Gate disabled")))
    result=ev.update(pool,[experience()],{"test":0},1)
    assert result["evolution"][0]["accepted"]
    assert any(s.name=="candidate" for s in pool.skills)


def test_no_gradient_ablation_uses_summary_path(offline,monkeypatch):
    pool=SecuritySkillPool();ev=Evolution(offline,{"threshold":1,"best_of_n":1,"ppo_gate":False,"semantic_gradient":False})
    seen=[]
    candidate=Skill(name="summary_candidate",initiation="valid",policy=["one"],termination="done")
    monkeypatch.setattr(ev,"evolve_without_gradient",lambda *a:(seen.append(True) or candidate,"NEW"))
    monkeypatch.setattr(ev,"evolve",lambda *a:(_ for _ in ()).throw(AssertionError("Semantic gradient disabled")))
    ev.update(pool,[experience()],{"test":0},1)
    assert seen


def test_fifo_ablation_is_a_real_maintenance_switch(offline,monkeypatch):
    pool=SecuritySkillPool();seen=[]
    monkeypatch.setattr(pool,"maintain",lambda *a:(_ for _ in ()).throw(AssertionError("Score disabled")))
    monkeypatch.setattr(pool,"maintain_fifo",lambda:seen.append(True) or [])
    Evolution(offline,{"score_maintenance":False}).update(pool,[],{},1)
    assert seen


def test_semantic_gradient_receives_all_step_history(offline,monkeypatch):
    ev=Evolution(lambda x: "unused",{"semantic_chunk_steps":2})
    seen=[]
    def gradient(self,skill,exp):
        seen.extend(json.loads(exp.trajectory))
        return {"initiation":"i","execution":"p","termination":"t"},True
    monkeypatch.setattr(OriginalEvolution,"_generate_semantic_gradient",gradient)
    exp=experience(count=5)
    for t in exp.transitions: t.update(observation_before={"visible":t["step"]},observation_after={"visible":t["step"]+1})
    ev._generate_semantic_gradient(Skill(),exp)
    assert [r["step"] for r in seen]==list(range(5))
    assert all("observation_before" in r and "observation_after" in r for r in seen)


def test_reward_ledger_matches_official_loop_without_fivefold_sum():
    rewards=[-1,-2,-9]; ledger=JointRewardLedger()
    official=[]
    for i,r in enumerate(rewards):
        terminal=i==2
        ledger.add({f"blue_agent_{j}":r for j in range(5)},terminal)
        if terminal: break
        official.append(r)
    assert ledger.total==sum(official)==-3
    assert sum(ledger.raw_rewards)==-12


def test_skills_off_has_same_prompt_and_action_as_acd(offline):
    c=validate({"profile":"paper-compatible","skip_busy_policy":False,"steps":6})
    env=make_environment(c,42);env.reset(seed=42)
    policy=DefensePolicy("blue_agent_4",offline,None,c)
    raw=env.raw_observations()["blue_agent_4"]
    state=make_state(env,"blue_agent_4",raw,policy.base)
    original_requests=[]
    class LegacyBackend:
        def generate(self,messages,**kwargs):
            original_requests.append(deepcopy(messages))
            return offline.generate(messages)["text"]
    original=LLMDefender("blue_agent_4",LegacyBackend())
    assert messages_for(state,policy.base)[0]==original.system_prompt
    action,detail=policy.decide(state,episode=0,step=0,seed=42)
    expected,old_detail=original.decide(raw,hosts=state["hosts"],subnets=state["subnets"],episode=0,step=0)
    assert original_requests[0]==messages_for(state,policy.base)
    assert str(action)==str(expected)
    for key in ("response","reason","parse_error","user_prompt"): assert detail[key]==old_detail[key]


def test_controllers_are_independent_and_persist_multiple_steps(offline):
    pool=SecuritySkillPool()
    def model(prompt,**kwargs): return "<status>CONTINUE</status>" if "<status>" in prompt else "<choice>DeceptionDeployment</choice>"
    first=SkillController(pool,model,{"epsilon":0,"max_skill_age":3})
    other=SkillController(pool,model,{"epsilon":0,"max_skill_age":3})
    state={"user_prompt":"visible","observation":{"phase":0},"allowed_actions":["DeployDecoy"]}
    skill=first.choose(state,0);assert first.choose(state,1) is skill
    assert other.active is None
    first.choose(state,3);assert first.last_event["events"][0]["reason"]=="maximum_age"


def test_proposed_checks_do_not_use_hidden_state(offline):
    skill=SecuritySkillPool().skills[2]
    state={"user_prompt":"No suspicious events","observation":{"phase":0,"message":[[0]*8]*4},"allowed_actions":["Restore","Analyse"]}
    ok,reason=CAGEVerifier().check(skill,state,[])
    assert not ok and reason=="no_observed_admin_compromise"


def test_training_eval_seed_overlap_rejected():
    with pytest.raises(ValueError,match="overlap"): validate({"train_seed":42,"eval_seed":42})


def test_resume_and_frozen_evaluation(offline,tmp_path):
    c=validate({"arm":"E1","composition":"all_llm","steps":6,"episodes":2,"eval_seed":10000})
    first=run(c,output=tmp_path/"eval",limit=1,client=offline)
    frozen=(tmp_path/"eval/frozen_pool.json").read_bytes()
    second=run(c,output=tmp_path/"eval",resume=True,client=TestModel())
    assert first["status"]=="partial" and second["status"]=="completed"
    assert len(second["episodes"])==2
    assert (tmp_path/"eval/frozen_pool.json").read_bytes()==frozen
    assert second["result_kind"]=="mock_test"


def test_connection_failure_produces_no_success_score(monkeypatch,tmp_path):
    c=validate({"steps":6})
    with pytest.raises(Exception): run(c,output=tmp_path/"failure")
    assert json.loads((tmp_path/"failure/failure.json").read_text())["status"]=="failed"
    assert not (tmp_path/"failure/summary.json").exists()


def test_hidden_simulator_fields_are_rejected():
    from research.adapter import validate_visible_observation
    with pytest.raises(ValueError,match="Hidden"):
        validate_visible_observation({"success":"TRUE","hidden_red_sessions":{"host":"root"}})


def test_pending_training_batch_resume_does_not_repeat_episode(offline,tmp_path):
    c=validate({"arm":"E2","composition":"all_llm","steps":6,"episodes":2,
        "train_episodes":2,"iterations":1,"evolution":{"threshold":99}})
    path=tmp_path/"train"
    run(c,stage="train",output=path,limit=1,client=offline)
    first=(path/"episode-0000.json").read_bytes()
    result=run(c,stage="train",output=path,resume=True,client=TestModel())
    assert (path/"episode-0000.json").read_bytes()==first
    assert result["status"]=="completed"
    assert json.loads((path/"checkpoint.json").read_text())["episodes_completed"]==2
    checkpoint=json.loads((path/"checkpoint.json").read_text())
    observed={t["seed"] for e in checkpoint["experience"]["fifo"] for t in e["transitions"]}
    assert observed <= {100,101} and 5000 not in observed
    assert result["history"][0]["validation_episodes"][0]["seed"]==5000


def test_paper_online_score_uses_only_active_scored_steps(offline):
    from research.paper_algorithm import update_online_scores
    pool=SecuritySkillPool(); skill=pool.skills[3]
    exp=experience(); exp.skill=skill.name
    exp.transitions=[{"joint_reward":4,"score_included":True,"skill_controller":{"events":[{"activated":skill.name}]}},
                     {"joint_reward":2,"score_included":True},
                     {"joint_reward":-100,"score_included":False}]
    update_online_scores(pool,[exp],1)
    assert skill.total_gain==2 and skill.frequency==1 and skill.avg_gain==2
    assert all(s.frequency==0 for s in pool.skills if s is not skill)


def test_paper_running_baseline_uses_complete_joint_series_once():
    from research.paper_algorithm import update_running_baselines
    baselines={}
    config={"gamma":.5,"ema_alpha":.1}
    update_running_baselines([{"scored_step_rewards":[1,2,0]}],baselines,"task",config)
    assert baselines["task"]==pytest.approx(4/3)
    assert baselines["task:step"]==1
    update_running_baselines([{"scored_step_rewards":[0,0,0]}],baselines,"task",config)
    assert baselines["task"]==pytest.approx(1.2)
    assert baselines["task:step"]==pytest.approx(.9)


def test_paper_pruning_never_labels_unobserved_skill_as_measured_gain(offline):
    from research.paper_algorithm import maintain_paper
    pool=SecuritySkillPool(); unseen=pool.skills[0]; loser=pool.skills[1]
    loser.frequency=1; loser.avg_gain=-1
    logs=maintain_paper(pool)
    assert unseen in pool.skills and unseen.frequency==0 and unseen.avg_gain==0
    assert loser not in pool.skills
    assert any(log["stage"]=="paper_nonpositive_observed_score" for log in logs)


def test_uncommitted_training_batch_cannot_be_evaluated(offline,tmp_path):
    c=validate({"arm":"E2","composition":"all_llm","steps":6,"episodes":2,
        "train_episodes":2,"iterations":1,"evolution":{"threshold":99}})
    run(c,stage="train",output=tmp_path/"train",limit=1,client=offline)
    c["pool"]=str(tmp_path/"train/checkpoint.json")
    with pytest.raises(ValueError,match="uncommitted"):
        run(c,output=tmp_path/"eval",client=TestModel())


def test_bad_termination_response_retains_skill_and_records_failure(offline):
    pool=SecuritySkillPool()
    def model(prompt,**kwargs):
        return "not valid XML" if "<status>" in prompt else "<choice>DeceptionDeployment</choice>"
    controller=SkillController(pool,model,{"epsilon":0,"max_skill_age":8})
    state={"user_prompt":"visible","observation":{"phase":0},"allowed_actions":["DeployDecoy"]}
    active=controller.choose(state,0)
    assert controller.choose(state,1) is active
    assert controller.last_event["events"][0]["error"]=="invalid_termination_response"


def test_reports_exclude_mock_rewards_and_preserve_unexecuted_cells(tmp_path):
    from research.reporting import report
    import csv
    (tmp_path/"summary.json").write_text(json.dumps({"config":{},"stage":"eval", "result_kind":"mock_test",
        "episodes":[{"joint_return":999999}]}))
    diagnostics=report(tmp_path)
    assert diagnostics["excluded_mock_summaries"]
    rows=list(csv.reader((tmp_path/"report/table1_overall.csv").open()))
    assert rows[1][3:]==["N/A"]*4
    assert "999999" not in (tmp_path/"report/table1_overall.csv").read_text()
    assert (tmp_path/"report/table4_ablation.csv").exists()


def test_training_continuation_keeps_pool_experience_and_seed_provenance(offline,tmp_path):
    from research.config import digest
    c=validate({"arm":"E2","composition":"all_llm","steps":6,"episodes":2,
        "train_episodes":1,"iterations":1,"validation_episodes":0,"evolution":{"threshold":99}})
    run(c,stage="train",output=tmp_path/"first",client=offline)
    previous=json.loads((tmp_path/"first/checkpoint.json").read_text())
    next_config=validate({**c,"train_seed":101,"train_episodes":4,"iterations":2,
                          "pool":str(tmp_path/"first/checkpoint.json")})
    output=tmp_path/'continued'
    run(next_config,stage='train',output=output,client=TestModel(),limit=2)
    intermediate=json.loads((output/'pool-iteration-0002.json').read_text())
    assert intermediate['training_completed'] is False
    assert intermediate['segment_completed_iterations']==1
    assert json.loads((output/'episode-0000.json').read_text())['input_pool_sha256']==digest(previous['pool'])
    run(next_config,stage='train',output=output,client=TestModel(),resume=True)
    continued=json.loads((tmp_path/"continued/checkpoint.json").read_text())
    assert [entry['iteration'] for entry in continued['history']]==[1,2,3]
    assert continued['inherited_optimizer_iterations']==1
    assert continued['segment_completed_iterations']==2
    assert continued['total_committed_iterations']==3
    assert continued['training_seeds']==[100,101,102,103,104]
    assert continued['total_training_episodes']==5
    assert continued['segment_training_episodes']==continued['episodes_completed']==4
    assert json.loads((output/'episode-0002.json').read_text())['input_pool_sha256']==digest(intermediate['pool'])
    assert json.loads((tmp_path/"continued/initial_pool.json").read_text())["pool"]==previous["pool"]
    assert len(continued["experience"]["fifo"])>=len(previous["experience"]["fifo"])


def test_resume_after_evolution_failure_reuses_completed_rollout(offline,tmp_path,monkeypatch):
    import research.experiment as runner
    c=validate({"arm":"E2","composition":"all_llm","steps":6,"episodes":2,
        "train_episodes":2,"iterations":1,"validation_episodes":0,"evolution":{"threshold":99}})
    directory=tmp_path/"train"
    run(c,stage="train",output=directory,limit=1,client=offline)
    original_update=Evolution.update
    monkeypatch.setattr(Evolution,"update",lambda *a,**k:(_ for _ in ()).throw(RuntimeError("interrupted before batch commit")))
    with pytest.raises(RuntimeError,match="interrupted"):
        run(c,stage="train",output=directory,resume=True,client=TestModel())
    collected=(directory/"episode-0001.json").read_bytes()
    trajectory=(directory/"episode-0001.jsonl").read_bytes()
    monkeypatch.setattr(Evolution,"update",original_update)
    monkeypatch.setattr(runner,"run_episode",lambda *a,**k:(_ for _ in ()).throw(AssertionError("Completed rollout must not be repeated")))
    result=run(c,stage="train",output=directory,resume=True,client=TestModel())
    assert result["status"]=="completed"
    assert collected==(directory/"episode-0001.json").read_bytes()
    assert trajectory==(directory/"episode-0001.jsonl").read_bytes()
    record=json.loads((directory/"interrupted/recovered-episode-0001.json").read_text())
    assert record["behavior_samples_unchanged"] and not record["legacy_evolution_rng_reseeded"]


@pytest.mark.parametrize("identity",[
    {"model":"unexpected-model","revision":"b968826d9c46dd6066d109eabc6255188de91218","weights_frozen":True},
    {"model":"Qwen/Qwen3-8B","revision":"b968826d9c46dd6066d109eabc6255188de91218","weights_frozen":False},
])
def test_wrong_or_trainable_policy_backend_is_rejected(monkeypatch,identity):
    from research.client import ModelClient,ModelFailure
    monkeypatch.setattr(ModelClient,"_request",lambda *a,**k:identity)
    with pytest.raises(ModelFailure):ModelClient(validate({})["model"])


@pytest.mark.parametrize('boundary',['sg_chunk','candidate_saved','score_sample'])
def test_durable_optimizer_resume_matches_continuous_without_repeating_calls(offline,tmp_path,monkeypatch,boundary):
    """Fault injection is contract verification, never Qwen/CAGE4 performance."""
    from research.client import ModelClient,TextModel
    from research.progress import BudgetExceeded,Journal
    backend={'completed':[],'fail':None,'failed':False,'new_scores':0}
    identity={'model':'Qwen/Qwen3-8B','revision':'fixture','weights_frozen':True,
        'scoring_mode':'incremental_teacher_forcing','dtype':'fixture','chat_template_sha256':'fixture'}
    def request(self,route,payload=None,endpoint=None):
        if route=='health':return identity
        if route=='tokenize':
            return {'prompt_token_ids':list(json.dumps(payload['messages'],ensure_ascii=False).encode()),'identity':identity,'prompt_tokens':30,'max_context':24576}
        if route=='generate':
            stage=payload['request_context'].get('stage')
            if backend['fail']=='sg_chunk' and stage=='sg_chunk' and payload['request_context'].get('chunk')==1 and not backend['failed']:
                backend['failed']=True;raise BudgetExceeded('Injected SG interruption')
            text=payload['messages'][-1]['content']
            answer=json.dumps({'is_related':True,'semantic_gradient':{'initiation':'','policy':'Inspect actual outcome','termination':''}}) if 'Skill Doctor' in text else json.dumps({'skill_name':'next','initiation':'When observed','policy':['Inspect actual outcome'],'termination':'After verification'})
            backend['completed'].append(('generate',payload['request_id'],payload['seed']))
            return {'text':answer,'token_ids':[1,2],'identity':identity,'prompt_tokens':30,'output_tokens':2,'seconds':.01}
        if route=='score':
            stage=payload['request_context'].get('stage')
            if stage=='new_score':
                backend['new_scores']+=1
                stop=(backend['fail']=='candidate_saved' and backend['new_scores']==1 or backend['fail']=='score_sample' and backend['new_scores']==2)
                if stop and not backend['failed']:
                    backend['failed']=True;raise BudgetExceeded('Injected scoring interruption')
            backend['completed'].extend(('score',key,None) for key in payload['request_ids'])
            return {'identity':identity,'scores':[{'sum':-1.0,'tokens':len(s['token_ids'])} for s in payload['samples']]}
        raise AssertionError(route)
    monkeypatch.setattr(ModelClient,'_request',request)
    exp=experience(count=3)
    for t in exp.transitions:
        t['state']=json.dumps([{'role':'system','content':'base'},{'role':'user','content':str(t['step'])}])
        t['observation_before']={'z':{'q':3,'a':1},'b':4}  # Noncanonical source order must survive resume.
    evolution_config={'schedule':'paper_batch','gate_mode':'paper_rtg','maintenance_mode':'paper_algorithm1',
        'best_of_n':2,'semantic_chunk_steps':1,'semantic_chunk_chars':48000}
    def execute(db):
        client=ModelClient({'endpoint':'http://fixture','model':identity['model'],'revision':'fixture','journal_path':str(db),
            'request_context':{'experiment':'fixture','arm':'B2'}})
        text=TextModel(client);text.purpose='evolution'
        pool=SecuritySkillPool(24);ev=Evolution(text,evolution_config)
        result=ev.update(pool,[exp],{'test':0.},1)
        return pool.snapshot(),result,text.snapshot()
    expected,complete,expected_counter=execute(tmp_path/'continuous.sqlite3')
    expected_requests=set(backend['completed'])
    backend.update(completed=[],fail=boundary,failed=False,new_scores=0)
    db=tmp_path/'resume.sqlite3'
    with pytest.raises(BudgetExceeded):execute(db)
    before=list(backend['completed'])
    assert before and Journal(db).tasks(('completed',))
    actual,resumed,counter=execute(db)
    assert actual==expected and counter==expected_counter
    assert resumed['evolution']==complete['evolution']
    assert resumed['optimizer_status']=='completed_changed'
    ids=[(route,key) for route,key,seed in backend['completed']]
    assert len(ids)==len(set(ids))  # Every completed logical response/sample is reused.
    assert set(backend['completed'])==expected_requests  # Exact prompts/seeds match continuous execution.
    assert all(item in backend['completed'] for item in before)
    after=len(backend['completed']);execute(db)
    assert len(backend['completed'])==after  # A committed loop is idempotent too.


def test_sg_parallel_reservations_and_deterministic_merge_match_serial(offline):
    from research.client import TextModel
    offline.config={'sg_workers':1};offline.journal=None
    def gradients(workers):
        offline.config['sg_workers']=workers;seen=[]
        original=offline.generate
        def generate(messages,**kwargs):
            seen.append((kwargs['seed'],hashlib.sha256(messages[0]['content'].encode()).hexdigest()))
            return original(messages,**kwargs)
        offline.generate=generate
        try:
            text=TextModel(offline);text.purpose='evolution';ev=Evolution(text,{'schedule':'paper_batch','semantic_chunk_steps':1})
            exp=experience(count=5);parent=SecuritySkillPool().skills[3]
            result=ev._generate_semantic_gradient(parent,exp)
            return result,ev.gradients,text.snapshot(),seen
        finally:offline.generate=original
    serial=gradients(1);parallel=gradients(3)
    assert serial[:3]==parallel[:3] and sorted(serial[3])==sorted(parallel[3])


def test_journal_rejects_live_duplicate_owner_and_preserves_budget_on_restart(tmp_path):
    from research.progress import Journal,TaskInProgress,BudgetExceeded
    db=tmp_path/'journal.sqlite3';j=Journal(db)
    payload={'z':{'q':3,'a':1},'b':4};j.checkpoint('ordered',payload)
    assert json.dumps(Journal(db).restore('ordered'))==json.dumps(payload)
    assert j.start('request','sg_chunk',{'seed':1})
    with pytest.raises(TaskInProgress):Journal(db).start('request','sg_chunk',{'seed':1})
    j.finish('request',{'exact':'response'})
    assert not j.start('request','sg_chunk',{'seed':1})
    j.budget_configure(8,36000,now=100)
    j.budget_start('two',2,now=100);j.budget_heartbeat('two',stop=True,now=1900)
    resumed=Journal(db);resumed.budget_configure(8,36000,now=1900)
    resumed.budget_start('one',1,now=1900)
    assert resumed.budget_usage(now=5500)['gpu_hours']==2
    with pytest.raises(ValueError):resumed.budget_configure(8,72000,now=5500)
    with pytest.raises(BudgetExceeded):resumed.check_budget(now=27101)


def test_preparation_resume_does_not_apply_running_baseline_or_online_scores_twice(offline,tmp_path,monkeypatch):
    from research.progress import Journal,BudgetExceeded
    c=validate({'arm':'E2','composition':'all_llm','steps':6,'episodes':1,'train_episodes':2,'iterations':1,
        'validation_episodes':0,'evolution':{'schedule':'paper_batch','gate_mode':'paper_rtg','maintenance_mode':'paper_algorithm1',
        'best_of_n':1,'semantic_chunk_steps':2}})
    def model(directory):
        obj=TestModel();obj.journal=Journal(directory/'journal.sqlite3');obj.base_context={'experiment':'fixture','arm':'B2'}
        return obj
    continuous=tmp_path/'continuous';continuous.mkdir()
    run(c,stage='train',output=continuous,client=model(continuous))
    expected=json.loads((continuous/'checkpoint.json').read_text())
    resumed=tmp_path/'resumed';resumed.mkdir()
    original=Evolution.update
    def interrupted(*args,**kwargs):raise BudgetExceeded('After baseline preparation, before optimizer')
    monkeypatch.setattr(Evolution,'update',interrupted)
    with pytest.raises(BudgetExceeded):run(c,stage='train',output=resumed,client=model(resumed))
    traces={p.name:p.read_bytes() for p in resumed.glob('episode-????.jsonl')}
    monkeypatch.setattr(Evolution,'update',original)
    run(c,stage='train',output=resumed,resume=True,client=model(resumed))
    actual=json.loads((resumed/'checkpoint.json').read_text())
    assert actual['baselines']==expected['baselines'] and actual['pool']==expected['pool']
    assert actual['experience']==expected['experience']
    assert {p.name:p.read_bytes() for p in resumed.glob('episode-????.jsonl')}==traces


@pytest.mark.parametrize('minimum,status',[(0,'completed_unchanged'),(2,'completed_no_eligible_skill')])
def test_completed_optimizer_status_is_independent_of_pool_change(offline,tmp_path,minimum,status):
    from research.progress import Journal
    from research.client import TextModel
    offline.journal=Journal(tmp_path/'journal.sqlite3');offline.base_context={'experiment':'fixture','arm':'B3' if minimum else 'B2'}
    model=TextModel(offline);model.purpose='evolution';pool=SecuritySkillPool(24)
    before=deepcopy(pool.snapshot());exp=experience()
    for row in exp.transitions:row.update(seed=11,episode_id=0)
    ev=Evolution(model,{'schedule':'paper_batch','gate_mode':'paper_rtg','maintenance_mode':'paper_algorithm1',
        'best_of_n':2,'minimum_independent_episodes':minimum})
    result=ev.update(pool,[exp],{'test':100.},1)
    assert result['optimizer_status']==status and not result['pool_changed'] and pool.snapshot()==before
    if minimum:assert not offline.calls and result['evolution'][0]['deferred']
    else:assert len(result['evolution'][0]['candidates'])==2 and not any(c['accepted'] for c in result['evolution'][0]['candidates'])


def test_public_variant_comparison_uses_distinct_configs_and_directories(offline,tmp_path,monkeypatch):
    import sys
    from research.__main__ import main
    seen=[]
    def record(config,stage,output,**kwargs):
        seen.append((config['comparison_id'],config['evolution']['minimum_independent_episodes'],stage,str(output)))
        return {'status':'completed'}
    monkeypatch.setattr('research.experiment.run',record)
    monkeypatch.setattr('scripts.report_experiment.report_experiment',lambda *a,**kw:None)
    monkeypatch.setattr(sys,'argv',['research','compare','--variants','B3-no-support','B3','--reds','finite','--train','--output',str(tmp_path)])
    main()
    assert [(label,minimum,stage) for label,minimum,stage,path in seen]==[
        ('B3-no-support',0,'train'),('B3-no-support',0,'eval'),('B3',2,'train'),('B3',2,'eval')]
    assert len({path for label,minimum,stage,path in seen})==4
