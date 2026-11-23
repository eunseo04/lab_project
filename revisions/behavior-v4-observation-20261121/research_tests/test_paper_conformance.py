"""Regression tests for paper equations, batch evidence, and CAGE4 execution."""
from copy import deepcopy
import hashlib
import json
import math

import numpy as np
import pytest

from research.client import ModelClient, ModelFailure
from research.adapter import DefensePolicy
from research.config import validate
from research.evolution import Evolution
from research.gate import gate_score
from research.paper_algorithm import maintain_paper, update_online_scores, update_running_baselines
from research.skills import CAGEVerifier, SecuritySkillPool
from research.upstream import Experience, OriginalEvolution, Skill
from test_research import offline, experience, TestModel


def test_paper_batch_visits_all_invoked_skills_using_only_current_evidence(offline, monkeypatch):
    pool = SecuritySkillPool()
    a, b, old = pool.skills[:3]
    a.frequency = b.frequency = 40
    ev = Evolution(offline, {"schedule":"paper_batch", "gate_mode":"paper_rtg",
        "maintenance_mode":"paper_algorithm1", "threshold":99, "max_evolutions":1, "best_of_n":1})
    historical = experience(); historical.skill = old.name
    ev.experience_buffer[old.name] = [historical]*100
    batch = []
    for sk in (a,b):
        exp = experience(); exp.skill = sk.name; batch.append(exp)
    monkeypatch.setattr(ev,"probabilities",lambda sk,exps:np.zeros(sum(len(e.transitions) for e in exps)))
    monkeypatch.setattr(ev,"evolve",lambda sk,exps:(Skill(name=sk.name+"_v2",initiation="visible",policy=["Act"],termination="done",version=2),"REFINE"))
    logs = ev.update(pool,batch,{"test":0},1)["evolution"]
    assert [x["parent"] for x in logs] == [a.name,b.name]
    assert all(x["evidence_source"]=="current_training_batch" and x["evidence_trajectories"]==1 for x in logs)
    assert all(x["accepted"] for x in logs)


def test_semantic_gradients_extracted_once_for_best_of_n(offline, monkeypatch):
    ev = Evolution(None, {"schedule":"paper_batch", "semantic_chunk_steps":2})
    exp = experience(count=5); calls=[]
    def gradient(self, skill, local):
        calls.append(json.loads(local.trajectory))
        return {"initiation":"i","execution":"p","termination":"t"},True
    monkeypatch.setattr(OriginalEvolution,"_generate_semantic_gradient",gradient)
    monkeypatch.setattr(ev,"_run_semantic_update",lambda *args:{"skill_name":"refined","initiation":"visible","policy":["Act"],"termination":"done"})
    parent = Skill(name="parent",version=1)
    ev.evolve(parent,[exp]); ev.evolve(parent,[exp]); ev.evolve(parent,[exp])
    assert len(calls)==3  # Three chunks total, not three chunks per candidate.
    assert [t["step"] for chunk in calls for t in chunk]==list(range(5))
    assert all("execution_contract" in chunk[0] for chunk in calls)


@pytest.mark.parametrize("suffix_already_exists",[False,True])
def test_colliding_candidate_is_scored_with_its_final_admitted_text(offline,monkeypatch,suffix_already_exists):
    pool=SecuritySkillPool(max_size=30);parent=pool.skills[0]
    base=parent.name+"_v2"
    for name in [base]+([base+"_c0"] if suffix_already_exists else []):
        pool.add_skill(Skill(name=name,initiation="visible",policy=["Act"],termination="done",version=2))
    ev=Evolution(offline,{"schedule":"paper_batch","gate_mode":"paper_rtg",
        "maintenance_mode":"paper_algorithm1","best_of_n":2})
    scored=[]
    def probabilities(skill,experiences):
        scored.append((skill.name,skill.format_for_llm()))
        return np.zeros(sum(len(e.transitions) for e in experiences))
    monkeypatch.setattr(ev,"probabilities",probabilities)
    monkeypatch.setattr(ev,"evolve",lambda sk,exps:(Skill(name=base,initiation="visible",policy=["Act"],termination="done",version=2),"REFINE"))
    exp=experience();exp.skill=parent.name
    log=ev.update(pool,[exp],{"test":0},1)["evolution"][0]
    accepted=next(c for c in log["candidates"] if c["accepted"])
    admitted=next(s for s in pool.skills if s.name==accepted["skill"]["name"])
    expected=base+("_c0_1" if suffix_already_exists else "_c0")
    assert admitted.name==expected
    assert scored[1]==(admitted.name,admitted.format_for_llm())
    assert scored[2][0]==base+"_c1"
    assert len({s.name for s in pool.skills})==len(pool.skills)


def test_no_gate_ablation_admits_siblings_with_distinct_logged_names(offline,monkeypatch):
    pool=SecuritySkillPool(max_size=30);parent=pool.skills[0]
    ev=Evolution(offline,{"schedule":"paper_batch","gate_mode":"paper_rtg",
        "maintenance_mode":"paper_algorithm1","best_of_n":2,"ppo_gate":False})
    monkeypatch.setattr(ev,"probabilities",lambda *args:pytest.fail("No-gate must not score probabilities"))
    monkeypatch.setattr(ev,"evolve",lambda sk,exps:(Skill(name=sk.name+"_v2",initiation="visible",policy=["Act"],termination="done",version=2),"REFINE"))
    exp=experience();exp.skill=parent.name
    log=ev.update(pool,[exp],{"test":0},1)["evolution"][0]
    accepted=[c["skill"]["name"] for c in log["candidates"] if c["accepted"]]
    assert accepted==[parent.name+"_v2",parent.name+"_v2_c1"]
    assert len({s.name for s in pool.skills})==len(pool.skills)
    assert set(accepted)<={s.name for s in pool.skills}


def state_and_history(action="Analyse host", invoked=True):
    raw={"phase":0,"success":"TRUE","message":[[0]*8]*4}
    state={"observation":raw,"user_prompt":"Visible observation", "allowed_actions":["Analyse"]}
    evidence={k:v for k,v in raw.items() if k not in ("action","phase","message")}
    fingerprint=hashlib.sha256(json.dumps(evidence,sort_keys=True).encode()).hexdigest()
    row={"requested_action":action,"policy_invoked":invoked,"request_applied":invoked,
        "parse_error":None,"evidence_hash":fingerprint,"skill_id":"HostInvestigation"}
    return state,[deepcopy(row),deepcopy(row)]


@pytest.mark.parametrize("invoked",[True,False])
def test_sleep_and_busy_waiting_are_not_repetition_failures(offline, invoked):
    state, history=state_and_history("Sleep",invoked)
    assert CAGEVerifier().check(SecuritySkillPool().skills[0],state,history)[0]


def test_repetition_requires_unchanged_current_observation(offline):
    skill=SecuritySkillPool().skills[0]
    state,history=state_and_history()
    assert CAGEVerifier().check(skill,state,history)==(False,"repeated_action_without_new_evidence")
    state["observation"]["observed_host"]={"Processes":[{"PID":123}]}
    assert CAGEVerifier().check(skill,state,history)[0]


def test_paper_ratio_uses_unmodified_probability_ratio():
    exp=experience(count=1); exp.transitions[0]["return_to_go"]=0
    paper,_=gate_score([exp],[0],[20],{"test":1},mode="paper_rtg")
    assert paper==pytest.approx(-math.exp(20))
    legacy,_=gate_score([exp],[0],[20],{"test":5},mode="official_code")
    assert legacy==pytest.approx(-math.exp(10))


def test_paper_first_batch_centers_negative_rewards_and_excludes_terminal(offline):
    baseline={}
    update_running_baselines([{"scored_step_rewards":[-2,-2,0],"scored_transitions":2}],baseline,"test",{"gamma":1,"ema_alpha":.1})
    assert baseline=={"test":-3,"test:step":-2}
    pool=SecuritySkillPool(); skill=pool.skills[0]
    exp=Experience(reward=-1,skill=skill.name,trajectory="",transitions=[
        {"joint_reward":-1,"score_included":True,"skill_controller":{"events":[{"activated":skill.name}]}},
        {"joint_reward":999,"score_included":False,"skill_controller":{"events":[{"activated":skill.name}]}}])
    update_online_scores(pool,[exp],baseline["test:step"])
    assert skill.frequency==1 and skill.avg_gain==1


def test_algorithm1_defers_pruning_until_capacity_is_exceeded(offline):
    pool=SecuritySkillPool(); pool.skills[0].frequency=1; pool.skills[0].avg_gain=-1
    original=deepcopy(pool.snapshot())
    assert maintain_paper(pool,capacity_only=True)==[]
    assert pool.snapshot()==original
    pool.max_size=3
    logs=maintain_paper(pool,capacity_only=True)
    assert len(pool.skills)<=3 and any(x["name"]==original["skills"][0]["name"] for x in logs)


def test_parallel_scoring_preserves_order_cache_and_checks_real_rescoring(monkeypatch):
    identity={"model":"Qwen/Qwen3-8B","revision":"pinned","dtype":"bfloat16",
        "chat_template_sha256":"same","scoring_mode":"incremental_teacher_forcing","weights_frozen":True}
    requests=[]
    def request(self, route, payload=None, endpoint=None):
        if route=="health":return identity
        if route=="generate":return {"identity":identity,"text":"x","token_ids":[11],"prompt_tokens":1,"output_tokens":1,"seconds":0,"generation_logprob":-11.}
        requests.append((endpoint,payload["samples"][0]["token_ids"][0]))
        return {"identity":identity,"scores":[{"sum":-float(payload["samples"][0]["token_ids"][0])}]}
    monkeypatch.setattr(ModelClient,"_request",request)
    client=ModelClient({"endpoint":"http://one","scoring_endpoints":["http://two"],"model":identity["model"],"revision":"pinned"})
    messages=[{"role":"user","content":"visible"}]
    client.generate(messages,return_logprobs=True)
    samples=[{"messages":messages,"token_ids":[n]} for n in (11,3,2)]
    assert client.score(samples)==[-11.,-3.,-2.]
    assert len(requests)==2
    assert client.score(samples,use_cache=False)==[-11.,-3.,-2.]
    assert len(requests)==5


def test_replica_with_different_template_is_rejected(monkeypatch):
    identity={"model":"Qwen/Qwen3-8B","revision":"pinned","dtype":"bfloat16",
        "chat_template_sha256":"same","scoring_mode":"incremental_teacher_forcing","weights_frozen":True}
    monkeypatch.setattr(ModelClient,"_request",lambda self,route,payload=None,endpoint=None:
        {**identity,"chat_template_sha256":"different"} if endpoint else identity)
    with pytest.raises(ModelFailure,match="Scorer weights"):
        ModelClient({"endpoint":"http://one","scoring_endpoints":["http://two"],"model":identity["model"],"revision":"pinned"})


def test_paper_profile_cannot_silently_mix_legacy_gate_or_maintenance():
    with pytest.raises(ValueError,match="paper_batch requires"):
        validate({"evolution":{"schedule":"paper_batch"}})


def test_explicit_contract_is_shared_by_baseline_and_proposed(offline):
    state={"agent":"blue_agent_4","observation":{"phase":0,"success":"TRUE","message":[[0]*8]*4},
        "user_prompt":"Visible state", "hosts":["assigned_host"],"subnets":["admin_network_subnet"],
        "allowed_actions":["Analyse"]}
    captured=[]
    generate=offline.generate
    def capture(messages,**kwargs):
        captured.append(deepcopy(messages));return generate(messages,**kwargs)
    offline.generate=capture
    for arm in ("E0","E3"):
        config=validate({"arm":arm,"action_contract":"explicit"})
        policy=DefensePolicy("blue_agent_4",offline,lambda *a,**k:"<choice>NONE</choice>",config,
                             SecuritySkillPool() if arm=="E3" else None)
        policy.decide(state,episode=0,step=0,seed=0)
    assert captured[0]==captured[1]
    prompt=captured[0][-1]["content"]
    assert '"assigned_host"' in prompt and '"admin_network_subnet"' in prompt
    assert "not a Skill name" in prompt


def test_character_bounded_semantic_chunks_do_not_drop_transitions(offline, monkeypatch):
    ev=Evolution(None,{"schedule":"paper_batch","semantic_chunk_steps":100,"semantic_chunk_chars":700})
    exp=experience(count=5);seen=[]
    for t in exp.transitions:t["observation_before"]={"visible":"x"*400}
    def gradient(self,sk,local):
        seen.extend(t["step"] for t in local.transitions)
        return {"initiation":"i","execution":"p","termination":"t"},True
    monkeypatch.setattr(OriginalEvolution,"_generate_semantic_gradient",gradient)
    ev._generate_semantic_gradient(Skill(name="parent"),exp)
    assert seen==list(range(5)) and len(ev.gradients)==5


def test_actor_json_instructions_never_enter_selector_or_termination(offline):
    state={"agent":"blue_agent_4","observation":{"phase":0,"success":"TRUE","message":[[0]*8]*4},
        "user_prompt":"Visible state", "hosts":["assigned_host"],"subnets":["admin_network_subnet"],
        "allowed_actions":["Analyse","DeployDecoy"]}
    meta_prompts=[]
    def meta(prompt,**kwargs):
        meta_prompts.append(prompt)
        return "<status>CONTINUE</status>" if "<status>" in prompt else "<choice>DeceptionDeployment</choice>"
    config=validate({"arm":"E3","action_contract":"explicit","selector":{"epsilon":0,"max_skill_age":8}})
    policy=DefensePolicy("blue_agent_4",offline,meta,config,SecuritySkillPool())
    for step in range(2):policy.decide(state,episode=0,step=step,seed=step)
    assert len(meta_prompts)==2 and "<status>" in meta_prompts[1]
    assert all("# EXECUTABLE ACTION CONTRACT" not in prompt and "Return one JSON object" not in prompt for prompt in meta_prompts)
    assert state["user_prompt"]=="Visible state"
