from copy import deepcopy
import json

import pytest

from research.adapter import DefensePolicy
from research.behavior import ObservableCaseLedger,readable_context
from research.behavior_skills import initiation_matches
from research.config import validate
from research.skills import CAGEVerifier,SecuritySkillPool
from test_research import offline


def state(raw=None):
    return {"agent":"blue_agent_4","hosts":["h1","h2"],"subnets":["admin_network_subnet"],
        "allowed_actions":["Analyse","Remove","Restore","DeployDecoy","Sleep","BlockTrafficZone","AllowTrafficZone"],
        "user_prompt":"Visible current observation", "observation":raw or {"phase":0,"success":"UNKNOWN","message":[[0]*8]*4}}


def row(step,action,after,invoked=True):
    return {"step":step,"requested_action":action,"response":None,"policy_invoked":invoked,
        "request_applied":invoked,"parse_error":None,"observation_after":after}


def ledger_with_warning():
    ledger=ObservableCaseLedger("blue_agent_4")
    ledger.context(state({"phase":0,"success":"UNKNOWN","h1":{"Processes":[{"PID":41}]}}),0)
    return ledger


def test_transient_busy_alert_survives_and_sleep_does_not_replace_pending_action():
    ledger=ledger_with_warning()
    ledger.transition(row(0,"Analyse h1",{"success":"IN_PROGRESS"}))
    ledger.transition(row(1,"Sleep",{"success":"IN_PROGRESS","h2":{"Processes":[{"PID":99}]}},False))
    result=ledger.transition(row(2,"Sleep",{"success":"TRUE","h1":{"Files":[]}},False))
    assert result["automatic_busy_wait"] and result["completed_action"]["target"]=="h1"
    context=ledger.context(state(),3)
    assert ledger.pending is None
    assert [c["host"] for c in context["open_cases"]]==["h2"]


def test_remove_true_keeps_a_verification_case_instead_of_claiming_cleanup():
    ledger=ledger_with_warning()
    ledger.transition(row(0,"Remove h1",{"success":"TRUE"}))
    case=ledger.context(state(),1)["open_cases"][0]
    assert case["verification_due"] and not case["response_due"]
    assert initiation_matches("EscalationDiagnosis",ledger.last_context)
    assert not initiation_matches("SelectiveDeception",ledger.last_context)


def test_old_file_after_remove_is_not_new_pid_and_does_not_force_repeat_removal():
    ledger=ledger_with_warning()
    ledger.transition(row(0,"Remove h1",{"success":"TRUE"}))
    ledger.context(state(),1)
    ledger.transition(row(1,"Analyse h1",{"success":"TRUE","h1":{"Files":[{"File Name":"cmd.sh"}]}}))
    context=ledger.context(state(),2)
    assert context["open_cases"]==[]
    assert ledger.cases["h1"]["last_analysis"]["reported_file_indicators"]==["cmd.sh"]


def test_new_pid_while_analysis_is_busy_reopens_response_even_with_old_files():
    ledger=ledger_with_warning()
    ledger.transition(row(0,"Remove h1",{"success":"TRUE"}))
    ledger.context(state(),1)
    ledger.transition(row(1,"Analyse h1",{"success":"IN_PROGRESS"}))
    ledger.transition(row(2,"Sleep",{"success":"TRUE","h1":{"Processes":[{"PID":42}],"Files":[{"File Name":"cmd.sh"}]}},False))
    context=ledger.context(state(),3)
    assert context["open_cases"][0]["response_due"]
    assert initiation_matches("RapidProcessResponse",context)


def test_admin_evidence_survives_quiet_tick_and_requires_recovery_not_more_remove():
    ledger=ledger_with_warning()
    ledger.transition(row(0,"Analyse h1",{"success":"TRUE","h1":{"Files":[{"File Name":"escalate.sh"}]}}))
    context=ledger.context(state(),10)
    assert initiation_matches("PrivilegedRecovery",context)
    assert not initiation_matches("RapidProcessResponse",context)
    ledger.transition(row(10,"Restore h1",{"success":"TRUE"}))
    assert ledger.context(state(),11)["open_cases"]==[]


def test_failed_decoy_is_not_counted_as_success_and_busy_ticks_count_once():
    ledger=ObservableCaseLedger("blue_agent_4");ledger.context(state(),0)
    ledger.transition(row(0,"DeployDecoy h1",{"success":"FALSE"}))
    assert not ledger.decoys
    ledger.transition(row(1,"DeployDecoy h1",{"success":"IN_PROGRESS"}))
    ledger.transition(row(2,"Sleep",{"success":"TRUE"},False))
    ledger.transition(row(3,"Sleep",{"success":"UNKNOWN"}))
    assert ledger.decoys["h1"]==1
    result=ledger.transition(row(4,"DeployDecoy h1",{"success":"TRUE"}))
    assert result["duplicate_successful_decoy"]


def test_repeat_analysis_diagnostic_is_based_on_accepted_completed_evidence():
    ledger=ledger_with_warning()
    ledger.transition(row(0,"Analyse h1",{"success":"TRUE","h1":{"Files":[]}}))
    ledger.context(state(),1)
    result=ledger.transition(row(1,"Analyse h1",{"success":"TRUE","h1":{"Files":[]}}))
    assert result["repeated_analysis_same_evidence"]


def test_peer_busy_bit_is_not_a_threat_and_agent_identity_is_preserved():
    ledger=ObservableCaseLedger("blue_agent_2")
    s=state();s["observation"]["message"][2][-1]=1
    context=ledger.context(s,0)
    assert context["peers"][2]["agent"]=="blue_agent_3"
    assert not initiation_matches("NetworkContainment",context)
    s["observation"]["message"][2][0]=1
    assert initiation_matches("NetworkContainment",ledger.context(s,1))


def test_unassigned_host_and_hidden_keys_never_enter_case_history():
    ledger=ObservableCaseLedger("blue_agent_4")
    s=state({"phase":0,"private_host":{"Processes":[{"PID":999}]},"true_state":{"red_session":"hidden"}})
    assert ledger.context(s,0)["open_cases"]==[]
    assert "hidden" not in json.dumps(ledger.last_context)


def test_response_seed_roundtrip_and_no_unbounded_wait_seed(offline):
    pool=SecuritySkillPool(24,seed_set="response_v1")
    assert len(pool.skills)==6 and all(s.name!="WaitVerification" for s in pool.skills)
    restored=SecuritySkillPool();restored.restore(pool.snapshot())
    assert restored.snapshot()==pool.snapshot()
    assert restored.seed_set=="response_v1"


def test_response_verifier_uses_retained_local_admin_evidence(offline):
    pool=SecuritySkillPool(24,seed_set="response_v1")
    ledger=ledger_with_warning();ledger.transition(row(0,"Analyse h1",{"success":"TRUE","h1":{"Files":[{"File Name":"escalate.sh"}]}}))
    s=state();s.update(behavior_context=ledger.context(s,2),behavior_seed_set="response_v1")
    skill=next(s for s in pool.skills if s.name=="PrivilegedRecovery")
    assert CAGEVerifier().check(skill,s,[])[0]


def test_common_evidence_context_has_identical_actor_input_with_skills_off(offline):
    captures=[];original=offline.generate
    def capture(messages,**kwargs):
        captures.append(deepcopy(messages));return original(messages,**kwargs)
    offline.generate=capture
    for arm in ("E0","E3"):
        config=validate({"arm":arm,"action_contract":"explicit","behavior_context":"evidence_v1","skill_seed_set":"response_v1"})
        policy=DefensePolicy("blue_agent_4",offline,lambda *a,**k:"<choice>NONE</choice>",config,
                             SecuritySkillPool(24,seed_set="response_v1") if arm=="E3" else None)
        policy.decide(state(),episode=0,step=0,seed=42)
    assert captures[0]==captures[1]
    assert "Sleep" in captures[0][-1]["content"] and "Monitor is automatic" in captures[0][-1]["content"]


def test_response_seeds_require_common_observable_context():
    with pytest.raises(ValueError,match="response_v1 skills require"):
        validate({"skill_seed_set":"response_v1"})


def test_readable_history_distinguishes_completed_removal_from_busy_wait():
    ledger=ledger_with_warning();ledger.transition(row(0,"Remove h1",{"success":"TRUE"}))
    text=readable_context(ledger.context(state(),1))
    assert "Availability now: FREE" in text and "Remove completed" in text
    assert "follow-up diagnosis" in text and "NOT been reviewed" in text
    assert '"action"' not in text  # Facts do not replace the actor's output contract.


def test_readable_history_keeps_quiet_sleep_available():
    ledger=ObservableCaseLedger("blue_agent_4")
    assert "Quiet Sleep remains available" in readable_context(ledger.context(state(),0))


def test_readable_followup_does_not_invent_compromise_for_an_unprompted_remove():
    ledger=ObservableCaseLedger("blue_agent_4");ledger.context(state(),0)
    ledger.transition(row(0,"Remove h1",{"success":"TRUE"}))
    text=readable_context(ledger.context(state(),1))
    assert "earlier own response awaiting review" in text and "admin-level" not in text
