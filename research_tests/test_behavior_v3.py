"""Isolated development fixes: no main campaign result is changed."""
import hashlib
import json

from research.behavior import ObservableCaseLedger, readable_context
from research.behavior_skills import initiation_matches
from research.skills import CAGEVerifier, SecuritySkillPool
from test_behavior import state, row
from test_research import offline


def connection(remote="10.0.2.5", remote_port=50001):
    return {"Processes":[{"Connections":[{"remote_address":remote,
        "remote_port":remote_port, "local_port":22}]}]}


def test_connection_only_lead_survives_busy_without_inventing_compromise():
    ledger=ObservableCaseLedger("blue_agent_4",connection_leads=True)
    ledger.context(state(),0)
    ledger.transition(row(0,"Analyse h2",{"success":"IN_PROGRESS"}))
    ledger.transition(row(1,"Sleep",{"success":"TRUE","h1":connection()},False))
    context=ledger.context(state(),2)
    assert context["open_cases"]==[]
    lead=context["investigation_leads"][0]
    assert lead["host"]=="h1" and lead["investigation_due"]
    assert "not proof of compromise" in readable_context(context)


def test_connection_review_cooldown_recurrence_and_expiry():
    ledger=ObservableCaseLedger("blue_agent_4",connection_leads=True)
    ledger.context(state({"h1":connection()}),0)
    ledger.transition(row(0,"Analyse h1",{"success":"TRUE"}))
    assert not ledger.context(state(),1)["investigation_leads"][0]["investigation_due"]
    ledger.context(state({"h1":connection(remote_port=50002)}),3)
    context=ledger.context(state(),4)
    assert len(context["investigation_leads"][0]["endpoints"])==1
    assert not context["investigation_leads"][0]["investigation_due"]
    assert ledger.context(state(),17)["investigation_leads"][0]["investigation_due"]
    assert ledger.context(state(),36)["investigation_leads"]==[]


def test_failed_analysis_does_not_consume_connection_lead():
    ledger=ObservableCaseLedger("blue_agent_4",connection_leads=True)
    ledger.context(state({"h1":connection()}),0)
    ledger.transition(row(0,"Analyse h1",{"success":"FALSE"}))
    assert ledger.context(state(),1)["investigation_leads"][0]["investigation_due"]


def test_connections_are_optional_and_only_from_assigned_hosts():
    assert "investigation_leads" not in ObservableCaseLedger("blue_agent_4").context(state(),0)
    ledger=ObservableCaseLedger("blue_agent_4",connection_leads=True)
    assert ledger.context(state({"other_host":connection()}),0)["investigation_leads"]==[]


def test_bounded_investigation_precedes_discretionary_decoy():
    ledger=ObservableCaseLedger("blue_agent_4",connection_leads=True)
    context=ledger.context(state({"h1":connection()}),0)
    assert initiation_matches("RiskBasedInvestigation",context)
    assert not initiation_matches("SelectiveDeception",context)
    ledger.transition(row(0,"Analyse h1",{"success":"TRUE"}))
    context=ledger.context(state(),1)
    assert not initiation_matches("RiskBasedInvestigation",context)
    assert initiation_matches("SelectiveDeception",context)


def test_repeat_verifier_uses_free_decisions_across_busy_ticks(offline):
    pool=SecuritySkillPool();skill=pool.skills[0]
    s=state();raw=s["observation"]
    evidence={str(k):v for k,v in raw.items() if k not in ("action","phase","message")}
    fingerprint=hashlib.sha256(json.dumps(evidence,sort_keys=True).encode()).hexdigest()
    decision={"policy_invoked":True,"request_applied":True,"parse_error":None,
        "requested_action":"Analyse h1","evidence_hash":fingerprint,"skill_id":skill.name}
    busy={"policy_invoked":False,"request_applied":False,"requested_action":"Sleep"}
    history=[decision,busy,decision,busy]
    assert CAGEVerifier().check(skill,s,history)[0]  # exact legacy behavior
    assert CAGEVerifier("accepted_decisions").check(skill,s,history)==(False,"repeated_action_without_new_evidence")
    sleep={**decision,"requested_action":"Sleep"}
    assert CAGEVerifier("accepted_decisions").check(skill,s,[decision,sleep,decision,busy])[0]


def test_repeat_verifier_still_allows_new_evidence(offline):
    pool=SecuritySkillPool();skill=pool.skills[0]
    decision={"policy_invoked":True,"request_applied":True,"parse_error":None,
        "requested_action":"Analyse h1","evidence_hash":"old","skill_id":skill.name}
    assert CAGEVerifier("accepted_decisions").check(skill,state(),[decision,decision])[0]
