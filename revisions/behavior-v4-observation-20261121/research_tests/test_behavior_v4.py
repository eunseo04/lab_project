"""CAGE4 connection endpoints are optional; failed responses remain unresolved."""
import json

from research.behavior import ObservableCaseLedger
from test_behavior import state,row,ledger_with_warning


def network(fields):
    return {"Processes":[{"Connections":[fields]}]}


def test_outgoing_connection_without_local_port_is_retained():
    ledger=ObservableCaseLedger("blue_agent_4",connection_leads=True)
    context=ledger.context(state({"h1":network({"local_address":"10.0.1.2","remote_address":"10.0.1.3","remote_port":22})}),0)
    assert context["investigation_leads"][0]["host"]=="h1"


def test_connection_without_remote_address_is_retained_as_uncertain():
    ledger=ObservableCaseLedger("blue_agent_4",connection_leads=True)
    context=ledger.context(state({"h1":network({"local_address":"10.0.1.2","local_port":52782})}),0)
    assert context["investigation_leads"][0]["investigation_due"]
    assert context["open_cases"]==[]


def test_partial_address_only_connection_is_not_silently_discarded():
    ledger=ObservableCaseLedger("blue_agent_4",connection_leads=True)
    context=ledger.context(state({"h1":network({"local_address":"10.0.1.2"})}),0)
    assert len(context["investigation_leads"])==1


def test_busy_outgoing_connection_survives_until_next_free_decision():
    ledger=ObservableCaseLedger("blue_agent_4",connection_leads=True)
    ledger.context(state(),0)
    ledger.transition(row(0,"Analyse h2",{"success":"IN_PROGRESS"}))
    ledger.transition(row(1,"Sleep",{"success":"TRUE","h1":network({"remote_port":22,"remote_address":"10.0.1.3"})},False))
    context=ledger.context(state(),2)
    assert context["investigation_leads"][0]["host"]=="h1"
    assert ledger.pending is None


def test_empty_connection_does_not_invent_an_investigation_lead():
    ledger=ObservableCaseLedger("blue_agent_4",connection_leads=True)
    assert ledger.context(state({"h1":network({})}),0)["investigation_leads"]==[]


def test_failed_remove_does_not_mark_flagged_evidence_responded():
    ledger=ledger_with_warning()
    ledger.transition(row(0,"Remove h1",{"success":"FALSE"}))
    assert ledger.context(state(),1)["open_cases"][0]["response_due"]


def test_failed_remove_and_unchanged_flagged_pid_do_not_close_the_case():
    ledger=ledger_with_warning()
    ledger.transition(row(0,"Remove h1",{"success":"FALSE"}));ledger.context(state(),1)
    ledger.transition(row(1,"Analyse h1",{"success":"TRUE","h1":{"Processes":[{"PID":41}]}}))
    assert ledger.context(state(),2)["open_cases"][0]["response_due"]


def test_pid_observed_after_remove_but_before_analysis_remains_unhandled():
    ledger=ledger_with_warning()
    ledger.transition(row(0,"Remove h1",{"success":"TRUE"}))
    ledger.context(state({"h1":{"Processes":[{"PID":42}]}}),1)
    ledger.transition(row(1,"Analyse h1",{"success":"TRUE","h1":{"Processes":[{"PID":42}]}}))
    assert ledger.context(state(),2)["open_cases"][0]["response_due"]


def test_accepted_case_insensitive_opcode_tracks_the_canonical_operation():
    ledger=ledger_with_warning()
    r=row(0,"Remove h1",{"success":"TRUE"});r["response"]=json.dumps({"action":"remove host:h1","reason":"respond"})
    completed=ledger.transition(r)["completed_action"]
    assert completed and completed["name"]=="Remove"
