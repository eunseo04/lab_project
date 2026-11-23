"""A successful Restore starts a new evidence epoch; old artifacts stay historical."""
from research.behavior import ObservableCaseLedger
from test_behavior import state,row


def admin_ledger():
    ledger=ObservableCaseLedger("blue_agent_4")
    ledger.context(state({"h1":{"Processes":[{"PID":41}],"Files":[{"File Name":"escalate.sh"}]}}),0)
    return ledger


def test_same_admin_indicator_after_restore_is_a_new_unresolved_case():
    ledger=admin_ledger();ledger.transition(row(0,"Restore h1",{"success":"TRUE"}))
    assert not ledger.context(state(),1)["open_cases"]
    context=ledger.context(state({"h1":{"Files":[{"File Name":"escalate.sh"}]}}),20)
    assert context["open_cases"][0]["observed_level"]==3
    assert context["open_cases"][0]["response_due"]


def test_reused_pid_after_restore_is_not_suppressed_by_previous_epoch():
    ledger=admin_ledger();ledger.transition(row(0,"Restore h1",{"success":"TRUE"}))
    context=ledger.context(state({"h1":{"Processes":[{"PID":41}]}}),20)
    case=context["open_cases"][0]
    assert case["observed_level"]==1 and case["file_indicators"]==[]
    assert case["response_due"] and case["flagged_pids"]==["41"]


def test_fresh_user_indicator_does_not_inherit_old_admin_classification():
    ledger=admin_ledger();ledger.transition(row(0,"Restore h1",{"success":"TRUE"}))
    case=ledger.context(state({"h1":{"Files":[{"File Name":"cmd.sh"}]}}),20)["open_cases"][0]
    assert case["observed_level"]==2 and case["file_indicators"]==["cmd.sh"]


def test_observed_pid_on_restore_completion_is_retained_in_the_new_epoch():
    ledger=admin_ledger()
    ledger.transition(row(0,"Restore h1",{"success":"TRUE","h1":{"Processes":[{"PID":42}]}}))
    case=ledger.context(state(),1)["open_cases"][0]
    assert case["observed_level"]==1 and case["flagged_pids"]==["42"]
    assert case["file_indicators"]==[] and case["response_due"]


def test_failed_restore_keeps_the_prior_admin_case():
    ledger=admin_ledger();ledger.transition(row(0,"Restore h1",{"success":"FALSE"}))
    assert ledger.context(state(),1)["open_cases"][0]["observed_level"]==3


def test_remove_from_before_restore_cannot_close_new_file_only_compromise():
    ledger=admin_ledger();ledger.transition(row(0,"Remove h1",{"success":"TRUE"}))
    ledger.context(state(),1);ledger.transition(row(1,"Restore h1",{"success":"TRUE"}))
    ledger.context(state({"h1":{"Files":[{"File Name":"cmd.sh"}]}}),3)
    ledger.transition(row(3,"Analyse h1",{"success":"TRUE","h1":{"Files":[{"File Name":"cmd.sh"}]}}))
    assert ledger.context(state(),4)["open_cases"][0]["response_due"]


def test_new_remove_after_restore_can_review_its_persistent_cmd_artifact():
    ledger=admin_ledger();ledger.transition(row(0,"Restore h1",{"success":"TRUE"}))
    ledger.context(state({"h1":{"Files":[{"File Name":"cmd.sh"}]}}),2)
    ledger.transition(row(2,"Remove h1",{"success":"TRUE"}));ledger.context(state(),3)
    ledger.transition(row(3,"Analyse h1",{"success":"TRUE","h1":{"Files":[{"File Name":"cmd.sh"}]}}))
    assert ledger.context(state(),4)["open_cases"]==[]


def test_failed_restore_does_not_replace_the_last_successful_epoch_boundary():
    ledger=admin_ledger();ledger.transition(row(0,"Restore h1",{"success":"TRUE"}))
    ledger.context(state({"h1":{"Files":[{"File Name":"cmd.sh"}]}}),2)
    ledger.transition(row(2,"Remove h1",{"success":"TRUE"}));ledger.context(state(),3)
    ledger.transition(row(3,"Restore h1",{"success":"FALSE"}));ledger.context(state(),4)
    ledger.transition(row(4,"Analyse h1",{"success":"TRUE","h1":{"Files":[{"File Name":"cmd.sh"}]}}))
    assert ledger.context(state(),5)["open_cases"]==[]
