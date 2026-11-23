"""Domain procedures for the frozen Skill-Pro policy, separate from legacy seeds."""

RESPONSE_SEEDS = [
    ("RapidProcessResponse",
     "An assigned host has a newly flagged process PID or observed user compromise that has not yet received a response for this evidence version.",
     ["Read the host-specific open case, flagged PID, evidence age, and prior outcomes; never confuse a peer's alert with local proof.",
      "For a freshly Monitor-flagged local PID, promptly Remove on that exact host rather than first spending another Analyse turn merely to rediscover the same warning.",
      "A TRUE Remove result only certifies completion. When available, review the case and Analyse once to check privilege escalation or remaining indicators; use PrivilegedRecovery if admin indicators are observed.",
      "Do not Remove again on the same historical file alone. Wait automatically during IN_PROGRESS; make a fresh decision as soon as completion is observable."],
     "The flagged evidence has received a response and its follow-up has been reviewed, or privileged recovery is needed.",
     ["Remove","Analyse","Restore","Sleep"]),
    ("EscalationDiagnosis",
     "A completed Remove needs follow-up, or an unresolved local case has new evidence that has not been analysed and cannot yet justify a specific response.",
     ["Select the exact open-case host needing diagnosis, including earlier alerts retained while another action was busy.",
      "Analyse once for the pending evidence version and inspect the returned file indicators after completion.",
      "Observed escalate.sh/escalate.exe justifies Restore; observed cmd.sh/cmd.exe before any response justifies Remove. A repeated old cmd file after Remove alone does not prove a fresh attacker session.",
      "Do not repeat Analyse on an unchanged, already reviewed evidence version. Stop when the result has been consumed and the next response or quiet observation is appropriate."],
     "The analysis result is reviewed, a response is indicated, or analysis failed and requires reassessment.",
     ["Analyse","Remove","Restore","Sleep"]),
    ("PrivilegedRecovery",
     "An unresolved assigned-host case contains observed admin privilege-escalation file indicators and no later successful Restore for that evidence.",
     ["Identify the exact host with observed escalate.sh/escalate.exe; peer severity or ordinary connection INFO alone is insufficient.",
      "Restore that host promptly. Remove cannot terminate privileged root/SYSTEM persistence; repeated Analyse cannot repair it.",
      "Account for the Restore availability cost, but do not indefinitely defer recovery of confirmed admin compromise solely because recovery is disruptive.",
      "Wait for completion and inspect TRUE/FALSE. A successful Restore completes this workflow; only new observed evidence reopens it."],
     "Restore completed and its status was read, or recovery failed and requires reassessment.",
     ["Restore","Analyse","Sleep"]),
    ("SelectiveDeception",
     "There is no outstanding local compromise response or verification, and a valid assigned host without an already successful own decoy is useful for one deliberate deployment.",
     ["Prioritize open local threat cases over discretionary deception; select an exact assigned host only after reading the case ledger.",
      "DeployDecoy once on a suitable host lacking a recorded successful own deployment. Do not spread or repeat decoys simply to remain active.",
      "Use the returned deployment status to verify execution; do not Analyse merely to prove that a completed decoy exists.",
      "After success stop. Reattempt a failure only after its observed cause or relevant evidence changes; quiet Sleep is a valid choice."],
     "Deployment completed or failed and its status has been inspected, or an urgent threat case needs attention.",
     ["DeployDecoy","Sleep"]),
    ("NetworkContainment",
     "Observed threat-origin bits or confirmed local compromise support a specific cross-zone containment decision; the peer busy bit alone is not a threat.",
     ["Read the explicitly decoded peer identities and threat-origin bits, current mission phase, and local case evidence.",
      "Use BlockTrafficZone only for a justified, exact destination subnet. Preserve availability by avoiding broad repeated blocks without new evidence.",
      "Record the actual result, stop after reassessing it, and leave recovery to a later evidence-based decision."],
     "The targeted block result has been reviewed or the observable containment trigger no longer applies.",
     ["BlockTrafficZone","Sleep"]),
    ("TrafficRecovery",
     "A successful own BlockTrafficZone is recorded and current observations support reassessing its availability cost.",
     ["Use only an exact subnet in own_observed_blocks; do not invent another defender's block.",
      "Check current local cases and decoded peer threat reports, then AllowTrafficZone when the earlier containment is no longer justified.",
      "Inspect completion and stop. Never repeatedly toggle the same subnet without a changed observed reason."],
     "Traffic recovery completed or failed and the result was reviewed.",
     ["AllowTrafficZone","Sleep"]),
]


def initiation_matches(family, context):
    """E3 initiation checks on the same observable context supplied to the actor."""
    cases=context.get("open_cases",[])
    if family=="RapidProcessResponse":
        return any(c["response_due"] and (c["flagged_pids"] or c["observed_level"]==2)
                   and c["observed_level"]<3 for c in cases)
    if family=="EscalationDiagnosis":
        return any(c["verification_due"] or c["analysis_due"] and c["observed_level"]<3 for c in cases)
    if family=="PrivilegedRecovery":return any(c["observed_level"]>=3 for c in cases)
    if family=="SelectiveDeception":return not cases and not context.get("pending_action")
    if family=="NetworkContainment":
        return any(p["detected_origin_agents"] or p["reported_compromise"]>=2 for p in context.get("peers",[]))
    if family=="TrafficRecovery":return bool(context.get("own_observed_blocks"))
    return True
