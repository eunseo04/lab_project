"""Action denominators and observable workflow diagnostics for real CAGE traces."""
from collections import Counter,defaultdict
from copy import deepcopy
import json
import re
import statistics

from research.behavior import ObservableCaseLedger,chosen_action,HOST_ACTIONS


def mean(values):return statistics.mean(values) if values else None


def episode_metrics(episode,rows,ledger_factory=ObservableCaseLedger):
    groups=defaultdict(list)
    for row in rows:groups[row["agent"]].append(row)
    result={}
    for agent,items in groups.items():
        role=items[0]["policy"]
        free=[r for r in items if r["request_applied"]]
        generated=[r for r in items if r.get("policy_invoked")]
        actions=lambda collection:dict(Counter((r.get("requested_action") or "None").split()[0] for r in collection))
        metrics={"role":role,"world_ticks":len(items),"free_opportunities":len(free),
            "busy_ticks":len(items)-len(free),"policy_generations":len(generated),
            "all_tick_requests":actions(items),"free_choice_actions":actions(free),
            "environment_last_action_ticks":dict(Counter(action["name"] for r in items for action in r["executed_actions"])),
            "generated_choice_actions":actions(generated),
            "valid_generated_choice_actions":actions([r for r in generated if not r.get("parse_error")]),
            "invalid_policy_fallback_actions":actions([r for r in free if r.get("parse_error")]),
            "automatic_monitor_ticks":sum(bool(r.get("automatic_monitor")) for r in items),
            "parse_errors":sum(bool(r.get("parse_error")) for r in generated),
            "learned_skill_decisions":sum((r.get("skill_version") or 0)>1 for r in generated),
            "skill_usage":dict(Counter(r.get("skill_id") or "NONE" for r in generated))}
        if role=="llm":
            hosts=set()
            for row in items:
                hosts.update(k for k,v in row["observation_before"].items() if isinstance(v,dict) and k not in ("true_state","global_state"))
                name,target=chosen_action(row)
                if name in HOST_ACTIONS and target:hosts.add(target)
            # The actor already receives its complete assigned target catalog.
            for row in generated[:1]:
                messages=json.loads(row["state"])
                match=re.search(r"Assigned host IDs: (\[.*?\])\n",messages[-1]["content"])
                if match:hosts=set(json.loads(match[1]))
            ledger=ledger_factory(agent)
            counts=Counter();completions=Counter();delays=[]
            decoy_services=defaultdict(set);decoy_ever=set();restored_decoys=set();decoy_details=Counter()
            reasons=Counter()
            opcode_variants=0
            for row in items:
                context=ledger.context({"hosts":sorted(hosts),"observation":row["observation_before"]},row["step"])
                if "behavior_context" in row:
                    assert row["behavior_context"]==context,(episode["seed"],agent,row["step"],"context replay differs")
                name,target=chosen_action(row)
                actual=(row.get("requested_action") or "Sleep").split()[0]
                response_name=None
                if row.get("response") and not row.get("parse_error"):
                    try:response_name=json.loads(row["response"])["action"].split()[0]
                    except (ValueError,KeyError,TypeError,AttributeError,IndexError):pass
                opcode_variants+=int(bool(row.get("policy_invoked") and response_name and
                    response_name!=actual and response_name.lower()==actual.lower()))
                if row.get("policy_invoked") and row["request_applied"]:
                    reason=(row.get("reason") or "").lower()
                    if "privileg" in reason or "admin" in reason:reasons["privilege_or_admin"]+=1
                    elif "user-level" in reason or "malicious" in reason:reasons["compromise_response"]+=1
                    elif "decoy" in reason:reasons["deception"]+=1
                    elif "traffic" in reason or "block" in reason:reasons["network_control"]+=1
                    elif "wait" in reason:reasons["waiting"]+=1
                    elif "no suspicious" in reason or "no immediate" in reason:reasons["quiet_observation"]+=1
                    elif "analys" in reason or "suspicious" in reason:reasons["investigation"]+=1
                    else:reasons["other"]+=1
                    waiting_for_action=bool(re.search(r"wait.*(?:ongoing|in.progress).*action|wait.*action.*(?:complet|finish)",reason))
                    if actual=="Sleep" and waiting_for_action and not context["pending_action"]:
                        counts["waiting_for_action_while_already_free"]+=1
                    cases=context["open_cases"]
                    if cases:counts["open_case_opportunities"]+=1
                    admin={c["host"] for c in cases if c["observed_level"]>=3}
                    urgent={c["host"] for c in cases if c["response_due"] and c["flagged_pids"] and c["observed_level"]<3}
                    if admin:
                        counts["observed_admin_opportunities"]+=1
                        counts["targeted_restore_with_admin_evidence"]+=int(actual=="Restore" and target in admin)
                    if actual=="Restore":
                        counts["restore_requests"]+=1
                        counts["restore_without_retained_admin_evidence"]+=int(target not in admin)
                    if actual=="DeployDecoy" and cases:counts["decoy_with_open_case"]+=1
                    if urgent:
                        counts["unhandled_flagged_pid_opportunities"]+=1
                        counts["targeted_response_to_flagged_pid"]+=int(actual in ("Remove","Restore") and target in urgent)
                    case=next((c for c in cases if c["host"]==target),None)
                    if case and actual in ("Remove","Restore"):delays.append(case["evidence_age"])
                replay=deepcopy(row)
                diagnostics=ledger.transition(replay)
                if "behavior_outcome" in row:assert diagnostics==row["behavior_outcome"]
                for key,value in diagnostics.items():
                    if isinstance(value,bool):counts[key]+=value
                completed=diagnostics["completed_action"]
                if completed:
                    completions[f"{completed['name']}:{completed['status']}"]+=1
                    host=completed.get("target")
                    if completed["status"]=="TRUE" and completed["name"]=="Restore":
                        decoy_services[host].clear()
                        if host in decoy_ever:restored_decoys.add(host)
                    if completed["status"]=="TRUE" and completed["name"]=="DeployDecoy":
                        # The public action returns the newly created service.
                        # Same-host deployments need not be the same service.
                        services={str(p["service_name"]) for p in row["observation_after"].get(host,{}).get("Processes",[])
                            if isinstance(p,dict) and p.get("service_name")}
                        decoy_details["successful_deployments"]+=1
                        if not services:decoy_details["service_identity_unavailable"]+=1
                        if host in decoy_ever:
                            decoy_details["same_host_successful_redeployments"]+=1
                            if services-decoy_services[host]:decoy_details["same_host_new_observed_service"]+=1
                            elif services:decoy_details["same_host_service_already_recorded_since_restore"]+=1
                            if host in restored_decoys:decoy_details["first_redeployment_after_own_restore"]+=1
                        decoy_services[host].update(services);decoy_ever.add(host);restored_decoys.discard(host)
            metrics.update(observable_diagnostics=dict(counts),completed_own_operations=dict(completions),
                decoy_deployment_diagnostics=dict(decoy_details),
                rule_based_reason_categories=dict(reasons),
                mean_observed_evidence_age_at_response=mean(delays),response_age_samples=len(delays),
                noncanonical_response_opcode=opcode_variants,stored_context_replay_verified=any("behavior_context" in r for r in items))
        else:
            metrics.update(observable_diagnostics=None,completed_own_operations=None,
                decoy_deployment_diagnostics=None,
                rule_based_reason_categories=None,
                mean_observed_evidence_age_at_response=None,response_age_samples=None,
                noncanonical_response_opcode=None,stored_context_replay_verified=None)
        result[agent]=metrics
    return result


def aggregate_behavior(episodes,agent="blue_agent_4"):
    items=[e["agents"][agent] for e in episodes]
    total=lambda key:sum(e[key] for e in items)
    combined={key:dict(sum((Counter(e[key]) for e in items),Counter()))
              for key in ("all_tick_requests","environment_last_action_ticks","free_choice_actions","generated_choice_actions",
                          "valid_generated_choice_actions","invalid_policy_fallback_actions","skill_usage")}
    combined.update({key:total(key) for key in ("world_ticks","free_opportunities","busy_ticks","policy_generations","automatic_monitor_ticks","parse_errors","learned_skill_decisions")})
    combined["learned_skill_fraction"]=total("learned_skill_decisions")/max(1,total("policy_generations"))
    combined["free_choice_percent"]={k:100*v/max(1,total("free_opportunities")) for k,v in combined["free_choice_actions"].items()}
    combined["explicit_passive_choices"]=sum(combined["free_choice_actions"].get(k,0) for k in ("Sleep","Monitor"))
    combined["explicit_passive_fraction"]=combined["explicit_passive_choices"]/max(1,total("free_opportunities"))
    if items and items[0]["observable_diagnostics"] is not None:
        diagnostics=dict(sum((Counter(e["observable_diagnostics"]) for e in items),Counter()))
        combined["observable_diagnostics"]=diagnostics
        combined["decoy_deployment_diagnostics"]=dict(sum((Counter(e.get("decoy_deployment_diagnostics") or {}) for e in items),Counter()))
        combined["repeated_analysis_fraction"]=diagnostics.get("repeated_analysis_same_evidence",0)/max(1,combined["generated_choice_actions"].get("Analyse",0))
        combined["sleep_with_open_case_fraction"]=diagnostics.get("sleep_with_open_case",0)/max(1,diagnostics.get("open_case_opportunities",0))
        combined["duplicate_decoy_fraction"]=diagnostics.get("duplicate_successful_decoy",0)/max(1,combined["generated_choice_actions"].get("DeployDecoy",0))
        combined["noncanonical_response_opcode"]=sum(e["noncanonical_response_opcode"] for e in items)
        combined["rule_based_reason_categories"]=dict(sum((Counter(e["rule_based_reason_categories"]) for e in items),Counter()))
    else:combined["observable_diagnostics"]=None
    return combined
