"""Instrument exact recorded replays; hidden state is for offline diagnosis only."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

import replay_behavior_pipeline as replay

ROOT=replay.ROOT
OUT=Path(os.environ.get('REWARD_ROOT_CAUSE_OUTPUT',str(ROOT/"research_outputs/rollout-v5/root-cause")))
native_factory=replay.make_environment


def services(state,host):
    return {str(name):{"active":bool(s.active),"reliability":s.get_service_reliability(),"process":s.process}
            for name,s in state.hosts[host].services.items()} if host in state.hosts else {}


def instrument(config,seed):
    env=native_factory(config,seed);native_reset=env.reset
    executions=[];mutators={};block_owners={};clock={"step":0}
    def reset(*args,**kwargs):
        result=native_reset(*args,**kwargs)
        ec=env.env.environment_controller;native_execute=ec.execute_action
        def execute(action):
            state=ec.state;name=type(action).__name__;agent=getattr(action,"agent",None)
            host=getattr(action,"hostname",None)
            ip=getattr(action,"ip_address",None)
            if host is None and ip in state.ip_addresses:host=state.ip_addresses[ip]
            changed_action=name in ("DegradeServices","Impact","Restore","Remove","DeployDecoy")
            before=services(state,host) if changed_action else None
            captured={}
            def profile(frame,event,arg):
                if event=="return" and frame.f_code is action.execute.__func__.__code__:
                    local=frame.f_locals
                    service=local.get("service_to_use")
                    if service is not None:
                        captured["selected_service"]={"reliability":service.get_service_reliability(),
                            "active":bool(service.active),"process":service.process}
                        captured["last_service_mutator"]=deepcopy(mutators.get((host,service.process)))
                    captured["session_present"]="session" in local
                    if "available_host_services" in local:captured["available_services"]=len(local["available_host_services"])
                    if "connection_failure_flag" in local:captured["connection_failure_flag"]=local["connection_failure_flag"]
            old_profile=sys.getprofile()
            if name in ("GreenLocalWork","GreenAccessService"):sys.setprofile(profile)
            try:obs=native_execute(action)
            finally:
                if name in ("GreenLocalWork","GreenAccessService"):sys.setprofile(old_profile)
            after=services(state,host) if changed_action else None
            record={"step":clock["step"],"order":len(executions),"phase":state.mission_phase,
                "agent":agent,"action":str(action),"name":name,"host":host,"success":str(obs.data.get("success")),**captured}
            for key in ('from_subnet','to_subnet'):
                if hasattr(action,key):record[key]=str(getattr(action,key))
            if name in ('BlockTrafficZone','AllowTrafficZone') and record['success']=='TRUE':
                # Native ControlTraffic stores blocks[to_subnet].append(from_subnet).
                edge=(record['to_subnet'],record['from_subnet'])
                if name=='BlockTrafficZone':block_owners[edge]={k:record[k] for k in ('step','order','phase','agent','name','from_subnet','to_subnet')}
                else:block_owners.pop(edge,None)
            if name=="GreenAccessService":
                record.update(destination=str(getattr(action,"dest_ip",None)),blocks=deepcopy(state.blocks))
                if captured.get('connection_failure_flag'):
                    src=state.hostname_subnet_map[state.ip_addresses[action.ip_address]].value
                    dst=state.hostname_subnet_map[state.ip_addresses[action.dest_ip]].value
                    record['blocking_edges']=[{'from_subnet':a,'to_subnet':b,'owner':deepcopy(block_owners.get((a,b)))}
                        for a,b in [(src,dst),(dst,src)] if b in state.blocks.get(a,[])]
            if changed_action:
                record.update(services_before=before,services_after=after)
                for service in after.values():
                    if service not in before.values():mutators[(host,service["process"])]=deepcopy(record)
            if name not in ("Sleep","Monitor"):executions.append(record)
            return obs
        ec.execute_action=execute
        return result
    env.reset=reset
    native_step=env.step
    def step(actions,messages=None):
        value=native_step(actions,messages=messages);clock["step"]+=1;return value
    env.step=step
    env.diagnostic_executions=executions
    instrument.last_env=env
    return env


def run(summary):
    summary=Path(summary).resolve();OUT.mkdir(parents=True,exist_ok=True)
    replay.ARTIFACT_ROOT=OUT;replay.make_environment=instrument
    replay.replay(summary)
    target=OUT/"reward-decomposition"/(summary.parent.name+".json")
    audit=json.loads(target.read_text());records=instrument.last_env.diagnostic_executions
    events=[];categories=Counter();category_rewards=Counter()
    by_key={(r["step"],r["agent"]):r for r in records if r["name"].startswith("Green")}
    original=json.loads(summary.read_text());rows=[json.loads(s) for s in Path(original["episodes"][0]["trace"]).read_text().splitlines()]
    blue_by_step={}
    for row in rows:blue_by_step.setdefault(row["step"],[]).append(row)
    scored={r["step"] for r in rows if r["score_included"]}
    for tick in audit["timeline"]:
        if tick["step"] not in scored:continue
        for event in tick["events"]:
            execution=by_key.get((tick["step"],event["agent"]))
            if event["kind"]=="RIA":category="successful_red_impact"
            elif not execution:category="unresolved_missing_execution"
            elif event["kind"]=="LWF":
                if not execution.get("session_present"):category="missing_green_session"
                elif execution.get("available_services")==0:category="no_active_local_service"
                elif execution.get("selected_service",{}).get("reliability",100)<100:
                    mutation=execution.get("last_service_mutator") or {}
                    category="red_degraded_service_reliability" if mutation.get("name")=="DegradeServices" else "unreliable_service_origin_unresolved"
                else:category="local_work_unresolved"
            elif execution.get("connection_failure_flag"):category="observed_zone_firewall_block"
            else:category="access_service_unresolved"
            blue=[{k:r.get(k) for k in ("agent","requested_action","executed_actions","skill_id","skill_version","action_status","request_applied","behavior_context")}
                  for r in blue_by_step[tick["step"]]]
            events.append({**event,"step":tick["step"],"classification":category,"green_execution":execution,
                "blue_at_tick":blue,"causal_scope":"Executed failure branch and service mutation only; not the total causal effect of a Blue policy."})
            categories[category]+=1;category_rewards[category]+=event["reward"]
    assert sum(category_rewards.values())+audit["components"].get("action_cost",0)==audit["joint_return"]
    result={"passed":True,"source":str(summary.relative_to(ROOT)),"source_summary_sha256":hashlib.sha256(summary.read_bytes()).hexdigest(),
        "components":audit["components"],"joint_return":audit["joint_return"],"seed":audit["seed"],
        "classifications":dict(categories),"classified_reward":dict(category_rewards),"events":events,"executions":records,
        "all_blue_observations_actions_rewards_matched":True,"new_performance_sample":False,"hidden_state_never_supplied_to_actor_or_semantic_gradient":True}
    path=OUT/(summary.parent.name+".json");path.write_text(json.dumps(result,default=str,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("events","executions")},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("summary");run(parser.parse_args().summary)
