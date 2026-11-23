"""Offline reward decomposition; replay must match all original observations.

No inferred reward components or simulator state are supplied to the actor,
semantic gradient or optimizer. The registered experiment code stays untouched.
"""
import argparse
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from acd.runner import seed_everything,load_agents
from acd.llm_agent import parse_response
from research.adapter import make_environment,plain,executed_actions,BLUE_AGENTS
from CybORG.Simulator.Actions import Sleep
from CybORG.Simulator.Actions.AbstractActions.Impact import Impact
from CybORG.Simulator.Actions.GreenActions import GreenAccessService,GreenLocalWork
from CybORG.Simulator.Scenarios.EnterpriseScenarioGenerator import SUBNET


def replay(path):
    summary=json.loads(path.read_text());config=summary["config"]
    assert summary["status"]=="completed" and len(summary["episodes"])==1
    episode=summary["episodes"][0];seed_everything(episode["seed"],config["threads"])
    env=make_environment(config,episode["seed"]);observations,_=env.reset(seed=episode["seed"])
    rl,_=load_agents({**config,"team":"all_rl" if config["composition"]=="all_rl" else "mixed",
                     "communication":"keep","deterministic_rl":False},192)
    ticks=defaultdict(dict)
    for line in Path(episode["trace"]).open():
        row=json.loads(line);ticks[row["step"]][row["agent"]]=row
    ec=env.env.environment_controller
    calculator=ec.team_reward_calculators["Blue"]["BlueRewardMachine"]
    original=calculator.calculate_reward;events=[]
    def observe_reward(current_state,action_dict,agent_observations,done,state):
        expected=original(current_state,action_dict,agent_observations,done,state)
        rewards=calculator.get_phase_rewards(state.mission_phase);current=[]
        for agent,actions in action_dict.items():
            if not actions:continue
            action=actions[0]
            if isinstance(action,Impact):host=action.hostname
            elif isinstance(action,(GreenAccessService,GreenLocalWork)):host=state.ip_addresses[action.ip_address]
            else:continue
            zone=state.hostname_subnet_map[host].value
            if not any(s.active for s in state.sessions[agent].values()):continue
            success=agent_observations[agent].observations[0].data["success"]
            kind=None
            if "green" in agent and success==False:kind="LWF" if isinstance(action,GreenLocalWork) else "ASF"
            elif "red" in agent and success and isinstance(action,Impact):kind="RIA"
            if kind:current.append({"kind":kind,"zone":zone,"host":host,"agent":agent,"reward":rewards[zone][kind],"phase":state.mission_phase})
        assert sum(e["reward"] for e in current)==expected
        events[:]=current
        return expected
    calculator.calculate_reward=observe_reward
    totals=Counter();phase_totals=defaultdict(Counter);zone_totals=defaultdict(Counter);timeline=[]
    for step,records in sorted(ticks.items()):
        before=env.raw_observations()
        assert all(plain(before[a])==records[a]["observation_before"] for a in BLUE_AGENTS),(step,"observation before differs")
        outgoing=env.paper_messages();actions={}
        for agent in BLUE_AGENTS:
            row=records[agent]
            if row["policy"]=="rl":
                actions[agent]=rl[agent].get_action(observations[agent])
                assert str(env.action_translator(agent,actions[agent]))==row["requested_action"],(step,agent,"RL action differs")
            elif not row.get("policy_invoked"):actions[agent]=Sleep()
            else:
                hosts={a.hostname for a,v in zip(env.actions(agent),env.action_mask(agent)) if v and getattr(a,"hostname",None)}
                action,_,error=parse_response(row["response"],agent,hosts,[s.value for s in SUBNET])
                assert error==row.get("parse_error") and str(action)==row["requested_action"]
                actions[agent]=action
        observations,rewards,_,_,_=env.step(actions,messages=outgoing)
        after=env.raw_observations()
        assert all(plain(after[a])==records[a]["observation_after"] for a in BLUE_AGENTS),(step,"observation after differs")
        assert all(executed_actions(env,a)==records[a]["executed_actions"] for a in BLUE_AGENTS),(step,"executed action differs")
        assert all(value==records["blue_agent_4"]["joint_reward"] for value in rewards.values()),(step,"joint reward differs")
        components=dict(ec.reward["Blue"])
        assert sum(components.values())==records["blue_agent_4"]["joint_reward"]
        if records["blue_agent_4"]["score_included"]:
            for event in events:
                totals[event["kind"]]+=event["reward"]
                phase_totals[str(event["phase"])][event["kind"]]+=event["reward"]
                zone_totals[event["zone"]][event["kind"]]+=event["reward"]
            cost=components["action_cost"];totals["action_cost"]+=cost
            phase_totals[str(after["blue_agent_4"]["phase"])]["action_cost"]+=cost
        timeline.append({"step":step,"score_included":records["blue_agent_4"]["score_included"],"components":components,"events":list(events)})
    assert sum(totals.values())==episode["joint_return"]
    result={"passed":True,"kind":"offline_exact_observation_and_reward_replay","source":str(path.relative_to(ROOT)),
        "seed":episode["seed"],"joint_return":episode["joint_return"],"components":dict(totals),
        "by_phase":dict(phase_totals),"by_zone":dict(zone_totals),"timeline":timeline,
        "supplied_to_actor_or_optimizer":False,"registered_simulator_reward_unchanged":True}
    target=ROOT/"research_outputs/behavior-v2/reward-decomposition"/(path.parent.name+".json")
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("timeline","by_phase","by_zone")},ensure_ascii=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("summaries",nargs="+")
    for arg in parser.parse_args().summaries:
        try:replay(Path(arg).resolve())
        except Exception as exc:
            target=ROOT/"research_outputs/behavior-v2/reward-decomposition"/f"failed-attempt-{time.time_ns()}.json"
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(json.dumps({"passed":False,"source":arg,"error":f"{type(exc).__name__}: {exc}",
                "pythonhashseed":os.environ.get("PYTHONHASHSEED"),"no_decomposition_reported":True},indent=2)+"\n")
            raise
