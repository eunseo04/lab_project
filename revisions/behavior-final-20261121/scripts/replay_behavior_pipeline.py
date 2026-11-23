"""Offline replay of original run_episode, with recorded LLM outputs only.

This produces diagnostics, never new model-performance samples. Every world
observation, requested/executed action and reward must equal the real source.
"""
from collections import Counter,defaultdict
from copy import deepcopy
import argparse
import json
import hashlib
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SOURCE_ROOT=Path(os.environ.get("BEHAVIOR_REPLAY_SOURCE_ROOT",str(ROOT))).resolve()
ARTIFACT_ROOT=Path(os.environ.get("BEHAVIOR_REPLAY_ARTIFACT_ROOT",str(ROOT/"research_outputs/behavior-v2"))).resolve()
sys.path.insert(0,str(SOURCE_ROOT))
import research.experiment as experiment
from research.adapter import make_environment
from research.client import TextModel
from research.skills import SecuritySkillPool
from research.upstream import configure_encoder
from CybORG.Simulator.Actions.AbstractActions.Impact import Impact
from CybORG.Simulator.Actions.GreenActions import GreenAccessService,GreenLocalWork


class RecordedInference:
    is_mock=True  # Keep replay outputs excluded from every performance reporter.
    def __init__(self,episode,rows):
        self.identity=episode["model_identity"];self.calls=[];self.counter=0
        self.recorded=iter(episode["calls"])
        self.policies=iter(r for r in rows if r.get("policy_invoked"))
    def embed(self,text):return [1.,0.]
    def generate(self,messages,*,purpose="policy",**kwargs):
        call=next(self.recorded);assert call["purpose"]==purpose
        self.calls.append(deepcopy(call));self.counter+=1
        row=next(self.policies) if purpose=="policy" else None
        return {"text":row["response"] if row else call["response"],"token_ids":row["token_ids"] if row else [],
            "generation_logprob":row.get("generation_logprob") if row else None,
            "identity":self.identity,"prompt_tokens":call["input_tokens"],"output_tokens":call["output_tokens"],
            "prompt_sha256":row.get("prompt_sha256") if row else None,"seconds":0.}


def replay(path):
    data=json.loads(path.read_text());episode=data["episodes"][0];config=data["config"]
    original=[json.loads(line) for line in Path(episode["trace"]).open()]
    client=RecordedInference(episode,original);configure_encoder(client.embed)
    pool=None
    if config["arm"]!="E0":
        snapshot=json.loads(Path(data["training_provenance"]["path"]).read_text())
        pool=SecuritySkillPool(config["pool_size"],seed_set=config["skill_seed_set"]);pool.restore(snapshot["pool"])
    timeline=[]
    def factory(case,seed):
        env=make_environment(case,seed);reset=env.reset
        def instrumented_reset(*args,**kwargs):
            result=reset(*args,**kwargs)
            ec=env.env.environment_controller;calc=ec.team_reward_calculators["Blue"]["BlueRewardMachine"]
            calculate=calc.calculate_reward;events=[]
            def observe(current,actions,observations,done,state):
                value=calculate(current,actions,observations,done,state)
                phase=state.mission_phase;weights=calc.get_phase_rewards(phase);current_events=[]
                for agent,choices in actions.items():
                    if not choices:continue
                    action=choices[0]
                    if isinstance(action,Impact):host=action.hostname
                    elif isinstance(action,(GreenAccessService,GreenLocalWork)):host=state.ip_addresses[action.ip_address]
                    else:continue
                    if not any(s.active for s in state.sessions[agent].values()):continue
                    success=observations[agent].observations[0].data["success"]
                    kind=None
                    if "green" in agent and success==False:kind="LWF" if isinstance(action,GreenLocalWork) else "ASF"
                    elif "red" in agent and success and isinstance(action,Impact):kind="RIA"
                    if kind:
                        zone=state.hostname_subnet_map[host].value
                        current_events.append({"kind":kind,"zone":zone,"host":host,"agent":agent,"phase":phase,"reward":weights[zone][kind]})
                assert sum(e["reward"] for e in current_events)==value
                events[:]=current_events
                return value
            calc.calculate_reward=observe
            step=env.step
            def instrumented_step(actions,messages=None):
                result=step(actions,messages=messages)
                timeline.append({"step":len(timeline),"components":deepcopy(ec.reward["Blue"]),"events":deepcopy(events)})
                return result
            env.step=instrumented_step
            return result
        env.reset=instrumented_reset
        return env
    experiment.make_environment=factory
    work=ARTIFACT_ROOT/"reward-replay-work"/path.parent.name
    work.mkdir(parents=True,exist_ok=True)
    result,_=experiment.run_episode(config,episode["seed"],0,client,pool,TextModel(client),work)
    replayed=[json.loads(line) for line in Path(result["trace"]).open()]
    assert len(replayed)==len(original)
    keys=("agent","step","requested_action","executed_actions","observation_before","observation_after","joint_reward","score_included")
    for before,after in zip(original,replayed):
        for key in keys:assert before[key]==after[key],(before["step"],before["agent"],key,"replay differs")
    assert result["joint_return"]==episode["joint_return"]
    totals=Counter();zones=defaultdict(Counter);phases=defaultdict(Counter)
    scored={r["step"] for r in original if r["score_included"]}
    for tick in timeline:
        if tick["step"] not in scored:continue
        totals["action_cost"]+=tick["components"]["action_cost"]
        for event in tick["events"]:
            totals[event["kind"]]+=event["reward"]
            zones[event["zone"]][event["kind"]]+=event["reward"]
            phases[str(event["phase"])][event["kind"]]+=event["reward"]
    assert sum(totals.values())==episode["joint_return"]
    output={"passed":True,"source":str(path.relative_to(ROOT)),"seed":episode["seed"],"joint_return":episode["joint_return"],
        "replay_source_root":str(SOURCE_ROOT),"source_summary_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_trace_sha256":hashlib.sha256(Path(episode["trace"]).read_bytes()).hexdigest(),
        "components":dict(totals),"by_zone":dict(zones),"by_reward_phase":dict(phases),"timeline":timeline,
        "all_blue_observations_actions_rewards_matched":True,"recorded_llm_outputs_replayed":True,
        "new_performance_sample":False,"decomposition_supplied_to_actor_or_optimizer":False}
    target=ARTIFACT_ROOT/"reward-decomposition"/(path.parent.name+".json")
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps({k:v for k,v in output.items() if k not in ("timeline","by_zone","by_reward_phase")},ensure_ascii=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("summaries",nargs="+")
    for arg in parser.parse_args().summaries:
        try:replay(Path(arg).resolve())
        except Exception as exc:
            target=ARTIFACT_ROOT/"reward-decomposition"/f"failed-pipeline-{time.time_ns()}.json"
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(json.dumps({"passed":False,"source":arg,"error":f"{type(exc).__name__}: {exc}","no_decomposition_reported":True},indent=2)+"\n")
            raise
