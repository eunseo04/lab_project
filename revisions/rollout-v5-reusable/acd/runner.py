from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
from pathlib import Path
import random
from statistics import mean, stdev
import time

import numpy as np
import torch
from CybORG.Simulator.Actions import Sleep
from CybORG.Agents.CybermonicAgents.cage4 import load

from . import ROOT
from .backends import create_backend
from .environment import BLUE_AGENTS, make_env
from .llm_agent import LLMDefender


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=json_default) + "\n")


def seed_everything(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)


def roles(team):
    if team == "none":
        return {a: "sleep" for a in BLUE_AGENTS}
    if team == "all_llm":
        return {a: "llm" for a in BLUE_AGENTS}
    return {a: ("llm" if team == "mixed" and a == "blue_agent_4" else "rl") for a in BLUE_AGENTS}


def load_agents(config, feature_dimension, training=False):
    agents, metadata = {}, {}
    manifest_path = Path(config["weights"]) / "training.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    if manifest and manifest.get("communication") != config["communication"]:
        raise ValueError("Checkpoint communication protocol does not match the experiment")
    for name, role in roles(config["team"]).items():
        if role != "rl":
            continue
        path = Path(config["weights"]) / f"gnn_ppo-{name[-1]}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Run ./run.sh train --config configs/train_paper_smoke.yaml first, or supply trained checkpoints.")
        agent = load(str(path))
        if agent.args[0] != feature_dimension:
            raise ValueError(f"{path.name} has {agent.args[0]} features, but {config['communication']} needs {feature_dimension}. "
                             "Public KEEP weights use 'communication: keep'. Train a separate checkpoint for paper communication.")
        agent.set_deterministic(config["deterministic_rl"])
        agents[name] = agent
        metadata[name] = {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "input_features": agent.args[0], "training": manifest}
    return agents, metadata


def evaluate(config, output=None):
    seed_everything(config["seed"], config["threads"])
    stamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S-%f")
    destination = Path(output) if output else Path(config["output"]) / f"{stamp}-{config['name']}"
    destination.mkdir(parents=True, exist_ok=False)
    agent_roles = roles(config["team"])
    backend = create_backend(config["model"]) if "llm" in agent_roles.values() else None
    weights_metadata = {}
    episode_results = []
    action_counts, invalid = Counter(), Counter()
    started = time.perf_counter()
    with (destination / "trace.jsonl").open("w", encoding="utf-8") as trace:
        for episode in range(config["episodes"]):
            episode_seed = config["seed"] + episode
            seed_everything(episode_seed, config["threads"])
            env = make_env(seed=episode_seed, steps=config["steps"], red=config["red"],
                           communication=config["communication"], horizon=config["horizon"])
            observations, _ = env.reset(seed=episode_seed)
            rl, weights_metadata = load_agents(config, env.feature_dimension)
            llm = {a: LLMDefender(a, backend, config.get("prompt"))
                   for a, role in agent_roles.items() if role == "llm"}
            if llm and episode == 0:
                write_json(destination / "system_prompts.json", {a: p.system_prompt for a, p in llm.items()})
            cumulative_reward = 0.0
            executed_steps = 0
            phases = set()
            for step in range(config["steps"]):
                raw = env.raw_observations()
                messages = env.paper_messages() if config["communication"] == "paper" else {a: v.copy() for a, v in env.msg.items()}
                actions, records = {}, {}
                for agent, role in agent_roles.items():
                    decision_start = time.perf_counter()
                    details = {}
                    if role == "llm":
                        actions[agent], details = llm[agent].decide(raw[agent], hosts=env.hosts(agent),
                            subnets=list(env.env.environment_controller.state.subnets), episode=episode, step=step)
                        if details.get("parse_error"):
                            invalid[agent] += 1
                    elif role == "rl":
                        actions[agent] = rl[agent].get_action(observations[agent])
                    else:
                        actions[agent] = Sleep()
                    requested = env.action_translator(agent, actions[agent])
                    busy = role == "rl" and actions[agent] is None
                    action_counts[f"{agent}:{'InProgress' if busy else type(requested).__name__}"] += 1
                    records[agent] = {"episode": episode, "seed": episode_seed, "step": step,
                        "agent": agent, "policy": role,
                        "requested_action": None if busy else str(requested), "policy_invoked": not busy,
                        "phase": raw[agent]["phase"], "message_sent": messages[agent],
                        "messages_received": raw[agent].get("message", []),
                        "decision_seconds": time.perf_counter() - decision_start, **details}
                observations, rewards, terminated, truncated, _ = env.step(actions, messages=messages)
                # Shared team reward: average duplicate per-agent rewards, never sum 5x.
                team_reward = mean(rewards[a] for a in BLUE_AGENTS)
                cumulative_reward += team_reward
                executed_steps += 1
                phases.add(int(env.env.environment_controller.state.mission_phase))
                for agent, record in records.items():
                    record.update(executed_action=str(env.env.get_last_action(agent)),
                        reward=rewards[agent], action_status=str(env.env.get_observation(agent).get("success")),
                        simulator_step=env.env.environment_controller.step_count)
                    trace.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")
                # Include terminal step reward before checking termination.
                if terminated.get("__all__") or truncated.get("__all__"):
                    break
            expected_steps = config["steps"] - (1 if config["horizon"] == "native" else 0)
            if executed_steps != expected_steps:
                raise RuntimeError(f"Episode ended at {executed_steps}, expected {expected_steps}")
            episode_results.append({"episode": episode, "seed": episode_seed,
                "steps": executed_steps, "simulator_steps": env.env.environment_controller.step_count,
                "mission_phases": sorted(phases), "reward": cumulative_reward})
            print(f"{config['name']}: episode {episode + 1}/{config['episodes']}, steps={executed_steps}, reward={cumulative_reward:.1f}", flush=True)
    backend_kind = config["model"].get("backend", "mock")
    result_kind = "simulation_evaluation"
    if "llm" in agent_roles.values():
        if backend_kind == "mock":
            result_kind = "offline_mock_pipeline_only"
        elif backend_kind == "replay":
            result_kind = "replay_pipeline_only"
    summary = {"config": config, "started_at_kst": stamp, "roles": agent_roles,
        "result_kind": result_kind,
        "paper_scores_reproduced": False,
        "notes": ["Public weights are not the paper's retrained 8-bit-communication policies.",
                  "Short training checkpoints validate execution only; they do not establish convergence."] if config["communication"] == "paper" else [],
        "episodes": episode_results, "weights": weights_metadata,
        "reward": {"mean": mean(e["reward"] for e in episode_results),
                   "stdev": stdev(e["reward"] for e in episode_results) if len(episode_results) > 1 else None},
        "elapsed_seconds": time.perf_counter() - started, "action_counts": dict(action_counts),
        "invalid_responses": dict(invalid),
        "sources": json.loads((ROOT / "sources.lock.json").read_text())}
    write_json(destination / "summary.json", summary)
    print("Saved", destination, flush=True)
    return destination, summary
