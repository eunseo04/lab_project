"""Train the published KEEP GCN/PPO networks on native or paper messages."""
from pathlib import Path
import hashlib
import math
from statistics import mean
import time

import numpy as np
import torch

from CybORG.Agents.CybermonicAgents.cage4 import InductiveGraphPPOAgent, load
from CybORG.Agents.CybermonicAgents.memory_buffer import MultiPPOMemory
from .environment import BLUE_AGENTS, make_env
from .runner import seed_everything, write_json


def fingerprint(agent):
    digest = hashlib.sha256()
    for module in (agent.actor, agent.critic):
        for tensor in module.state_dict().values():
            digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def rollout(agents, config, seed):
    """Credit each multistep action with all rewards through its completion.

    Keep terminal transitions at each episode boundary so discounted returns
    never leak between episodes. The original PPO implementation is unchanged.
    """
    seed_everything(seed, config["threads"])
    env = make_env(seed=seed, steps=config["steps"], red=config["red"],
                   communication=config["communication"], horizon=config["horizon"])
    states, _ = env.reset(seed=seed)
    memories = MultiPPOMemory(config["training"]["batch_size"])
    pending = {}
    total = 0.0
    for step in range(config["steps"]):
        actions = {}
        for i, name in enumerate(BLUE_AGENTS):
            if states[name][1]:
                actions[name] = None
            else:
                action, value, probability = agents[i].get_action(states[name])
                pending[i] = [states[name][0], action, value, probability, 0.0]
                actions[name] = action
        next_states, rewards, terminated, truncated, _ = env.step(actions)
        total += mean(rewards.values())
        terminal = step == config["steps"] - 1 or truncated.get("__all__", False)
        for i, name in enumerate(BLUE_AGENTS):
            if i not in pending:
                continue
            pending[i][4] += rewards[name]
            if terminal or not next_states[name][1]:
                state, action, value, probability, reward = pending.pop(i)
                memories.remember(i, state, action, value, probability, reward, int(terminal))
        states = next_states
        if terminal:
            break
    # The last finished action can precede a terminal busy period. Always seal it.
    for memory in memories.mems:
        if memory.t:
            memory.t[-1] = 1
    return memories.mems, total


def train(config, output=None):
    hp = config.get("training", {})
    defaults = {"episodes": 50000, "rollout_episodes": 25, "batch_size": 2500,
                "epochs": 4, "hidden": 256, "embedding": 128,
                "actor_lr": 0.0003, "critic_lr": 0.001, "clip": 0.2}
    hp = {**defaults, **hp}
    for k in ("episodes", "rollout_episodes", "batch_size", "epochs", "hidden", "embedding"):
        if type(hp[k]) is not int or hp[k] < 1:
            raise ValueError(f"training.{k} must be a positive integer")
    if hp["episodes"] % hp["rollout_episodes"]:
        raise ValueError("training.episodes must be divisible by training.rollout_episodes")
    config = {**config, "training": hp}
    destination = Path(output or config["weights"])
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Training output already contains files: {destination}; choose a new output directory")
    destination.mkdir(parents=True, exist_ok=True)
    seed_everything(config["seed"], config["threads"])
    probe = make_env(seed=config["seed"], steps=config["steps"], communication=config["communication"], horizon=config["horizon"])
    feature_dimension = probe.feature_dimension
    agents = [InductiveGraphPPOAgent(feature_dimension, bs=hp["batch_size"], epochs=hp["epochs"],
        a_kwargs={"lr": hp["actor_lr"], "hidden1": hp["hidden"], "hidden2": hp["embedding"]},
        c_kwargs={"lr": hp["critic_lr"], "hidden1": hp["hidden"], "hidden2": hp["embedding"]},
        clip=hp["clip"], concat_edges=True) for _ in BLUE_AGENTS]
    for agent in agents:
        agent.train()
    before = [fingerprint(a) for a in agents]
    history = []
    started = time.perf_counter()
    for first in range(0, hp["episodes"], hp["rollout_episodes"]):
        batch = [[] for _ in BLUE_AGENTS]
        returns = []
        for offset in range(hp["rollout_episodes"]):
            memories, reward = rollout(agents, config, config["seed"] + first + offset)
            for i, memory in enumerate(memories):
                batch[i].append(memory)
            returns.append(reward)
        counts, losses = [], []
        for i, agent in enumerate(agents):
            counts.append(sum(len(m.a) for m in batch[i]))
            # PyTorch's upstream std() normalization needs at least 2 samples.
            if counts[-1] < 2:
                raise ValueError("Too few completed actions for PPO; increase steps or rollout_episodes")
            agent.memory.mems = batch[i]
            # The upstream container defaults to 5 buffers, but here each buffer
            # is an episode. Process every collected episode, including N != 5.
            agent.memory.tot = len(batch[i])
            loss = agent.learn()
            if not math.isfinite(loss):
                raise RuntimeError(f"Non-finite PPO loss for agent {i}")
            losses.append(loss)
            agent.save(str(destination / f"gnn_ppo-{i}.pt"))
        row = {"episodes_completed": first + hp["rollout_episodes"], "reward_mean": mean(returns),
               "losses": losses, "transitions_per_agent": counts}
        history.append(row)
        write_json(destination / "history.json", history)
        print(f"PPO episodes={row['episodes_completed']}, reward={row['reward_mean']:.1f}, transitions={counts}", flush=True)
    after = [fingerprint(a) for a in agents]
    reload_matches = []
    for i, agent in enumerate(agents):
        restored = load(str(destination / f"gnn_ppo-{i}.pt"))
        reload_matches.append(fingerprint(restored) == fingerprint(agent))
    result = {"config": config, "communication": config["communication"],
        "input_features": feature_dimension, "episodes_trained": hp["episodes"],
        "validation_only": hp["episodes"] < 50000,
        "policy_quality_evaluated": False, "elapsed_seconds": time.perf_counter() - started,
        "parameters_changed": [a != b for a, b in zip(before, after)],
        "save_reload_matches": reload_matches, "history": history,
        "collector_fixes": ["Single environment step for all five agents", "Rewards attributed through multistep action completion", "Terminal transitions sealed per episode", "All rollout episode buffers included in each PPO update"]}
    if not all(result["parameters_changed"]) or not all(reload_matches):
        raise RuntimeError("Training did not change/reload all five agents correctly")
    write_json(destination / "training.json", result)
    print("Saved training checkpoints", destination, flush=True)
    return destination, result
