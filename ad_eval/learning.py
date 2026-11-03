"""Reward-updated strategy UCT and action priors, independent of the paper scorer."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import random

STRATEGIES = {
    "red": {
        "explore": "Reduce uncertainty about the exposed services and current access, using observations.",
        "advance": "Advance the current verified attack phase toward the user/root objective.",
        "validate": "Verify the current foothold and retrieve objective evidence without assuming success.",
    },
    "blue": {
        "inspect": "Investigate the reported weakness and its underlying configuration from actual evidence.",
        "mitigate": "Apply a targeted remediation of the observed weakness while preserving legitimate service.",
        "validate": "Verify remediation and legitimate service behavior; repair incomplete protection.",
    },
}


def empty_planner_state() -> dict:
    return {role: {"strategies": {}, "actions": {}} for role in STRATEGIES}


def load_planner_state(path: str | None) -> dict:
    if not path:
        return empty_planner_state()
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(result) != set(STRATEGIES):
        raise ValueError("planner state must contain separate red and blue policies")
    return result


def posterior(stat: dict | None) -> float:
    stat = stat or {}
    return (1. + float(stat.get("reward_sum", 0.))) / (2. + int(stat.get("visits", 0)))


def action_priors(state: dict, role: str) -> dict[str, float]:
    return {key: posterior(stat) for key, stat in state[role]["actions"].items()}


class StrategyMCTS:
    """UCT over high-level strategies; simulated values never become observed rewards."""
    def __init__(self, state: dict, role: str, *, rng: random.Random, n_sims: int = 32):
        self.state, self.role, self.rng, self.n_sims = state, role, rng, n_sims
        if role not in STRATEGIES or n_sims < 1:
            raise ValueError("valid role and positive simulation budget required")

    def search(self) -> tuple[str, str]:
        names = list(STRATEGIES[self.role])
        priors = {key: posterior(self.state[self.role]["strategies"].get(key)) for key in names}
        n, w = dict.fromkeys(names, 0), dict.fromkeys(names, 0.)
        for total in range(1, self.n_sims + 1):
            key = max(names, key=lambda key: float("inf") if not n[key] else
                      w[key] / n[key] + 1.4 * priors[key] * math.sqrt(math.log(total + 1) / n[key]))
            n[key] += 1
            w[key] += float(self.rng.random() < priors[key])
        selected = max(names, key=lambda key: w[key] / n[key] if n[key] else -1.)
        return selected, STRATEGIES[self.role][selected]


def trajectory_reward(team_score: dict, role: str) -> float:
    """Observed terminal return in [0,1]; no manual detection or rollout win rate."""
    if team_score.get("evidence", {}).get("invalid_match"):
        raise ValueError("invalid match must not update a learning policy")
    if role == "red":
        user, root = team_score.get("user_flag"), team_score.get("root_flag")
        if not isinstance(user, bool) or not isinstance(root, bool):
            raise ValueError("red trajectory requires observed user/root outcomes")
        return max((int(user) + 2 * int(root)) / 3., .2 if team_score.get("initial_access") is True else 0.)
    patch, available = team_score.get("vulnerability_patched"), team_score.get("full_availability")
    if not isinstance(patch, bool) or not isinstance(available, bool):
        raise ValueError("blue trajectory requires observed patch and availability outcomes")
    return .6 * int(patch) + .4 * int(available)


def update_planner(state: dict, trajectory: dict, reward: float) -> dict:
    if not math.isfinite(reward) or not 0 <= reward <= 1:
        raise ValueError("observed trajectory reward must lie in [0,1]")
    result = deepcopy(state)
    role = trajectory["role"]
    # One trajectory is one observation for each used strategy/action, avoiding
    # extra credit merely from repeating the same command many times.
    for table, field in (("strategies", "strategy_id"), ("actions", "tech_id")):
        for key in {step.get(field) for step in trajectory["steps"]} - {None, ""}:
            stat = result[role][table].setdefault(key, {"visits": 0, "reward_sum": 0.})
            stat["visits"] += 1
            stat["reward_sum"] += reward
    return result
