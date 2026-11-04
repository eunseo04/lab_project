"""Persistent strategy lineage and observed action priors, separate from scoring."""
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
    return normalize_planner({role: {"strategies": {}, "actions": {}} for role in STRATEGIES})


def normalize_planner(state: dict) -> dict:
    """Migrate legacy statistics without discarding their observed rewards."""
    result = deepcopy(state)
    if set(result) != set(STRATEGIES):
        raise ValueError("planner state must contain separate red and blue policies")
    for role, seeds in STRATEGIES.items():
        policy = result[role]
        policy.setdefault("strategies", {})
        policy.setdefault("actions", {})
        tree = policy.setdefault("tree", {"nodes": {}, "incumbent": None})
        for key, instruction in seeds.items():
            tree["nodes"].setdefault(key, {"id": key, "parent_id": None, "children": [],
                                        "instruction": instruction, "status": "active",
                                        "preferred_tactics": [], "preferred_techniques": [],
                                        "max_depth": 4, "discount": .95})
        for key, node in tree["nodes"].items():
            if key != node["id"] or node.get("parent_id") == key:
                raise ValueError("invalid strategy identity/parent")
            ancestor, visited = node.get("parent_id"), {key}
            while ancestor is not None:
                if ancestor in visited or ancestor not in tree["nodes"]:
                    raise ValueError("strategy tree contains a cycle or missing parent")
                visited.add(ancestor)
                ancestor = tree["nodes"][ancestor].get("parent_id")
    return result


def strategy_node(state: dict, role: str, key: str) -> dict:
    return normalize_planner(state)[role]["tree"]["nodes"][key]


def add_strategy(state: dict, role: str, node: dict) -> dict:
    result = normalize_planner(state)
    nodes = result[role]["tree"]["nodes"]
    if node["id"] in nodes or node["parent_id"] not in nodes:
        raise ValueError("strategy requires a new ID and an existing parent")
    nodes[node["id"]] = deepcopy(node)
    nodes[node["id"]].update(status="candidate", children=[])
    nodes[node["parent_id"]]["children"].append(node["id"])
    return normalize_planner(result)


def commit_comparison(state: dict, role: str, key: str, report: dict) -> dict:
    result = normalize_planner(state)
    tree = result[role]["tree"]
    node = tree["nodes"][key]
    if report["candidate_id"] != key or report["parent_id"] != node["parent_id"]:
        raise ValueError("comparison does not evaluate this strategy lineage")
    if report["status"] not in {"accepted", "rejected", "inconclusive"}:
        raise ValueError("invalid comparison status")
    if report["status"] == "accepted" and (not report.get("valid") or not report.get("pairs")):
        raise ValueError("strategy promotion requires valid paired self-play evidence")
    if report["status"] == "accepted":
        from .self_play import compare_returns
        if compare_returns(report["pairs"], report.get("minimum_gain", 0.))[0] != "accepted":
            raise ValueError("paired returns do not support strategy promotion")
    node["status"] = "active" if report["status"] == "accepted" else report["status"]
    node["comparison"] = deepcopy(report)
    if report["status"] == "accepted":
        tree["incumbent"] = key
    return result


def load_planner_state(path: str | None) -> dict:
    if not path:
        return empty_planner_state()
    return normalize_planner(json.loads(Path(path).read_text(encoding="utf-8")))


def posterior(stat: dict | None) -> float:
    stat = stat or {}
    return (1. + float(stat.get("reward_sum", 0.))) / (2. + int(stat.get("visits", 0)))


def action_priors(state: dict, role: str) -> dict[str, float]:
    return {key: posterior(stat) for key, stat in state[role]["actions"].items()}


class StrategyMCTS:
    """Select a validated lineage; candidate nodes run only in pinned comparisons.

    UCT traverses accepted children using observed trajectory statistics. Search
    visits are local and never masquerade as real trajectory observations.
    """
    def __init__(self, state: dict, role: str, *, rng: random.Random, n_sims: int = 32,
                 team: str | None = None):
        self.state, self.role, self.rng, self.n_sims = normalize_planner(state), role, rng, n_sims
        self.team = team
        if role not in STRATEGIES or n_sims < 1:
            raise ValueError("valid role and positive simulation budget required")

    def search(self) -> tuple[str, str]:
        policy = self.state[self.role]
        tree, stats = policy["tree"], policy["strategies"]
        forced = policy.get("selection_overrides", {}).get(self.team)
        if forced:
            node = tree["nodes"][forced]
            if node["status"] not in {"active", "candidate"}:
                raise ValueError("cannot run a rejected strategy")
            return forced, node["instruction"]
        if tree.get("incumbent"):
            node = tree["nodes"][tree["incumbent"]]
            return node["id"], node["instruction"]
        nodes = tree["nodes"]
        roots = [key for key, node in nodes.items() if node["parent_id"] is None and node["status"] == "active"]
        visits, values = {}, {}
        for total in range(1, self.n_sims + 1):
            options, path = roots, []
            while options:
                selected = max(options, key=lambda key: posterior(stats.get(key)) +
                               1.4 * math.sqrt(math.log(total + 1) / (1 + visits.get(key, 0))))
                path.append(selected)
                options = [key for key in nodes[selected]["children"] if nodes[key]["status"] == "active"]
            value = posterior(stats.get(path[-1]))
            for key in path:
                visits[key] = visits.get(key, 0) + 1
                values[key] = values.get(key, 0.) + value
        selected = max(visits, key=lambda key: (values[key] / visits[key], visits[key]))
        return selected, nodes[selected]["instruction"]


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
    result = normalize_planner(state)
    role = trajectory["role"]
    # One trajectory is one observation for each used strategy/action, avoiding
    # extra credit merely from repeating the same command many times.
    for table, field in (("strategies", "strategy_id"), ("actions", "tech_id")):
        keys = {step.get(field) for step in trajectory["steps"]} - {None, ""}
        if table == "strategies":
            # Credit each ancestor at most once per observed trajectory.
            nodes = result[role]["tree"]["nodes"]
            for key in list(keys):
                parent = nodes.get(key, {}).get("parent_id")
                while parent:
                    keys.add(parent)
                    parent = nodes[parent].get("parent_id")
        for key in keys:
            stat = result[role][table].setdefault(key, {"visits": 0, "reward_sum": 0.})
            stat["visits"] += 1
            stat["reward_sum"] += reward
    return result
