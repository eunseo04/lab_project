"""Controlled strategy comparisons using real, freshly reset closed-loop games.

Only one team's one role changes between arms. Both opponents, role weights,
sampling, seed, range images and observations protocol remain pinned. These
development games are excluded from measured campaign scores and GRPO datasets.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path

from .docker_lab import COMPOSE, ROOT, HOSTS, KALI, cmd
from .learning import posterior, normalize_planner, trajectory_reward
from .models import verify_model


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def range_identity() -> dict:
    containers = [*HOSTS.values(), *KALI.values()]
    images = cmd(["docker", "inspect", "-f", "{{.Image}}", *containers]).splitlines()
    if len(images) != 4 or any(not image.startswith("sha256:") for image in images):
        raise RuntimeError("cannot establish self-play range image identity")
    return {"images": dict(zip(containers, images)),
            "compose_sha256": hashlib.sha256(COMPOSE.read_bytes()).hexdigest(),
            "dataset_sha256": hashlib.sha256((ROOT / "attack_dataset.json").read_bytes()).hexdigest(),
            "worker_source_sha256": {name: hashlib.sha256((ROOT / "ad_eval" / name).read_bytes()).hexdigest()
                                     for name in ("proposed_worker.py", "learning.py", "mcts.py", "policy.py", "inference.py")}}


def policy_identity(models) -> dict:
    result = {}
    for role, spec in models.items():
        checked = verify_model(spec)
        result[role] = {"spec": asdict(spec), "server_model": {
            key: checked["server_model"].get(key) for key in ("id", "root", "parent")},
            "checkpoint": checked.get("checkpoint_artifact")}
    return result


def incumbent_for(state, role, trajectories=()) -> str:
    policy = normalize_planner(state)[role]
    tree = policy["tree"]
    if tree.get("incumbent"):
        return tree["incumbent"]
    counts = {}
    for trajectory in trajectories:
        if trajectory["role"] == role:
            for step in trajectory["steps"]:
                key = step.get("strategy_id")
                if key in tree["nodes"] and tree["nodes"][key]["status"] == "active":
                    counts[key] = counts.get(key, 0) + 1
    active = [key for key, node in tree["nodes"].items() if node["status"] == "active"]
    return max(active, key=lambda key: (counts.get(key, 0), posterior(policy["strategies"].get(key))))


def pinned_state(state, role, team, strategy_id, incumbents):
    result = normalize_planner(state)
    for current_role in ("red", "blue"):
        result[current_role]["selection_overrides"] = {side: incumbents[current_role] for side in ("1", "2")}
    result[role]["selection_overrides"][team] = strategy_id
    return result


def compare_returns(pairs: list[dict], margin: float) -> tuple[str, bool, float | None]:
    if not math.isfinite(margin) or not 0 <= margin < 1:
        raise ValueError("invalid promotion margin")
    valid = (len(pairs) >= 2 and {pair.get("team") for pair in pairs} == {"1", "2"} and
             all(pair.get("valid") and all(type(pair.get(key)) in {int, float} and
                 math.isfinite(pair[key]) and 0 <= pair[key] <= 1
                 for key in ("baseline_return", "candidate_return")) for pair in pairs))
    if not valid:
        return "inconclusive", False, None
    deltas = [pair["candidate_return"] - pair["baseline_return"] for pair in pairs]
    gain = sum(deltas) / len(deltas)
    # Conservative observed-gain gate; not a statistical significance claim.
    accepted = gain > margin and all(delta >= 0 for delta in deltas)
    return "accepted" if accepted else "rejected", True, gain


class PairedSelfPlay:
    def __init__(self, args, *, runner=None, environment=None, policies=None):
        self.args, self.runner = args, runner
        self.environment = environment or range_identity
        self.policies = policies or policy_identity

    def evaluate(self, *, state, role, parent_id, candidate_id, models, rep, directory,
                 incumbents) -> dict:
        if self.runner is None:
            from .run import run_one
            runner = run_one
        else:
            runner = self.runner
        args = self.args
        frozen = self.policies(models)
        pairs = []
        directory = Path(directory)
        if (directory / "protocol.json").exists():
            raise FileExistsError("refusing to overwrite strategy comparison evidence")
        save(directory / "protocol.json", {
            "role": role, "parent_id": parent_id, "candidate_id": candidate_id,
            "role_policies": frozen, "incumbents": incumbents,
            "pairs": args.strategy_self_play_pairs, "promotion_min_mean_gain": args.strategy_min_gain,
            "mcts_simulations": int(os.environ.get("MAYHEM_SIMS", "32")),
            "promotion_rule": "all_valid_mean_gain_above_margin_no_negative_pair",
            "learning_during_comparison": False, "range_reset_each_arm": True,
            "observations": "same_live_Wazuh_host_time_filter_protocol_not_replayed_alerts",
            "limits": "fresh random flags/credentials; equal seeds do not ensure deterministic GPU or scheduling"})
        for index in range(args.strategy_self_play_pairs):
            team = str(1 + index % 2)  # compare from both network positions
            seed = (args.seed + 1_000_000 + rep * 1000 + index * 4) % (2**31 - 4)
            arms = {}
            order = ("baseline", "candidate") if (index + rep) % 2 else ("candidate", "baseline")
            pair = {"index": index, "team": team, "seed": seed, "order": list(order), "valid": False}
            for arm in order:
                arm_dir = directory / f"pair{index:03d}" / arm
                strategy_id = parent_id if arm == "baseline" else candidate_id
                planner = pinned_state(state, role, team, strategy_id, incumbents)
                snapshot = arm_dir / "planner.json"
                save(snapshot, planner)
                local_args = deepcopy(args)
                local_args.out_dir = str(arm_dir)
                local_args.adaptive_proposed = False
                local_args.match_seconds = args.strategy_self_play_seconds or args.match_seconds
                local_args.interval = min(args.interval, local_args.match_seconds / 4)
                # run_one applies the normal rep offset to the shared base seed.
                local_args.seed = seed - 4 * (rep - 1)
                try:
                    if self.policies(models) != frozen:
                        raise RuntimeError("role model identity changed during strategy comparison")
                    match = runner(local_args, "proposed", rep, {}, role_models=models, planner_state=snapshot)
                    identity = self.environment()
                    if self.policies(models) != frozen:
                        raise RuntimeError("role model weights changed during strategy comparison")
                    if match.get("worker_exit_codes"):
                        raise ValueError("self-play worker failed")
                    audit = match.get("strategy_selection_audit", {})
                    if len(audit) != 4 or any(not row.get("valid") for row in audit.values()):
                        raise ValueError("workers did not verify all pinned strategy selections")
                    initial = match.get("initial_conditions", {})
                    if (initial.get("reset") != "compose_down_volumes_then_recreate" or
                            any(initial.get("patches", {}).get(side, {}).get("vulnerability_patched") is not False or
                                initial.get("services", {}).get(side, {}).get("up") is not True for side in ("1", "2"))):
                        raise ValueError("self-play lacks fresh vulnerable and available initial-state evidence")
                    score = next(row for row in match["teams"] if row["team"] == team)
                    # A failed opponent invalidates the comparison as well.
                    if len(match["teams"]) != 2 or any(row.get("evidence", {}).get("invalid_match") for row in match["teams"]):
                        raise ValueError("invalid self-play team or opponent")
                    if any(not isinstance(row.get("full_availability"), bool) for row in match["teams"]):
                        raise ValueError("missing self-play availability evidence")
                    reward = trajectory_reward(score, role)
                    arms[arm] = {"return": reward, "environment": identity,
                                 "inference": match["inference"], "duration_s": match["duration_s"],
                                 "score": str(arm_dir / f"rep{rep:03d}_proposed" / "score.json"),
                                 "role_models": match["role_models"], "strategy_selection_audit": audit,
                                 "initial_conditions": initial}
                    if arms[arm]["role_models"] != {key: asdict(spec) for key, spec in models.items()}:
                        raise ValueError("self-play did not use the frozen role policies")
                except Exception as exc:
                    arms[arm] = {"error": type(exc).__name__ + ": " + str(exc)}
            pair["arms"] = arms
            if all("return" in arms[arm] for arm in ("baseline", "candidate")):
                a, b = arms["baseline"], arms["candidate"]
                pair["valid"] = all(a[key] == b[key] for key in ("environment", "inference", "duration_s", "role_models"))
                pair.update(baseline_return=a["return"], candidate_return=b["return"])
            pairs.append(pair)
            save(directory / "pairs.json", {"pairs": pairs})
        status, valid, gain = compare_returns(pairs, args.strategy_min_gain)
        report = {"role": role, "parent_id": parent_id, "candidate_id": candidate_id,
                  "status": status, "valid": valid, "mean_paired_gain": gain, "pairs": pairs,
                  "minimum_gain": args.strategy_min_gain,
                  "protocol": str(directory / "protocol.json"),
                  "metric": "existing_observed_trajectory_return_no_GRPO_semantic_diversity",
                  "statistical_significance_claimed": False}
        save(directory / "comparison.json", report)
        return report
