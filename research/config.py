from copy import deepcopy
from pathlib import Path
import hashlib
import json
import yaml
from . import ROOT

ARMS = ("E0", "E1", "E2", "E3")
REDS = ("finite", "aggressive", "stealthy", "impact", "degrade")


def load(path):
    config = yaml.safe_load(Path(path).read_text())
    return validate(config)


def validate(config):
    c = deepcopy(config)
    defaults = {"arm": "E0", "composition": "hybrid", "red": "finite", "steps": 500,
        "episodes": 100, "train_episodes": 6, "iterations": 3, "train_seed": 100,
        "eval_seed": 10000, "profile": "controlled-comparison", "scoring": "official",
        "communication": "paper_projected_keep", "threads": 1,
        "weights": "third_party/cage-4-submission/weights", "pool_size": 10,
        "output": "research_outputs", "skip_busy_policy": True, "action_contract": "legacy",
        "behavior_context": "legacy", "skill_seed_set": "legacy", "case_presentation": "json"}
    for key, value in defaults.items(): c.setdefault(key, value)
    c.setdefault("validation_episodes", 1)
    c.setdefault("validation_seed", 5000)
    c.setdefault("model", {"endpoint": "http://127.0.0.1:8018", "model": "Qwen/Qwen3-8B",
        "revision": "b968826d9c46dd6066d109eabc6255188de91218", "temperature": 1.0,
        "max_tokens": 192, "timeout_seconds": 240})
    c.setdefault("selector", {"epsilon": 0.1, "max_skill_age": 8})
    c.setdefault("evolution", {})
    for key, value in {"enabled": True, "ppo_gate": True, "semantic_gradient": True,
        "score_maintenance": True, "gate_mode": "official_code", "threshold": 6,
        "best_of_n": 3, "max_evolutions": 2, "clip": 0.2, "gamma": 0.99,
        "ema_alpha": 0.1, "acceptance_margin": 0.0, "semantic_chunk_steps": 8,
        "maintenance_mode": "official_code", "schedule": "official_code"}.items():
        c["evolution"].setdefault(key, value)
    if c["arm"] not in ARMS or c["red"] not in REDS: raise ValueError("Invalid experiment arm/red policy")
    if c["composition"] not in ("hybrid", "all_llm", "no_blue", "all_rl"):
        raise ValueError("composition must be hybrid/all_llm/no_blue/all_rl")
    if c["profile"] not in ("paper-compatible", "controlled-comparison"):
        raise ValueError("Invalid action/prompt comparison profile")
    if c["scoring"] not in ("official", "terminal_inclusive"): raise ValueError("Invalid scoring profile")
    if c["action_contract"] not in ("legacy", "explicit"):
        raise ValueError("action_contract must be legacy or explicit")
    if c["behavior_context"] not in ("legacy", "evidence_v1"):
        raise ValueError("Invalid observable behavior context")
    if c["case_presentation"] not in ("json","readable"):
        raise ValueError("Invalid case history presentation")
    if c["skill_seed_set"] not in ("legacy", "response_v1"):
        raise ValueError("Invalid versioned skill seed set")
    if c["skill_seed_set"]=="response_v1" and c["behavior_context"]!="evidence_v1":
        raise ValueError("response_v1 skills require observable evidence_v1 context")
    if c["profile"]=="paper-compatible" and (c["behavior_context"]!="legacy" or c["skill_seed_set"]!="legacy"):
        raise ValueError("Unmodified paper-compatible mode requires legacy state and skill seeds")
    if c["profile"] == "paper-compatible" and c["action_contract"] != "legacy":
        raise ValueError("Unmodified ACD paper-compatible prompts require the legacy action contract")
    if c["communication"] not in ("paper_projected_keep", "keep_native"): raise ValueError("Invalid communication profile")
    for key in ("steps", "episodes", "train_episodes", "iterations", "threads", "pool_size"):
        if type(c[key]) is not int or c[key] < 1: raise ValueError(f"{key} must be a positive integer")
    if c["steps"] < 3: raise ValueError("CC4 requires at least 3 steps")
    if c["iterations"] > c["train_episodes"]: raise ValueError("iterations cannot exceed training episodes")
    if type(c["validation_episodes"]) is not int or c["validation_episodes"] < 0:
        raise ValueError("validation_episodes must be a nonnegative integer")
    e = c["evolution"]
    if e["gate_mode"] not in ("official_code", "paper_rtg"): raise ValueError("Invalid PPO gate mode")
    if e["maintenance_mode"] not in ("official_code", "paper_score", "paper_algorithm1"):
        raise ValueError("Invalid maintenance mode")
    if e["schedule"] not in ("official_code", "paper_batch"):
        raise ValueError("Invalid Skill-Pro evolution schedule")
    if e["schedule"] == "paper_batch" and (e["gate_mode"] != "paper_rtg" or e["maintenance_mode"] != "paper_algorithm1"):
        raise ValueError("paper_batch requires paper_rtg and paper_algorithm1; do not mix algorithm variants")
    for key in ("threshold", "best_of_n", "max_evolutions", "semantic_chunk_steps"):
        if type(e[key]) is not int or e[key] < 1: raise ValueError(f"evolution.{key} must be positive")
    if not 0 < e["gamma"] <= 1 or not 0 < e["ema_alpha"] <= 1 or not 0 < e["clip"] < 1:
        raise ValueError("Invalid discount, EMA rate or PPO clipping interval")
    train_seeds = set(range(c["train_seed"], c["train_seed"]+c["train_episodes"]))
    eval_seeds = set(range(c["eval_seed"], c["eval_seed"]+c["episodes"]))
    if train_seeds & eval_seeds: raise ValueError("Training and evaluation seed sets overlap")
    validation_seeds=set(range(c["validation_seed"],c["validation_seed"]+c["validation_episodes"]))
    if validation_seeds & (train_seeds | eval_seeds): raise ValueError("Validation seeds overlap with training/evaluation")
    if c["profile"] == "paper-compatible" and c["skip_busy_policy"]:
        raise ValueError("Paper-compatible mode requires skip_busy_policy: false")
    if c["model"].get("temperature", 1.0) <= 0: raise ValueError("Positive policy temperature required")
    if not c["model"].get("model") or not c["model"].get("revision"):
        raise ValueError("Explicit model ID and pinned revision are required; no model fallback")
    for key in ("weights", "output", "pool"):
        if c.get(key):
            p = Path(c[key]).expanduser()
            c[key] = str(p if p.is_absolute() else ROOT/p)
    return c


def digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def compatible_digest(config):
    """Normalize newly explicit defaults without changing a recorded manifest."""
    c = validate(config)
    c.pop("maintenance_mode", None)  # former unused top-level default
    return digest(c)
