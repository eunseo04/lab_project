from copy import deepcopy
from pathlib import Path
import yaml
from . import ROOT

TEAMS = ("none", "all_rl", "all_llm", "mixed")
REDS = ("finite", "aggressive", "stealthy", "impact", "degrade")
LEGACY_ENCODER = "legacy_local_v1"
ORIGINAL_ENCODER = "acd_original_v1"
COMMUNICATION_ENCODERS = (LEGACY_ENCODER, ORIGINAL_ENCODER)


def validate_communication_encoder(communication, encoder):
    if encoder not in COMMUNICATION_ENCODERS:
        raise ValueError(f"Unknown communication encoder: {encoder}")
    if communication in ("keep", "keep_native") and encoder != LEGACY_ENCODER:
        raise ValueError("Native KEEP communication does not use the ACD encoder")
    return encoder


def load_config(path):
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")
    return validate(config)


def validate(config):
    c = deepcopy(config)
    c.setdefault("name", "experiment")
    c.setdefault("team", "mixed")
    c.setdefault("red", "finite")
    c.setdefault("communication", "paper")
    c.setdefault("horizon", "fixed")
    c.setdefault("episodes", 2)
    c.setdefault("steps", 500)
    c.setdefault("seed", 42)
    c.setdefault("threads", 1)
    c.setdefault("deterministic_rl", False)
    c.setdefault("weights", "third_party/cage-4-submission/weights")
    c.setdefault("model", {"backend": "mock", "enabled": False})
    c.setdefault("output", "runs")
    for key in ("episodes", "steps", "threads"):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if c["steps"] < 3:
        raise ValueError("CC4 needs at least 3 steps for its three mission phases")
    if type(c["seed"]) is not int or c["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if c["team"] not in TEAMS or c["red"] not in REDS:
        raise ValueError(f"team must be in {TEAMS}; red must be in {REDS}")
    if c["communication"] not in ("keep", "paper"):
        raise ValueError("communication must be keep or paper")
    c["communication_encoder"] = validate_communication_encoder(
        c["communication"], c.get("communication_encoder", LEGACY_ENCODER))
    if c["horizon"] not in ("native", "fixed"):
        raise ValueError("horizon must be native or fixed")
    if not isinstance(c["model"], dict):
        raise ValueError("model must be a mapping")
    for key in ("weights", "output", "prompt"):
        if c.get(key):
            p = Path(c[key]).expanduser()
            c[key] = str(p if p.is_absolute() else ROOT / p)
    model = c["model"]
    if model.get("replay_path"):
        p = Path(model["replay_path"]).expanduser()
        model["replay_path"] = str(p if p.is_absolute() else ROOT / p)
    if "api_key" in model:
        raise ValueError("Use api_key_env (an environment variable name), not a key in YAML")
    return c
