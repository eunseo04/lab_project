"""Explicit base/GRPO model selection and checkpoint registration, without GPU imports."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path

import requests

from .llm import endpoint_base, served_model


@dataclass(frozen=True)
class ModelSpec:
    model: str
    endpoint: str
    kind: str
    checkpoint: str | None = None


def select_model(args, condition: str) -> ModelSpec:
    if condition in {"cai", "proposed_base"}:
        return ModelSpec(served_model(args.base_model), endpoint_base(args.endpoint), "base")
    if condition != "proposed" or not args.trained_model:
        raise ValueError("proposed requires an explicit --trained-model served GRPO policy")
    if served_model(args.trained_model) == served_model(args.base_model) and not args.trained_endpoint:
        raise ValueError("base and GRPO need distinct served names or distinct endpoints")
    return ModelSpec(served_model(args.trained_model),
                     endpoint_base(args.trained_endpoint or args.endpoint), "grpo",
                     args.grpo_checkpoint)


def checkpoint_info(path: str | Path) -> dict:
    directory = Path(path).expanduser().resolve(strict=True)
    config = directory / "adapter_config.json"
    if not config.is_file():
        raise ValueError("pass the exact LoRA checkpoint directory containing adapter_config.json")
    weights = [p for p in directory.glob("adapter_model.*") if p.suffix in {".safetensors", ".bin"}]
    if not weights:
        raise ValueError("checkpoint has no adapter weights")
    data = json.loads(config.read_text(encoding="utf-8"))
    hashes = {}
    for item in [config, *sorted(weights)]:
        digest = hashlib.sha256()
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        hashes[item.name] = digest.hexdigest()
    return {"path": str(directory), "base_model": data.get("base_model_name_or_path"),
            "sha256": hashes,
            "training_provenance": "user-supplied GRPO checkpoint; algorithm not inferred from weights"}


def verify_model(spec: ModelSpec) -> dict:
    headers = {"Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "EMPTY")}
    response = requests.get(spec.endpoint + "/models", headers=headers, timeout=15)
    response.raise_for_status()
    card = next((row for row in response.json()["data"] if row["id"] == spec.model), None)
    if card is None:
        raise RuntimeError(f"model {spec.model!r} is absent from {spec.endpoint}/models")
    result = {**asdict(spec), "server_model": card}
    if spec.checkpoint:
        info = checkpoint_info(spec.checkpoint)
        # vLLM reports the adapter directory as root. A friendly alias alone is
        # insufficient to verify that the intended weights were registered.
        if str(card.get("root", "")).rstrip("/") != info["path"]:
            raise RuntimeError("served adapter root differs from the requested GRPO checkpoint")
        result["checkpoint_artifact"] = info
    return result


def load_adapter(spec: ModelSpec) -> dict:
    if not spec.checkpoint:
        raise ValueError("--checkpoint is required to register a GRPO LoRA adapter")
    info = checkpoint_info(spec.checkpoint)
    response = requests.post(
        spec.endpoint + "/load_lora_adapter",
        headers={"Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "EMPTY")},
        json={"lora_name": spec.model, "lora_path": info["path"]}, timeout=120,
    )
    response.raise_for_status()
    return verify_model(spec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Register an existing GRPO LoRA checkpoint in vLLM")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--trained-model", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    spec = ModelSpec(served_model(args.trained_model), endpoint_base(args.endpoint), "grpo", args.checkpoint)
    print(json.dumps(load_adapter(spec), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
