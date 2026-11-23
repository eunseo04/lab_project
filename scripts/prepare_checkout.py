"""Prepare portable run configs without changing registered research snapshots."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDED_ROOT = "/home/jeonges/proposed"


def prepare(root=ROOT, destination=None):
    root = Path(root).resolve()
    source = root / "revisions/rollout-v5-reusable/research_configs"
    destination = Path(destination) if destination is not None else root / ".local-configs"
    destination.mkdir(parents=True, exist_ok=True)
    generated = []
    for label in ("B0", "B1", "B2", "B3", "B4"):
        text = (source / f"rollout-{label}.yaml").read_text()
        text = text.replace(RECORDED_ROOT, str(root))
        target = destination / f"rollout-{label}.yaml"
        target.write_text(text)
        generated.append(str(target))
    return generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps({"root": str(ROOT), "generated_configs": prepare()}, indent=2))
