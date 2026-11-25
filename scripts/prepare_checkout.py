"""Prepare current B0-B4 configs without modifying source templates."""
import argparse
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def prepare(root=ROOT, destination=None):
    root = Path(root).resolve()
    source = root / 'research_configs'
    destination = Path(destination) if destination is not None else root / ".local-configs"
    destination.mkdir(parents=True, exist_ok=True)
    generated = []
    for label in ("B0", "B1", "B2", "B3", "B4", "B3-no-support"):
        config = yaml.safe_load((source / f"rollout-{label}.yaml").read_text())
        for field in ('weights','output','pool'):
            if config.get(field):
                path=Path(config[field]).expanduser()
                config[field]=str(path if path.is_absolute() else root/path)
        target = destination / f"rollout-{label}.yaml"
        target.write_text(yaml.safe_dump(config,sort_keys=False))
        generated.append(str(target))
    return generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path)
    args=parser.parse_args()
    print(json.dumps({"root": str(ROOT), "generated_configs": prepare(destination=args.destination)}, indent=2))
