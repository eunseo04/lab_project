import json
from pathlib import Path

# attack_dataset.json is produced by dataset.py (build_dataset -> json.dump).
# Resolve it relative to this file / project root so the loader works from any CWD,
# and read as UTF-8 (the ATT&CK descriptions contain non-ASCII characters).
_DATASET_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _DATASET_DIR.parent

_SEARCH_PATHS = [
    _DATASET_DIR / "attack_dataset.json",
    _PROJECT_ROOT / "attack_dataset.json",
    Path("attack_dataset.json"),
]


def dataset_path():
    for p in _SEARCH_PATHS:
        if p.exists():
            return p
    return None


def load_attack_dataset():
    path = dataset_path()
    if path is None:
        raise FileNotFoundError(
            "attack_dataset.json not found. Generate it first with "
            "`python dataset/dataset.py` (requires network access to the "
            "mitre-attack/attack-stix-data repo). Searched: "
            + ", ".join(str(p) for p in _SEARCH_PATHS)
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
