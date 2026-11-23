"""Retrieve exactly the checked source revisions for a fresh setup."""
from pathlib import Path
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
lock = json.loads((ROOT / "sources.lock.json").read_text())
for name, source in lock["sources"].items():
    path = ROOT / "third_party" / name
    if not path.exists():
        subprocess.run(["git", "clone", source["url"], str(path)], check=True)
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if actual != source["commit"]:
        dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True)
        if dirty.strip():
            raise RuntimeError(f"Source has local changes; cannot change its revision: {path}")
        subprocess.run(["git", "-C", str(path), "fetch", source["url"], source["commit"]], check=True)
        subprocess.run(["git", "-C", str(path), "checkout", "--detach", source["commit"]], check=True)
print("Source revisions match sources.lock.json")
