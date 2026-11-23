"""Obtain all four pinned research sources without editing local changes."""
from pathlib import Path
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
lock = json.loads((ROOT / "research_sources.lock.json").read_text())
for name, source in lock["sources"].items():
    path = ROOT / "third_party" / name
    if not path.exists():
        subprocess.run(["git", "clone", source["url"], str(path)], check=True)
    dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True)
    changes = [line for line in dirty.splitlines()
               if not (line.startswith("?? ") and ("__pycache__/" in line or line.endswith(".pyc")))]
    if changes:
        raise RuntimeError(f"Local changes preserved; source verification failed: {path}")
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if actual != source["commit"]:
        subprocess.run(["git", "-C", str(path), "fetch", source["url"], source["commit"]], check=True)
        subprocess.run(["git", "-C", str(path), "checkout", "--detach", source["commit"]], check=True)
print("All four source revisions match research_sources.lock.json; source files are clean (Python bytecode ignored)")
