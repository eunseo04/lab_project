"""Audit reachable local Python modules without importing GPU or legacy modules."""
from __future__ import annotations

import ast
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = ("ad_eval/run.py", "ad_eval/rescore.py", "ad_eval/models.py",
               "ad_eval/cai_parallel.py", "ad_eval/proposed_worker.py", "ad_eval/grpo_train.py", "train.py")
LEGACY = {"pipeline", "llm", "redtools", "wazuh_client", "environment", "eval", "evaluator",
          "strategy_improve", "playbook_gen"}


def audit(root: Path = ROOT) -> dict:
    files = {p.relative_to(root).as_posix(): p for p in root.glob("*.py")}
    for directory in ("ad_eval", "scripts", "tests"):
        for p in (root / directory).glob("*.py"):
            files[p.relative_to(root).as_posix()] = p
    modules = {name.removesuffix(".py").replace("/", "."): path for name, path in files.items()}
    visited = set()
    missing = []
    external = set()
    legacy_edges = []
    todo = [name.removesuffix(".py").replace("/", ".") for name in ENTRYPOINTS]
    todo += [name for name in modules if name.startswith("tests.")]
    while todo:
        name = todo.pop()
        if name in visited:
            continue
        if name not in modules:
            missing.append(name)
            continue
        visited.add(name)
        tree = ast.parse(modules[name].read_text(encoding="utf-8"))
        package = name.split(".")[:-1]
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Import):
                targets = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package[:len(package) - node.level + 1]
                    module = ".".join(base + ([node.module] if node.module else []))
                    targets = [module] if node.module else [module + "." + a.name for a in node.names]
                else:
                    targets = [node.module or ""]
            for target in targets:
                if name.startswith("ad_eval.") and target.split(".")[0] in LEGACY:
                    legacy_edges.append([name, target])
                if target in modules:
                    todo.append(target)
                elif target.startswith("ad_eval."):
                    missing.append(target)
                else:
                    external.add(target.split(".")[0])
    return {"reachable_local_modules": sorted(visited), "missing_local_modules": sorted(set(missing)),
            "evaluation_legacy_imports": legacy_edges, "external_imports": sorted(external),
            "unused_top_level_python": sorted(name for name in modules
                                              if "." not in name and name not in visited
                                              and name != "run_cai_mayhem"),
            "scope": "static Python imports; runtime assets are listed in README.md"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = audit()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.check and (result["missing_local_modules"] or result["evaluation_legacy_imports"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
