"""Persistent, deduplicated strategic improvements ordered by observed need."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class IdeaQueue:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.items = json.loads(self.path.read_text()) if self.path.exists() else []

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.items, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(self.path)

    def generate(self, reflection: dict, parent_id: str):
        for finding in reflection["findings"]:
            proposal = finding.get("proposal", finding["hypothesis"])
            identity = [reflection["role"], parent_id, finding["kind"], proposal]
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
            item = next((item for item in self.items if item["id"] == key), None)
            if item is None:
                item = {"id": key, "role": reflection["role"], "parent_id": parent_id,
                        "kind": finding["kind"], "proposal": proposal,
                        "status": "pending", "sources": [], "evaluations": []}
                self.items.append(item)
            if not any(source["reflection_id"] == reflection["reflection_id"] for source in item["sources"]):
                if item["status"] == "inconclusive":
                    item["status"] = "pending"
                item["sources"].append({"reflection_id": reflection["reflection_id"],
                                        "trajectory_id": reflection["trajectory_id"],
                                        "step_ids": finding["step_ids"],
                                        "terminal_fields": finding.get("terminal_fields", []),
                                        "priority_hint": finding.get("priority", 1),
                                        "observed_return": reflection["observed_return"]})
            # This heuristic ranks testing needs, never claims a measured gain.
            gap = sum(1 - source["observed_return"] for source in item["sources"]) / len(item["sources"])
            item["priority"] = round(gap + .1 * min(5, len(item["sources"])) +
                                     (.25 if item["kind"] in {"policy_rejection", "availability_regression"} else 0) +
                                     .2 * max(source.get("priority_hint", 1) for source in item["sources"]) / 5., 6)
        self.save()

    def next(self, role: str, parent_id: str) -> dict | None:
        candidates = [item for item in self.items if item["role"] == role and
                      item["parent_id"] == parent_id and item["status"] == "pending"]
        return max(candidates, key=lambda item: (item["priority"], item["id"])) if candidates else None

    def finish(self, key: str, report: dict):
        item = next(item for item in self.items if item["id"] == key)
        item["status"] = report["status"]
        item["evaluations"].append(report)
        self.save()
