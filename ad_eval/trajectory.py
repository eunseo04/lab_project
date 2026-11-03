"""Export complete role trajectories and replay prefixes without secret flag text."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re


def sanitize(value):
    if isinstance(value, str):
        return re.sub(r"FLAG\{[^\s{}]+\}", "[REDACTED_FLAG]", value)
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def export_trajectories(directory: Path, match: dict, events: dict[tuple[str, str], list[dict]]) -> list[Path]:
    timeline = []
    for (team, role), stream in events.items():
        for event in stream:
            if event.get("kind") == "tool_result" and event.get("command") and event.get("decision_id"):
                timeline.append({key: event.get(key) for key in
                                 ("decision_id", "command", "started_utc", "ended_utc")}
                                | {"team": team, "role": role})
    timeline.sort(key=lambda event: event.get("ended_utc") or "")
    exported = []
    for team in ("1", "2"):
        for role in ("red", "blue"):
            stream = events[(team, role)]
            outcomes = {e["decision_id"]: e for e in stream
                        if e.get("kind") in {"tool_result", "action_rejected"} and e.get("decision_id")}
            steps = []
            for event in stream:
                if event.get("kind") != "policy_decision" or event.get("decision_id") not in outcomes:
                    continue  # exclude a generation still in flight at the deadline
                outcome = outcomes[event["decision_id"]]
                before = event["started_utc"]
                prefix = [e for e in timeline if e["decision_id"] != event["decision_id"]
                          and e.get("ended_utc") and e["ended_utc"] <= before]
                steps.append({"row_id": event["decision_id"], "prompt": event["messages"],
                              "completion": event["completion"], "command": event["command"],
                              "strategy_id": event.get("strategy_id"), "tech_id": event.get("tech_id"),
                              "observation": outcome.get("output", ""),
                              "rejected": outcome.get("kind") == "action_rejected",
                              "replay_prefix": prefix})
            trajectory = {"trajectory_id": f"{match['match_id']}:team{team}:{role}",
                          "match_id": match["match_id"], "rep": match["rep"], "team": team, "role": role,
                          "steps": sanitize(steps), "inference": match["inference"],
                          "model": match.get("role_models", {}).get(role, match.get("model")),
                          "team_score": next(row for row in match["teams"] if row["team"] == team),
                          "replay_semantics": "fresh private range; completed prior tool calls serialized by completion time"}
            path = directory / f"team{team}_{role}.trajectory.json"
            path.write_text(json.dumps(trajectory, ensure_ascii=False, indent=2), encoding="utf-8")
            path.chmod(0o600)
            exported.append(path)
    return exported
