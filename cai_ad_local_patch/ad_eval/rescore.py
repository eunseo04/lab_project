"""Recompute paper metrics from saved measurements after blind human review.

Never rerun the stochastic match just to add a human detection label.
Raw records.json is retained unchanged, and score_reviewed.json is separate.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .evidence import merge_manual_review
from .scoring import score_match, summarize_matches


def rescore_directory(out_dir: Path, labels: dict) -> dict:
    matches = []
    for path in sorted(out_dir.glob("rep*_*/*records.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        records = []
        for rec in raw["records"]:
            key = f'{raw["rep"]}:{raw["condition"]}:{rec["team"]}'
            records.append(merge_manual_review(rec, labels.get(key, {})))
        match = score_match(records[0], records[1], raw["duration_s"], raw["interval_s"])
        match["match_id"] = raw["match_id"]
        match["source"] = "local_mayhem_cyber_range_not_HTB"
        (path.parent / "score_reviewed.json").write_text(
            json.dumps(match, ensure_ascii=False, indent=2), encoding="utf-8")
        matches.append(match)
    if not matches:
        raise FileNotFoundError(f"no records.json found under {out_dir}")
    return {"matches": matches, "by_condition": summarize_matches(matches),
            "reviewed_labels": sorted(labels), "scoring": "local_cai_ad_v1"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="output/ad_eval")
    p.add_argument("--labels", required=True, help="JSON with independently reviewed match labels")
    a = p.parse_args()
    out_dir = Path(a.out_dir)
    labels = json.loads(Path(a.labels).read_text(encoding="utf-8"))
    result = rescore_directory(out_dir, labels)
    path = out_dir / "summary_reviewed.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ad_eval] reviewed summary: {path}")


if __name__ == "__main__":
    main()
