"""
ATT&CK x Wazuh coverage analysis, scoped to the Target environment.

Turns the raw 168/858 (~19.6%) ID-mapping baseline into something actionable:

  Step 1  select techniques that actually apply to the Target (platform filter)
  Step 2  for each, list the logs needed to detect it (data_components / analytics)
  Step 4  flag the Target-relevant techniques Wazuh does NOT already map an ID for
          -> exactly the set that may need custom rules (don't write all 690)

The global 19.6% mixes in Windows/macOS/cloud/identity techniques that never run on
a Linux container + web app range, so it understates real coverage. Coverage computed
*within the Target set* is the honest baseline.

Inputs (produced earlier in the repo):
  attack_dataset.json                      <- dataset/dataset.py
  wazuh-docker/single-node/wazuh_mitre_ids.txt  <- IDs scraped from Wazuh rules

Usage:
  python coverage.py                       # financial_linux_web profile, prints + writes
  python coverage.py --profile all         # no platform filter (reproduces 858/168/690)
  python coverage.py --profile linux
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_DEFAULT_DATASET = _ROOT / "attack_dataset.json"
_DEFAULT_IDS = _ROOT / "wazuh-docker" / "single-node" / "wazuh_mitre_ids.txt"

_TID_RE = re.compile(r"T\d{4}(?:\.\d{3})?")

# Platform sets per Target profile. The financial cyber range is a Linux Flask
# web app + Postgres + attacker container, so Linux + Containers (+ network recon)
# are the platforms whose techniques can plausibly fire there.
TARGET_PROFILES = {
    "all": None,
    "linux": {"Linux"},
    "containers": {"Containers"},
    "financial_linux_web": {"Linux", "Containers", "Network Devices", "PRE"},
}


def load_dataset(path: Path = _DEFAULT_DATASET) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def load_wazuh_ids(path: Path = _DEFAULT_IDS) -> set[str]:
    text = Path(path).read_text(encoding="utf-8")
    return {t for t in text.split() if _TID_RE.fullmatch(t)}


def _valid_tids(dataset: dict) -> set[str]:
    return {tid for tid in dataset if _TID_RE.fullmatch(tid)}


def platforms_match(info: dict, profile_platforms: set[str] | None) -> bool:
    if profile_platforms is None:
        return True
    return bool(set(info.get("platforms", [])) & profile_platforms)


def select_target_techniques(dataset: dict, profile: str) -> dict:
    """Step 1: techniques that apply to the Target platform profile."""
    if profile not in TARGET_PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose from {list(TARGET_PROFILES)}")
    platforms = TARGET_PROFILES[profile]
    return {
        tid: info for tid, info in dataset.items()
        if _TID_RE.fullmatch(tid) and platforms_match(info, platforms)
    }


def required_logs(info: dict) -> list[str]:
    """Step 2: log sources needed to detect a technique (from the dataset)."""
    logs = []
    for comp in info.get("data_components", []) or []:
        if isinstance(comp, dict):
            ls = comp.get("log_source", {}) or {}
            label = comp.get("name") or ls.get("name") or ls.get("channel")
            if label:
                logs.append(label)
    # fall back to analytics names if no explicit data components
    if not logs:
        for an in info.get("analytics", []) or []:
            if isinstance(an, dict) and an.get("name"):
                logs.append(an["name"])
    return sorted(set(logs))


def coverage_report(dataset: dict, wazuh_ids: set[str], profile: str = "financial_linux_web") -> dict:
    target = select_target_techniques(dataset, profile)
    target_ids = set(target)

    all_ids = _valid_tids(dataset)
    global_matched = all_ids & wazuh_ids

    matched = target_ids & wazuh_ids          # covered (ID mapped in a Wazuh rule)
    uncovered = target_ids - wazuh_ids         # candidates for custom rules (step 4)

    needs_rules = []
    for tid in sorted(uncovered):
        info = target[tid]
        needs_rules.append({
            "tech_id": tid,
            "name": info.get("name"),
            "tactics": info.get("tactics", []),
            "platforms": info.get("platforms", []),
            "required_logs": required_logs(info),
        })

    return {
        "profile": profile,
        "global": {
            "total_techniques": len(all_ids),
            "wazuh_mapped": len(global_matched),
            "coverage_pct": round(100 * len(global_matched) / max(len(all_ids), 1), 1),
        },
        "target": {
            "selected_techniques": len(target_ids),
            "covered": len(matched),
            "uncovered": len(uncovered),
            "coverage_pct": round(100 * len(matched) / max(len(target_ids), 1), 1),
        },
        "covered_ids": sorted(matched),
        "needs_custom_rules": needs_rules,
    }


def covered_id_set(profile: str = "financial_linux_web",
                   dataset_path: Path = _DEFAULT_DATASET,
                   ids_path: Path = _DEFAULT_IDS) -> set[str]:
    """Helper for pipeline.py: Target techniques Wazuh already maps an ID for."""
    dataset = load_dataset(dataset_path)
    wazuh_ids = load_wazuh_ids(ids_path)
    return set(select_target_techniques(dataset, profile)) & wazuh_ids


def _print(report: dict) -> None:
    g, t = report["global"], report["target"]
    print(f"[profile: {report['profile']}]")
    print(f"  global   : {g['wazuh_mapped']}/{g['total_techniques']} "
          f"ID-mapped ({g['coverage_pct']}%)  <- the raw baseline")
    print(f"  TARGET   : {t['covered']}/{t['selected_techniques']} "
          f"ID-mapped ({t['coverage_pct']}%)  <- meaningful coverage")
    print(f"  need custom rules (Target, uncovered): {t['uncovered']}")
    print("\n  Top uncovered Target techniques (candidates for custom rules):")
    for item in report["needs_custom_rules"][:15]:
        logs = ", ".join(item["required_logs"][:3]) or "-"
        print(f"    {item['tech_id']:10} {str(item['name'])[:38]:38} logs: {logs}")


def main():
    ap = argparse.ArgumentParser(description="ATT&CK x Wazuh coverage (Target-scoped)")
    ap.add_argument("--profile", default="financial_linux_web", choices=list(TARGET_PROFILES))
    ap.add_argument("--dataset", default=str(_DEFAULT_DATASET))
    ap.add_argument("--ids", default=str(_DEFAULT_IDS))
    ap.add_argument("--out-dir", default="output/coverage")
    args = ap.parse_args()

    dataset = load_dataset(Path(args.dataset))
    wazuh_ids = load_wazuh_ids(Path(args.ids))
    report = coverage_report(dataset, wazuh_ids, profile=args.profile)
    _print(report)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"coverage_{args.profile}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / f"needs_custom_rules_{args.profile}.json").write_text(
        json.dumps(report["needs_custom_rules"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[coverage] wrote artifacts -> {out}/")


if __name__ == "__main__":
    main()
