"""
metrics.py -- cumulative attack / detection / containment rates for the self-play loop.

Reads the per-episode trajectories (output/pipeline/trajectories.json, one trajectory
per iteration) and counts, cumulatively across iterations:

  attempted   red attacks executed
  succeeded   red attacks whose command succeeded on the target
  detected    red attacks flagged by Wazuh
  applied     blue defenses actually applied
  mitigated   attacks blue contained (defense matched the active technique)

Rates:
  attack_success_rate = succeeded / attempted          ("공격률" / 공격 성공률)
  detection_rate      = detected  / attempted          ("공격 감지율")
  mitigation_rate     = mitigated / detected           (탐지된 것 중 봉쇄 비율)
  containment_rate    = mitigated / attempted          (전체 공격 중 봉쇄 비율)

Usage:
  python metrics.py                                   # reads output/pipeline/trajectories.json
  python metrics.py --traj path/to/trajectories.json --out-dir output/pipeline
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def _rate(a: int, b: int) -> float:
    return round(a / b, 4) if b else 0.0


def _episode_counts(steps: list[dict]) -> dict:
    red = [s for s in steps if s.get("role") == "red"]
    blue = [s for s in steps if s.get("role") == "blue"]
    return {
        "attempted": len(red),
        "succeeded": sum(1 for s in red if (s.get("result") or {}).get("success")),
        "detected": sum(1 for s in red if s.get("detected")),
        "applied": sum(1 for s in blue if s.get("applied")),
        "mitigated": sum(1 for s in blue if s.get("mitigated")),
    }


def compute_metrics(trajectories: list[dict]) -> dict:
    """trajectories: list of {'steps': [...], ...}, one per iteration."""
    per_iter = []
    cum = {"attempted": 0, "succeeded": 0, "detected": 0, "applied": 0, "mitigated": 0}
    for i, traj in enumerate(trajectories, 1):
        c = _episode_counts(traj.get("steps", []))
        for k in cum:
            cum[k] += c[k]
        per_iter.append({
            "iteration": i,
            **c,
            "attack_success_rate": _rate(c["succeeded"], c["attempted"]),
            "detection_rate": _rate(c["detected"], c["attempted"]),
            "containment_rate": _rate(c["mitigated"], c["attempted"]),
            # cumulative
            "cum_attempted": cum["attempted"],
            "cum_succeeded": cum["succeeded"],
            "cum_detected": cum["detected"],
            "cum_mitigated": cum["mitigated"],
            "cum_attack_success_rate": _rate(cum["succeeded"], cum["attempted"]),
            "cum_detection_rate": _rate(cum["detected"], cum["attempted"]),
            "cum_containment_rate": _rate(cum["mitigated"], cum["attempted"]),
        })
    summary = {
        "iterations": len(trajectories),
        "totals": cum,
        "attack_success_rate": _rate(cum["succeeded"], cum["attempted"]),
        "detection_rate": _rate(cum["detected"], cum["attempted"]),
        "containment_rate": _rate(cum["mitigated"], cum["attempted"]),
    }
    return {"summary": summary, "per_iteration": per_iter}


def write_metrics(metrics: dict, out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    rows = metrics["per_iteration"]
    if rows:
        with open(os.path.join(out_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def print_metrics(metrics: dict) -> None:
    s = metrics["summary"]
    print("\n" + "=" * 70)
    print("CUMULATIVE METRICS (attack / detection)")
    print("=" * 70)
    print(f"  iter | attempt succ det mitig | cum_atkRate cum_detRate cum_containRate")
    for r in metrics["per_iteration"]:
        print(f"  {r['iteration']:>4} | {r['attempted']:>7} {r['succeeded']:>4} "
              f"{r['detected']:>3} {r['mitigated']:>5} | "
              f"{r['cum_attack_success_rate']:>10} {r['cum_detection_rate']:>11} "
              f"{r['cum_containment_rate']:>15}")
    t = s["totals"]
    print("-" * 70)
    print(f"  TOTAL attempts={t['attempted']} succeeded={t['succeeded']} "
          f"detected={t['detected']} mitigated={t['mitigated']}")
    print(f"  >> attack success rate : {s['attack_success_rate']*100:.1f}%")
    print(f"  >> detection rate      : {s['detection_rate']*100:.1f}%")
    print(f"  >> containment rate    : {s['containment_rate']*100:.1f}% (of all attacks)")


def load_and_report(traj_path: str, out_dir: str) -> dict:
    with open(traj_path, "r", encoding="utf-8") as f:
        trajectories = json.load(f)
    metrics = compute_metrics(trajectories)
    write_metrics(metrics, out_dir)
    print_metrics(metrics)
    print(f"\n[metrics] wrote metrics.json + metrics.csv -> {out_dir}/")
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Cumulative attack/detection metrics")
    ap.add_argument("--traj", default="output/pipeline/trajectories.json")
    ap.add_argument("--out-dir", default="output/pipeline")
    args = ap.parse_args()
    if not os.path.exists(args.traj):
        raise SystemExit(f"trajectories file not found: {args.traj} (run main.py first)")
    load_and_report(args.traj, args.out_dir)


if __name__ == "__main__":
    main()
