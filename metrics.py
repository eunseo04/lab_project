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
    def succ(s):
        return bool((s.get("result") or {}).get("success"))
    attempted = len(red)
    succeeded = sum(1 for s in red if succ(s))
    # Each attempt is exactly one of: blocked / contained / breached.
    #   blocked   = attack did not execute on the target
    #   contained = executed BUT blue mitigated the same technique
    #   breached  = executed AND not contained  (this is NET attack success)
    contained = sum(1 for s in red if succ(s) and s.get("mitigated"))
    breached = sum(1 for s in red if succ(s) and not s.get("mitigated"))
    blocked = attempted - succeeded
    return {
        "attempted": attempted,
        "succeeded": succeeded,        # raw: attack command ran (ignores defense)
        "breached": breached,          # net attack success (got through)
        "contained": contained,        # succeeded but neutralized by blue
        "blocked": blocked,            # failed to execute
        "detected": sum(1 for s in red if s.get("detected")),
    }


def compute_metrics(trajectories: list[dict]) -> dict:
    """trajectories: list of {'steps': [...], ...}, one per iteration."""
    per_iter = []
    cum = {"attempted": 0, "succeeded": 0, "breached": 0, "contained": 0, "blocked": 0, "detected": 0}
    for i, traj in enumerate(trajectories, 1):
        c = _episode_counts(traj.get("steps", []))
        for k in cum:
            cum[k] += c[k]
        per_iter.append({
            "iteration": i,
            **c,
            # raw execution (ignores defense) -- does NOT move with containment
            "attack_success_rate": _rate(c["succeeded"], c["attempted"]),
            # the partition: these three sum to 1.0 and trade off against each other
            "breach_rate": _rate(c["breached"], c["attempted"]),        # net attack success
            "containment_rate": _rate(c["contained"], c["attempted"]),  # 방어(봉쇄)율
            "block_rate": _rate(c["blocked"], c["attempted"]),
            "detection_rate": _rate(c["detected"], c["attempted"]),
            # cumulative
            "cum_attempted": cum["attempted"],
            "cum_succeeded": cum["succeeded"],
            "cum_detected": cum["detected"],
            "cum_breached": cum["breached"],
            "cum_contained": cum["contained"],
            "cum_attack_success_rate": _rate(cum["succeeded"], cum["attempted"]),
            "cum_breach_rate": _rate(cum["breached"], cum["attempted"]),
            "cum_containment_rate": _rate(cum["contained"], cum["attempted"]),
            "cum_block_rate": _rate(cum["blocked"], cum["attempted"]),
            "cum_detection_rate": _rate(cum["detected"], cum["attempted"]),
        })
    summary = {
        "iterations": len(trajectories),
        "totals": cum,
        "attack_success_rate": _rate(cum["succeeded"], cum["attempted"]),  # raw execution
        "breach_rate": _rate(cum["breached"], cum["attempted"]),           # net attack success
        "containment_rate": _rate(cum["contained"], cum["attempted"]),     # defense rate
        "block_rate": _rate(cum["blocked"], cum["attempted"]),
        "detection_rate": _rate(cum["detected"], cum["attempted"]),
        # breach + containment + block == 1.0
        "defense_rate": _rate(cum["contained"] + cum["blocked"], cum["attempted"]),
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
    print("\n" + "=" * 72)
    print("CUMULATIVE METRICS  (each attempt = blocked | contained | breached)")
    print("=" * 72)
    print(f"  iter | attempt breach contain block det | cum_breach cum_contain cum_block")
    for r in metrics["per_iteration"]:
        print(f"  {r['iteration']:>4} | {r['attempted']:>7} {r['breached']:>6} "
              f"{r['contained']:>7} {r['blocked']:>5} {r['detected']:>3} | "
              f"{r['cum_breach_rate']:>10} {r['cum_containment_rate']:>11} {r['cum_block_rate']:>9}")
    t = s["totals"]
    print("-" * 72)
    print(f"  TOTAL attempts={t['attempted']} breached={t['breached']} "
          f"contained={t['contained']} blocked={t['blocked']} detected={t['detected']}")
    print(f"  >> breach rate (net attack success) : {s['breach_rate']*100:.1f}%")
    print(f"  >> containment rate (defense/bongswae): {s['containment_rate']*100:.1f}%")
    print(f"  >> block rate                       : {s['block_rate']*100:.1f}%")
    print(f"  >> defense rate (contain+block)     : {s['defense_rate']*100:.1f}%"
          f"   [breach + defense = 100%]")
    print(f"  (raw attack execution rate = {s['attack_success_rate']*100:.1f}%, ignores defense)")
    print(f"  >> detection rate                   : {s['detection_rate']*100:.1f}%")


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
