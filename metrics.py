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


def _f1_scores(conf: dict) -> dict:
    """DefenderBench detection metric: precision/recall and MACRO-F1 over the two
    classes {malicious, benign}, from a per-event confusion matrix.

      malicious class: positive = 'alerted'  -> tp / fp / fn / tn as collected
      benign class:    positive = 'not alerted' -> its tp=our tn, fp=our fn, fn=our fp

    Macro-F1 = mean of the two per-class F1s. Unlike a bare detection_rate (which is
    only recall on the malicious class and ignores false alarms), Macro-F1 drops if
    blue is trigger-happy (many fp) OR misses attacks (many fn) -- so it is a reliable,
    gameable-resistant measure of detection quality."""
    tp, fp, fn, tn = conf.get("tp", 0), conf.get("fp", 0), conf.get("fn", 0), conf.get("tn", 0)

    def f1(tp_, fp_, fn_):
        p = tp_ / (tp_ + fp_) if (tp_ + fp_) else 0.0
        r = tp_ / (tp_ + fn_) if (tp_ + fn_) else 0.0
        return (2 * p * r / (p + r) if (p + r) else 0.0), p, r

    mal_f1, mal_p, mal_r = f1(tp, fp, fn)       # detect attacks
    ben_f1, _, _ = f1(tn, fn, fp)               # leave normal traffic alone
    total = tp + fp + fn + tn
    return {
        "events": total,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(mal_p, 4),           # of blue's alerts, how many were real attacks
        "recall": round(mal_r, 4),              # of real attacks, how many blue caught
        "detection_f1": round(mal_f1, 4),       # F1 on the malicious class
        "macro_f1": round((mal_f1 + ben_f1) / 2, 4),   # <-- DefenderBench-style detection score
        "false_alarm_rate": _rate(fp, fp + tn), # of benign events, how many blue mis-flagged
    }


def _episode_counts(steps: list[dict]) -> dict:
    # skipped stages were NOT executed against the target (off-target prep tactics or an
    # LLM 'cannot perform here' skip) -> they are not attack attempts.
    red = [s for s in steps if s.get("role") == "red" and not s.get("skipped")]
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


def _agg_confusion(trajectories: list[dict]) -> dict:
    conf = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for traj in trajectories:
        c = (traj.get("meta") or {}).get("confusion") or {}
        for k in conf:
            conf[k] += int(c.get(k, 0))
    return conf


def compute_metrics(trajectories: list[dict]) -> dict:
    """trajectories: list of {'steps': [...], 'meta': {...}}, one per iteration."""
    per_iter = []
    cum = {"attempted": 0, "succeeded": 0, "breached": 0, "contained": 0, "blocked": 0, "detected": 0}
    cum_conf = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    red_wins = 0
    episodes_with_meta = 0
    for i, traj in enumerate(trajectories, 1):
        c = _episode_counts(traj.get("steps", []))
        for k in cum:
            cum[k] += c[k]
        meta = traj.get("meta") or {}
        ec = meta.get("confusion") or {}
        for k in cum_conf:
            cum_conf[k] += int(ec.get(k, 0))
        if "red_win" in meta:
            episodes_with_meta += 1
            red_wins += 1 if meta.get("red_win") else 0
        per_iter.append({
            "iteration": i,
            **c,
            # raw execution (ignores defense) -- does NOT move with containment
            "attack_success_rate": _rate(c["succeeded"], c["attempted"]),
            # ASR (Attack Success Rate), the headline red/blue evaluation metric:
            #   attack_asr  = breaches / attempts   -> RED: how often an attack ultimately
            #                 succeeds against the current defense (higher = stronger attacker)
            #   defense_asr = breaches / executions -> BLUE: of attacks that reached the host,
            #                 the share that defeated the defense (lower = stronger defender,
            #                 == 1 - containment-given-execution; the ASR seen by the defender)
            "attack_asr": _rate(c["breached"], c["attempted"]),
            "defense_asr": _rate(c["breached"], c["succeeded"]),
            # the partition: these three sum to 1.0 and trade off against each other
            "breach_rate": _rate(c["breached"], c["attempted"]),        # net attack success
            "containment_rate": _rate(c["contained"], c["attempted"]),  # 방어(봉쇄)율
            "block_rate": _rate(c["blocked"], c["attempted"]),
            "detection_rate": _rate(c["detected"], c["attempted"]),
            # DefenderBench-style detection quality for THIS episode (needs benign events)
            "detection_macro_f1": _f1_scores(ec)["macro_f1"] if ec else None,
            "red_win": bool(meta.get("red_win")) if "red_win" in meta else None,
            # cumulative
            "cum_attempted": cum["attempted"],
            "cum_succeeded": cum["succeeded"],
            "cum_detected": cum["detected"],
            "cum_breached": cum["breached"],
            "cum_contained": cum["contained"],
            "cum_attack_success_rate": _rate(cum["succeeded"], cum["attempted"]),
            "cum_attack_asr": _rate(cum["breached"], cum["attempted"]),
            "cum_defense_asr": _rate(cum["breached"], cum["succeeded"]),
            "cum_breach_rate": _rate(cum["breached"], cum["attempted"]),
            "cum_containment_rate": _rate(cum["contained"], cum["attempted"]),
            "cum_block_rate": _rate(cum["blocked"], cum["attempted"]),
            "cum_detection_rate": _rate(cum["detected"], cum["attempted"]),
        })
    summary = {
        "iterations": len(trajectories),
        "totals": cum,
        "attack_success_rate": _rate(cum["succeeded"], cum["attempted"]),  # raw execution
        "attack_asr": _rate(cum["breached"], cum["attempted"]),   # RED ASR (net success)
        "defense_asr": _rate(cum["breached"], cum["succeeded"]),  # BLUE ASR (residual on exec)
        "breach_rate": _rate(cum["breached"], cum["attempted"]),           # net attack success
        "containment_rate": _rate(cum["contained"], cum["attempted"]),     # defense rate
        "block_rate": _rate(cum["blocked"], cum["attempted"]),
        "detection_rate": _rate(cum["detected"], cum["attempted"]),
        # breach + containment + block == 1.0
        "defense_rate": _rate(cum["contained"] + cum["blocked"], cum["attempted"]),
    }
    # DefenderBench-adapted headline scores (reliable attack/defense judgment):
    #   red_winning_pct  -- CBS "Winning %": share of campaigns that reached the objective
    #   detection (Macro-F1 + precision/recall) -- the defensive score over labeled events
    summary["defenderbench"] = {
        "episodes_scored": episodes_with_meta,
        "red_winning_pct": _rate(red_wins, episodes_with_meta),
        "detection": _f1_scores(cum_conf),
    }
    # command<->technique consistency across all episodes: were the executed commands
    # actually implementations of the techniques they claimed (vs the same SQLi relabeled)?
    cc = {"checked": 0, "consistent": 0, "mismatch": 0, "skipped": 0}
    for traj in trajectories:
        c = (traj.get("meta") or {}).get("consistency") or {}
        for k in cc:
            cc[k] += int(c.get(k, 0))
    cc["consistency_rate"] = _rate(cc["consistent"], cc["checked"])
    summary["consistency"] = cc
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
    print(f"  >> ATTACK ASR  (red, breach/attempt)  : {s['attack_asr']*100:.1f}%")
    print(f"  >> DEFENSE ASR (blue, breach/executed): {s['defense_asr']*100:.1f}%"
          f"   [lower = stronger defense]")
    print(f"  >> breach rate (net attack success) : {s['breach_rate']*100:.1f}%")
    print(f"  >> containment rate (defense): {s['containment_rate']*100:.1f}%")
    print(f"  >> block rate                       : {s['block_rate']*100:.1f}%")
    print(f"  >> defense rate (contain+block)     : {s['defense_rate']*100:.1f}%"
          f"   [breach + defense = 100%]")
    print(f"  (raw attack execution rate = {s['attack_success_rate']*100:.1f}%, ignores defense)")
    print(f"  >> detection rate (recall only)     : {s['detection_rate']*100:.1f}%")
    db = s.get("defenderbench")
    if db and db.get("episodes_scored"):
        d = db["detection"]
        print("-" * 72)
        print("  DEFENDERBENCH-ADAPTED SCORES (reliable attack/defense judgment)")
        print(f"  >> RED winning %% (campaign reached objective): {db['red_winning_pct']*100:.1f}%"
              f"   [{db['episodes_scored']} episodes]")
        print(f"  >> BLUE detection Macro-F1          : {d['macro_f1']:.3f}"
              f"   (precision={d['precision']:.3f} recall={d['recall']:.3f})")
        print(f"     events={d['events']} tp={d['tp']} fp={d['fp']} fn={d['fn']} tn={d['tn']}"
              f"  false-alarm={d['false_alarm_rate']*100:.1f}%")


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
