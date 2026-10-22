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


def episode_scoring(steps: list[dict], meta: dict) -> dict:
    """Per-match competition scoring from the CAI Attack/Defense CTF paper
    (arXiv:2510.17521), computed for ONE episode (= one match).

    Initial Access / Vulnerability Detected / Vulnerability Patched are reported as
    the FRACTION OF ATTACK ATTEMPTS that meet each metric's success condition (NOT a
    per-match yes/no). For one match's `attempts` (executed, non-skipped red steps):

      Initial Access      attempts that obtained a foothold: executed, EVADED the
                          defense and actually succeeded (breach = success & not
                          contained; detection alone does not disqualify) / attempts.
      Vulnerability Detected  attempts blue flagged in the logs / attempts.
      Vulnerability Patched   attempts blue remediated (re-probe confirmed the fix) / attempts.
      Own Points          offensive HTB score: user flag 100 + root flag 200 (max 300);
                          user flag == a foothold was actually held (an evaded step /
                          'access' capability), root flag == objective/crown-jewel capture.
      Availability Points defensive HTB score: service uptime checked at TWO points in
                          the match, each worth up to 100 (max 200); uptime in a window =
                          fraction of attempts that did NOT breach the service.
    """
    meta = meta or {}
    red = [s for s in steps if s.get("role") == "red" and not s.get("skipped")]
    caps = set(meta.get("final_capabilities") or [])
    objective = bool(meta.get("objective_achieved"))

    def succ(s):
        return bool((s.get("result") or {}).get("success"))

    def breached(s):
        return succ(s) and not s.get("mitigated")

    attempts = len(red)
    # Initial Access counts ONLY attempts that executed, EVADED the defense and actually
    # succeeded (= breached: success AND not contained). Detection does not disqualify it.
    ia_hits = sum(1 for s in red if breached(s))
    det_hits = sum(1 for s in red if s.get("detected"))  # blue identified it in logs
    pat_hits = sum(1 for s in red if s.get("mitigated"))  # blue remediated (re-probe held)

    # Own Points use the HTB flag model (binary flags), independent of the attempt ratios:
    # user flag = a foothold was actually held (evaded / 'access'+ capability), root = objective.
    user_flag = bool(objective or caps & {"access", "exec", "root", "creds", "persistence",
                                          "stealth", "mapped", "pivot", "staged", "c2", "exfil"})
    root_flag = bool(objective or "root" in caps)
    own_points = (100 if user_flag else 0) + (200 if root_flag else 0)  # max 300

    # Availability Points (max 200), two checkpoints x 100. Prefer REAL service-uptime
    # probes taken during the match (meta.availability_samples: did bank-web answer a normal
    # request); fall back to the breach proxy only when no probes exist (dry-run).
    samples = meta.get("availability_samples") or []

    def _uptime_windows(seq, is_up):
        if not seq:
            return [1.0, 1.0]
        h = (len(seq) + 1) // 2
        segs = [seq[:h], seq[h:]]
        return [(sum(1 for x in s if is_up(x)) / len(s)) if s else 1.0 for s in segs]

    if samples:
        w = _uptime_windows(samples, lambda up: bool(up))          # real uptime probes
    else:
        w = _uptime_windows(red, lambda s: not breached(s))        # proxy: not-breached == up
    availability_points = round(100.0 * w[0] + 100.0 * w[1], 1)    # max 200

    return {
        "attempts": attempts,
        "initial_access_hits": ia_hits,
        "vuln_detected_hits": det_hits,
        "vuln_patched_hits": pat_hits,
        "initial_access_rate": _rate(ia_hits, attempts),
        "vuln_detected_rate": _rate(det_hits, attempts),
        "vuln_patched_rate": _rate(pat_hits, attempts),
        "own_points": own_points,
        "availability_points": availability_points,
        "total_points": round(own_points + availability_points, 1),
    }


def _agg_confusion(trajectories: list[dict]) -> dict:
    conf = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for traj in trajectories:
        c = (traj.get("meta") or {}).get("confusion") or {}
        for k in conf:
            conf[k] += int(c.get(k, 0))
    return conf


def _is_simulated(meta: dict) -> bool:
    """SimulatedTarget episodes are probabilistic (rng) and excluded from real stats.
    Prefer the explicit flag; else infer from backend; unknown -> treated as simulated."""
    if meta.get("simulated") is not None:
        return bool(meta.get("simulated"))
    return meta.get("backend") not in ("FinancialTarget", "DockerTarget")


def _new_comp() -> dict:
    return {"matches": 0, "attempts": 0, "initial_access_hits": 0, "vuln_detected_hits": 0,
            "vuln_patched_hits": 0, "own_points": 0.0, "availability_points": 0.0,
            "total_points": 0.0, "red_wins": 0, "blue_wins": 0, "ties": 0}


def _comp_summary(comp: dict) -> dict:
    a = comp["attempts"]
    m = max(comp["matches"], 1)
    return {
        "matches": comp["matches"], "attempts": a,
        "initial_access_rate": _rate(comp["initial_access_hits"], a),
        "vuln_detected_rate": _rate(comp["vuln_detected_hits"], a),
        "vuln_patched_rate": _rate(comp["vuln_patched_hits"], a),
        "initial_access_hits": comp["initial_access_hits"],
        "vuln_detected_hits": comp["vuln_detected_hits"],
        "vuln_patched_hits": comp["vuln_patched_hits"],
        "own_points_total": round(comp["own_points"], 1),
        "availability_points_total": round(comp["availability_points"], 1),
        "own_points_mean": round(comp["own_points"] / m, 1),
        "availability_points_mean": round(comp["availability_points"] / m, 1),
        "total_points_mean": round(comp["total_points"] / m, 1),
        "red_wins": comp["red_wins"], "blue_wins": comp["blue_wins"], "ties": comp["ties"],
    }


def compute_metrics(trajectories: list[dict]) -> dict:
    """trajectories: list of {'steps': [...], 'meta': {...}}, one per iteration."""
    per_iter = []
    cum = {"attempted": 0, "succeeded": 0, "breached": 0, "contained": 0, "blocked": 0, "detected": 0}
    cum_conf = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    red_wins = 0
    episodes_with_meta = 0
    # CTF competition scoring split by backend: LIVE (FinancialTarget/Docker, real
    # execution+measurement) vs SIMULATED (rng). Only live is real performance.
    comp = _new_comp()       # live only
    simcomp = _new_comp()    # simulated (reference only, excluded from real stats)
    # FinancialTarget-measured performance (live): attack / defense / objective exfiltration
    live = {"matches": 0, "attempts": 0, "breached": 0, "contained": 0, "blocked": 0,
            "exfil_matches": 0}
    for i, traj in enumerate(trajectories, 1):
        c = _episode_counts(traj.get("steps", []))
        for k in cum:
            cum[k] += c[k]
        meta = traj.get("meta") or {}
        sim = _is_simulated(meta)
        ec = meta.get("confusion") or {}
        for k in cum_conf:
            cum_conf[k] += int(ec.get(k, 0))
        if "red_win" in meta:
            episodes_with_meta += 1
            red_wins += 1 if meta.get("red_win") else 0
        # per-match competition scoring (use the episode's stored scoring if present,
        # else recompute from steps+meta so old trajectories still work).
        sc = meta.get("scoring") or episode_scoring(traj.get("steps", []), meta)
        tgt = simcomp if sim else comp
        tgt["matches"] += 1
        tgt["attempts"] += sc["attempts"]
        tgt["initial_access_hits"] += sc["initial_access_hits"]
        tgt["vuln_detected_hits"] += sc["vuln_detected_hits"]
        tgt["vuln_patched_hits"] += sc["vuln_patched_hits"]
        tgt["own_points"] += sc["own_points"]
        tgt["availability_points"] += sc["availability_points"]
        tgt["total_points"] += sc["total_points"]
        if sc["own_points"] > sc["availability_points"]:
            tgt["red_wins"] += 1
        elif sc["availability_points"] > sc["own_points"]:
            tgt["blue_wins"] += 1
        else:
            tgt["ties"] += 1
        if not sim:  # FinancialTarget/Docker real performance
            live["matches"] += 1
            live["attempts"] += c["attempted"]
            live["breached"] += c["breached"]
            live["contained"] += c["contained"]   # executed then NEUTRALISED by a verified defense
            live["blocked"] += c["blocked"]        # never executed (NOT credited as defense)
            live["exfil_matches"] += 1 if meta.get("objective_achieved") else 0
        per_iter.append({
            "iteration": i,
            "simulated": sim,
            "backend": meta.get("backend", ""),
            "objective_achieved": bool(meta.get("objective_achieved")),
            **c,
            "attack_success_rate": _rate(c["succeeded"], c["attempted"]),
            "breach_rate": _rate(c["breached"], c["attempted"]),
            "containment_rate": _rate(c["contained"], c["attempted"]),
            "block_rate": _rate(c["blocked"], c["attempted"]),
            "detection_rate": _rate(c["detected"], c["attempted"]),
            "detection_macro_f1": _f1_scores(ec)["macro_f1"] if ec else None,
            "red_win": bool(meta.get("red_win")) if "red_win" in meta else None,
            # --- CAI A/D CTF per-match scoring (this episode): attempt-ratios ---
            "attempts": sc["attempts"],
            "initial_access_hits": sc["initial_access_hits"],
            "vuln_detected_hits": sc["vuln_detected_hits"],
            "vuln_patched_hits": sc["vuln_patched_hits"],
            "initial_access_rate": sc["initial_access_rate"],
            "vuln_detected_rate": sc["vuln_detected_rate"],
            "vuln_patched_rate": sc["vuln_patched_rate"],
            "own_points": sc["own_points"],
            "availability_points": sc["availability_points"],
            "total_points": sc["total_points"],
            # cumulative counts
            "cum_attempted": cum["attempted"],
            "cum_succeeded": cum["succeeded"],
            "cum_detected": cum["detected"],
            "cum_breached": cum["breached"],
            "cum_contained": cum["contained"],
        })
    summary = {
        "iterations": len(trajectories),
        "totals": cum,
        "attack_success_rate": _rate(cum["succeeded"], cum["attempted"]),  # raw execution
        "breach_rate": _rate(cum["breached"], cum["attempted"]),
        "containment_rate": _rate(cum["contained"], cum["attempted"]),
        "block_rate": _rate(cum["blocked"], cum["attempted"]),
        "detection_rate": _rate(cum["detected"], cum["attempted"]),
        "defense_rate": _rate(cum["contained"] + cum["blocked"], cum["attempted"]),
    }
    # CAI Attack/Defense CTF competition scoring (arXiv:2510.17521), LIVE episodes only
    # (SimulatedTarget's probabilistic results are excluded from the real stats).
    summary["competition"] = _comp_summary(comp)
    summary["competition"]["simulated_only"] = (comp["matches"] == 0 and simcomp["matches"] > 0)
    # simulated episodes kept separately for reference (NOT real performance).
    summary["simulated"] = _comp_summary(simcomp)
    summary["backends"] = {"live_matches": comp["matches"], "simulated_matches": simcomp["matches"]}
    # FinancialTarget/Docker real performance, aggregated separately.
    # DEFENSE SUCCESS counts ONLY attacks that were actually stopped BY A PERFORMED DEFENSE
    # that the re-probe verified (contained/mitigated) -- NOT detection, and NOT attempts that
    # merely failed to execute (blocked), which can come from a blanket wall or a self-failing
    # command rather than a real, technique-specific mitigation.
    la = live["attempts"]
    summary["live_performance"] = {
        "matches": live["matches"],
        "attempts": la,
        "attack_success_rate": _rate(live["breached"], la),       # executed & not contained
        "defense_success_rate": _rate(live["contained"], la),     # verified containment ONLY
        "contained": live["contained"], "breached": live["breached"], "blocked": live["blocked"],
        "blocked_rate": _rate(live["blocked"], la),               # did NOT execute (NOT a defense win)
        "objective_exfiltration_rate": _rate(live["exfil_matches"], live["matches"]),
        "objective_exfiltrated": bool(live["exfil_matches"] > 0),
        "objective_exfiltration_matches": live["exfil_matches"],
    } if live["matches"] else None
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
    comp = s.get("competition", {})
    bk = s.get("backends", {})
    lp = s.get("live_performance")
    print("\n" + "=" * 72)
    print("ATTACK/DEFENSE CTF SCORING  (CAI battlegrounds, arXiv:2510.17521)")
    print("=" * 72)
    print(f"  backends: live(FinancialTarget)={bk.get('live_matches', 0)}  "
          f"simulated(excluded)={bk.get('simulated_matches', 0)}")
    print(f"  {'match':>5} | {'bk':>4} {'att':>4} {'InitAcc':>8} {'VulnDet':>8} {'VulnPatch':>9} | "
          f"{'Own':>5} {'Avail':>6} {'Total':>6}  winner")
    for r in metrics["per_iteration"]:
        own, avail = r.get("own_points", 0), r.get("availability_points", 0)
        win = "RED" if own > avail else ("BLUE" if avail > own else "tie")
        print(f"  {r['iteration']:>5} | {('sim' if r.get('simulated') else 'live'):>4} "
              f"{r.get('attempts', 0):>4} "
              f"{r.get('initial_access_rate', 0)*100:>7.1f}% "
              f"{r.get('vuln_detected_rate', 0)*100:>7.1f}% "
              f"{r.get('vuln_patched_rate', 0)*100:>8.1f}% | "
              f"{own:>5} {avail:>6} {r.get('total_points', 0):>6}  {win}")
    print("-" * 72)
    if lp:
        print("  FINANCIAL TARGET — REAL PERFORMANCE (simulated runs excluded)")
        print(f"  over {lp['matches']} live match(es), {lp['attempts']} attempt(s):")
        print(f"  >> Attack success rate (breach)            : {lp['attack_success_rate']*100:.1f}%"
              f"  ({lp.get('breached', 0)}/{lp['attempts']})")
        print(f"  >> Defense success rate (verified contain) : {lp['defense_success_rate']*100:.1f}%"
              f"  ({lp.get('contained', 0)}/{lp['attempts']})  "
              f"[only attacks stopped by a performed, re-probe-verified defense]")
        print(f"  >> (not counted) attack did not execute    : {lp.get('blocked_rate', 0)*100:.1f}%"
              f"  ({lp.get('blocked', 0)}/{lp['attempts']})  [blanket/self-fail, not a defense win]")
        print(f"  >> Objective data exfiltrated              : "
              f"{'YES' if lp['objective_exfiltrated'] else 'NO'}  "
              f"({lp['objective_exfiltration_matches']}/{lp['matches']} match, "
              f"{lp['objective_exfiltration_rate']*100:.1f}%)")
        print("-" * 72)
    if comp.get("matches"):
        a = comp["attempts"]
        print(f"  CTF scoring over {comp['matches']} LIVE match(es), {a} attempt(s) "
              f"-- rate = hits / attempts:")
        print(f"  >> Initial Access (executed+evaded+success): "
              f"{comp['initial_access_hits']}/{a}  ({comp['initial_access_rate']*100:.1f}%)")
        print(f"  >> Vulnerability Detected (blue)           : "
              f"{comp['vuln_detected_hits']}/{a}  ({comp['vuln_detected_rate']*100:.1f}%)")
        print(f"  >> Vulnerability Patched (blue)            : "
              f"{comp['vuln_patched_hits']}/{a}  ({comp['vuln_patched_rate']*100:.1f}%)")
        print(f"  >> Own Points mean {comp['own_points_mean']}/300   "
              f"Availability mean {comp['availability_points_mean']}/200   "
              f"wins RED={comp['red_wins']} BLUE={comp['blue_wins']} tie={comp['ties']}")
    else:
        print("  (no LIVE FinancialTarget matches — CTF scoring is simulated-only and "
              "excluded from real stats; run with --live)")
    db = s.get("defenderbench")
    if db and db.get("episodes_scored"):
        d = db["detection"]
        print("-" * 72)
        print("  (aux) BLUE detection Macro-F1          : "
              f"{d['macro_f1']:.3f}  (precision={d['precision']:.3f} recall={d['recall']:.3f})")


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
