"""Run both conditions on the SAME local Docker battleground and score fairly.

Usage (inside lab_project):
  python -m ad_eval.run --conditions cai --base-model openai/Qwen3-8B --endpoint http://127.0.0.1:8000 --i-own-lab
  python -m ad_eval.run --conditions cai,proposed --base-model openai/Qwen3-8B --trained-model Qwen3-8B-grpo --endpoint http://127.0.0.1:8000 --i-own-lab

Only an isolated, deliberately vulnerable range that you own is allowed.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .agents import make_workers
from .docker_lab import (ROOT, HOSTS, KALI, IPS, OPPONENT, cmd, compose_up,
                         current_flags, patch_evidence, service_probe, target_audit, root_ssh_credentials)
from .evidence import flags_and_access, merge_manual_review
from .scoring import score_match, summarize_matches


def dump(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def preflight(args) -> None:
    if not args.i_own_lab:
        raise ValueError("add --i-own-lab after verifying these Docker hosts are yours")
    if not shutil.which("docker"):
        raise RuntimeError("docker command not installed")
    if args.repeats < 1 or args.match_seconds < 10 or args.interval < 1:
        raise ValueError("repeats>=1, match-seconds>=10, interval>=1 are required")
    if args.interval >= args.match_seconds / 3:
        raise ValueError("monitor interval must be < match duration / 3")
    if "cai" in args.conditions and not (shutil.which(os.environ.get("CAI_BIN", "cai")) or Path(os.environ.get("CAI_BIN", "cai")).is_file()):
        raise RuntimeError("actual CAI CLI not found; install official cai-framework")
    if "proposed" in args.conditions:
        if not args.trained_model or args.trained_model == args.base_model:
            raise ValueError("proposed requires distinct served --trained-model (actual GRPO adapter)")
        if not args.endpoint:
            raise ValueError("proposed requires --endpoint for a live vLLM adapter")
        from wazuh_client import get_wazuh_client
        if not get_wazuh_client().ping():
            raise RuntimeError("proposed requires REAL reachable Wazuh, not synthetic alerts")
    cmd(["docker", "version", "--format", "{{.Server.Version}}"], timeout=15)


def _wazuh_alert_telemetry(log: str) -> bool:
    """Supplementary SIEM telemetry, NOT the paper's manually adjudicated
    agent vulnerability detection metric. The latter requires review in BOTH
    experimental conditions, to avoid biased measurement."""
    seen = False
    for line in log.splitlines():
        if not line.startswith("AD_EVAL_EVENT "):
            continue
        try:
            payload = json.loads(line[len("AD_EVAL_EVENT "):])
        except json.JSONDecodeError:
            continue
        if payload.get("kind") == "wazuh_alert_observed" and payload.get("tech_id"):
            seen = True
    return seen


def run_one(args, condition: str, rep: int, labels: dict) -> dict:
    slug = f"rep{rep:03d}_{condition}"
    out = Path(args.out_dir).resolve() / slug
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n[ad_eval] {slug}: resetting identical Docker hosts", flush=True)
    compose_up()
    # Initial secrets and subsequent rotations are observed only by the trusted
    # scorer, never passed to Red or Blue agent prompts.
    flags = {t: {"user": set(), "root": set()} for t in HOSTS}
    def collect_flags():
        for t in HOSTS:
            pair = current_flags(t)
            flags[t]["user"].add(pair["user"])
            flags[t]["root"].add(pair["root"])
    collect_flags()
    model = args.base_model if condition == "cai" else args.trained_model
    workers = make_workers(condition, out, model=model,
                           endpoint=args.endpoint, seed=rep,
                           require_wazuh=(condition == "proposed"),
                           blue_credentials=root_ssh_credentials())
    samples = {t: [] for t in HOSTS}
    worker_exit = {}
    start_utc = datetime.now(timezone.utc).isoformat()
    # Snapshot audit logs before any agent actions, to avoid attributing stale events.
    audit_prefix = {t: len(target_audit(t)) for t in HOSTS}
    began = time.monotonic()
    try:
        # Four agents are launched independently, immediately, with separate contexts.
        for w in workers:
            w.start()
        deadline = began + args.match_seconds
        next_probe = began
        while True:
            now = time.monotonic()
            if now >= deadline:
                break
            if now >= next_probe:
                elapsed = max(0., now - began)
                for t in HOSTS:
                    p = service_probe(t)
                    samples[t].append({"elapsed_s": round(elapsed, 2), **p})
                collect_flags()
                for w in workers:
                    if w.process is not None and w.process.poll() is not None:
                        worker_exit[f"{w.team}_{w.role}"] = w.process.returncode
                next_probe += args.interval
                if next_probe < time.monotonic():
                    next_probe = time.monotonic() + args.interval
            time.sleep(min(0.2, max(0., next_probe - time.monotonic()),
                           max(0., deadline - time.monotonic())))
        # End-of-window sample: required by scoring to rule out a last-second outage.
        for t in HOSTS:
            samples[t].append({"elapsed_s": float(args.match_seconds), **service_probe(t)})
        collect_flags()
    finally:
        for w in workers:
            if w.process is not None and w.process.poll() is not None:
                worker_exit[f"{w.team}_{w.role}"] = w.process.returncode
            w.stop()
    audit = {t: target_audit(t)[audit_prefix[t]:] for t in HOSTS}
    records = []
    for t in HOSTS:
        redlog = (out / f"team{t}_red.log").read_text(encoding="utf-8", errors="replace")
        bluelog = (out / f"team{t}_blue.log").read_text(encoding="utf-8", errors="replace")
        gt = flags[OPPONENT[t]]
        ev = flags_and_access(redlog, audit[OPPONENT[t]], gt["user"], gt["root"])
        patch = patch_evidence(t)
        siem_alert = _wazuh_alert_telemetry(bluelog) if condition == "proposed" else None
        rec = {
            "team": t, "condition": condition,
            "initial_access": ev["initial_access"],
            "user_flag": ev["user_flag"], "root_flag": ev["root_flag"],
            "vulnerability_detected": None,  # human-adjudicated, both CAI and proposed
            "vulnerability_patched": patch["vulnerability_patched"],
            "availability_samples": samples[t],
            # CAI SDK usage traces should be parsed separately; never invent cost.
            "input_tokens": None, "output_tokens": None, "cost_usd": None,
            "evidence": {**ev["evidence"], **patch,
                         "blue_wazuh_alert_telemetry": siem_alert,
                         "logs": [f"team{t}_red.log", f"team{t}_blue.log"]},
        }
        # If agents failed to initialize, lack of progress is NOT a fair zero.
        if (worker_exit.get(f"{t}_red") not in (None, 0) or
                worker_exit.get(f"{t}_blue") not in (None, 0)):
            rec["availability_samples"] = []
            for key in ("initial_access", "user_flag", "root_flag",
                        "vulnerability_detected", "vulnerability_patched"):
                rec[key] = None
            rec["evidence"]["invalid_worker_exit"] = worker_exit
        manual = labels.get(f"{rep}:{condition}:{t}", {})
        rec = merge_manual_review(rec, manual)
        records.append(rec)
    dump(out / "records.json", {
        "match_id": slug, "condition": condition, "rep": rep,
        "duration_s": args.match_seconds, "interval_s": args.interval,
        "records": records,
    })
    match = score_match(records[0], records[1], args.match_seconds, args.interval)
    match["match_id"] = slug
    match["started_utc"] = start_utc
    match["model"] = model
    match["source"] = "local_mayhem_cyber_range_not_HTB"
    match["worker_exit_codes"] = worker_exit
    dump(out / "score.json", match)
    print(f"[ad_eval] {slug} winner={match['winner']} (unknown if scoring incomplete)", flush=True)
    return match


def main() -> None:
    ap = argparse.ArgumentParser(description="Local CAI A&D comparison with strict team-level scoring")
    ap.add_argument("--conditions", default="cai,proposed",
                    help="comma-separated cai,proposed")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--match-seconds", type=float, default=900.)
    ap.add_argument("--interval", type=float, default=15.)
    ap.add_argument("--base-model", required=True,
                    help="CAI model identifier, e.g. openai/Qwen3-8B")
    ap.add_argument("--trained-model", default=None,
                    help="vLLM registered GRPO LoRA model name, NOT an untrained fallback")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000")
    ap.add_argument("--out-dir", default="output/ad_eval")
    ap.add_argument("--labels", default=None,
                    help="optional reviewed JSON keyed by 'rep:condition:team'; use rescore later for post-hoc review")
    ap.add_argument("--i-own-lab", action="store_true",
                    help="I own and authorize this isolated vulnerable Docker range")
    args = ap.parse_args()
    args.conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    if not args.conditions or any(x not in {"cai", "proposed"} for x in args.conditions):
        ap.error("--conditions must contain cai and/or proposed")
    preflight(args)
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8")) if args.labels else {}
    matches = []
    # Counterbalance which condition runs first across repetitions.
    for rep in range(1, args.repeats + 1):
        order = args.conditions if rep % 2 else list(reversed(args.conditions))
        for cond in order:
            matches.append(run_one(args, cond, rep, labels))
            dump(Path(args.out_dir) / "summary.json", {
                "schema_version": "local_cai_ad_v1", "matches": matches,
                "by_condition": summarize_matches(matches),
                "caveats": [
                    "Local Docker challenge, NOT original proprietary HTB battlegrounds",
                    "Availability scoring is explicitly specified local 2-checkpoint analogue",
                    "CAI original paper used manually adjudicated battle logs",
                    "CAI baseline (CLI workers) and ours (MCTS workers) run as independent processes",
                    "No baseline efficacy or GRPO improvement is assumed prior to live tests",
                    "Vulnerability detection requires human review for both arms",
                ]})
    print("[ad_eval] completed. Results: " + str(Path(args.out_dir) / "summary.json"))


if __name__ == "__main__":
    main()
