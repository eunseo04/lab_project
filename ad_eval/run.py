"""Compare native CAI/base Qwen with MCTS+SIEM/GRPO on the shared local range."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

from .agents import make_workers
from .docker_lab import (ROOT, HOSTS, IPS, OPPONENT, COMPOSE, cmd, compose_up,
                         current_flags, execution_audit, patch_evidence, service_probe,
                         target_audit, root_ssh_credentials)
from .evidence import flags_and_access, merge_manual_review
from .models import select_model, verify_model
from .scoring import score_match, summarize_matches


def dump(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    path.chmod(0o600)


def read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("AD_EVAL_EVENT "):
            line = line[len("AD_EVAL_EVENT "):]
        elif not line.startswith("{"):
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # an in-flight line at the deadline has no completed evidence
    return result


def usage_totals(events_by_role: dict[str, list[dict]], input_price: float | None,
                 output_price: float | None) -> dict:
    totals = {"input_tokens": 0, "output_tokens": 0}
    observed = True
    for events in events_by_role.values():
        usage = [e for e in events if e.get("kind") == "llm_usage"]
        if not usage or not usage[-1].get("complete"):
            observed = False
            continue
        started = sum(e.get("kind") == "llm_request_started" for e in events)
        if started > usage[-1].get("requests", 0):
            observed = False
        for key in totals:
            value = usage[-1].get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                observed = False
            else:
                totals[key] += value
    if not observed:
        totals = {"input_tokens": None, "output_tokens": None}
    cost = None
    if observed and input_price is not None and output_price is not None:
        cost = round((totals["input_tokens"] * input_price +
                      totals["output_tokens"] * output_price) / 1_000_000, 8)
    return {**totals, "cost_usd": cost,
            "usage_complete": observed,
            "cost_basis": "configured_USD_per_million_tokens" if cost is not None else None}


def validate_args(args) -> None:
    args.conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    if not args.conditions or len(set(args.conditions)) != len(args.conditions) or any(
            x not in {"cai", "proposed", "proposed_base"} for x in args.conditions):
        raise ValueError("conditions must be unique cai,proposed,proposed_base names")
    if (args.repeats < 1 or not math.isfinite(args.match_seconds) or args.match_seconds < 10 or
            not math.isfinite(args.interval) or not 0 < args.interval < args.match_seconds / 3):
        raise ValueError("repeats>=1, match-seconds>=10, 0<interval<duration/3 required")
    for value in (args.input_usd_per_million, args.output_usd_per_million):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError("token prices must be finite and nonnegative")
    for condition in args.conditions:
        select_model(args, condition)


def preflight(args) -> dict:
    if not shutil.which("docker"):
        raise RuntimeError("Docker CLI not installed")
    if "cai" in args.conditions:
        try:
            version = importlib.metadata.version("cai-framework")
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError("run with the Python environment containing cai-framework") from exc
    else:
        version = None
    models = {c: verify_model(select_model(args, c)) for c in args.conditions}
    if any(c != "cai" for c in args.conditions):
        from .wazuh import get_wazuh_client
        if not get_wazuh_client().ping():
            raise RuntimeError("proposed requires a reachable real Wazuh Indexer")
    cmd(["docker", "version", "--format", "{{.Server.Version}}"], timeout=15)
    return {"cai_package_version": version, "python": sys.executable, "models": models,
            "compose_sha256": hashlib.sha256(COMPOSE.read_bytes()).hexdigest(),
            "dataset_sha256": hashlib.sha256((ROOT / "attack_dataset.json").read_bytes()).hexdigest(),
            "baseline_execution": "native_CAI_parallel_two_isolated_agents_per_team",
            "score_version": "local_cai_ad_v2"}


def collect_records(args, condition: str, rep: int, out: Path, samples: dict,
                    flags: dict, worker_exit: dict, audit: dict, host_events: dict) -> list[dict]:
    records = []
    # A failed opponent changes the defensive task too: invalidate the entire match.
    invalid = bool(worker_exit)
    for team in HOSTS:
        event_paths = {role: out / (f"team{team}_{role}.events.jsonl" if condition == "cai"
                                   else f"team{team}_{role}.log") for role in ("red", "blue")}
        events = {role: read_events(path) for role, path in event_paths.items()}
        if condition == "cai" and any(not any(e.get("kind") == "agent_started" for e in events[r])
                                      for r in ("red", "blue")):
            invalid = True
        if any(e.get("kind") == "worker_error" for seq in events.values() for e in seq):
            invalid = True
        tools = [e for e in events["red"] if e.get("kind") == "tool_result"]
        red_output = "\n".join(str(e.get("output", "")) for e in tools)
        gt = flags[OPPONENT[team]]
        ev = flags_and_access(red_output, audit[OPPONENT[team]], gt["user"], gt["root"],
                              tool_events=tools, host_events=host_events[OPPONENT[team]],
                              target_ip=IPS[OPPONENT[team]],
                              source_ip="172.32.10.101" if team == "1" else "172.32.10.102")
        patch = patch_evidence(team)
        usage = usage_totals(events, args.input_usd_per_million, args.output_usd_per_million)
        records.append({"team": team, "condition": condition,
                        "initial_access": ev["initial_access"], "user_flag": ev["user_flag"],
                        "root_flag": ev["root_flag"], "vulnerability_detected": None,
                        "vulnerability_patched": patch["vulnerability_patched"],
                        "availability_samples": samples[team], **usage,
                        "evidence": {**ev["evidence"], **patch,
                                     "blue_wazuh_alert_telemetry": any(e.get("kind") == "wazuh_alert_observed"
                                                                       for e in events["blue"]),
                                     "event_logs": [p.name for p in event_paths.values()]}})
    if invalid:
        for record in records:
            record["availability_samples"] = []
            for key in ("initial_access", "user_flag", "root_flag", "vulnerability_detected", "vulnerability_patched"):
                record[key] = None
            record["evidence"]["invalid_match"] = {"worker_exit_codes": worker_exit,
                                                       "reason": "worker_exit_or_error_or_missing_native_agent"}
    return records


def run_one(args, condition: str, rep: int, labels: dict) -> dict:
    slug = f"rep{rep:03d}_{condition}"
    out = Path(args.out_dir).resolve() / slug
    if out.exists():
        raise FileExistsError(f"refusing to overwrite experiment evidence: {out}")
    out.mkdir(parents=True)
    print(f"[ad_eval] {slug}: recreating the shared range", flush=True)
    compose_up()
    if condition != "cai":
        from .wazuh import get_wazuh_client, verify_lab_ingestion
        dump(out / "wazuh_preflight.json", verify_lab_ingestion(get_wazuh_client()))
    initial_patches = {team: patch_evidence(team) for team in HOSTS}
    if any(p["vulnerability_patched"] is not False for p in initial_patches.values()):
        raise RuntimeError("fresh range does not expose both expected weaknesses")
    flags = {team: {"user": set(), "root": set()} for team in HOSTS}

    def collect_flags():
        for team in HOSTS:
            pair = current_flags(team)
            for kind in ("user", "root"):
                flags[team][kind].add(pair[kind])

    collect_flags()
    model = select_model(args, condition)
    workers = make_workers(condition, out, model=model.model, endpoint=model.endpoint,
                           seed=rep, blue_credentials=root_ssh_credentials())
    samples = {team: [] for team in HOSTS}
    worker_exit = {}
    audit_prefix = {team: len(target_audit(team)) for team in HOSTS}
    event_prefix = {team: len(execution_audit(team)) for team in HOSTS}
    start_utc = datetime.now(timezone.utc).isoformat()
    began = time.monotonic()
    import signal
    import threading

    def terminate_workers():
        for worker in workers:
            if worker.process is not None and worker.process.poll() is None:
                try:
                    os.killpg(worker.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    watchdog = threading.Timer(args.match_seconds, terminate_workers)
    watchdog.daemon = True
    watchdog.start()

    def probe(team):
        elapsed = time.monotonic() - began
        return team, {"elapsed_s": round(elapsed, 4), **service_probe(team)}

    try:
        for worker in workers:
            worker.start()
        deadline = began + args.match_seconds
        next_probe = began
        with ThreadPoolExecutor(max_workers=2) as pool:
            while time.monotonic() < deadline:
                if time.monotonic() >= next_probe:
                    for team, sample in pool.map(probe, HOSTS):
                        samples[team].append(sample)
                    collect_flags()
                    next_probe = time.monotonic() + args.interval
                for worker in workers:
                    if time.monotonic() < deadline and worker.process.poll() is not None:
                        worker_exit[f"{worker.team}_{worker.role}"] = worker.process.returncode
                if worker_exit:
                    break
                time.sleep(min(0.2, max(0., deadline - time.monotonic())))
    finally:
        # Signal EVERY worker at the common deadline before waiting for any of them.
        watchdog.cancel()
        terminate_workers()
        for worker in workers:
            worker.stop()
        # The ephemeral Blue password in native /parallel config is no longer needed.
        for path in out.glob("team*_parallel.json"):
            path.unlink()
    with ThreadPoolExecutor(max_workers=2) as pool:
        for team, sample in pool.map(probe, HOSTS):
            samples[team].append(sample)
    collect_flags()
    audit = {team: target_audit(team)[audit_prefix[team]:] for team in HOSTS}
    host_events = {team: execution_audit(team)[event_prefix[team]:] for team in HOSTS}
    for team in HOSTS:
        dump(out / f"host{team}_executions.json", host_events[team])
    records = collect_records(args, condition, rep, out, samples, flags, worker_exit, audit, host_events)
    # Always retain raw observations before applying human labels.
    dump(out / "records.json", {"match_id": slug, "condition": condition, "rep": rep,
                               "duration_s": args.match_seconds, "interval_s": args.interval,
                               "records": records})
    reviewed = [merge_manual_review(rec, labels.get(f"{rep}:{condition}:{rec['team']}", {}))
                for rec in records]
    match = score_match(*reviewed, args.match_seconds, args.interval)
    match.update(match_id=slug, started_utc=start_utc, model=model.model,
                 model_kind=model.kind, source="local_mayhem_cyber_range_not_HTB",
                 worker_exit_codes=worker_exit)
    dump(out / "score.json", match)
    print(f"[ad_eval] {slug}: winner={match['winner']}", flush=True)
    return match


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", default="cai,proposed")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--match-seconds", type=float, default=900.)
    parser.add_argument("--interval", type=float, default=15.)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--trained-model", help="vLLM model ID for the GRPO checkpoint")
    parser.add_argument("--grpo-checkpoint", help="exact local LoRA checkpoint path, checked against vLLM root")
    parser.add_argument("--endpoint", default=os.environ.get("QWEN_ENDPOINT", "http://127.0.0.1:8000"))
    parser.add_argument("--trained-endpoint", help="optional separate server for the trained model")
    parser.add_argument("--out-dir", default="output/ad_eval")
    parser.add_argument("--labels", help="review labels keyed by rep:condition:team")
    parser.add_argument("--input-usd-per-million", type=float)
    parser.add_argument("--output-usd-per-million", type=float)
    parser.add_argument("--dry-run", action="store_true", help="print model/execution plan, without network or Docker")
    parser.add_argument("--preflight-only", action="store_true", help="check dependencies, served models and Indexer, without resetting hosts")
    return parser


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    plan = {c: vars(select_model(args, c)) for c in args.conditions}
    if args.dry_run:
        print(json.dumps({"conditions": plan, "duration_s": args.match_seconds,
                          "cai_workers": "2 native /parallel sessions, each with Red+Blue",
                          "proposed_workers": "4 isolated MCTS/SIEM workers"}, indent=2))
        return
    manifest = preflight(args)
    if args.preflight_only:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    out_dir = Path(args.out_dir)
    if (out_dir / "manifest.json").exists():
        raise FileExistsError("choose a fresh --out-dir for each experiment campaign")
    dump(out_dir / "manifest.json", manifest)
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8")) if args.labels else {}
    matches = []
    for rep in range(1, args.repeats + 1):
        order = args.conditions if rep % 2 else list(reversed(args.conditions))
        for condition in order:
            matches.append(run_one(args, condition, rep, labels))
            dump(out_dir / "summary.json", {"schema_version": "local_cai_ad_v2", "matches": matches,
                                           "by_condition": summarize_matches(matches),
                                           "caveats": ["Local range, model and availability checker differ from the HTB paper",
                                                       "Detection and uncorrelated initial access require independent human review",
                                                       "Team observations in a match are paired; Wilson intervals are descriptive"]})
    print(f"[ad_eval] completed: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
