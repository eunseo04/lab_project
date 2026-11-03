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
from dataclasses import asdict

from .agents import make_workers
from .docker_lab import (ROOT, HOSTS, IPS, OPPONENT, COMPOSE, cmd, compose_up,
                         current_flags, execution_audit, patch_evidence, service_probe,
                         target_audit, root_ssh_credentials)
from .evidence import flags_and_access, merge_manual_review
from .models import select_model, verify_model
from .inference import DEFAULT_INFERENCE, InferenceSettings
from .scoring import score_match, summarize_matches
from .adaptive import AdaptiveExperiment, enabled as adaptive_enabled, learning_preflight, validate_learning_args
from .grpo_rewards import config_from_args as red_reward_config


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


def inference_settings(args, rep: int = 1) -> InferenceSettings:
    return InferenceSettings(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
                             top_k=args.top_k, min_p=args.min_p, repetition_penalty=args.repetition_penalty,
                             frequency_penalty=args.frequency_penalty, presence_penalty=args.presence_penalty,
                             enable_thinking=args.enable_thinking, seed=args.seed + 4 * (rep - 1))


def campaign_plan(args) -> list[dict]:
    plan = []
    for rep in range(1, args.repeats + 1):
        order = args.conditions if rep % 2 else list(reversed(args.conditions))
        for condition in order:
            plan.append({"rep": rep, "condition": condition,
                         "inference": inference_settings(args, rep).to_dict()})
    return plan


def closed_loop_configuration(args) -> dict:
    adaptive = "proposed" in args.conditions and adaptive_enabled(args)
    red_reward = red_reward_config(args, required=False) if adaptive else None
    return {"feedback_within_match": True,
            "model_weights": {condition: "trajectory_grpo" if adaptive and condition == "proposed" else "fixed"
                              for condition in args.conditions},
            "mcts_reward_updates": adaptive, "reset_environment_between_matches": True,
            "preserve_proposed_planner_and_weights": adaptive,
            "grpo_updates_per_proposed_match": 4 if adaptive else 0,
            "planned_grpo_updates": 4 * args.repeats if adaptive else 0,
            "red_grpo_reward": red_reward.to_dict() if red_reward else None}


def inference_audit(events: list[dict], expected: InferenceSettings) -> dict:
    declarations = [e.get("inference") for e in events if e.get("kind") == "inference_settings"]
    requests = [e.get("inference") for e in events
                if e.get("kind") in {"inference_request", "llm_request_started"} and "inference" in e]
    mismatches = sum(value != expected.request_fields() for value in [*declarations, *requests])
    usage = [e for e in events if e.get("kind") == "llm_usage"]
    completed = usage[-1].get("requests", 0) if usage else 0
    unlogged = max(0, completed - len(requests))
    matched = bool(declarations) and mismatches == 0 and unlogged == 0
    return {"expected": expected.request_fields(), "requests_observed": len(requests),
            "mismatches": mismatches, "unlogged_requests": unlogged, "valid": matched,
            "status": "mismatch_or_missing_configuration" if not matched else
                      "verified_client_requests" if requests else "configured_no_requests"}


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
    inference_settings(args)
    inference_settings(args, args.repeats).for_role("2", "blue")
    validate_learning_args(args)


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
    learning = learning_preflight(args) if "proposed" in args.conditions and adaptive_enabled(args) else None
    return {"cai_package_version": version, "python": sys.executable, "models": models,
            "compose_sha256": hashlib.sha256(COMPOSE.read_bytes()).hexdigest(),
            "dataset_sha256": hashlib.sha256((ROOT / "attack_dataset.json").read_bytes()).hexdigest(),
            "baseline_execution": "native_CAI_parallel_two_isolated_agents_per_team",
            "score_version": "local_cai_ad_v2", "inference": inference_settings(args).to_dict(),
            "repeats_per_condition": args.repeats, "schedule": campaign_plan(args),
            "closed_loop": closed_loop_configuration(args), "learning": learning}


def collect_records(args, condition: str, rep: int, out: Path, samples: dict,
                    flags: dict, worker_exit: dict, audit: dict, host_events: dict) -> list[dict]:
    records = []
    # A failed opponent changes the defensive task too: invalidate the entire match.
    invalid = bool(worker_exit)
    for team in HOSTS:
        event_paths = {role: out / (f"team{team}_{role}.events.jsonl" if condition == "cai"
                                   else f"team{team}_{role}.log") for role in ("red", "blue")}
        events = {role: read_events(path) for role, path in event_paths.items()}
        inference = inference_settings(args, rep)
        audits = {role: inference_audit(events[role], inference.for_role(team, role))
                  for role in ("red", "blue")}
        if any(not audit["valid"] for audit in audits.values()):
            invalid = True
        cycles = {role: {"started": sum(e.get("kind") == "closed_loop_cycle_started" for e in seq),
                         "completed": sum(e.get("kind") == "closed_loop_cycle_completed" for e in seq)}
                  for role, seq in events.items()}
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
                        "inference_audit": audits, "closed_loop_cycles": cycles,
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
                                                       "reason": "worker_error_missing_agent_or_inference_mismatch"}
    return records


def run_one(args, condition: str, rep: int, labels: dict, *, adaptive=None) -> dict:
    slug = f"rep{rep:03d}_{condition}"
    out = Path(args.out_dir).resolve() / slug
    if out.exists():
        raise FileExistsError(f"refusing to overwrite experiment evidence: {out}")
    out.mkdir(parents=True)
    role_models, planner_state = adaptive.prepare(rep) if adaptive else (None, None)
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
    inference = inference_settings(args, rep)
    workers = make_workers(condition, out, model=model.model, endpoint=model.endpoint,
                           seed=inference.seed, blue_credentials=root_ssh_credentials(), inference=inference,
                           role_models=role_models, planner_state=planner_state)
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
                               "inference": inference.to_dict(),
                               "role_models": {role: asdict(spec) for role, spec in role_models.items()} if role_models else None,
                               "planner_input": str(planner_state) if planner_state else None,
                               "records": records})
    reviewed = [merge_manual_review(rec, labels.get(f"{rep}:{condition}:{rec['team']}", {}))
                for rec in records]
    match = score_match(*reviewed, args.match_seconds, args.interval)
    match.update(match_id=slug, started_utc=start_utc, model=model.model,
                 model_kind=model.kind, source="local_mayhem_cyber_range_not_HTB",
                 worker_exit_codes=worker_exit, rep=rep, inference=inference.to_dict())
    if role_models:
        match["model"] = {role: spec.model for role, spec in role_models.items()}
        match["role_models"] = {role: asdict(spec) for role, spec in role_models.items()}
        match["planner_input"] = str(planner_state)
    dump(out / "score.json", match)
    if adaptive:
        events = {(team, role): read_events(out / f"team{team}_{role}.log")
                  for team in HOSTS for role in ("red", "blue")}
        match["learning_updates"] = adaptive.update(match, out, events)
        dump(out / "score.json", match)
    print(f"[ad_eval] {slug}: winner={match['winner']}", flush=True)
    return match


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", default="cai,proposed")
    parser.add_argument("--repeats", type=int, default=10, help="complete closed-loop matches per condition")
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
    parser.add_argument("--temperature", type=float, default=DEFAULT_INFERENCE.temperature)
    parser.add_argument("--top-p", type=float, default=DEFAULT_INFERENCE.top_p)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_INFERENCE.max_tokens)
    parser.add_argument("--top-k", type=int, default=DEFAULT_INFERENCE.top_k)
    parser.add_argument("--min-p", type=float, default=DEFAULT_INFERENCE.min_p)
    parser.add_argument("--repetition-penalty", type=float, default=DEFAULT_INFERENCE.repetition_penalty)
    parser.add_argument("--frequency-penalty", type=float, default=DEFAULT_INFERENCE.frequency_penalty)
    parser.add_argument("--presence-penalty", type=float, default=DEFAULT_INFERENCE.presence_penalty)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=DEFAULT_INFERENCE.seed)
    parser.add_argument("--adaptive-proposed", action=argparse.BooleanOptionalAction, default=True,
                        help="reward-update proposed MCTS and train/deploy GRPO after each role trajectory")
    parser.add_argument("--training-python", default=sys.executable)
    parser.add_argument("--training-base-model", help="HF model ID or local base weights for GRPO")
    parser.add_argument("--training-cuda-devices", help="CUDA_VISIBLE_DEVICES for the isolated training process")
    parser.add_argument("--red-grpo-checkpoint", help="optional separate initial Red LoRA")
    parser.add_argument("--blue-grpo-checkpoint", help="optional separate initial Blue LoRA")
    parser.add_argument("--grpo-num-generations", type=int, default=4)
    parser.add_argument("--grpo-epochs", type=float, default=1.)
    parser.add_argument("--grpo-max-steps", type=int, default=-1, help="-1 trains the whole trajectory dataset")
    parser.add_argument("--grpo-learning-rate", type=float, default=1e-5)
    parser.add_argument("--grpo-beta", type=float, default=0., help="GRPO KL coefficient; 0 uses no extra reference model")
    parser.add_argument("--red-reward-alpha", type=float, help="explicit experimental weight for verified attack reward")
    parser.add_argument("--red-reward-beta", type=float, help="explicit experimental semantic weight; separate from --grpo-beta KL")
    parser.add_argument("--red-reward-gamma", type=float, help="explicit experimental successful-history duplication penalty weight")
    parser.add_argument("--red-semantic-model", help="frozen judge model; defaults to the base model")
    parser.add_argument("--red-semantic-endpoint", help="frozen judge endpoint; defaults to the base endpoint")
    parser.add_argument("--red-embedding-model", help="frozen local/cached SentenceTransformer, or served embedding model ID")
    parser.add_argument("--red-embedding-endpoint", help="optional /v1/embeddings server instead of a local encoder")
    parser.add_argument("--red-embedding-revision", help="pinned revision for a cached local embedding model")
    parser.add_argument("--red-diversity-cosine-weight", type=float, default=.5)
    parser.add_argument("--red-diversity-success-threshold", type=float, default=.2)
    parser.add_argument("--red-diversity-history-limit", type=int, default=0, help="0 compares against all verified successful commands")
    parser.add_argument("--dry-run", action="store_true", help="print model/execution plan, without network or Docker")
    parser.add_argument("--preflight-only", action="store_true", help="check dependencies, served models and Indexer, without resetting hosts")
    return parser


def run_campaign(args, labels: dict) -> list[dict]:
    matches = []
    schedule = campaign_plan(args)
    learner = AdaptiveExperiment(args) if "proposed" in args.conditions and adaptive_enabled(args) else None
    if learner:
        learner.initialize_policies()  # validate dynamic registration before spending a measured match
    for index, item in enumerate(schedule, 1):
        print(f"[ad_eval] workflow {index}/{len(schedule)}: "
              f"rep={item['rep']} condition={item['condition']}", flush=True)
        if learner is not None and item["condition"] == "proposed":
            match = run_one(args, item["condition"], item["rep"], labels, adaptive=learner)
        else:
            match = run_one(args, item["condition"], item["rep"], labels)
        matches.append(match)
        dump(Path(args.out_dir) / "summary.json", {
            "schema_version": "local_cai_ad_v2", "matches": matches,
            "by_condition": summarize_matches(matches),
            "inference": inference_settings(args).to_dict(),
            "progress": {"requested_per_condition": args.repeats, "total_requested": len(schedule),
                         "total_completed": len(matches), "complete": len(matches) == len(schedule),
                         "completed_by_condition": {condition: sum(m["condition"] == condition for m in matches)
                                                    for condition in args.conditions}},
            "closed_loop": closed_loop_configuration(args),
            "completed_grpo_updates": sum(len(match.get("learning_updates", [])) for match in matches),
            "caveats": ["Local range, model and availability checker differ from the HTB paper",
                        "Detection and uncorrelated initial access require independent human review",
                        "Team observations in a match are paired; Wilson intervals are descriptive",
                        "Equal sampling settings and seeds do not guarantee deterministic GPU outputs"]})
    return matches


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
                          "repeats_per_condition": args.repeats, "total_matches": len(campaign_plan(args)),
                          "inference": inference_settings(args).to_dict(), "schedule": campaign_plan(args),
                          "closed_loop": closed_loop_configuration(args),
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
    run_campaign(args, labels)
    print(f"[ad_eval] completed: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
