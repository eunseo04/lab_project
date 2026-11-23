"""Audit archived Qwen experiments and export comparable metrics; no model calls.

Run from the project root with .venv/bin/python scripts/analyze_qwen_closed_loop.py.
Raw benchmark files are read only. Short validation and long held-out evaluation
are kept in separate tables/figures. Joint rewards are counted once per tick.
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import re
import statistics
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "analysis_outputs/qwen_comparison"
SOURCES = {}


def read(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    content = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(content).hexdigest()
    return json.loads(content)


def trace(episode):
    assert episode["status"] == "completed" and episode["kind"] == "real_cage4"
    path = Path(episode["trace"])
    if not path.is_absolute():
        path = ROOT / path
    content = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(content).hexdigest()
    rows = [json.loads(line) for line in content.splitlines() if line]
    ticks = defaultdict(list)
    for row in rows:
        ticks[row["step"]].append(row)
    assert len(ticks) == episode["joint_transitions"]
    reward, scored = 0.0, 0
    for group in ticks.values():
        assert len(group) == 5 and len({r["agent"] for r in group}) == 5
        assert len({(r["joint_reward"], r["score_included"], r["terminal"]) for r in group}) == 1
        if group[0]["score_included"]:
            reward += group[0]["joint_reward"]
            scored += 1
    assert reward == episode["joint_return"] and scored == episode["scored_transitions"]
    return rows


def write_csv(name, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def error_category(row):
    error = row.get("parse_error")
    if not error:
        return None
    text = row["response"].strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return "json_syntax"
    if error.startswith("Unknown subnet:"):
        return "unknown_subnet"
    if error.startswith("Host is outside"):
        return "invalid_host"
    if isinstance(value, dict) and value.get("action") == "WaitVerification":
        return "skill_name_as_action"
    return "action_or_schema"


def audit_case(path):
    summary = read(path / "summary.json")
    assert summary["status"] == "completed" and summary["result_kind"] == "real_cage4"
    assert summary["pool_frozen"]
    config = summary["config"]
    for key, value in {"composition": "hybrid", "steps": 500, "scoring": "official", "communication": "paper_projected_keep"}.items():
        assert config[key] == value
    episodes = summary["episodes"]
    assert len(episodes) == 2 and [e["seed"] for e in episodes] == [10000, 10001]
    arm, red = config["arm"], config["red"]
    values = [e["joint_return"] for e in episodes]
    metrics = {"red": red, "arm": arm, "n": len(values), "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values), "seed_10000": values[0], "seed_10001": values[1],
        "eval_wall_s_total": sum(e["execution_time_s"] for e in episodes),
        "nominal_steps": 500, "joint_transitions_per_episode": 499, "scored_transitions_per_episode": 498}
    diagnostics = {"red": red, "arm": arm}
    actions, errors, skills, calls, terms = Counter(), Counter(), Counter(), Counter(), Counter()
    false_status, phase_rewards = Counter(), Counter()
    repeated, usage, all_rows = [], [], []
    for episode in episodes:
        identity = episode["model_identity"]
        assert identity["model"] == "Qwen/Qwen3-8B" and identity["weights_frozen"]
        assert identity["revision"] == "b968826d9c46dd6066d109eabc6255188de91218"
        rows = trace(episode)
        all_rows.extend(rows)
        calls.update(c["purpose"] for c in episode["calls"])
        blue = [r for r in rows if r["agent"] == "blue_agent_4"]
        assert len(blue) == 499 and all(r["policy"] == "llm" for r in blue)
        for i, row in enumerate(blue):
            action = row["requested_action"].split()[0]
            actions[action] += 1
            if row["score_included"]:
                phase_rewards[row["mission_phase"]] += row["joint_reward"]
            if row["action_status"] == "FALSE":
                false_status.update(e["name"] for e in row["executed_actions"])
            if row.get("policy_invoked"):
                skills[row.get("skill_id") or "NONE"] += 1
                category = error_category(row)
                if category:
                    errors[category] += 1
                if (row.get("skill_version") or 0) > 1:
                    usage.append({"red": red, "arm": arm, "seed": row["seed"], "step": row["step"],
                        "skill": row["skill_id"], "action": row["requested_action"], "parse_error": row.get("parse_error")})
            for event in (row.get("skill_controller") or {}).get("events", []):
                if not event.get("terminated"):
                    continue
                terms[event["reason"]] += 1
                if event["reason"] == "repeated_action_without_new_evidence":
                    previous, last = blue[i-2:i]
                    assert last["requested_action"] == previous["requested_action"]
                    assert last["evidence_hash"] == previous["evidence_hash"]
                    repeated.append({"red": red, "seed": row["seed"], "step": row["step"], "skill": event["terminated"],
                        "repeated_action": last["requested_action"],
                        "both_prior_policy_skipped": not previous["policy_invoked"] and not last["policy_invoked"],
                        "current_evidence_changed": row["evidence_hash"] != last["evidence_hash"],
                        "previous_action_just_completed": last["observation_before"].get("success") == "IN_PROGRESS"
                            and last["observation_after"].get("success") != "IN_PROGRESS"})
    assert sum(phase_rewards.values()) == sum(values)
    decisions = sum(skills.values())
    assert decisions == calls["policy"]
    diagnostics.update(policy_decisions=decisions, parse_or_target_errors=sum(errors.values()),
        error_rate_pct=100*sum(errors.values())/decisions, error_categories=dict(errors),
        requested_actions=dict(actions), false_status_by_executed_action=dict(false_status),
        policy_decisions_by_skill=dict(skills), model_calls_by_purpose=dict(calls),
        model_calls_total=sum(calls.values()), policy_output_at_max_tokens=sum(
            c["purpose"] == "policy" and c["output_tokens"] >= config["model"]["max_tokens"]
            for e in episodes for c in e["calls"]), terminations=dict(terms),
        repeated_sleep_terminations=sum(e["repeated_action"] == "Sleep" for e in repeated),
        repeated_both_prior_policy_skipped=sum(e["both_prior_policy_skipped"] for e in repeated),
        repeated_current_evidence_changed=sum(e["current_evidence_changed"] for e in repeated),
        repeated_previous_action_just_completed=sum(e["previous_action_just_completed"] for e in repeated),
        evolved_policy_decisions=len(usage), evolved_policy_pct=100*len(usage)/decisions,
        phase_mean_rewards={str(k): v/len(episodes) for k, v in sorted(phase_rewards.items())})
    metrics.update(model_calls=diagnostics["model_calls_total"], policy_decisions=decisions,
        parse_or_target_errors=sum(errors.values()))
    return metrics, diagnostics, repeated, usage, all_rows


def loop_records():
    groups = [
        ("main_500_E2", ["research_outputs/real-500/E2-hybrid-finite/train"]),
        ("main_500_E3", ["research_outputs/real-500/E3-hybrid-finite/train"]),
        ("pilot_12_E2", ["research_outputs/real-closed-loop/E2-hybrid-finite/train", "research_outputs/real-continued/E2-hybrid-finite/train"]),
        ("paper_rtg_20_E2", ["research_outputs/real-paper-rtg/E2-hybrid-finite/train"])]
    result, priority = [], []
    for group, paths in groups:
        histories, episodes = {}, {}
        summaries = []
        for path in paths:
            summary = read(Path(path) / "summary.json")
            summaries.append(summary)
            for episode in summary["episodes"]:
                assert episode["seed"] not in episodes
                trace(episode)
                episodes[episode["seed"]] = episode
            for history in summary["history"]:
                iteration = history["iteration"]
                if iteration in histories:
                    assert history == histories[iteration], "Inherited history changed"
                histories[iteration] = history
        ordered_episodes = sorted(episodes.values(), key=lambda e: e["seed"])
        offset, previous_pool_size = 0, 7
        for iteration, history in sorted(histories.items()):
            # Local episode counters restart at warm-start; history iteration IDs do not.
            training = ordered_episodes[offset:offset+2]
            offset += 2
            assert statistics.mean(e["joint_return"] for e in training) == history["reward_mean"]
            validation = history["validation_episodes"]
            for episode in validation:
                trace(episode)
            evolutions = history["evolution"]
            assert len(evolutions) == 1
            ev = evolutions[0]
            best = max(ev["candidates"], key=lambda c: c["score"])
            ds = best["diagnostics"]
            advantages = [a for d in ds for a in d["advantages"]]
            if summaries[-1]["config"]["evolution"]["gate_mode"] == "official_code":
                old_self = statistics.mean(advantages)
            else:
                old_self = statistics.mean(statistics.mean(d["advantages"]) for d in ds)
            row = {"protocol": group, "arm": summaries[-1]["config"]["arm"], "iteration": iteration,
                "nominal_steps": summaries[-1]["config"]["steps"],
                "train_seeds": [e["seed"] for e in training], "train_returns": [e["joint_return"] for e in training],
                "train_mean": history["reward_mean"], "ema_baseline": history["baseline"],
                "validation_seeds": [e["seed"] for e in validation],
                "validation_mean": history["validation_reward_mean"],
                "parent": ev["parent"], "parent_frequency": ev["old_skill"]["frequency"],
                "candidate_scores": [c["score"] for c in ev["candidates"]],
                "accepted": [c["skill"]["name"] for c in ev["candidates"] if c["accepted"]],
                "gate_valid_historical_decisions": len(advantages), "gate_evidence_seeds": sorted({
                    t["seed"] for d in ds for t in d["transitions"]}),
                "gate_old_self_score": old_self, "gate_best_candidate_score": best["score"],
                "gate_candidate_minus_parent_self": best["score"]-old_self,
                "pool_before": previous_pool_size, "pool_after": history["pool_size"],
                "semantic_chunks": len(ev["semantic_gradients"]), "evolution_wall_s": history["seconds"],
                "error": ev.get("error"), "maintenance": history["maintenance"]}
            result.append(row)
            previous_pool_size = history["pool_size"]
        if group in ("main_500_E2", "main_500_E3"):
            from research.upstream import get_priority
            history = histories[1]
            parent = history["evolution"][0]["old_skill"]
            pool = [SimpleNamespace(**(parent if s["name"] == parent["name"] else s))
                for s in history["pool_snapshot"]["skills"] if s["version"] == 1]
            checkpoint = read(Path(paths[0]) / "checkpoint.json")
            counts = Counter(e["skill"] for e in checkpoint["experience"]["fifo"])
            ready = []
            for skill in pool:
                score = get_priority(skill, pool, 1)
                item = {"protocol": group, "skill": skill.name, "frequency": skill.frequency,
                    "historical_trajectories": counts[skill.name], "ready": counts[skill.name] >= 2,
                    "last_evolved_before_iteration": skill.last_evolved_iter, "priority": score,
                    "chosen": skill.name == parent["name"]}
                priority.append(item)
                if item["ready"]:
                    ready.append(item)
            assert sorted(ready, key=lambda r: r["priority"], reverse=True)[0]["chosen"]
    return result, priority


def figures(main, loops, references):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, red, title in zip(axes, ("finite", "aggressive"), ("FiniteState (training policy)", "AggressiveFSM (frozen-pool transfer)")):
        rows = [r for r in main if r["red"] == red]
        xs = list(range(len(rows)))
        ax.errorbar(xs, [r["mean"] for r in rows], yerr=[r["sample_std"] for r in rows], fmt="o", capsize=5, color="#174f79")
        for x, row in zip(xs, rows):
            ax.scatter([x-.045, x+.045], [row["seed_10000"], row["seed_10001"]], s=25, color="#b85621", alpha=.8)
            ax.annotate(f'{row["mean"]:g}', (x, row["mean"]), xytext=(7, 5), textcoords="offset points", fontsize=9)
        ax.set_xticks(xs, [r["arm"] for r in rows])
        ax.set_xlim(-.5, max(xs)+.6)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Official Blue joint return (higher is better)")
    fig.suptitle("Qwen3-8B / Hybrid / 500 nominal steps / N=2 per arm", fontsize=12)
    fig.text(.5, .01, "Bars: sample SD; orange dots: paired seeds 10000, 10001. E3 uses FiniteState-trained skills.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .055, 1, .93))
    fig.savefig(OUT / "qwen_500_comparison.png", dpi=180)
    plt.close(fig)
    pilot = [r for r in loops if r["protocol"] == "pilot_12_E2"]
    xs = [0]+[r["iteration"] for r in pilot]
    ys = [references["E1"]]+[r["validation_mean"] for r in pilot]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, "o-", color="#174f79", label="E2 Skill-Pro (iteration 0: static pool)")
    ax.axhline(references["E0"], color="#b85621", linestyle="--", label="E0 Qwen ACD: same seed / same horizon")
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:g}", (x, y), xytext=(4, 7), textcoords="offset points")
    ax.set_xticks(xs)
    ax.set_xlabel("Completed closed-loop iteration")
    ax.set_ylabel("Official Blue joint return (higher is better)")
    ax.set_title("12-step E2 pilot / validation seed 5000 only (N=1)")
    ax.margins(y=.15)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "qwen_12_step_closed_loop.png", dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metrics, diagnostics, repeated, usage, traces = [], [], [], [], {}
    cases = [("real-500", f"{arm}-hybrid-finite") for arm in ("E0", "E1", "E2", "E3")]
    cases += [("real-transfer", f"{arm}-hybrid-aggressive") for arm in ("E0", "E3")]
    for folder, case in cases:
        metric, diagnostic, repetitions, reused, rows = audit_case(ROOT / "research_outputs" / folder / case / "eval")
        metrics.append(metric); diagnostics.append(diagnostic)
        repeated.extend(repetitions); usage.extend(reused); traces[(metric["red"], metric["arm"])] = rows
    equality_fields = ("step", "seed", "agent", "observation_before", "observation_after", "requested_action",
        "executed_actions", "joint_reward", "response", "skill_id", "skill_version", "policy_invoked", "comm_vector")
    left, right = traces[("finite", "E1")], traces[("finite", "E2")]
    assert len(left) == len(right) == 4990
    assert all(all(a.get(k) == b.get(k) for k in equality_fields) for a, b in zip(left, right))
    del traces
    pairs = []
    for red, target, baseline in [("finite", "E1", "E0"), ("finite", "E2", "E1"),
        ("finite", "E3", "E2"), ("finite", "E3", "E0"), ("aggressive", "E3", "E0")]:
        a = next(r for r in metrics if r["red"] == red and r["arm"] == target)
        b = next(r for r in metrics if r["red"] == red and r["arm"] == baseline)
        differences = [a[f"seed_{s}"]-b[f"seed_{s}"] for s in (10000, 10001)]
        pairs.append({"red": red, "comparison": f"{target}-{baseline}", "paired_seeds": [10000,10001],
            "paired_differences": differences, "mean_difference": statistics.mean(differences)})
    loops, priorities = loop_records()
    references = {}
    for arm in ("E0", "E1"):
        summary = read(OUT / f"reference/{arm}-hybrid-finite/eval/summary.json")
        assert summary["config"]["steps"] == 12 and len(summary["episodes"]) == 1
        assert summary["episodes"][0]["seed"] == 5000 and summary["pool_frozen"]
        trace(summary["episodes"][0])
        references[arm] = summary["episodes"][0]["joint_return"]
    for name, rows in [("main_metrics.csv", metrics), ("paired_differences.csv", pairs),
        ("closed_loop_metrics.csv", loops), ("policy_diagnostics.csv", diagnostics),
        ("phase_metrics.csv", [{"red": d["red"], "arm": d["arm"], "phase": int(phase), "mean_joint_return": value}
            for d in diagnostics for phase, value in d["phase_mean_rewards"].items()]),
        ("evolution_priorities.csv", priorities), ("repeated_action_events.csv", repeated), ("evolved_skill_usage.csv", usage)]:
        write_csv(name, rows)
    for relative in ("scripts/analyze_qwen_closed_loop.py", "research/skills.py", "research/evolution.py",
        "research/gate.py", "research/adapter.py", "third_party/Skill-Pro/Skills/skill_evolution.py"):
        SOURCES[relative] = hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
    audited_episodes = sum(path.endswith(".jsonl") for path in SOURCES)
    report = {"model": "Qwen/Qwen3-8B", "metric": "original CAGE4 joint return; higher is better",
        "composition": "Hybrid: four frozen KEEP agents + one Qwen Blue agent", "main": metrics,
        "paired_differences": pairs, "closed_loops": loops, "policy_diagnostics": diagnostics,
        "evolution_priorities": priorities, "evolved_skill_usage": usage,
        "short_reference_seed_5000": references, "e1_e2_equal_trace_rows": 4990,
        "audit": {"passed": True, "unique_real_episode_traces": audited_episodes,
            "joint_reward_counting": "one contribution per scored tick, never five times",
            "e1_e2_equal_blue_records": 4990, "resumed_history_deduplicated": True},
        "source_sha256": dict(sorted(SOURCES.items())), "limitations": [
            "Main held-out evaluations have two seeds only; no significance claim.",
            "500-step E2/E3 each completed one loop only; loops 2+ are not measured.",
            "12-step three-loop pilot is E2, not E3; its single validation seed is reused.",
            "12/20-step scores must not be compared numerically with 500-step scores.",
            "No action-causal or LWF/ASF/RIA reward attribution is inferred from joint traces.",
            "No initial-pool E3 verifier-only held-out control exists."]}
    (OUT / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    figures(metrics, loops, references)
    print(json.dumps({"output": str(OUT), "long_eval_episodes_audited": 12, "short_reference_episodes_audited": 2,
        "closed_loop_rows": len(loops), "unique_real_episode_traces_audited": audited_episodes,
        "e1_e2_equal_trace_rows": 4990, "main": metrics,
        "reference": references}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
