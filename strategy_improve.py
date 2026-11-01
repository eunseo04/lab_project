"""High-level STRATEGY-IMPROVEMENT loop (Idea Queue + UCB), layered ON TOP of the low-level
action MCTS in pipeline.py -- the two-level structure from the paper.

Separation of concerns (do NOT merge these):
  * Action learning (pipeline): MCTS.priors are per-technique (tech_id) action priors, updated by
    EMA from step rewards. UNCHANGED here.
  * Strategy improvement (this module): an LLM proposes STRUCTURED, independently-testable strategy
    ideas (idea_id != tech_id). Each idea is tried in a candidate episode and scored by its
    IMPROVEMENT over a baseline episode (z = S_idea - S_baseline). A UCB bandit over the idea queue
    balances trying new ideas vs. re-using ones that improved performance.

An idea changes HOW an agent selects/forms actions; it is delivered to the low level via the
`strategy_idea` argument threaded into pipeline.llm_enrich_action / llm_enrich_defense and
run_episode(red_strategy_idea=/blue_strategy_idea=). Prompt injection is the baseline lever; a
structural idea (e.g. a candidate-scoring rule) would additionally need a code hook at its
`implementation_point` -- recorded in the idea so it is explicit.
"""
from __future__ import annotations
import os
import json
import math
import hashlib
from pathlib import Path
from typing import Optional, Callable

_IDEA_FIELDS = ("idea_id", "hypothesis", "change_to_test", "implementation_point",
                "evaluation_metric", "expected_effect")


# ---------------------------------------------------------------------------
# Idea queue + UCB (separate from the action-level MCTS priors)
# ---------------------------------------------------------------------------
class IdeaQueue:
    """Per-ROLE queue of strategy ideas with UCB(i) = mean_improvement_i + c*sqrt(ln N / n_i).
    Persisted to JSON so the statistics survive across processes/experiments."""

    def __init__(self, role: str, path: Optional[str] = None):
        self.role = role
        self.path = Path(path) if path else (Path(__file__).resolve().parent /
                                             "output" / f"idea_queue_{role}.json")
        self.stats: dict = {}
        self.total_evaluations: int = 0
        self.load()

    # --- persistence ---
    def load(self):
        try:
            if self.path.exists():
                d = json.loads(self.path.read_text())
                self.stats = d.get("stats", {})
                self.total_evaluations = int(d.get("total_evaluations", 0))
        except Exception as e:
            print(f"[idea:{self.role}] load skipped: {e}")

    def save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(
                {"role": self.role, "total_evaluations": self.total_evaluations,
                 "stats": self.stats}, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"[idea:{self.role}] save skipped: {e}")

    # --- queue ops ---
    def add(self, idea: dict) -> str:
        iid = idea.get("idea_id") or _idea_id_for(idea)
        idea["idea_id"] = iid
        self.stats.setdefault(iid, {"idea": idea, "count": 0,
                                    "improvement_sum": 0.0, "mean_improvement": 0.0})
        return iid

    def select_ucb(self, c: float = 1.0) -> dict:
        if not self.stats:
            raise ValueError("No strategy ideas available")
        # explore any never-evaluated idea first (UCB's infinite bonus at n_i == 0)
        untested = [it for it in self.stats.values() if it["count"] == 0]
        if untested:
            return untested[0]["idea"]
        total = max(self.total_evaluations, 1)

        def ucb(it):
            return it["mean_improvement"] + c * math.sqrt(math.log(total) / it["count"])
        best = max(self.stats.values(), key=ucb)
        return best["idea"]

    def update(self, idea_id: str, z: float):
        """Record an improvement score z = S_idea - S_baseline for this idea's evaluation."""
        it = self.stats.get(idea_id)
        if it is None:
            return
        it["count"] += 1
        it["improvement_sum"] += float(z)
        it["mean_improvement"] = round(it["improvement_sum"] / it["count"], 4)
        self.total_evaluations += 1
        self.save()

    def ranking(self) -> list:
        return sorted(self.stats.values(),
                      key=lambda it: (it["mean_improvement"], it["count"]), reverse=True)


def _idea_id_for(idea: dict) -> str:
    seed = (idea.get("change_to_test") or "") + "|" + (idea.get("implementation_point") or "")
    return "idea_" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Episode summary + role-specific scoring (both used by the loop)
# ---------------------------------------------------------------------------
def summarize_trajectory(traj) -> str:
    """Compact, LLM-friendly summary of ONE episode: outcome + the CAI-style scoring."""
    meta = getattr(traj, "meta", {}) or {}
    sc = meta.get("scoring", {}) or {}
    conf = meta.get("confusion", {}) or {}
    red_steps = [s for s in getattr(traj, "steps", []) if s.get("role") == "red" and not s.get("skipped")]
    techs = [f"{s.get('stage')}:{(s.get('action') or {}).get('tech_id')}"
             f"{'*' if s.get('advanced') else ''}{'!det' if s.get('detected') else ''}"
             f"{'~pat' if s.get('mitigated') else ''}" for s in red_steps][:12]
    return (f"objective_achieved={meta.get('objective_achieved')} "
            f"IA_rate={sc.get('initial_access_rate')} det_rate={sc.get('vuln_detected_rate')} "
            f"patch_rate={sc.get('vuln_patched_rate')} own_pts={sc.get('own_points')} "
            f"avail_pts={sc.get('availability_points')} confusion={conf} "
            f"red_return={round(getattr(traj,'red_return',0.0),2)} "
            f"blue_return={round(getattr(traj,'blue_return',0.0),2)}\nsteps={techs}")


def evaluate_episode(traj, role: str) -> float:
    """Role-specific episode score in [0,1] (fixed scale so z = S_idea - S_baseline is meaningful).
      RED  : offense = 0.5*Initial-Access-rate + 0.5*(Own Points / 300)  (foothold + flags)
      BLUE : defense = 0.4*patch_rate + 0.3*detection-F1 + 0.3*(Availability/200), where the F1
             rewards catching attacks WITHOUT false alarms (so recall-only trigger-happiness and
             service outages do not inflate the score)."""
    meta = getattr(traj, "meta", {}) or {}
    sc = meta.get("scoring", {}) or {}
    if role == "red":
        ia = float(sc.get("initial_access_rate", 0.0) or 0.0)
        own = float(sc.get("own_points", 0.0) or 0.0) / 300.0
        return round(max(0.0, min(1.0, 0.5 * ia + 0.5 * own)), 4)
    # blue
    conf = meta.get("confusion", {}) or {}
    tp, fp, fn = float(conf.get("tp", 0)), float(conf.get("fp", 0)), float(conf.get("fn", 0))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    patch = float(sc.get("vuln_patched_rate", 0.0) or 0.0)
    avail = float(sc.get("availability_points", 0.0) or 0.0) / 200.0
    return round(max(0.0, min(1.0, 0.4 * patch + 0.3 * f1 + 0.3 * avail)), 4)


# ---------------------------------------------------------------------------
# LLM idea generation (structured, testable; idea_id != tech_id)
# ---------------------------------------------------------------------------
def generate_strategy_idea(llm, role: str, trajectory_summary: str, strategy_history) -> Optional[dict]:
    """Ask the LLM for ONE structured, independently-testable strategy improvement (JSON).
    Returns None if no LLM / parse fails. The idea must change HOW the agent selects or executes
    actions -- not merely restate results, and never claim success before evaluation."""
    if llm is None or not getattr(llm, "available", False):
        return None
    goal = ("advance the attack while evading detection and containment" if role == "red"
            else "detect attacks and verify EFFECTIVE containment without false alarms or outages")
    hist = strategy_history
    if isinstance(strategy_history, dict):
        hist = "\n".join(f"- {iid}: mean_improvement={it.get('mean_improvement')} "
                         f"n={it.get('count')} :: {(it.get('idea') or {}).get('change_to_test','')[:120]}"
                         for iid, it in list(strategy_history.items())[:12]) or "(none yet)"
    system = (
        f"You are improving the {role.upper()} strategy in a red/blue self-play security "
        f"experiment. Goal: {goal}. Propose ONE concrete, INDEPENDENTLY TESTABLE strategy "
        f"improvement that changes HOW the agent selects or executes actions. Do NOT merely "
        f"restate observed results, and do NOT claim an idea succeeded before it is evaluated. "
        f"Return ONLY a JSON object with keys: idea_id (short stable slug), hypothesis, "
        f"change_to_test, implementation_point, evaluation_metric, expected_effect.")
    prompt = (f"Previous episode results:\n{trajectory_summary}\n\n"
              f"Previously tested ideas and their outcomes:\n{hist}\n\n"
              f"Respond with the JSON object only.")
    text = llm.complete(prompt, system=system) or ""
    idea = _parse_idea_json(text)
    if not idea:
        return None
    idea.setdefault("role", role)
    for k in _IDEA_FIELDS:
        idea.setdefault(k, "")
    if not idea.get("idea_id"):
        idea["idea_id"] = _idea_id_for(idea)
    if not (idea.get("change_to_test") or "").strip():
        return None               # an idea with no testable change is useless
    return idea


def _parse_idea_json(text: str) -> Optional[dict]:
    """Extract a JSON object from the LLM reply (tolerant: fenced blocks, json_repair fallback)."""
    import re as _re
    t = (text or "").strip()
    m = _re.search(r"\{.*\}", t, _re.DOTALL)
    blob = m.group(0) if m else t
    for loader in (json.loads,):
        try:
            d = loader(blob)
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    try:
        import json_repair
        d = json_repair.loads(blob)
        return d if isinstance(d, dict) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# The High-level loop: generate -> add -> UCB-select -> baseline vs candidate -> z -> update
# ---------------------------------------------------------------------------
def run_strategy_loop(role: str, iterations: int,
                      run_episode_fn: Callable[[Optional[dict], int], object],
                      llm, *, queue_path: Optional[str] = None, c: float = 1.0,
                      reset_fn: Optional[Callable[[], None]] = None,
                      seed_base: int = 0, repeats: int = 1) -> dict:
    """Drive `iterations` of the strategy-improvement loop for ONE role.

    run_episode_fn(strategy_idea, seed) -> Trajectory : the caller wires this to pipeline.run_episode
      with its target + MCTS (so the ACTION-level MCTS priors persist across calls as before). The
      selected idea is passed through to run_episode(red_strategy_idea=/blue_strategy_idea=).
    reset_fn() : optional environment reset run BEFORE the baseline and BEFORE the candidate, so the
      candidate is NOT advantaged/penalised by state blue left behind in the baseline (fair A/B).
    Each iteration: an LLM idea is generated from the last summary, added to the queue; UCB picks
    the idea to test THIS iteration; we run a BASELINE (no idea) and a CANDIDATE (with idea) under
    the SAME seed (optionally averaged over `repeats`), compute z = S_candidate - S_baseline, and
    update that idea's queue statistics. Returns the history + final ranking."""
    q = IdeaQueue(role, path=queue_path)
    history = []
    last_summary = "(no prior episode)"
    for it in range(1, iterations + 1):
        idea = generate_strategy_idea(llm, role, last_summary, q.stats)
        if idea:
            q.add(idea)
        if not q.stats:
            print(f"[strategy:{role}] iter {it}: no ideas yet (LLM unavailable?) -> skip")
            continue
        selected = q.select_ucb(c=c)
        iid = selected["idea_id"]
        seed = seed_base + it
        base_scores, cand_scores, zs = [], [], []
        reps = max(1, repeats)
        for r in range(reps):
            # SAME seed for the baseline/candidate of a repeat so the ONLY difference is the idea;
            # reset_fn fully restores the environment BEFORE each run (fix: env isolation).
            if reset_fn:
                reset_fn()
            base_traj = run_episode_fn(None, seed + 1000 * r)        # baseline: NO idea
            sb = evaluate_episode(base_traj, role); base_scores.append(sb)
            if reset_fn:
                reset_fn()
            cand_traj = run_episode_fn(selected, seed + 1000 * r)    # candidate: WITH idea
            sc = evaluate_episode(cand_traj, role); cand_scores.append(sc)
            zs.append(sc - sb)
            last_summary = summarize_trajectory(cand_traj)
        s_base = sum(base_scores) / reps
        s_cand = sum(cand_scores) / reps
        z = round(s_cand - s_base, 4)
        # variability of the improvement across repeats (fix: record mean AND variance)
        mean_z = sum(zs) / reps
        var_z = sum((x - mean_z) ** 2 for x in zs) / reps
        std_z = round(var_z ** 0.5, 4)
        q.update(iid, z)
        rec = {"iter": it, "idea_id": iid, "repeats": reps,
               "s_baseline": round(s_base, 4), "s_candidate": round(s_cand, 4),
               "improvement_z": z, "z_std": std_z, "z_per_repeat": [round(x, 4) for x in zs],
               "mean_improvement": q.stats[iid]["mean_improvement"], "n": q.stats[iid]["count"]}
        history.append(rec)
        print(f"[strategy:{role}] iter {it}: idea={iid} S_base={s_base:.3f} S_cand={s_cand:.3f} "
              f"z={z:+.3f}±{std_z:.3f} (n_rep={reps}; mean_improvement={rec['mean_improvement']}, "
              f"n_eval={rec['n']})")
    top = q.ranking()[:5]
    print(f"[strategy:{role}] top ideas by mean improvement:")
    for it_ in top:
        print(f"    {it_['idea']['idea_id']:16} mean_z={it_['mean_improvement']:+.3f} n={it_['count']} "
              f":: {(it_['idea'].get('change_to_test') or '')[:80]}")
    return {"role": role, "history": history, "ranking": top}


# ---------------------------------------------------------------------------
# Convenience: bind pipeline.run_episode into a run_episode_fn(idea, seed) for the loop.
# The SAME red_mcts/blue_mcts are reused across calls, so the ACTION-level MCTS priors persist
# (unchanged learning); the strategy idea is routed to the correct role slot of run_episode.
# ---------------------------------------------------------------------------
def make_episode_runner(role: str, *, target, memory, defense, llm=None,
                        objective=None, n_sims: int = 24,
                        red_priors_ref: Optional[dict] = None,
                        blue_priors_ref: Optional[dict] = None, **episode_kw):
    """Build run_episode_fn(idea, seed) with FULL A/B ISOLATION (pre-connection fixes):
      * MCTS state: a FRESH MCTS is built per call from a DEEP COPY of the reference action priors
        (red_priors_ref / blue_priors_ref), so baseline and candidate START IDENTICAL and neither
        run's prior updates leak into the other or back into the reference (clean z).
      * run_episode mutable state (learned_covered / infeasible / mismatch_hints) is created FRESH
        per call -- never shared across the A/B pair -- so blue-deployed detection coverage or
        pruned-technique sets from one run cannot contaminate the other.
      * same seed per A/B pair -> same MCTS/search + benign/detection RNG, so the only difference
        is the injected strategy idea.
    The reference priors stay FROZEN during the strategy loop (the loop learns STRATEGY ideas, not
    action priors); to also evolve the action policy, update the reference dicts deliberately
    between iterations (kept separate so it never confounds a z measurement)."""
    import pipeline as P
    import random
    import copy as _copy
    red_priors_ref = {} if red_priors_ref is None else red_priors_ref
    blue_priors_ref = {} if blue_priors_ref is None else blue_priors_ref

    def _run(idea, seed):
        rm = P.MCTS(memory, action_priors=_copy.deepcopy(red_priors_ref), n_sims=n_sims,
                    rng=random.Random(seed), llm=llm, objective=objective)
        bm = P.MCTS(memory, action_priors=_copy.deepcopy(blue_priors_ref), n_sims=n_sims,
                    rng=random.Random(seed + 1), llm=llm, objective=objective)
        kw = dict(episode_kw)
        kw["red_strategy_idea"] = idea if role == "red" else None
        kw["blue_strategy_idea"] = idea if role == "blue" else None
        kw["learned_covered"] = set()        # independent per run (no cross-contamination)
        kw["infeasible"] = set()
        kw["mismatch_hints"] = {}
        return P.run_episode(memory, target, rm, bm, defense,
                             rng=random.Random(seed), llm=llm, **kw)
    return _run


def live_reset_fn(target):
    """THOROUGH environment reset for a live target (fix: control firewall + patch + privesc AND
    the dynamically-deployed Wazuh rules, not just the vuln sentinel). Verifies the baseline is
    restored so one A/B run never inherits the other's defences."""
    def _reset():
        if hasattr(target, "flush_rules"):
            try:
                target.flush_rules()            # remove blue-authored Wazuh detection rules
            except Exception:
                pass
        if hasattr(target, "reset_defenses"):
            target.reset_defenses()             # iptables flush + re-arm sudoers + rm patch sentinel
    return _reset


if __name__ == "__main__":
    import argparse, random
    import pipeline as P
    ap = argparse.ArgumentParser(description="High-level strategy-improvement (Idea Queue + UCB)")
    ap.add_argument("--role", choices=["red", "blue"], default="red")
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--c", type=float, default=1.0, help="UCB exploration strength")
    ap.add_argument("--repeats", type=int, default=1, help="A/B repeats per idea (same seeds)")
    ap.add_argument("--live", action="store_true",
                    help="run against the LIVE ShellTarget range (dry_run=False) on the isolated "
                         "mayhem docker network; otherwise a SimulatedTarget demo (no containers)")
    ap.add_argument("--max-steps", type=int, default=None, help="cap episode length (A/B speed)")
    a = ap.parse_args()
    memory = P.load_shared_memory(profile=("shell_linux" if a.live else "shell_linux"))
    llm = P.get_llm() if P.get_llm is not None else None
    n_sims = int(os.environ.get("MAYHEM_SIMS", "24"))
    if a.live:
        # SAFETY: the ShellTarget range runs on the dedicated, isolated shell_cyber_range docker
        # network against a TEST host only. Real command execution + real Wazuh measurement.
        tgt = P.ShellTarget(build=False)
        dry = False
        reset_fn = live_reset_fn(tgt)
    else:
        tgt = P.SimulatedTarget(rng=random.Random(0))
        dry = True
        reset_fn = (lambda: tgt.reset_defenses()) if hasattr(tgt, "reset_defenses") else None
    obj = tgt.objective()
    defense = P.DefenseGraph(memory)
    ep_kw = {"dry_run": dry}
    if a.max_steps:
        ep_kw["max_steps"] = a.max_steps
    # FROZEN reference action priors shared as the identical START for every baseline & candidate
    # (deep-copied per run inside make_episode_runner so A/B never cross-contaminate).
    red_ref, blue_ref = {}, {}
    runner = make_episode_runner(a.role, target=tgt, memory=memory, defense=defense, llm=llm,
                                 objective=obj, n_sims=n_sims,
                                 red_priors_ref=red_ref, blue_priors_ref=blue_ref, **ep_kw)
    # verify reset works once up front (live only)
    if a.live and reset_fn:
        reset_fn()
    out = run_strategy_loop(a.role, a.iterations, runner, llm, c=a.c,
                            reset_fn=reset_fn, repeats=a.repeats)
    if a.live and reset_fn:
        reset_fn()   # restore the range to a clean baseline after the experiment
    print(f"[strategy] done ({'LIVE' if a.live else 'SIM'}): {len(out['history'])} iters, "
          f"{len(out['ranking'])} ideas ranked.")
