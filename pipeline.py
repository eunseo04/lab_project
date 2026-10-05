"""
Closed-loop orchestrator wiring the diagram's boxes to the real modules in this repo.

Flow (see the architecture figure):

    Docker environment  -> caibench.ctf(name=...)            (environment/ctf.py)
    Shared ATT&CK memory-> dataset_loader.load_attack_dataset (dataset/)
    MCTS Strategy       -> MCTS (this file) over cyber actions
    Red/Blue Action     -> generate_candidates(role, state, memory)
    Red/Blue Team -> Target -> execute_action()             (CTF.get_shell, guarded)
    Wazuh Alert         -> wazuh_log.extract_wazuh_context   (wazuh_log.py)
    (detect) -> defend  -> DefenseGraph.lookup()
    GRPO policy improve -> export_grpo_dataset()  ->  train.py  (reward.py / trl)
    LLM Improve         -> llm_improve()  (updates strategy priors / MCTS graph)
    Closed-loop         -> run_closed_loop()

The subsystems in this repo (CAI Docker env, Strategist MCTS, TriPlay-RL GRPO) were
standalone. This module is the connective tissue. Steps that need live infrastructure
(a Docker daemon, a running Wazuh/SIEM, GPUs for GRPO) are guarded: with --dry-run the
target is simulated and a synthetic Wazuh alert is used, so the whole loop runs locally
and proves the wiring. Swap `SimulatedTarget` for `DockerTarget` and the synthetic alert
for a real Wazuh pull to go live.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# --- real repo modules (the pieces we are connecting) -----------------------
from wazuh_log import extract_wazuh_context  # Wazuh Alert box

try:
    from dataset.dataset_loader import load_attack_dataset  # shared memory box
except Exception:  # pragma: no cover - dataset package import guard
    load_attack_dataset = None


# A tiny built-in ATT&CK slice so --dry-run works without attack_dataset.json.
# (Generate the full one with `python dataset/dataset.py`.)
_FALLBACK_MEMORY = {
    "T1595": {"name": "Active Scanning", "tactics": ["reconnaissance"],
              "platforms": ["Linux"], "detection_strategies": [{"name": "Network traffic analysis"}]},
    "T1190": {"name": "Exploit Public-Facing Application", "tactics": ["initial-access"],
              "platforms": ["Linux"], "detection_strategies": [{"name": "WAF / app log monitoring"}]},
    "T1059": {"name": "Command and Scripting Interpreter", "tactics": ["execution"],
              "platforms": ["Linux"], "detection_strategies": [{"name": "Process command-line monitoring"}]},
    "T1548": {"name": "Abuse Elevation Control Mechanism", "tactics": ["privilege-escalation"],
              "platforms": ["Linux"], "detection_strategies": [{"name": "Audit sudo/suid usage"}]},
    "T1005": {"name": "Data from Local System", "tactics": ["collection"],
              "platforms": ["Linux"], "detection_strategies": [{"name": "File access auditing"}]},
    "T1041": {"name": "Exfiltration Over C2 Channel", "tactics": ["exfiltration"],
              "platforms": ["Linux"], "detection_strategies": [{"name": "Egress / DNS monitoring"}]},
}

# Rough ordering of the financial-range kill chain; used to reward red progress.
_TACTIC_ORDER = [
    "reconnaissance", "initial-access", "execution",
    "privilege-escalation", "collection", "exfiltration",
]


def load_shared_memory() -> dict:
    """ATT&CK techniques shared by both Red and Blue (the 'shared-memory' dataset)."""
    if load_attack_dataset is not None:
        try:
            return load_attack_dataset()
        except FileNotFoundError:
            pass
    print("[pipeline] attack_dataset.json not found -> using built-in ATT&CK sample.")
    return dict(_FALLBACK_MEMORY)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
@dataclass
class CyberState:
    """Abstract state of one engagement, shared by the MCTS and the executor."""
    executed: tuple = ()        # techniques red has run
    detected: tuple = ()        # techniques Wazuh flagged
    defended: tuple = ()        # techniques blue has mitigated
    foothold: int = 0           # index into _TACTIC_ORDER reached by red
    step: int = 0

    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    def tactic_reached(self) -> str:
        return _TACTIC_ORDER[min(self.foothold, len(_TACTIC_ORDER) - 1)]


# ---------------------------------------------------------------------------
# Strategy generation (Red + Blue action candidates from shared memory)
# ---------------------------------------------------------------------------
def generate_candidates(role: str, state: CyberState, memory: dict, k: int = 4) -> list[dict]:
    """Produce actionable strategy candidates for red or blue from the ATT&CK memory.

    This is the LLM 'idea generation' seam: here it is rule-based over the dataset,
    but `llm_improve` can replace/augment the ranking with model proposals.
    """
    target_tactic = state.tactic_reached()
    if role == "red":
        pool = [
            {"role": "red", "tech_id": tid, "name": info.get("name", tid),
             "tactics": info.get("tactics", [])}
            for tid, info in memory.items()
            if tid not in state.executed
        ]
        # prefer techniques matching the tactic red is currently trying to advance
        pool.sort(key=lambda a: (target_tactic not in a["tactics"], a["tech_id"]))
    else:  # blue: defend what has been executed/detected but not yet defended
        pool = []
        for tid in (state.detected or state.executed):
            info = memory.get(tid, {})
            if tid in state.defended:
                continue
            strat = (info.get("detection_strategies") or [{}])
            pool.append({"role": "blue", "tech_id": tid,
                         "name": f"Defend {info.get('name', tid)}",
                         "defense": strat[0].get("name", "generic monitoring")})
        if not pool:
            pool = [{"role": "blue", "tech_id": None, "name": "Harden & monitor baseline",
                     "defense": "baseline hardening"}]
    return pool[:k]


# ---------------------------------------------------------------------------
# MCTS over cyber actions  (action-level search; "MCTS Strategy" box)
# ---------------------------------------------------------------------------
class MCTS:
    """Minimal UCT search that ranks candidate actions by simulated win-rate.

    The rollout model stands in for 'execute in target + SIEM reaction' so search
    can run before touching the real environment. Priors come from `action_priors`
    (updated by llm_improve), realizing the MCTS-graph-update feedback edge.
    """

    def __init__(self, memory: dict, action_priors: Optional[dict] = None,
                 n_sims: int = 64, c_uct: float = 1.4, rng: Optional[random.Random] = None):
        self.memory = memory
        self.priors = action_priors or {}
        self.n_sims = n_sims
        self.c_uct = c_uct
        self.rng = rng or random.Random(0)

    def search(self, state: CyberState, role: str = "red") -> tuple[dict, float]:
        candidates = generate_candidates(role, state, self.memory)
        if not candidates:
            return {}, 0.0
        N = {i: 0 for i in range(len(candidates))}
        W = {i: 0.0 for i in range(len(candidates))}
        total = 0
        for _ in range(self.n_sims):
            total += 1
            i = self._select(candidates, N, W, total)
            reward = self._rollout(state, candidates[i], role)
            N[i] += 1
            W[i] += reward
        best = max(range(len(candidates)), key=lambda i: (W[i] / N[i]) if N[i] else 0.0)
        winrate = (W[best] / N[best]) if N[best] else 0.0
        chosen = dict(candidates[best])
        chosen["winrate"] = round(winrate, 3)
        return chosen, winrate

    def _select(self, candidates, N, W, total) -> int:
        best_i, best_u = 0, -1e9
        for i, c in enumerate(candidates):
            prior = self.priors.get(c.get("tech_id") or c["name"], 0.5)
            if N[i] == 0:
                u = prior + self.c_uct  # encourage unexplored
            else:
                q = W[i] / N[i]
                u = q + self.c_uct * prior * math.sqrt(math.log(total + 1) / N[i])
            if u > best_u:
                best_i, best_u = i, u
        return best_i

    def _rollout(self, state: CyberState, action: dict, role: str) -> float:
        """Stochastic self-play rollout. Replace with real Target+Wazuh outcomes."""
        prior = self.priors.get(action.get("tech_id") or action["name"], 0.5)
        if role == "red":
            advances = action.get("tactics") and state.tactic_reached() in action["tactics"]
            p_success = 0.4 + 0.4 * prior + (0.1 if advances else 0.0)
            success = self.rng.random() < p_success
            detected = self.rng.random() < 0.5  # SIEM coverage proxy
            reward = 0.0
            reward += 1.0 if success else -0.2
            reward -= 0.6 if detected else 0.0
            return max(0.0, min(1.0, 0.5 + 0.5 * reward))
        else:  # blue win-rate: catching/defending the active technique
            p_defend = 0.3 + 0.6 * prior
            return 1.0 if self.rng.random() < p_defend else 0.2


# ---------------------------------------------------------------------------
# Target execution (Docker vs simulated)
# ---------------------------------------------------------------------------
class SimulatedTarget:
    """Stand-in for the Docker target: deterministic-ish outcome, no daemon needed."""

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random(1)

    def run(self, action: dict) -> dict:
        success = self.rng.random() < float(action.get("winrate", 0.5))
        return {
            "action": action,
            "success": success,
            "stdout": f"[sim] executed {action.get('tech_id')} ({action.get('name')}) -> "
                      f"{'ok' if success else 'blocked'}",
        }


class DockerTarget:
    """Live target backed by the CAI CTF container (environment/ctf.py via caibench)."""

    def __init__(self, ctf_name: str):
        import caibench
        self.ctf = caibench.ctf(name=ctf_name)
        self.ctf.start_ctf()

    def run(self, action: dict) -> dict:
        cmd = action.get("command") or f"echo run {action.get('tech_id')}"
        out = self.ctf.get_shell(cmd)
        return {"action": action, "success": bool(out), "stdout": out or ""}

    def close(self):
        try:
            self.ctf.cleanup()
        except Exception as e:  # pragma: no cover
            print(f"[pipeline] target cleanup error: {e}")


# ---------------------------------------------------------------------------
# Detection (Wazuh) + graph-based defense lookup
# ---------------------------------------------------------------------------
def synthetic_wazuh_alert(action: dict, host_ip: str = "172.30.20.10") -> dict:
    """A Wazuh-shaped alert for dry-run, parseable by wazuh_log.extract_wazuh_context."""
    return {
        "timestamp": "2026-10-06T00:00:00.000+0000",
        "agent": {"name": "financial-bank-web", "ip": host_ip},
        "data": {"srcip": "172.30.10.100", "srcuser": "www-data"},
        "rule": {"id": "100100", "level": 10,
                 "description": f"Possible {action.get('name')} ({action.get('tech_id')})",
                 "groups": ["attack", action.get("tech_id", "unknown")]},
        "decoder": {"name": "json"},
    }


class DefenseGraph:
    """Maps a detected technique to a stored defense action (the 'graph lookup' step)."""

    def __init__(self, memory: dict):
        self.memory = memory

    def lookup(self, tech_id: Optional[str]) -> dict:
        info = self.memory.get(tech_id, {}) if tech_id else {}
        strat = (info.get("detection_strategies") or [{}])
        return {"tech_id": tech_id,
                "defense": strat[0].get("name", "isolate host + rotate credentials")}


# ---------------------------------------------------------------------------
# Trajectory recording + GRPO export (bridge to train.py / reward.py)
# ---------------------------------------------------------------------------
@dataclass
class Trajectory:
    steps: list = field(default_factory=list)
    red_return: float = 0.0
    blue_return: float = 0.0


def export_grpo_dataset(trajectories: list[Trajectory], out_path: str, role: str = "red") -> str:
    """Write a GRPO-ready JSONL (chat `prompt` rows) consumable by train.py.

    train.py does `Dataset.from_json(path)` and GRPOTrainer samples completions per
    `prompt`; reward.py scores them. We emit one row per trajectory step: the prompt
    frames the state+strategy-generation task so the policy learns to pick actions.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rows = []
    sys_msg = (f"You are the {role.upper()} security agent. Given the engagement state "
               f"and the shared MITRE ATT&CK memory, propose the next best action.")
    for traj in trajectories:
        for st in traj.steps:
            if st.get("role") != role:
                continue
            user = (f"State: foothold={st['state']['foothold']} "
                    f"executed={st['state']['executed']} detected={st['state']['detected']}\n"
                    f"Candidate chosen by MCTS: {st['action'].get('tech_id')} "
                    f"({st['action'].get('name')}), winrate={st['action'].get('winrate')}")
            rows.append({"prompt": [{"role": "system", "content": sys_msg},
                                    {"role": "user", "content": user}],
                         "meta": {"reward_hint": st.get("reward", 0.0)}})
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[pipeline] wrote {len(rows)} GRPO rows -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# LLM improve (feedback -> strategy priors / MCTS graph update)
# ---------------------------------------------------------------------------
def llm_improve(trajectories: list[Trajectory], priors: dict) -> dict:
    """Close the loop: adjust action priors from observed returns.

    This is the 'LLM Improve' box (Strategy 개선 / Value·승률 재계산 / MCTS Graph 업데이트).
    A real implementation calls an LLM to rewrite strategies; here we do a transparent
    value update so the behaviour is inspectable and testable.
    """
    new = dict(priors)
    for traj in trajectories:
        for st in traj.steps:
            key = st["action"].get("tech_id") or st["action"].get("name")
            if key is None:
                continue
            old = new.get(key, 0.5)
            # nudge toward realized reward (EMA)
            new[key] = round(0.8 * old + 0.2 * float(st.get("reward", 0.5)), 3)
    return new


# ---------------------------------------------------------------------------
# Episode + closed loop
# ---------------------------------------------------------------------------
def run_episode(memory: dict, target, mcts: MCTS, defense: DefenseGraph,
                max_steps: int = 6, dry_run: bool = True,
                rng: Optional[random.Random] = None) -> Trajectory:
    rng = rng or random.Random(2)
    state = CyberState()
    traj = Trajectory()
    for _ in range(max_steps):
        # 1) Red searches + acts
        red_action, red_wr = mcts.search(state, role="red")
        if not red_action:
            break
        result = target.run(red_action)

        # 2) SIEM (Wazuh) detection
        alert = synthetic_wazuh_alert(red_action) if dry_run else target.pull_wazuh_alert()
        ctx = extract_wazuh_context(alert)
        detected = ctx["event"]["level"] and int(ctx["event"]["level"]) >= 7 and rng.random() < 0.6

        # 3) Blue defends via graph lookup when detected
        defense_action = None
        if detected:
            defense_action = defense.lookup(red_action.get("tech_id"))

        # 4) advance state
        executed = tuple(sorted(set(state.executed) | {red_action["tech_id"]}))
        det = tuple(sorted(set(state.detected) | ({red_action["tech_id"]} if detected else set())))
        defd = tuple(sorted(set(state.defended) | ({red_action["tech_id"]} if defense_action else set())))
        foothold = state.foothold + (1 if result["success"] and not detected else 0)
        new_state = CyberState(executed=executed, detected=det, defended=defd,
                               foothold=foothold, step=state.step + 1)

        red_reward = (1.0 if result["success"] else 0.0) - (0.7 if detected else 0.0)
        blue_reward = (1.0 if detected else 0.0) + (0.5 if defense_action else 0.0)
        traj.red_return += red_reward
        traj.blue_return += blue_reward
        traj.steps.append({
            "role": "red", "state": asdict(state), "action": red_action,
            "result": {"success": result["success"], "stdout": result["stdout"][:120]},
            "wazuh": ctx, "detected": bool(detected),
            "defense": defense_action, "reward": round(red_reward, 3),
        })
        if defense_action:
            traj.steps.append({
                "role": "blue", "state": asdict(new_state), "action": defense_action,
                "reward": round(blue_reward, 3),
            })

        state = new_state
        if state.foothold >= len(_TACTIC_ORDER) - 1:  # red reached exfiltration
            break
    return traj


def run_closed_loop(iterations: int = 3, ctf_name: Optional[str] = None,
                    dry_run: bool = True, out_dir: str = "output/pipeline",
                    seed: int = 0) -> list[Trajectory]:
    rng = random.Random(seed)
    memory = load_shared_memory()
    priors: dict = {}
    defense = DefenseGraph(memory)
    target = SimulatedTarget(rng) if dry_run else DockerTarget(ctf_name or "cowsay")

    all_traj: list[Trajectory] = []
    try:
        for it in range(1, iterations + 1):
            mcts = MCTS(memory, action_priors=priors, rng=random.Random(seed + it))
            traj = run_episode(memory, target, mcts, defense, dry_run=dry_run, rng=rng)
            all_traj.append(traj)
            print(f"[iter {it}] steps={len(traj.steps)} "
                  f"red_return={traj.red_return:.2f} blue_return={traj.blue_return:.2f}")

            # GRPO trajectory-level improvement (export -> train.py consumes this)
            export_grpo_dataset([traj], os.path.join(out_dir, f"grpo_red_iter{it}.jsonl"), role="red")

            # LLM improve: recompute priors / update MCTS graph (closed-loop edge)
            priors = llm_improve([traj], priors)
    finally:
        if not dry_run and hasattr(target, "close"):
            target.close()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(out_dir, "trajectories.json"), "w", encoding="utf-8") as f:
        json.dump([asdict(t) for t in all_traj], f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "priors.json"), "w", encoding="utf-8") as f:
        json.dump(priors, f, ensure_ascii=False, indent=2)
    print(f"[pipeline] done. artifacts in {out_dir}/")
    return all_traj


def parse_args():
    ap = argparse.ArgumentParser(description="Red/Blue MCTS-GRPO closed-loop orchestrator")
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--ctf-name", type=str, default=None, help="CTF name for live DockerTarget")
    ap.add_argument("--dry-run", action="store_true",
                    help="Simulate target + Wazuh (no Docker/SIEM needed)")
    ap.add_argument("--live", dest="dry_run", action="store_false",
                    help="Use the real Docker target (requires daemon + Wazuh)")
    ap.add_argument("--out-dir", type=str, default="output/pipeline")
    ap.add_argument("--seed", type=int, default=0)
    ap.set_defaults(dry_run=True)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_closed_loop(iterations=args.iterations, ctf_name=args.ctf_name,
                    dry_run=args.dry_run, out_dir=args.out_dir, seed=args.seed)
