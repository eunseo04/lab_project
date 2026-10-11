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

try:
    import coverage as _coverage  # ATT&CK x Wazuh coverage (Target-scoped)
except Exception:  # pragma: no cover
    _coverage = None

try:
    from llm import get_llm  # OpenAI client (idea generation + improve)
except Exception:  # pragma: no cover
    get_llm = None


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


def load_shared_memory(profile: str = "financial_linux_web") -> dict:
    """ATT&CK techniques shared by both Red and Blue (the 'shared-memory' dataset).

    Scoped to the Target `profile` and annotated with `wazuh_covered` so the
    detection step reflects real ATT&CK x Wazuh coverage (see coverage.py).
    """
    memory = None
    if load_attack_dataset is not None:
        try:
            memory = load_attack_dataset()
        except FileNotFoundError:
            memory = None
    if memory is None:
        print("[pipeline] attack_dataset.json not found -> using built-in ATT&CK sample.")
        memory = dict(_FALLBACK_MEMORY)

    # Scope to Target platforms + annotate Wazuh coverage when coverage.py is usable.
    covered: set = set()
    if _coverage is not None:
        try:
            memory = _coverage.select_target_techniques(memory, profile)
            covered = _coverage.covered_id_set(profile)
            print(f"[pipeline] shared memory scoped to '{profile}': "
                  f"{len(memory)} techniques, {len(covered)} Wazuh-covered.")
        except Exception as e:
            print(f"[pipeline] coverage scoping skipped ({e}); using full memory.")
    for tid, info in memory.items():
        info["wazuh_covered"] = tid in covered
    return memory


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
def generate_candidates(role: str, state: CyberState, memory: dict, k: int = 4, llm=None) -> list[dict]:
    """Produce actionable strategy candidates for red or blue from the ATT&CK memory.

    The candidate *set* comes from the shared ATT&CK memory. When an LLM is available
    it reorders/selects the most promising techniques for the current state (the
    'LLM decides strategy / next action' step); MCTS then simulates to pick the best.
    Falls back to a rule-based ordering with no LLM.
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

    if llm is not None and getattr(llm, "available", False) and pool:
        pool = _llm_reorder_candidates(role, state, pool, llm, k)
    return pool[:k]


def _llm_reorder_candidates(role: str, state: CyberState, pool: list[dict], llm, k: int) -> list[dict]:
    """Ask the LLM to pick the most promising next techniques for this state."""
    shortlist = pool[: max(k * 3, 12)]  # keep the prompt small
    menu = "\n".join(f"- {c['tech_id']}: {c['name']}" for c in shortlist if c.get("tech_id"))
    if not menu:
        return pool
    system = (f"You are the {role.upper()} agent in an authorized financial cyber-range. "
              f"From the candidate MITRE ATT&CK techniques, choose the {k} best to try "
              f"NEXT given the state, most promising first. Reply with ONLY the technique "
              f"IDs separated by commas (e.g. T1190,T1059).")
    user = (f"State: foothold={state.foothold} ({state.tactic_reached()}), "
            f"executed={list(state.executed)}, detected={list(state.detected)}.\n"
            f"Candidates:\n{menu}")
    text = llm.complete(user, system=system)
    if not text:
        return pool
    import re as _re
    picked = _re.findall(r"T\d{4}(?:\.\d{3})?", text)
    if not picked:
        return pool
    by_id = {c["tech_id"]: c for c in pool}
    ordered = [by_id[t] for t in picked if t in by_id]
    # append any remaining rule-based candidates so MCTS still has breadth
    ordered += [c for c in pool if c not in ordered]
    return ordered


def llm_enrich_action(action: dict, state: CyberState, memory: dict, llm=None,
                      target_ip: str = "target") -> dict:
    """Ask the LLM to turn a chosen ATT&CK technique into a concrete, runnable
    action (command + rationale) against the Target. Falls back to a plain echo
    when no LLM is available, so the loop still runs offline.
    """
    if not action or action.get("tech_id") is None:
        return action
    if llm is None or not getattr(llm, "available", False):
        action.setdefault("command", f"echo run {action.get('tech_id')}")
        return action
    info = memory.get(action["tech_id"], {})
    system = ("You are a red-team operator working ONLY inside an authorized, isolated "
              "financial cyber-range (Docker). Given one MITRE ATT&CK technique and the "
              f"engagement state, output ONE shell command to attempt it against the lab "
              f"target at {target_ip} (use real tools: nmap, curl, nc, hydra, sqlmap...), "
              "and a one-line rationale. Respond EXACTLY as two lines: "
              "'CMD: <command>' then 'WHY: <rationale>'.")
    user = (f"Technique {action['tech_id']} ({action.get('name')}). "
            f"Tactics: {info.get('tactics', [])}. Target IP: {target_ip}. "
            f"State: foothold={state.foothold}, executed={list(state.executed)}.")
    text = llm.complete(user, system=system)
    cmd = f"echo run {action['tech_id']}"
    why = ""
    for line in (text or "").splitlines():
        if line.upper().startswith("CMD:"):
            cmd = line.split(":", 1)[1].strip() or cmd
        elif line.upper().startswith("WHY:"):
            why = line.split(":", 1)[1].strip()
    action["command"] = cmd
    if why:
        action["rationale"] = why
    return action


def llm_enrich_defense(defense: dict, state: CyberState, memory: dict, llm=None,
                       target_ip: str = "target") -> dict:
    """Ask the LLM to turn a chosen defense into a concrete mitigation command run
    ON the target host (e.g. iptables block, kill process, add Wazuh rule, rotate
    creds). Falls back to an echo when no LLM is available.
    """
    if not defense or defense.get("tech_id") is None:
        defense.setdefault("command", f"echo apply {defense.get('defense','hardening')}")
        return defense
    if llm is None or not getattr(llm, "available", False):
        defense.setdefault("command", f"echo defend {defense.get('tech_id')}")
        return defense
    info = memory.get(defense["tech_id"], {})
    strat = (info.get("detection_strategies") or [{}])[0].get("name", "monitoring")
    system = ("You are the BLUE defender of an authorized financial cyber-range (Linux). "
              "Given the attacker technique just observed, output ONE shell command to "
              "CONTAIN or MITIGATE it on the host (iptables/ufw block, kill PID, disable "
              "account, tighten perms, add an auditd/Wazuh rule), and a one-line rationale. "
              "Respond EXACTLY as: 'CMD: <command>' then 'WHY: <rationale>'.")
    user = (f"Observed technique {defense['tech_id']} ({info.get('name')}). "
            f"Detection strategy: {strat}. Host: {target_ip}. "
            f"Already defended: {list(state.defended)}.")
    text = llm.complete(user, system=system)
    cmd = f"echo defend {defense['tech_id']}"
    why = ""
    for line in (text or "").splitlines():
        if line.upper().startswith("CMD:"):
            cmd = line.split(":", 1)[1].strip() or cmd
        elif line.upper().startswith("WHY:"):
            why = line.split(":", 1)[1].strip()
    defense["command"] = cmd
    if why:
        defense["rationale"] = why
    return defense


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
                 n_sims: int = 64, c_uct: float = 1.4, rng: Optional[random.Random] = None,
                 llm=None):
        self.memory = memory
        self.priors = action_priors or {}
        self.n_sims = n_sims
        self.c_uct = c_uct
        self.rng = rng or random.Random(0)
        self.llm = llm  # LLM proposes/reorders candidate actions (strategy generation)

    def search(self, state: CyberState, role: str = "red") -> tuple[dict, float]:
        candidates = generate_candidates(role, state, self.memory, llm=self.llm)
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

    def target_ip(self) -> str:
        return "172.30.20.10"

    def run(self, action: dict) -> dict:
        success = self.rng.random() < float(action.get("winrate", 0.5))
        return {
            "action": action,
            "success": success,
            "stdout": f"[sim] executed {action.get('tech_id')} ({action.get('name')}) -> "
                      f"{'ok' if success else 'blocked'}",
        }

    def defend(self, defense_action: dict) -> dict:
        applied = self.rng.random() < float(defense_action.get("winrate", 0.6))
        return {
            "defense": defense_action,
            "applied": applied,
            "stdout": f"[sim] applied defense for {defense_action.get('tech_id')} -> "
                      f"{'mitigated' if applied else 'failed'}",
        }


class DockerTarget:
    """Live target backed by the CAI CTF container (environment/ctf.py via caibench).

    Red-team execution model:
      * attacker_container set  -> run real tools (nmap, curl, nc, sqlmap, ...) FROM
        the attacker container AGAINST the target IP over the network (a true remote
        attack). This is the cross-platform replacement for the POSIX-only CAI tool
        stack (tool/ needs termios/pty, which do not exist on Windows).
      * otherwise               -> exec the command INSIDE the target container via
        ctf.get_shell (post-exploitation / in-host actions).
    """

    def __init__(self, ctf_name: str, attacker_container: Optional[str] = None):
        import caibench
        self.ctf = caibench.ctf(name=ctf_name)
        self.ctf.start_ctf()
        self.ip = self.ctf.get_ip()
        self.attacker_container = attacker_container or os.environ.get("ATTACKER_CONTAINER") or None

    def target_ip(self) -> str:
        return self.ip or "target"

    def run(self, action: dict) -> dict:
        """Red attack: tools run FROM the attacker container (remote) or IN the target."""
        import redtools
        cmd = action.get("command") or f"echo run {action.get('tech_id')}"
        if self.attacker_container:
            out = redtools.run_command(cmd, container=self.attacker_container)
        else:
            out = redtools.run_command(cmd, ctf=self.ctf)
        return {"action": action, "success": bool(out and "error" not in out.lower()),
                "stdout": out or ""}

    def defend(self, defense_action: dict) -> dict:
        """Blue defense: mitigation runs ON the target host (containment/hardening)."""
        import redtools
        cmd = defense_action.get("command") or f"echo apply {defense_action.get('defense')}"
        out = redtools.run_command(cmd, ctf=self.ctf)
        return {"defense": defense_action, "applied": bool(out and "error" not in out.lower()),
                "stdout": out or ""}

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
def llm_improve(trajectories: list[Trajectory], priors: dict, llm=None,
                role: str = "red") -> tuple[dict, str]:
    """Close the loop for one role: adjust that role's action priors from observed
    returns, and (if an LLM is available) produce a strategy-improvement note.

    This is the 'LLM Improve' box (Strategy 개선 / Value·승률 재계산 / MCTS Graph 업데이트).
    Called once per role so red and blue self-improve independently.
    Returns (new_priors, note).
    """
    new = dict(priors)
    for traj in trajectories:
        for st in traj.steps:
            if st.get("role") != role:
                continue
            key = st["action"].get("tech_id") or st["action"].get("name")
            if key is None:
                continue
            old = new.get(key, 0.5)
            new[key] = round(0.8 * old + 0.2 * float(st.get("reward", 0.5)), 3)  # EMA

    note = ""
    if llm is not None and getattr(llm, "available", False):
        summary = []
        for traj in trajectories:
            steps = [(s['role'], s['action'].get('tech_id'), s.get('detected'), s.get('reward'))
                     for s in traj.steps]
            summary.append(f"red_return={traj.red_return:.2f} blue_return={traj.blue_return:.2f} "
                           f"steps={steps}")
        goal = ("advance the attack while evading Wazuh" if role == "red"
                else "detect and contain the attacks earlier")
        system = (f"You are improving the {role.upper()} agent in a red/blue self-play "
                  f"security loop. Its goal: {goal}. Given the last episode's trajectory, "
                  f"suggest in 3 bullets how {role.upper()} should adapt next iteration.")
        note = llm.complete("\n".join(summary), system=system)
    return new, note


# ---------------------------------------------------------------------------
# Episode + closed loop
# ---------------------------------------------------------------------------
def run_episode(memory: dict, target, red_mcts: MCTS, blue_mcts: MCTS, defense: DefenseGraph,
                max_steps: int = 6, dry_run: bool = True,
                rng: Optional[random.Random] = None, llm=None) -> Trajectory:
    """One red-vs-blue self-play episode. BOTH agents use MCTS to choose actions,
    the LLM turns them into concrete commands, and both actually run against the
    Target (attack / mitigation)."""
    rng = rng or random.Random(2)
    state = CyberState()
    traj = Trajectory()
    tgt_ip = target.target_ip() if hasattr(target, "target_ip") else "target"
    for _ in range(max_steps):
        # 1) RED: MCTS picks a technique, LLM makes it a command, execute on target
        red_action, _ = red_mcts.search(state, role="red")
        if not red_action:
            break
        red_action = llm_enrich_action(red_action, state, memory, llm=llm, target_ip=tgt_ip)
        result = target.run(red_action)

        # 2) SIEM (Wazuh) detection -- coverage-driven
        alert = synthetic_wazuh_alert(red_action) if dry_run else target.pull_wazuh_alert()
        ctx = extract_wazuh_context(alert)
        covered = bool(memory.get(red_action["tech_id"], {}).get("wazuh_covered"))
        p_detect = 0.85 if covered else 0.15
        high_sev = ctx["event"]["level"] and int(ctx["event"]["level"]) >= 7
        detected = bool(high_sev and rng.random() < p_detect)

        executed = tuple(sorted(set(state.executed) | {red_action["tech_id"]}))
        det = tuple(sorted(set(state.detected) | ({red_action["tech_id"]} if detected else set())))
        mid_state = CyberState(executed=executed, detected=det, defended=state.defended,
                               foothold=state.foothold, step=state.step)

        # 3) BLUE: MCTS picks a defense, LLM makes it a mitigation command, apply it.
        #    Blue acts on what it detected; if nothing detected it still hardens.
        defense_action, _ = blue_mcts.search(mid_state, role="blue")
        applied = False
        if defense_action and defense_action.get("tech_id") is not None:
            defense_action = llm_enrich_defense(defense_action, mid_state, memory,
                                                llm=llm, target_ip=tgt_ip)
            dres = target.defend(defense_action)
            applied = dres.get("applied", False)

        # 4) advance state (a mitigation on the active technique removes red's foothold gain)
        mitigated = applied and defense_action and defense_action.get("tech_id") == red_action["tech_id"]
        defd = tuple(sorted(set(state.defended) |
                            ({defense_action["tech_id"]} if (applied and defense_action and defense_action.get("tech_id")) else set())))
        foothold = state.foothold + (1 if (result["success"] and not detected and not mitigated) else 0)
        new_state = CyberState(executed=executed, detected=det, defended=defd,
                               foothold=foothold, step=state.step + 1)

        # 5) rewards: red advances & evades; blue detects & contains
        red_reward = (1.0 if result["success"] else 0.0) - (0.7 if detected else 0.0) - (0.5 if mitigated else 0.0)
        blue_reward = (1.0 if detected else -0.3) + (0.7 if mitigated else 0.0)
        traj.red_return += red_reward
        traj.blue_return += blue_reward
        traj.steps.append({
            "role": "red", "state": asdict(state), "action": red_action,
            "result": {"success": result["success"], "stdout": result["stdout"][:120]},
            "wazuh": ctx, "detected": bool(detected), "reward": round(red_reward, 3),
        })
        if defense_action and defense_action.get("tech_id") is not None:
            traj.steps.append({
                "role": "blue", "state": asdict(mid_state), "action": defense_action,
                "applied": bool(applied), "mitigated": bool(mitigated),
                "reward": round(blue_reward, 3),
            })

        state = new_state
        if state.foothold >= len(_TACTIC_ORDER) - 1:  # red reached exfiltration
            break
    return traj


def run_closed_loop(iterations: int = 3, ctf_name: Optional[str] = None,
                    dry_run: bool = True, out_dir: str = "output/pipeline",
                    seed: int = 0, profile: str = "financial_linux_web",
                    use_llm: bool = True, attacker_container: Optional[str] = None) -> list[Trajectory]:
    rng = random.Random(seed)
    memory = load_shared_memory(profile=profile)
    red_priors: dict = {}
    blue_priors: dict = {}
    defense = DefenseGraph(memory)
    target = (SimulatedTarget(rng) if dry_run
              else DockerTarget(ctf_name or "cowsay", attacker_container=attacker_container))
    llm = get_llm() if (use_llm and get_llm is not None) else None

    all_traj: list[Trajectory] = []
    red_notes: list[str] = []
    blue_notes: list[str] = []
    try:
        for it in range(1, iterations + 1):
            # Both agents search with their OWN learned priors (self-play).
            red_mcts = MCTS(memory, action_priors=red_priors, rng=random.Random(seed + it), llm=llm)
            blue_mcts = MCTS(memory, action_priors=blue_priors, rng=random.Random(seed + 100 + it), llm=llm)
            traj = run_episode(memory, target, red_mcts, blue_mcts, defense,
                               dry_run=dry_run, rng=rng, llm=llm)
            all_traj.append(traj)
            print(f"[iter {it}] steps={len(traj.steps)} "
                  f"red_return={traj.red_return:.2f} blue_return={traj.blue_return:.2f}")

            # GRPO export for BOTH roles -> train.py consumes each policy's dataset
            export_grpo_dataset([traj], os.path.join(out_dir, f"grpo_red_iter{it}.jsonl"), role="red")
            export_grpo_dataset([traj], os.path.join(out_dir, f"grpo_blue_iter{it}.jsonl"), role="blue")

            # LLM improve each role independently (value/winrate recompute + notes)
            red_priors, rnote = llm_improve([traj], red_priors, llm=llm, role="red")
            blue_priors, bnote = llm_improve([traj], blue_priors, llm=llm, role="blue")
            if rnote:
                red_notes.append(f"# iter {it}\n{rnote}")
            if bnote:
                blue_notes.append(f"# iter {it}\n{bnote}")
    finally:
        if not dry_run and hasattr(target, "close"):
            target.close()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if red_notes:
        (Path(out_dir) / "llm_improve_red.md").write_text("\n\n".join(red_notes), encoding="utf-8")
    if blue_notes:
        (Path(out_dir) / "llm_improve_blue.md").write_text("\n\n".join(blue_notes), encoding="utf-8")
    with open(os.path.join(out_dir, "trajectories.json"), "w", encoding="utf-8") as f:
        json.dump([asdict(t) for t in all_traj], f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "priors.json"), "w", encoding="utf-8") as f:
        json.dump({"red": red_priors, "blue": blue_priors}, f, ensure_ascii=False, indent=2)
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
    ap.add_argument("--profile", type=str, default=os.environ.get("TARGET_PROFILE", "financial_linux_web"))
    ap.add_argument("--attacker-container", default=os.environ.get("ATTACKER_CONTAINER"),
                    help="Container to launch real attack tools from (--live)")
    ap.add_argument("--no-llm", dest="use_llm", action="store_false",
                    help="Disable the LLM (rule-based only)")
    ap.set_defaults(dry_run=True, use_llm=True)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_closed_loop(iterations=args.iterations, ctf_name=args.ctf_name,
                    dry_run=args.dry_run, out_dir=args.out_dir, seed=args.seed,
                    profile=args.profile, use_llm=args.use_llm,
                    attacker_container=args.attacker_container)
