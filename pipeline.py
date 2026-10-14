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
def is_covered(tid: str, memory: dict, learned_covered=None) -> bool:
    """A technique is detectable if Wazuh ships a rule for it OR blue has deployed one."""
    base = bool(memory.get(tid, {}).get("wazuh_covered"))
    return base or (learned_covered is not None and tid in learned_covered)


def generate_candidates(role: str, state: CyberState, memory: dict, k: int = 4, llm=None,
                        learned_covered=None, evasive: bool = False) -> list[dict]:
    """Produce actionable strategy candidates for red or blue from the ATT&CK memory.

    The candidate *set* comes from the shared ATT&CK memory. When an LLM is available
    it reorders/selects the most promising techniques for the current state (the
    'LLM decides strategy / next action' step); MCTS then simulates to pick the best.
    Falls back to a rule-based ordering with no LLM.

    Blue candidates carry `deploy_rule=True` when the technique is NOT yet detectable:
    applying such a defense adds a Wazuh detection rule (coverage expansion), so blue's
    self-improvement grows SIEM coverage over iterations.
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
        if evasive:
            # EVASIVE campaign: among on-tactic techniques, prefer the lowest-signal
            # ones -- those NOT yet covered by a Wazuh rule -- so the action blends into
            # the benign baseline instead of tripping an existing detection. This is what
            # drives the "hide in normal logs" behaviour; blue must grow coverage to catch up.
            pool.sort(key=lambda a: (target_tactic not in a["tactics"],
                                     is_covered(a["tech_id"], memory, learned_covered),
                                     a["tech_id"]))
    else:  # blue: defend what has been executed/detected but not yet defended
        pool = []
        for tid in (state.detected or state.executed):
            info = memory.get(tid, {})
            if tid in state.defended:
                continue
            strat = (info.get("detection_strategies") or [{}])
            covered = is_covered(tid, memory, learned_covered)
            if covered:
                pool.append({"role": "blue", "tech_id": tid, "deploy_rule": False,
                             "name": f"Contain {info.get('name', tid)}",
                             "defense": strat[0].get("name", "generic monitoring")})
            else:
                # the technique evaded the SIEM -> blue can author a detection rule
                pool.append({"role": "blue", "tech_id": tid, "deploy_rule": True,
                             "name": f"Deploy detection rule for {info.get('name', tid)}",
                             "defense": strat[0].get("name", "new Wazuh rule")})
        if not pool:
            pool = [{"role": "blue", "tech_id": None, "deploy_rule": False,
                     "name": "Harden & monitor baseline", "defense": "baseline hardening"}]

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
                 llm=None, learned_covered=None, evasive: bool = False):
        self.memory = memory
        self.priors = action_priors or {}
        self.n_sims = n_sims
        self.c_uct = c_uct
        self.rng = rng or random.Random(0)
        self.llm = llm  # LLM proposes/reorders candidate actions (strategy generation)
        self.learned_covered = learned_covered  # blue-deployed detection coverage
        self.evasive = evasive  # red plans a stealth campaign that blends into benign noise

    def search(self, state: CyberState, role: str = "red") -> tuple[dict, float]:
        candidates = generate_candidates(role, state, self.memory, llm=self.llm,
                                         learned_covered=self.learned_covered,
                                         evasive=self.evasive and role == "red")
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
            # evasive red plans against detection: uncovered techniques are quieter,
            # and getting caught is weighted much more heavily (stealth is the objective).
            covered = is_covered(action.get("tech_id"), self.memory, self.learned_covered)
            p_detect = (0.75 if covered else 0.15) if self.evasive else 0.5
            detected = self.rng.random() < p_detect
            reward = 0.0
            reward += 1.0 if success else -0.2
            reward -= (1.1 if self.evasive else 0.6) if detected else 0.0
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

    def run(self, action: dict, defended: bool = False) -> dict:
        base = float(action.get("winrate", 0.5))
        # a re-attempt against a correctly-applied defense should mostly fail
        p = base * 0.15 if defended else base
        success = self.rng.random() < p
        return {
            "action": action,
            "success": success,
            "stdout": f"[sim] {'re-attempt ' if defended else ''}executed {action.get('tech_id')} "
                      f"({action.get('name')}) -> {'ok' if success else 'blocked'}",
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
        from wazuh_client import get_wazuh_client
        self.ctf = caibench.ctf(name=ctf_name)
        self.ctf.start_ctf()
        self.ip = self.ctf.get_ip()
        self.attacker_container = attacker_container or os.environ.get("ATTACKER_CONTAINER") or None
        self.wazuh = get_wazuh_client()
        self.agent_name = os.environ.get("WAZUH_AGENT_NAME") or ctf_name

    def target_ip(self) -> str:
        return self.ip or "target"

    def run(self, action: dict, defended: bool = False) -> dict:
        """Red attack: tools run FROM the attacker container (remote) or IN the target.
        `defended` just marks a re-attempt probe; reality (the command result) decides."""
        import redtools
        cmd = action.get("command") or f"echo run {action.get('tech_id')}"
        if self.attacker_container:
            out = redtools.run_command(cmd, container=self.attacker_container)
        else:
            out = redtools.run_command(cmd, ctf=self.ctf)
        return {"action": action, "success": bool(out and "error" not in out.lower()),
                "stdout": out or ""}

    def wazuh_detect(self, since, tech_id: str):
        """Return (detected: bool, context: dict) from REAL Wazuh alerts since `since`."""
        if not (self.wazuh and self.wazuh.available):
            return None, None  # caller falls back
        alerts = self.wazuh.query_alerts(since=since, agent_name=self.agent_name)
        tids = self.wazuh.technique_ids(alerts)
        detected = tech_id in tids
        ctx = extract_wazuh_context(alerts[0]) if alerts else None
        return detected, ctx

    def run_benign(self, command: str) -> str:
        """Execute a NORMAL (benign) command on the target host itself (not the attacker
        box) so Wazuh observes genuine routine activity."""
        import redtools
        return redtools.run_command(command, ctf=self.ctf)

    def measure_benign_fp(self, benign_events: list[dict]):
        """Run benign commands ON the target, then ask Wazuh whether it alerted in that
        window -> REAL false positives. Returns {"events", "fp", "tn", "alerts"} or None
        if Wazuh is unavailable (caller falls back to the proxy model).

        Time-separation is the key to attribution: benign commands run in their OWN window
        (before the attack), so any alert here is a false alarm on normal activity, not the
        attack. The alert count is clamped to the number of benign commands."""
        if not (self.wazuh and self.wazuh.available):
            return None
        from datetime import datetime as _dt, timezone as _tz
        import time as _time
        t0 = _dt.now(_tz.utc)
        for ev in benign_events:
            cmd = ev.get("command")
            if cmd:
                self.run_benign(cmd)
        _time.sleep(float(os.environ.get("WAZUH_POLL_WAIT", "3")))  # let alerts land
        alerts = self.wazuh.query_alerts(since=t0, agent_name=self.agent_name)
        # Benign window must not count attack-tagged alerts (shouldn't appear here, but be safe).
        fp_alerts = [a for a in alerts if not self.wazuh.technique_ids([a])]
        n = len(benign_events)
        fp_count = min(len(fp_alerts), n)   # each benign event alerts at most once
        events = []
        for i, ev in enumerate(benign_events):
            events.append({"malicious": False, "kind": ev["decoder"]["name"],
                           "level": ev["rule"]["level"], "alerted": i < fp_count, "live": True})
        return {"events": events, "fp": fp_count, "tn": n - fp_count, "alerts": len(alerts)}

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
    meta: dict = field(default_factory=dict)  # campaign/win info for DefenderBench-style scoring


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
                rng: Optional[random.Random] = None, llm=None, learned_covered=None,
                evasive: bool = False, benign_rate: int = 4) -> Trajectory:
    """One red-vs-blue self-play episode. BOTH agents use MCTS to choose actions,
    the LLM turns them into concrete commands, and both actually run against the
    Target (attack / mitigation). Blue can DEPLOY detection rules, which grow
    `learned_covered` so future attacks with that technique are detectable.

    EVASIVE mode: this is ONE continuous campaign that advances through the kill
    chain, with the attacker's action interleaved among `benign_rate` normal log
    events each step (see benign_activity.py). Blue is scored on whether it tells
    the malicious event apart from the benign baseline (precision/recall -> Macro-F1),
    not merely on how often it alarms. Red prefers low-signal (uncovered) techniques,
    so it hides in the noise until blue grows detection coverage to catch up."""
    rng = rng or random.Random(2)
    if learned_covered is None:
        learned_covered = set()
    state = CyberState()
    traj = Trajectory()
    tgt_ip = target.target_ip() if hasattr(target, "target_ip") else "target"
    from benign_activity import BenignActivityGenerator
    benign_gen = BenignActivityGenerator(rng=rng, agent_name=getattr(target, "agent_name", "target"))
    # per-event confusion tally across the whole episode (DefenderBench detection = Macro-F1)
    conf = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    fp_base = float(os.environ.get("BLUE_FP_RATE", "0.03"))  # blue's baseline false-alarm rate
    for _ in range(max_steps):
        # 1) RED: MCTS picks a technique, LLM makes it a command, execute on target
        red_action, _ = red_mcts.search(state, role="red")
        if not red_action:
            break
        red_action = llm_enrich_action(red_action, state, memory, llm=llm, target_ip=tgt_ip)
        from datetime import datetime as _dt, timezone as _tz

        # 1b) BENIGN window FIRST (time-separated from the attack). In --live mode these
        #     run as real commands on the target and we read Wazuh for TRUE false positives;
        #     otherwise they are scored by the proxy model further below.
        n_benign = benign_rate if (evasive or benign_rate) else 0
        live_benign = None
        if not dry_run and hasattr(target, "measure_benign_fp") and n_benign:
            live_benign = target.measure_benign_fp(benign_gen.batch(n_benign))

        attack_time = _dt.now(_tz.utc)
        result = target.run(red_action)

        # 2) SIEM (Wazuh) detection
        covered = is_covered(red_action["tech_id"], memory, learned_covered)
        live_detect = None
        if not dry_run and hasattr(target, "wazuh_detect"):
            import time as _time
            _time.sleep(float(os.environ.get("WAZUH_POLL_WAIT", "3")))  # let the alert land
            live_detect, live_ctx = target.wazuh_detect(attack_time, red_action["tech_id"])
        if live_detect is not None:
            # REAL Wazuh: detected iff an alert tagged with this technique actually fired
            detected = bool(live_detect)
            ctx = live_ctx or extract_wazuh_context(synthetic_wazuh_alert(red_action))
        else:
            # simulated proxy: coverage-driven probability (base rules OR blue-deployed)
            ctx = extract_wazuh_context(synthetic_wazuh_alert(red_action))
            p_detect = 0.85 if covered else 0.15
            high_sev = ctx["event"]["level"] and int(ctx["event"]["level"]) >= 7
            detected = bool(high_sev and rng.random() < p_detect)

        # 2b) BENIGN baseline: score blue's discrimination on normal activity.
        #     A false positive is blue alerting on a benign event. This is what makes
        #     "detection" a precision/recall measure (Macro-F1) instead of a recall-only
        #     rate that a trigger-happy detector could max out.
        events = []
        if live_benign is not None:
            # REAL false positives: benign commands actually ran on the target and Wazuh
            # either alerted on them (FP) or not (TN).
            for e in live_benign["events"]:
                conf["fp" if e["alerted"] else "tn"] += 1
                events.append(e)
        else:
            # proxy: FP chance rises with how alarm-like the event looks (_noise) on top of
            # a baseline false-alarm rate (dry-run, or live without Wazuh available).
            for bev in benign_gen.batch(n_benign, base_ts=attack_time):
                fp = rng.random() < min(0.9, fp_base + bev.get("_noise", 0.0))
                conf["fp" if fp else "tn"] += 1
                events.append({"malicious": False, "kind": bev["decoder"]["name"],
                               "level": bev["rule"]["level"], "alerted": bool(fp)})
        # the one malicious event this step
        conf["tp" if detected else "fn"] += 1
        events.append({"malicious": True, "tech_id": red_action["tech_id"],
                       "kind": "attack", "alerted": bool(detected)})

        executed = tuple(sorted(set(state.executed) | {red_action["tech_id"]}))
        det = tuple(sorted(set(state.detected) | ({red_action["tech_id"]} if detected else set())))
        mid_state = CyberState(executed=executed, detected=det, defended=state.defended,
                               foothold=state.foothold, step=state.step)

        # 3) BLUE: MCTS picks a defense, LLM makes it a mitigation command, apply it.
        #    A defense on an uncovered technique DEPLOYS a detection rule (coverage++).
        defense_action, _ = blue_mcts.search(mid_state, role="blue")
        applied = False
        rule_deployed = False
        if defense_action and defense_action.get("tech_id") is not None:
            defense_action = llm_enrich_defense(defense_action, mid_state, memory,
                                                llm=llm, target_ip=tgt_ip)
            dres = target.defend(defense_action)
            applied = dres.get("applied", False)
            if applied and defense_action.get("deploy_rule"):
                # blue authored a new Wazuh rule -> this technique is now detectable
                tid = defense_action["tech_id"]
                if tid not in learned_covered:
                    learned_covered.add(tid)
                    rule_deployed = True

        # 4) VERIFY the defense: re-attempt the same attack. Containment counts ONLY if
        #    the attack no longer succeeds after the defense (reliable criterion), not
        #    merely because the mitigation command ran.
        mitigated = False
        reverified = None
        if (applied and defense_action and result["success"]
                and defense_action.get("tech_id") == red_action["tech_id"]):
            reprobe = target.run(red_action, defended=True)
            reverified = not reprobe["success"]
            mitigated = bool(reverified)
        defd = tuple(sorted(set(state.defended) |
                            ({defense_action["tech_id"]} if (applied and defense_action and defense_action.get("tech_id")) else set())))
        foothold = state.foothold + (1 if (result["success"] and not detected and not mitigated) else 0)
        new_state = CyberState(executed=executed, detected=det, defended=defd,
                               foothold=foothold, step=state.step + 1)

        # 5) rewards: red advances & evades; blue detects, contains & grows coverage
        red_reward = (1.0 if result["success"] else 0.0) - (0.7 if detected else 0.0) - (0.5 if mitigated else 0.0)
        blue_reward = (1.0 if detected else -0.3) + (0.7 if mitigated else 0.0) + (0.6 if rule_deployed else 0.0)
        traj.red_return += red_reward
        traj.blue_return += blue_reward
        traj.steps.append({
            "role": "red", "state": asdict(state), "action": red_action,
            "result": {"success": result["success"], "stdout": result["stdout"][:120]},
            "wazuh": ctx, "detected": bool(detected), "covered": bool(covered),
            "mitigated": bool(mitigated),  # was this successful attack contained by blue?
            "events": events,             # benign+malicious stream for precision/recall
            "evaded": bool(result["success"] and not detected and not mitigated),
            "reward": round(red_reward, 3),
        })
        if defense_action and defense_action.get("tech_id") is not None:
            traj.steps.append({
                "role": "blue", "state": asdict(mid_state), "action": defense_action,
                "applied": bool(applied), "mitigated": bool(mitigated),
                "verified": reverified,  # True/False if a re-probe confirmed the block; None if not tested
                "rule_deployed": bool(rule_deployed), "reward": round(blue_reward, 3),
            })

        state = new_state
        if state.foothold >= len(_TACTIC_ORDER) - 1:  # red reached exfiltration
            break
    # DefenderBench-style episode outcome: red "wins" (CBS Winning %) iff the campaign
    # reached its final objective (exfiltration tactic). conf[] carries the per-event
    # confusion matrix used for blue's Macro-F1.
    traj.meta = {
        "evasive": bool(evasive),
        "red_win": bool(state.foothold >= len(_TACTIC_ORDER) - 1),
        "final_foothold": int(state.foothold),
        "objective_steps": len(_TACTIC_ORDER) - 1,
        "confusion": conf,
    }
    return traj


def load_state(out_dir: str) -> dict:
    """Load persisted learning state (priors + blue-deployed coverage) from a prior run."""
    empty = {"red_priors": {}, "blue_priors": {}, "learned_covered": [], "runs": 0, "history": []}
    path = os.path.join(out_dir, "state.json")
    if not os.path.exists(path):
        return dict(empty)
    try:
        with open(path, "r", encoding="utf-8") as f:
            s = json.load(f)
        for k, v in empty.items():
            s.setdefault(k, v)
        return s
    except Exception as e:
        print(f"[pipeline] could not load state ({e}); starting fresh.")
        return dict(empty)


def save_state(out_dir: str, red_priors: dict, blue_priors: dict,
               learned_covered: set, runs: int, history: list) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(out_dir, "state.json"), "w", encoding="utf-8") as f:
        json.dump({"red_priors": red_priors, "blue_priors": blue_priors,
                   "learned_covered": sorted(learned_covered), "runs": runs,
                   "history": history},
                  f, ensure_ascii=False, indent=2)


def run_closed_loop(iterations: int = 3, ctf_name: Optional[str] = None,
                    dry_run: bool = True, out_dir: str = "output/pipeline",
                    seed: int = 0, profile: str = "financial_linux_web",
                    use_llm: bool = True, attacker_container: Optional[str] = None,
                    fresh: bool = False, evasive: bool = False,
                    benign_rate: int = 4) -> list[Trajectory]:
    rng = random.Random(seed)
    memory = load_shared_memory(profile=profile)

    # Resume learned state across runs (priors + blue-deployed detection coverage),
    # so each `main.py` run continues evolving instead of restarting from scratch.
    state0 = {"red_priors": {}, "blue_priors": {}, "learned_covered": [], "runs": 0, "history": []} \
        if fresh else load_state(out_dir)
    red_priors: dict = dict(state0["red_priors"])
    blue_priors: dict = dict(state0["blue_priors"])
    learned_covered: set = set(state0["learned_covered"])
    prior_runs = state0["runs"]
    history: list = list(state0.get("history", []))
    learned_at_start = len(learned_covered)
    if prior_runs and not fresh:
        print(f"[pipeline] resumed from {out_dir}/state.json: runs={prior_runs}, "
              f"red_priors={len(red_priors)}, blue_priors={len(blue_priors)}, "
              f"blue-deployed rules={len(learned_covered)}")
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
            # blue_mcts sees learned_covered so it won't re-deploy rules it already has.
            red_mcts = MCTS(memory, action_priors=red_priors, rng=random.Random(seed + it),
                            llm=llm, learned_covered=learned_covered, evasive=evasive)
            blue_mcts = MCTS(memory, action_priors=blue_priors, rng=random.Random(seed + 100 + it),
                             llm=llm, learned_covered=learned_covered)
            traj = run_episode(memory, target, red_mcts, blue_mcts, defense,
                               dry_run=dry_run, rng=rng, llm=llm, learned_covered=learned_covered,
                               evasive=evasive, benign_rate=benign_rate)
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

    # record coverage history for this run (for the dashboard growth curve)
    base_covered = sum(1 for v in memory.values() if v.get("wazuh_covered"))
    target_total = len(memory)
    try:
        import coverage as _cov
        dataset_total = len(_cov._valid_tids(_cov.load_dataset()))
    except Exception:
        dataset_total = target_total
    learned_n = len(learned_covered)
    history.append({
        "run": prior_runs + 1,
        "base_covered": base_covered,          # Wazuh ships rules for these
        "blue_rules": learned_n,               # cumulative blue-deployed rules
        "new_rules": learned_n - learned_at_start,
        "total_covered": base_covered + learned_n,
        "target_total": target_total,          # techniques in Target profile
        "dataset_total": dataset_total,        # techniques in attack_dataset.json
        "target_coverage": round((base_covered + learned_n) / max(target_total, 1), 4),
        "dataset_coverage": round((base_covered + learned_n) / max(dataset_total, 1), 4),
    })

    # persist learned state so the NEXT run resumes and keeps evolving
    save_state(out_dir, red_priors, blue_priors, learned_covered, prior_runs + 1, history)
    print(f"[pipeline] saved state.json: runs={prior_runs + 1}, "
          f"blue rules={learned_n} | coverage target={100*(base_covered+learned_n)/max(target_total,1):.1f}% "
          f"dataset={100*(base_covered+learned_n)/max(dataset_total,1):.1f}%")

    # cumulative attack / detection metrics
    try:
        import metrics as _metrics
        m = _metrics.compute_metrics([asdict(t) for t in all_traj])
        _metrics.write_metrics(m, out_dir)
        s = m["summary"]
        db = s.get("defenderbench", {})
        det = db.get("detection", {})
        extra = ""
        if db.get("episodes_scored"):
            extra = (f" | RED win%={db['red_winning_pct']*100:.1f}% "
                     f"BLUE Macro-F1={det.get('macro_f1', 0):.3f} "
                     f"(P={det.get('precision', 0):.2f} R={det.get('recall', 0):.2f})")
        print(f"[pipeline] metrics: breach(net)={s['breach_rate']*100:.1f}% "
              f"containment={s['containment_rate']*100:.1f}% block={s['block_rate']*100:.1f}% "
              f"detection={s['detection_rate']*100:.1f}%{extra} -> {out_dir}/metrics.json")
    except Exception as e:  # pragma: no cover
        print(f"[pipeline] metrics computation skipped: {e}")

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
    ap.add_argument("--fresh", action="store_true",
                    help="Ignore saved state.json and start learning from scratch")
    ap.set_defaults(dry_run=True, use_llm=True)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_closed_loop(iterations=args.iterations, ctf_name=args.ctf_name,
                    dry_run=args.dry_run, out_dir=args.out_dir, seed=args.seed,
                    profile=args.profile, use_llm=args.use_llm,
                    attacker_container=args.attacker_container, fresh=args.fresh)
