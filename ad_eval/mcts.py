"""Multi-step stochastic UCT over abstract states; simulations are not evidence.

Only the executor can establish real access, capabilities, flags or mitigations.
Search transitions remain local to each call and never enter the final scorer.
"""
from __future__ import annotations
import json
import math
import random
from dataclasses import dataclass, asdict, field, replace
from pathlib import Path
from typing import Optional


def load_shared_memory(profile: str = "linux", path: Path | None = None) -> dict:
    """Load the frozen source dataset, without training overlays or exploit commands."""
    if profile != "linux":
        raise ValueError("the evaluation range supports the linux profile")
    source = path or Path(__file__).resolve().parents[1] / "attack_dataset.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError("ATT&CK memory must be a nonempty object")
    memory = {}
    for tid, info in raw.items():
        platforms = {str(p).lower() for p in info.get("platforms", [])}
        if platforms and "linux" not in platforms:
            continue
        memory[tid] = {key: info[key] for key in
                       ("name", "tactics", "platforms", "detection_strategies") if key in info}
    for tid in environment_infeasible(memory, profile):
        del memory[tid]
    if not memory:
        raise ValueError("no Linux techniques in the frozen dataset")
    return memory

_TACTIC_ORDER = [
    "reconnaissance", "resource-development", "initial-access", "execution",
    "persistence", "privilege-escalation", "defense-impairment",
    "credential-access", "discovery", "lateral-movement",
    "collection", "command-and-control", "exfiltration",
]


_STAGE_CAPABILITY = {
    "reconnaissance": "recon",
    "resource-development": "infra",
    "initial-access": "access",       # first foothold / shell on the host
    "execution": "exec",              # code execution on the host
    "persistence": "persistence",
    "privilege-escalation": "root",   # elevated privileges
    "defense-impairment": "stealth",
    "credential-access": "creds",
    "discovery": "mapped",
    "lateral-movement": "pivot",
    "collection": "staged",           # target data located/staged
    "command-and-control": "c2",
    "exfiltration": "exfil",          # data moved out == objective reached
}


def stage_capability(stage: str) -> Optional[str]:
    return _STAGE_CAPABILITY.get(stage)


def prerequisite_capability(stage: str) -> Optional[str]:
    """The capability a technique at `stage` REQUIRES before it can be attempted
    (the previous stage's capability). The first stage has no prerequisite."""
    try:
        i = _TACTIC_ORDER.index(stage)
    except ValueError:
        return None
    if i <= 0:
        return None
    return _STAGE_CAPABILITY.get(_TACTIC_ORDER[i - 1])


@dataclass
class CyberState:
    """Abstract state of one engagement, shared by the MCTS and the executor.

    At the root, the executor supplies its observed planning state. Descendants
    are hypothetical transition-model states, never verified environment facts.
    Simulated capabilities and flags are never returned to the executor as evidence."""
    executed: tuple = ()        # techniques red has run
    detected: tuple = ()        # techniques Wazuh flagged
    defended: tuple = ()        # techniques blue has mitigated
    capabilities: tuple = ()    # verified stage capabilities red currently holds
    foothold: int = 0           # kill-chain stages cleared so far (derived progress index)
    flag: bool = False          # objective captured (environment-verified)
    step: int = 0

    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    def tactic_reached(self) -> str:
        return _TACTIC_ORDER[min(self.foothold, len(_TACTIC_ORDER) - 1)]

    def has(self, cap: Optional[str]) -> bool:
        """True if red holds this capability (None means 'no prerequisite')."""
        return cap is None or cap in self.capabilities


def is_covered(tid: str, memory: dict, learned_covered=None) -> bool:
    """A technique is detectable if Wazuh ships a rule for it OR blue has deployed one."""
    base = bool(memory.get(tid, {}).get("wazuh_covered"))
    return base or (learned_covered is not None and tid in learned_covered)


_ENV_INFEASIBLE_NAME_PATTERNS = (
    "phishing", "spearphishing", "supply chain", "trusted relationship",
    "hardware additions", "drive-by", "drive by", "removable media",
    "replication through", "wi-fi", "wifi", "wireless", "bluetooth",
    "mobile", "android", "ios ", "sms", "voice", "qr code", "physical",
    "cloud account", "office application", "email",
)


_LAB_PROFILE_HINTS = ("financial", "web", "linux", "container", "bank")


def environment_infeasible(memory: dict, profile: str) -> set:
    """Technique ids that cannot be performed in this target environment at all
    (capability the range simply does not have). Name-pattern based so it is robust to
    id churn; gated to lab profiles so a generic 'all' profile is unaffected."""
    p = (profile or "").lower()
    if not any(h in p for h in _LAB_PROFILE_HINTS):
        return set()
    out = set()
    for tid, info in memory.items():
        name = (info.get("name") or "").lower()
        if any(pat in name for pat in _ENV_INFEASIBLE_NAME_PATTERNS):
            out.add(tid)
    return out


def generate_candidates(role: str, state: CyberState, memory: dict, k: int = 4, llm=None,
                        learned_covered=None, evasive: bool = False,
                        objective: Optional["Objective"] = None,
                        focus_tid: Optional[str] = None,
                        allow_novel: bool = False, persist_path=None,
                        infeasible=None) -> list[dict]:
    """Produce actionable strategy candidates for red or blue from the ATT&CK memory.

    The candidate *set* comes from the shared ATT&CK memory. When an LLM is available
    it reorders/selects the most promising techniques for the current state (the
    'LLM decides strategy / next action' step); MCTS then simulates to pick the best.
    Falls back to a rule-based ordering with no LLM.

    When `allow_novel` is set (red + LLM), red first INVENTS new actions for the current
    stage that are not already in the shared memory, registers them (so they can be
    chosen this very turn) and persists them to the generated overlay so the attack
    dataset keeps expanding -- this is what lets the attacker diversify itself instead
    of being confined to the original dataset.

    Blue candidates carry `deploy_rule=True` when the technique is NOT yet detectable:
    applying such a defense adds a Wazuh detection rule (coverage expansion), so blue's
    self-improvement grows SIEM coverage over iterations.
    """
    target_tactic = state.tactic_reached()
    infeasible = infeasible or set()
    new_ids: list[str] = []
    if role == "red":
        # IDEATION: let the attack LLM invent brand-new actions for this stage first, so
        # the pool below (built from memory) already contains them.
        if allow_novel:
            raise ValueError("evaluation uses frozen techniques; novel generation belongs to training")
        # exclude techniques that are impossible in THIS environment (pruned once, not
        # re-offered and SKIPped every step).
        all_unexec = [
            {"role": "red", "tech_id": tid, "name": info.get("name", tid),
             "tactics": info.get("tactics", [])}
            for tid, info in memory.items()
            if tid not in state.executed and tid not in infeasible
        ]
        # STAGED kill chain WITH PRECONDITIONS: only offer techniques for the stage red
        # is currently on, and only if red actually HOLDS that stage's prerequisite
        # capability (the previous stage's power). The campaign advances one tactic at a
        # time and never skips a stage it has not truly cleared. If the current stage is
        # exhausted, look ahead ONLY to a later stage whose prerequisite red already holds
        # (so a failed step can never be treated as if it had succeeded).
        def _on_stage(a, tac):
            return tac in (a.get("tactics") or [])
        pool = []
        if state.has(prerequisite_capability(target_tactic)):
            pool = [a for a in all_unexec if _on_stage(a, target_tactic)]
        if not pool:
            for nxt in _TACTIC_ORDER[state.foothold + 1:]:
                if not state.has(prerequisite_capability(nxt)):
                    break  # cannot reach this stage yet -> do not skip ahead
                pool = [a for a in all_unexec if _on_stage(a, nxt)]
                if pool:
                    break
        if not pool:
            # Current stage is exhausted of UNTRIED techniques and no reachable look-ahead.
            # Retry THIS stage's own techniques (including already-executed ones) so red keeps
            # pushing the stage it is stuck on. Do NOT fall back to all_unexec: offering
            # unrelated off-stage techniques here just gets them SKIPped and wrongly pruned
            # as environment-infeasible (removing techniques later stages still need).
            pool = [{"role": "red", "tech_id": tid, "name": info.get("name", tid),
                     "tactics": info.get("tactics", [])}
                    for tid, info in memory.items()
                    if target_tactic in (info.get("tactics") or []) and tid not in infeasible]
        pool.sort(key=lambda a: a["tech_id"])
        if evasive:
            # EVASIVE campaign: among the stage's techniques, prefer the lowest-signal
            # ones -- those NOT yet covered by a Wazuh rule -- so the action blends into
            # the benign baseline instead of tripping an existing detection. This is what
            # drives the "hide in normal logs" behaviour; blue must grow coverage to catch up.
            pool.sort(key=lambda a: (is_covered(a["tech_id"], memory, learned_covered),
                                     a["tech_id"]))
        # Front-load the techniques invented THIS turn so they are not dropped by the
        # top-k cut and actually get evaluated by MCTS (otherwise synthetic .9xx ids,
        # which sort last, would rarely be tried and red would never explore them).
        if new_ids:
            front = [a for a in pool if a["tech_id"] in new_ids]
            pool = front + [a for a in pool if a["tech_id"] not in new_ids]
    else:  # blue: defend what has been executed/detected but not yet defended
        def _mk_defense(tid: str) -> Optional[dict]:
            info = memory.get(tid, {})
            if tid in state.defended:
                return None
            strat = (info.get("detection_strategies") or [{}])
            covered = is_covered(tid, memory, learned_covered)
            if covered:
                return {"role": "blue", "tech_id": tid, "deploy_rule": False,
                        "name": f"Contain {info.get('name', tid)}",
                        "defense": strat[0].get("name", "generic monitoring")}
            # the technique evaded the SIEM -> blue can author a detection rule
            return {"role": "blue", "tech_id": tid, "deploy_rule": True,
                    "name": f"Deploy detection rule for {info.get('name', tid)}",
                    "defense": strat[0].get("name", "new Wazuh rule")}

        # REACTIVE DEFENSE: anchor on the live incoming attack (focus_tid) first so blue
        # actually contains THIS turn's technique -- containment only counts when blue
        # defends the technique red just used, and a plain [:k] over all past techniques
        # would often drop the current one. The current attack leads the candidate list.
        ordered_tids = []
        if focus_tid and focus_tid not in state.defended:
            ordered_tids.append(focus_tid)
        for tid in (state.detected or state.executed):
            if tid not in ordered_tids:
                ordered_tids.append(tid)
        pool = [d for d in (_mk_defense(t) for t in ordered_tids) if d]
        if not pool:
            pool = [{"role": "blue", "tech_id": None, "deploy_rule": False,
                     "name": "Harden & monitor baseline", "defense": "baseline hardening"}]

    if llm is not None and getattr(llm, "available", False) and pool:
        pool = _llm_reorder_candidates(role, state, pool, llm, k, objective=objective)
        # even after an LLM reorder, keep the live attack reachable within the top-k.
        if role == "blue" and focus_tid:
            focus = next((c for c in pool if c.get("tech_id") == focus_tid), None)
            if focus:
                pool = [focus] + [c for c in pool if c is not focus]
    return pool[:k]


def _llm_reorder_candidates(role: str, state: CyberState, pool: list[dict], llm, k: int,
                            objective: Optional["Objective"] = None) -> list[dict]:
    """Ask the LLM to pick the most promising next techniques for this state, told the
    objective and the verified capabilities so it chooses goal-relevant, low-detection
    moves rather than arbitrary techniques."""
    shortlist = pool[: max(k * 3, 12)]  # keep the prompt small
    menu = "\n".join(f"- {c['tech_id']}: {c['name']}" for c in shortlist if c.get("tech_id"))
    if not menu:
        return pool
    # ROLE-AWARE strategy framing: red plans toward the objective flag; blue plans to
    # detect and contain the attacker's observed activity (its goal is NOT red's goal).
    if role == "blue":
        goal = (" Your objective as the defender is to DETECT and CONTAIN the attacker's "
                "current activity: prefer defenses that reliably stop the executed/detected "
                "techniques and expand detection coverage for techniques that evaded Wazuh.")
        pick = f"choose the {k} best DEFENSES to deploy NEXT against the observed attack"
    else:
        g = ""
        if objective is not None:
            g = (f" The campaign objective is to {objective.name} by reaching the "
                 f"'{objective.capture_tactic}' stage (capture {objective.target_file}). "
                 f"Prefer techniques that advance toward that goal while minimizing the "
                 f"chance of Wazuh detection.")
        goal = g
        pick = f"choose the {k} best techniques to try NEXT"
    system = (f"You are the {role.upper()} agent in an authorized Linux Docker cyber-range. "
              f"From the candidates, {pick} given the state, most promising first.{goal} "
              f"Reply with ONLY the technique IDs separated by commas (e.g. T1190,T1059).")
    user = (f"State: stage={state.tactic_reached()} (foothold={state.foothold}), "
            f"capabilities={list(state.capabilities)}, executed={list(state.executed)}, "
            f"detected={list(state.detected)}, defended={list(state.defended)}, "
            f"flag_captured={state.flag}.\n"
            f"Selected strategy: {objective.description if objective else ''}\n"
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


def step_reward(role: str, *, success: bool, detected: bool,
                mitigated: bool = False, rule_deployed: bool = False,
                evasive: bool = False, captured: bool = False,
                on_stage: bool = True) -> float:
    """Local search shaping, independent of GRPO and final evaluation scores.

    MCTS applies the Red shaping here to hypothetical transitions. Actual
    strategy comparisons use observed trajectory_reward, not this utility.
    """
    if role == "red":
        # follows the MCTS rollout shaping: success vs -0.2, detection penalty
        # weighted much harder under an evasive (stealth) campaign.
        r = (1.0 if success else -0.2)
        r -= (1.1 if evasive else 0.6) if detected else 0.0
        r -= 0.5 if mitigated else 0.0  # containment only known on a realized step
        r -= 0.3 if not on_stage else 0.0  # off-stage effort does not progress the goal
        r += 3.0 if captured else 0.0      # objective reached: dominates the return
        return r
    # blue: reward detection, containment and growing coverage (MCTS has no
    # separately-shaped blue rollout reward, so this is the single blue definition).
    r = (1.0 if detected else -0.3)
    r += 0.7 if mitigated else 0.0
    r += 0.6 if rule_deployed else 0.0
    return r


@dataclass
class SearchEdge:
    action: dict
    prior: float
    visits: int = 0
    value_sum: float = 0.
    outcomes: dict = field(default_factory=dict)


@dataclass
class SearchNode:
    state: CyberState
    visits: int = 0
    value_sum: float = 0.
    edges: list = field(default_factory=list)


class MCTS:
    """Finite-horizon stochastic UCT: select, transition, expand and back up.

    An edge can lead to several outcome states, rather than freezing its first
    random outcome. Descendant candidates are regenerated from their own state.
    LLM ranking is used at the real root only, never as simulated observation.
    """
    def __init__(self, memory: dict, action_priors: Optional[dict] = None,
                 n_sims: int = 64, c_uct: float = 1.4, rng: Optional[random.Random] = None,
                 llm=None, learned_covered=None, evasive: bool = False,
                 objective: Optional["Objective"] = None,
                 allow_novel: bool = False, generated_path=None, infeasible=None,
                 max_depth: int = 4, discount: float = .95):
        if n_sims < 1 or not 1 <= max_depth <= 16 or not 0 < discount <= 1:
            raise ValueError("invalid MCTS budget, horizon or discount")
        if allow_novel:
            raise ValueError("search uses frozen techniques; improve strategies separately")
        self.memory, self.priors = memory, action_priors or {}
        self.n_sims, self.c_uct = n_sims, c_uct
        self.rng, self.llm = rng or random.Random(0), llm
        self.learned_covered, self.evasive = learned_covered, evasive
        self.objective = objective or Objective()
        self.infeasible = infeasible or set()
        self.max_depth, self.discount = max_depth, discount
        self.last_search = {}

    def _prior(self, action, strategy):
        value = self.priors.get(action.get("tech_id"), .5)
        if action.get("tech_id") in strategy.get("preferred_techniques", []):
            value *= 2.
        tactics = action.get("tactics", self.memory.get(action.get("tech_id"), {}).get("tactics", []))
        if set(tactics) & set(strategy.get("preferred_tactics", [])):
            value *= 1.5
        return max(.01, value)

    def _expand(self, node, role, focus_tid, strategy, root=False):
        objective = replace(self.objective, description=(self.objective.description +
                            "\n" + strategy.get("instruction", "")).strip())
        candidates = generate_candidates(role, node.state, self.memory, k=12,
                        llm=self.llm if root else None, learned_covered=self.learned_covered,
                        evasive=self.evasive, objective=objective, focus_tid=focus_tid,
                        infeasible=self.infeasible)
        candidates.sort(key=lambda action: self._prior(action, strategy), reverse=True)
        candidates = candidates[:4]
        total = sum(self._prior(action, strategy) for action in candidates) or 1.
        node.edges = [SearchEdge(action, self._prior(action, strategy) / total) for action in candidates]

    def transition(self, state: CyberState, action: dict, role: str,
                   focus_tid: Optional[str] = None) -> tuple[CyberState, float]:
        """Sample a *hypothetical* next state; never mutate an executor state."""
        tid = action.get("tech_id")
        prior = max(0., min(1., self.priors.get(tid, .5)))
        executed, detected, defended = set(state.executed), set(state.detected), set(state.defended)
        caps, foothold, captured = set(state.capabilities), state.foothold, state.flag
        if role == "red":
            stage = state.tactic_reached()
            on_stage = stage in action.get("tactics", []) and state.has(prerequisite_capability(stage))
            success = self.rng.random() < (.4 + .4 * prior + (.1 if on_stage else 0.))
            covered = is_covered(tid, self.memory, self.learned_covered) or tid in defended
            detection = self.rng.random() < ((.75 if covered else .15) if self.evasive else .5)
            if tid:
                executed.add(tid)
                if detection:
                    detected.add(tid)
            mitigated = tid in defended
            if success and on_stage and not mitigated:
                cap = stage_capability(stage)
                if cap:
                    caps.add(cap)
                foothold = min(foothold + 1, len(_TACTIC_ORDER) - 1)
                captured = captured or (stage == self.objective.capture_tactic and not detection)
            shaped = step_reward(role, success=success, detected=detection, mitigated=mitigated,
                                 captured=captured, on_stage=on_stage, evasive=self.evasive)
            reward = max(0., min(1., .5 + .25 * shaped))
        else:
            probability = .3 + .6 * prior + (.4 if tid and tid == focus_tid else 0.)
            probability += .1 if action.get("deploy_rule") else 0.
            success = self.rng.random() < min(1., probability)
            if success and tid:
                defended.add(tid)
                detected.add(tid)
            reward = 1. if success else .2
        result = CyberState(tuple(sorted(executed)), tuple(sorted(detected)), tuple(sorted(defended)),
                            tuple(sorted(caps)), foothold, bool(captured), state.step + 1)
        return result, reward

    def search(self, state: CyberState, role: str = "red", focus_tid: Optional[str] = None,
               strategy: Optional[dict] = None) -> tuple[dict, float]:
        if role not in {"red", "blue"}:
            raise ValueError("expected red/blue")
        strategy = strategy or {}
        horizon = int(strategy.get("max_depth", self.max_depth))
        discount = float(strategy.get("discount", self.discount))
        if not 1 <= horizon <= 16 or not 0 < discount <= 1:
            raise ValueError("invalid strategy search horizon/discount")
        root = SearchNode(state)
        self._expand(root, role, focus_tid, strategy, root=True)
        deepest = 0
        for _ in range(self.n_sims):
            node, path = root, []
            for depth in range(horizon):
                if node.state.flag:
                    break
                if not node.edges:
                    self._expand(node, role, focus_tid, strategy)
                if not node.edges:
                    break
                unvisited = [edge for edge in node.edges if not edge.visits]
                edge = max(unvisited, key=lambda edge: edge.prior) if unvisited else max(
                    node.edges, key=lambda edge: edge.value_sum / edge.visits + self.c_uct *
                    edge.prior * math.sqrt(math.log(node.visits + 1) / edge.visits))
                next_state, reward = self.transition(node.state, edge.action, role, focus_tid)
                child = edge.outcomes.setdefault(next_state.key(), SearchNode(next_state))
                path.append((node, edge, reward))
                node = child
                deepest = max(deepest, depth + 1)
            # Back up normalized discounted returns at every ancestor. They are
            # search estimates only; persistent stats use real trajectory_reward.
            value, mass = 0., 0.
            node.visits += 1
            for parent, edge, reward in reversed(path):
                value, mass = reward + discount * value, 1. + discount * mass
                estimate = value / mass
                edge.visits += 1
                edge.value_sum += estimate
                parent.visits += 1
                parent.value_sum += estimate
        self.last_tree = root
        self.last_search = {"simulations": self.n_sims, "max_depth_reached": deepest,
                            "horizon": horizon, "discount": discount, "root_visits": root.visits,
                            "edges": [{"tech_id": edge.action.get("tech_id"), "visits": edge.visits,
                                       "value": edge.value_sum / edge.visits if edge.visits else None,
                                       "outcome_states": len(edge.outcomes)} for edge in root.edges],
                            "value_source": "hypothetical_model_not_environment_evidence"}
        if not root.edges or not deepest:
            return {}, 0.
        best = max(root.edges, key=lambda edge: (edge.visits, edge.value_sum / edge.visits if edge.visits else -1.))
        value = best.value_sum / best.visits
        chosen = dict(best.action, search_value=round(value, 4))
        return chosen, value


@dataclass
class Objective:
    """A concrete, environment-verifiable goal for the episode, e.g. reading a
    protected flag file. Campaign success is decided by actually capturing this (the
    environment confirms it), NOT by reaching an internal progress score."""
    name: str = "capture_flag"
    challenge: Optional[str] = None     # CTF challenge key (live) or "sim" (dry-run)
    capture_tactic: str = "exfiltration"  # kill-chain stage at which the flag is taken
    target_file: str = "/root/flag.txt"   # the specific file red must access
    description: str = ""
    attack_surface: str = ""            # REAL exploitable endpoints, fed to the LLM so it
