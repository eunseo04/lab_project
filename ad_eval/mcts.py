"""UCT action ranking extracted from pipeline.py; rollouts are NOT evaluation results.

Evaluation uses a frozen ATT&CK dataset. Financial targets, generated overlays,
GRPO training, synthetic SIEM events and legacy campaign orchestration stay outside
this module. The UCT selection and rollout reward formulas retain their source behavior.
"""
from __future__ import annotations
import json
import math
import random
from dataclasses import dataclass, asdict
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

    `capabilities` and `flag` are the VERIFIED facts the executor threads from turn to
    turn: capabilities are the stage powers red has actually earned (preconditions for
    later stages) and `flag` is True only once the environment confirmed the objective
    was captured. The MCTS reads these so it plans from real state, not assumptions."""
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
    """One source of truth for the per-action / per-step reward.

    This is the MCTS-side reward: it drives action-level search (rollouts),
    episode returns, action priors and llm_improve (strategy reward). It used to
    be duplicated by a divergent inline formula inside run_episode; that copy has
    been removed and both the MCTS rollout and the realized self-play step now
    call this function, so there is a single reward definition.

    It is GOAL-AWARE: capturing the objective flag dominates (`captured`), and an
    off-stage action that cannot advance the campaign is penalized (`on_stage`) so
    the agent pursues the goal instead of spraying unrelated techniques.

    IMPORTANT: GRPO does NOT use this. GRPO scores whole trajectories with the
    TriPlay reward (reward.py). Keep the two separate: MCTS = action/strategy
    reward here; GRPO = trajectory-level TriPlay judging.
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


class MCTS:
    """Minimal UCT search that ranks candidate actions by simulated win-rate.

    The rollout model stands in for 'execute in target + SIEM reaction' so search
    can run before touching the real environment. Priors come from `action_priors`
    (updated by llm_improve), realizing the MCTS-graph-update feedback edge.
    """

    def __init__(self, memory: dict, action_priors: Optional[dict] = None,
                 n_sims: int = 64, c_uct: float = 1.4, rng: Optional[random.Random] = None,
                 llm=None, learned_covered=None, evasive: bool = False,
                 objective: Optional["Objective"] = None,
                 allow_novel: bool = False, generated_path=None, infeasible=None):
        self.memory = memory
        self.priors = action_priors or {}
        self.infeasible = infeasible if infeasible is not None else set()
        self.n_sims = n_sims
        self.c_uct = c_uct
        self.rng = rng or random.Random(0)
        self.llm = llm  # LLM proposes/reorders candidate actions (strategy generation)
        self.learned_covered = learned_covered  # blue-deployed detection coverage
        self.evasive = evasive  # red plans a stealth campaign that blends into benign noise
        self.objective = objective or Objective()  # the goal MCTS plans toward
        self.allow_novel = allow_novel  # red may INVENT new actions (grows the attack dataset)
        self.generated_path = generated_path  # where invented techniques are persisted
        self._proposed_stages: set = set()  # ideate at most once per stage per episode

    def search(self, state: CyberState, role: str = "red",
               focus_tid: Optional[str] = None) -> tuple[dict, float]:
        self._focus_tid = focus_tid  # blue: the live attack technique to contain
        # Throttle ideation to the FIRST time red reaches a given stage this episode, so
        # the LLM invents fresh actions per stage without a proposal call every single step.
        allow_novel = False
        if role == "red" and self.allow_novel:
            stg = state.tactic_reached()
            if stg not in self._proposed_stages:
                self._proposed_stages.add(stg)
                allow_novel = True
        candidates = generate_candidates(role, state, self.memory, llm=self.llm,
                                         learned_covered=self.learned_covered,
                                         evasive=self.evasive and role == "red",
                                         objective=self.objective, focus_tid=focus_tid,
                                         allow_novel=allow_novel,
                                         persist_path=self.generated_path,
                                         infeasible=self.infeasible)
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
        """Stochastic, GOAL-AWARE self-play rollout: it weighs both the chance of
        advancing toward the objective and the detection risk, so MCTS prefers actions
        that make verifiable progress to the flag over merely trying many techniques."""
        prior = self.priors.get(action.get("tech_id") or action["name"], 0.5)
        if role == "red":
            stage = state.tactic_reached()
            on_stage = bool(action.get("tactics") and stage in action["tactics"])
            # precondition: this technique can only progress if red holds the stage's
            # prerequisite capability; otherwise it is wasted effort this turn.
            can_progress = on_stage and state.has(prerequisite_capability(stage))
            p_success = 0.4 + 0.4 * prior + (0.1 if can_progress else 0.0)
            success = self.rng.random() < p_success
            # evasive red plans against detection: uncovered techniques are quieter,
            # and getting caught is weighted much more heavily (stealth is the objective).
            covered = is_covered(action.get("tech_id"), self.memory, self.learned_covered)
            p_detect = (0.75 if covered else 0.15) if self.evasive else 0.5
            detected = self.rng.random() < p_detect
            # objective: if this is the capture stage and the step would succeed+evade,
            # simulate capturing the flag (dominant reward) so search drives to the goal.
            is_capture = can_progress and stage == self.objective.capture_tactic
            captured = bool(is_capture and success and not detected)
            # single reward definition (shared with run_episode); normalize the
            # raw step reward to a [0,1] win-rate for UCT backups.
            reward = step_reward("red", success=success, detected=detected,
                                 evasive=self.evasive, captured=captured,
                                 on_stage=can_progress)
            # keep the win-rate in [0,1]; the capture bonus widens the top end.
            return max(0.0, min(1.0, 0.5 + 0.25 * reward))
        else:  # blue win-rate: catching/defending the active technique
            p_defend = 0.3 + 0.6 * prior
            # prefer defending the LIVE incoming attack (focus) -- that is what can be
            # contained this turn; and deploying a rule for an evaded technique raises
            # future detection, so reward it slightly.
            if getattr(self, "_focus_tid", None) and action.get("tech_id") == self._focus_tid:
                p_defend = min(1.0, p_defend + 0.4)
            if action.get("deploy_rule"):
                p_defend = min(1.0, p_defend + 0.1)
            return 1.0 if self.rng.random() < p_defend else 0.2


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
