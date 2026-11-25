"""CAGE4 skill seeds and per-agent option state, using unchanged Skill-Pro types."""
from dataclasses import asdict
import hashlib
import json
import re
import random
from .upstream import Skill, OriginalSkillPool

SEEDS = [
    ("HostInvestigation", "When an assigned host has suspicious process/connection events that need investigation.",
     ["Identify an assigned host with observable suspicious events.", "Analyse that host.", "Wait for completion and inspect the actual result before escalation."],
     "Analysis has completed and its observable findings have been inspected, or no relevant host remains.", ["Analyse"]),
    ("ThreatRemoval", "When observable events support a user-level compromise or a malicious process on an assigned host.",
     ["Check the affected host and previous analysis/monitor evidence.", "Remove malicious processes on that host.", "Verify the returned status; re-analyse if evidence is inconclusive."],
     "Removal completed and subsequent observations no longer support the same malicious process, or removal failed and further analysis is needed.", ["Remove", "Analyse"]),
    ("HostRecovery", "When an assigned host has observable admin-level compromise and removal is insufficient.",
     ["Confirm observable admin-compromise evidence on the specific host.", "Restore the host, considering its availability cost.", "Wait for the restore action and check the observed result."],
     "Restore has completed with observable confirmation, or the action has failed.", ["Restore", "Analyse"]),
    ("DeceptionDeployment", "When the defender is available and a legitimate assigned host can host a useful decoy.",
     ["Choose a valid assigned host without repeating a failed deployment blindly.", "DeployDecoy on the host.", "Wait for completion and examine status and subsequent monitor events."],
     "Deployment has completed or failed and its result has been checked.", ["DeployDecoy", "Analyse"]),
    ("NetworkContainment", "When observable repeated malicious connections or communicated threat justify limiting a zone.",
     ["Check the visible mission phase and relevant communication vectors.", "Choose a valid subnet for BlockTrafficZone without gratuitous availability loss.", "Observe success/failure and reassess the containment need."],
     "The block action result is known and the immediate containment decision has been reassessed.", ["BlockTrafficZone"]),
    ("TrafficRecovery", "When the defender has observable evidence of its previous traffic block and can reassess its need.",
     ["Identify the previously blocked subnet from observed action history.", "Check current phase and available threat evidence.", "AllowTrafficZone when recovery is justified and check the actual status."],
     "Traffic restoration has completed or failed and its result has been inspected.", ["AllowTrafficZone"]),
    ("WaitVerification", "When an action is still in progress or an already initiated response must be checked before another intervention.",
     ["Inspect Last Action Status and the busy bit.", "Sleep while the previous action remains in progress.", "Read the completed action status and automatic Monitor observations before continuing."],
     "The outstanding action has completed and its observable outcome is available.", ["Sleep"]),
]


class SecuritySkillPool(OriginalSkillPool):
    def __init__(self, max_size=10, seed_set="legacy"):
        if seed_set not in ("legacy","response_v1"):raise ValueError("Unknown skill seed set")
        self.seed_set=seed_set
        super().__init__(max_size=max_size)

    def _initialize_seeds(self):
        from .behavior_skills import RESPONSE_SEEDS
        seeds=SEEDS if self.seed_set=="legacy" else RESPONSE_SEEDS
        for name, initiation, policy, termination, actions in seeds:
            skill = Skill(name=name, initiation=initiation, policy=policy, termination=termination, version=1)
            skill.allowed_actions = list(actions)
            skill.family = name
            self.skills.append(skill)
            self._update_emb_cache(skill)

    def snapshot(self):
        data={"max_size": self.max_size, "skills": [{**asdict(s),
            "allowed_actions": getattr(s, "allowed_actions", []),
            "family": getattr(s, "family", re.sub(r"_v\d+$", "", s.name))} for s in self.skills]}
        if self.seed_set!="legacy":data["seed_set"]=self.seed_set
        return data

    def restore(self, data):
        fields = set(Skill.__dataclass_fields__)
        self.seed_set=data.get("seed_set","legacy")
        if self.seed_set not in ("legacy","response_v1"):raise ValueError("Unknown stored skill seed set")
        self.skills, self._skill_embs = [], {}
        self.max_size = data["max_size"]
        for row in data["skills"]:
            skill = Skill(**{k: v for k, v in row.items() if k in fields})
            skill.allowed_actions = row.get("allowed_actions", [])
            skill.family = row.get("family", re.sub(r"_v\d+$", "", skill.name))
            self.skills.append(skill)
            self._update_emb_cache(skill)


def structurally_valid(skill):
    return (isinstance(skill, Skill) and bool(skill.name) and skill.name != "None"
        and isinstance(skill.initiation, str) and bool(skill.initiation.strip())
        and isinstance(skill.termination, str) and bool(skill.termination.strip())
        and isinstance(skill.policy, list) and 1 <= len(skill.policy) <= 30
        and all(isinstance(s, str) and s.strip() for s in skill.policy))


class CAGEVerifier:
    """E3 only: checks are based exclusively on this Blue's observation/history."""
    def __init__(self, history_mode="legacy_ticks", revision="v4"):
        if history_mode not in ("legacy_ticks", "accepted_decisions"):
            raise ValueError("Invalid verification history mode")
        self.history_mode = history_mode
        if revision not in ("v4","v5"):raise ValueError("Invalid domain verification revision")
        self.revision=revision

    def check(self, skill, state, history):
        if not structurally_valid(skill):
            return False, "invalid_skill_structure"
        raw = state["observation"]
        if raw.get("phase") not in (0, 1, 2):
            return False, "unknown_mission_phase"
        for message in raw.get("message", []):
            if len(message) != 8 or any(x not in (0, 1, False, True) for x in message):
                return False, "invalid_communication_vector"
        status = str(raw.get("success", "UNKNOWN"))
        if "IN_PROGRESS" in status:
            return False, "action_in_progress"
        decisions = history
        if self.history_mode == "accepted_decisions":
            # Retain deliberate free Sleep: it breaks a consecutive intervention
            # sequence. Only automatic busy rows and invalid requests are skipped.
            decisions = [h for h in history if h.get("policy_invoked")
                and h.get("request_applied") and not h.get("parse_error")]
        if len(decisions) >= 2:
            last, previous = decisions[-1], decisions[-2]
            evidence = {str(k): v for k, v in raw.items() if k not in ("action", "phase", "message")}
            current_hash = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
            # Busy placeholders and deliberate waiting are not repeated policy
            # interventions. Check the CURRENT observation, not just old hashes.
            real_decisions = all(h.get("policy_invoked") and h.get("request_applied")
                and not h.get("parse_error") and h.get("requested_action") != "Sleep"
                for h in (last, previous))
            if (real_decisions and last["requested_action"] == previous["requested_action"]
                    and last["evidence_hash"] == previous["evidence_hash"]
                    and last["evidence_hash"] == current_hash
                    and previous.get("skill_id") == skill.name
                    and last.get("skill_id") == skill.name):
                return False, "repeated_action_without_new_evidence"
        family = getattr(skill, "family", skill.name)
        response_profile=state.get("behavior_seed_set")=="response_v1"
        if response_profile:
            from .behavior_skills import initiation_matches
            if not initiation_matches(family,state["behavior_context"],revision=self.revision):
                return False,"observable_case_initiation_not_satisfied"
        text = state["user_prompt"].lower()
        if not response_profile and family == "HostRecovery" and "critical:" not in text:
            return False, "no_observed_admin_compromise"
        if not response_profile and family == "ThreatRemoval" and not any(s in text for s in ("warning:", "alert:", "critical:")):
            return False, "no_observed_malicious_process_evidence"
        if not response_profile and family == "TrafficRecovery" and not any("BlockTraffic" in (h.get("requested_action") or "") for h in history):
            return False, "no_observed_previous_block"
        if not response_profile and family == "NetworkContainment":
            alerts = any(s in text for s in ("warning:", "alert:", "critical:"))
            peer_threat = any(any(m[:7]) for m in raw.get("message", []))
            if not alerts and not peer_threat:
                return False, "no_observed_containment_trigger"
        if "FALSE" in status and history and history[-1].get("skill_id") == skill.name:
            return False, "previous_action_failed_reassess_skill"
        if not set(getattr(skill, "allowed_actions", [])) <= set(state["allowed_actions"]):
            return False, "skill_requires_unavailable_action"
        return True, "observable_checks_passed"

class SkillController:
    def __init__(self, pool, text_model, config, proposed=False):
        self.pool, self.model, self.config = pool, text_model, config
        self.active, self.started = None, None
        self.history = []
        self.proposed = proposed
        self.verifier = CAGEVerifier(config.get("verification_history", "legacy_ticks"),config.get("domain_verification_revision","v4"))
        self.last_event = {}

    def choose(self, state, step):
        events = []
        selection_audit=None
        if self.active is not None:
            if step - self.started >= self.config.get("max_skill_age", 8):
                events.append({"terminated": self.active.name, "reason": "maximum_age"})
                self.active = None
            elif self.proposed:
                valid, reason = self.verifier.check(self.active, state, self.history)
                if not valid:
                    events.append({"terminated": self.active.name, "reason": reason})
                    self.active = None
            if self.active is not None:
                prompt = f"""[ROLE]
You are a Meta-Controller supervising an AI Agent.
[CURRENT STATE]
{state['user_prompt']}
[ACTIVE OPTION]
Name: {self.active.name}
Initiation: {self.active.initiation}
Policy: {self.active.policy}
Termination: {self.active.termination}
Decide whether the skill should STOP now because termination is achieved or initiation is no longer satisfied.
Output EXACTLY <status>DONE</status> or <status>CONTINUE</status>."""
                response = self.model(prompt, purpose="termination")
                match = re.search(r"<status>\s*(DONE|CONTINUE)\s*</status>", response, re.I)
                if match and match[1].upper() == "DONE":
                    events.append({"terminated": self.active.name, "reason": "model_termination"})
                    self.active = None
                elif match is None:
                    events.append({"error": "invalid_termination_response", "raw": response})
        if self.active is None:
            candidates = list(self.pool.skills)
            calls=[];before_rng=random.getstate()
            def observed_selector(prompt):
                response=self.model(prompt)
                calls.append({'prompt':prompt,'response':response})
                return response
            try:
                if self.proposed:
                    self.pool.skills = [s for s in candidates if self.verifier.check(s, state, self.history)[0]]
                offered=[{'name':s.name,'initiation':s.initiation} for s in self.pool.skills]
                self.active = self.pool.select_skill(state["user_prompt"], llm_agent=observed_selector,
                    select_type="llm_model", epsilon=self.config.get("epsilon", 0.1))
                selection_audit={'source':'llm' if calls else 'epsilon_exploration',
                    'state_prompt':state['user_prompt'],'offered_skills':offered,
                    'model_calls':calls,'selected':self.active.name if self.active else None,
                    'python_rng_before':before_rng,'python_rng_after':random.getstate(),
                    'input_unchanged':True}
            finally:
                self.pool.skills = candidates
            if self.active is not None:
                self.started = step
                events.append({"activated": self.active.name, "version": self.active.version})
        self.last_event = {"events": events, "active": self.active.name if self.active else None,
                           "started": self.started, "age": step-self.started if self.active else 0,
                           "selection_audit":selection_audit}
        return self.active


def probe_selection(previous, proposed, records, client, rule=None, data_role='training'):
    """Run the unchanged upstream selector on paired recorded states, never infer reward.

    An independent TextModel reservation and restored Python RNG keep the live
    optimizer counter/RNG untouched. The primitive actor's appended action contract
    is removed only to recover the exact earlier selector state.
    """
    from contextlib import nullcontext
    from .client import TextModel,stable_seed
    from .progress import digest,rng_snapshot,rng_restore
    from .attribution import selection_shift
    from copy import deepcopy
    if data_role not in ('training','validation','historical_training'):
        raise ValueError('Final-test/development states cannot decide Skill admission')
    rule=rule or {};forbidden=set(rule.get('forbidden_seeds',[]))
    choices=[];seen=set()
    for row in records:
        if row.get('seed') in forbidden:raise ValueError('Final-test state leaked into selector verification')
        if rule.get('allowed_seeds') is not None and row.get('seed') not in rule['allowed_seeds']:
            raise ValueError('Selector state is outside registered training/validation seeds')
        if not row.get('policy_invoked') or not row.get('user_prompt'):continue
        prompt=row['user_prompt'].split('\n\n# EXECUTABLE ACTION CONTRACT\n',1)[0]
        state_id=digest([row.get('seed'),row.get('agent'),row.get('step'),prompt])
        if state_id not in seen:choices.append((row,prompt,state_id));seen.add(state_id)
    maximum=rule.get('maximum_states',24)
    if type(maximum) is not int or maximum<1:raise ValueError('Positive selector probe state count required')
    if not choices:raise ValueError('No recorded selector states')
    indices=sorted({i*len(choices)//min(maximum,len(choices)) for i in range(min(maximum,len(choices)))})
    saved=rng_snapshot();text=TextModel(client)
    try:
        pools={}
        for name,snapshot in (('previous',previous),('candidate',proposed)):
            pool=SecuritySkillPool(snapshot['max_size'],snapshot.get('seed_set','legacy'));pool.restore(snapshot);pools[name]=pool
        if rule.get('include_controls',False):
            old={s.name:s for s in pools['previous'].skills};clone=deepcopy(pools['candidate'])
            for s in clone.skills:
                if s.name not in old and getattr(s,'parent_id',None) in old:s.initiation=old[s.parent_id].initiation
            pools['new_entry_old_initiation']=clone
            reordered=deepcopy(pools['candidate']);reordered.skills=list(reversed(reordered.skills));pools['candidate_reversed_order']=reordered
        results={name:[] for name in pools}
        for index in indices:
            row,prompt,state_id=choices[index];seed=stable_seed(row.get('seed'),row.get('step'),row.get('agent'),'selector-probe-v1')
            for name,pool in pools.items():
                random.seed(seed);calls=[]
                reservation={'seed':seed,'counter':1,'purpose':'skill_selection','max_tokens':48}
                def ask(message):
                    answer=text.invoke(message,reservation);calls.append({'prompt':message,'response':answer});return answer
                scope=client.context(stage='selector_probe',probe_role=name,state_id=state_id,logical_sampling_seed=seed) if hasattr(client,'context') else nullcontext()
                with scope:selected=pool.select_skill(prompt,llm_agent=ask,select_type='llm_model',epsilon=rule.get('epsilon',.1))
                results[name].append({'state_id':state_id,'sampling_seed':seed,'record_seed':row.get('seed'),
                    'step':row.get('step'),'choice':selected.name if selected else None,
                    'selector_source':'llm' if calls else 'epsilon_exploration','model_calls':calls})
    finally:rng_restore(saved)
    result=selection_shift(results['previous'],results['candidate'],rule)
    result.update(probes=results,data_role=data_role,live_rng_restored=True,
        previous_pool_sha256=digest(previous),candidate_pool_sha256=digest(proposed),
        controls_note='Entry name/order/count and initiation affect selection; policy/termination text are absent from the upstream selection prompt.')
    return result
