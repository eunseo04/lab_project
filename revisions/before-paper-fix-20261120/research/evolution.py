"""Non-parametric Skill-Pro evolution, preserving official components and modes."""
from collections import defaultdict
from copy import deepcopy
import json
import time
import numpy as np

from .upstream import (OriginalEvolution, Experience, ExperiencePool, GoldenExperiencePool,
                       build_success_failure_eval_batch, get_priority)
from .skills import structurally_valid
from .gate import gate_score


class Evolution(OriginalEvolution):
    def __init__(self, model, config):
        super().__init__(model, threshold=config.get("threshold", 6), epsilon=config.get("clip", 0.2))
        self.config = config
        self.gradients = []
        self.fifo = ExperiencePool(max_size=config.get("experience_capacity", 1000))
        self.golden = GoldenExperiencePool(max_size=config.get("golden_capacity", 20))

    def _generate_semantic_gradient(self, old_skill, experience):
        # Every temporal chunk is presented; never replace history with only the
        # final reward. Context overflow/error rejects evolution rather than hiding it.
        width = self.config.get("semantic_chunk_steps", 8)
        chunks = [experience.transitions[i:i+width] for i in range(0, len(experience.transitions), width)]
        gradients = []
        for chunk in chunks:
            visible = [{k: t.get(k) for k in ("step", "agent", "observation_before", "requested_action",
                "executed_actions", "action_status", "observation_after", "joint_reward", "skill_id",
                "skill_version", "skill_terminated", "response", "parse_error", "policy_invoked", "request_applied")} for t in chunk]
            local = Experience(reward=experience.reward, skill=experience.skill,
                trajectory=json.dumps(visible, ensure_ascii=False), transitions=chunk, env_name=experience.env_name)
            gradient, related = super()._generate_semantic_gradient(old_skill, local)
            self.gradients.append({"skill": old_skill.name, "steps": [t["step"] for t in chunk],
                                   "gradient": gradient, "related": related})
            if related:
                gradients.append(gradient)
        combined = {key: "\n".join(g[key] for g in gradients if g.get(key))
                    for key in ("initiation", "execution", "termination")}
        return combined, bool(gradients)

    def probabilities(self, skill, experiences):
        samples = []
        for exp in experiences:
            for transition in exp.transitions:
                if transition.get("parse_error") or not transition.get("token_ids"):
                    raise ValueError("Unparseable/non-generated action is not PPO verification evidence")
                messages = json.loads(transition["state"])
                messages[0]["content"] += "\n# ACTIVE SECURITY SKILL\n" + skill.format_for_llm()
                samples.append({"messages": messages, "token_ids": transition["token_ids"]})
        values = self.evolver_model.score(samples)
        if len(values) != len(samples) or not np.isfinite(values).all():
            raise ValueError("Missing/non-finite exact log probabilities; retain previous skill")
        return values

    def update(self, pool, experiences, baselines, iteration):
        started = time.perf_counter()
        logs = []
        for exp in experiences:
            self.fifo.add(exp)
            self.golden.add(exp)
            if exp.skill and exp.skill != "None":
                self.experience_buffer[exp.skill].append(exp)
        ready = [s for s in pool.skills if len(self.experience_buffer[s.name]) >= self.threshold]
        ready.sort(key=lambda s: get_priority(s, pool.skills, iteration), reverse=True)
        for parent in ready[:self.config.get("max_evolutions", 2)]:
            before_skills, before_cache = deepcopy(pool.skills), deepcopy(pool._skill_embs)
            evidence = build_success_failure_eval_batch(self.experience_buffer[parent.name], self.threshold)
            gate_evidence = []
            for exp in evidence:
                valid = [t for t in exp.transitions if t.get("policy_invoked") and t.get("request_applied")
                         and not t.get("parse_error") and t.get("token_ids")]
                if valid:
                    gate_evidence.append(Experience(reward=exp.reward, skill=exp.skill, trajectory=exp.trajectory,
                        transitions=valid, env_name=exp.env_name, step_count=exp.step_count))
            record = {"iteration": iteration, "parent": parent.name, "old_skill": parent.__dict__.copy(),
                      "candidates": [], "accepted": False, "error": None}
            start_gradient = len(self.gradients)
            try:
                gate_enabled = self.config.get("ppo_gate", True)
                old = self.probabilities(parent, gate_evidence) if gate_enabled else None
                candidates = []
                for index in range(self.config.get("best_of_n", 3)):
                    if self.config.get("semantic_gradient", True):
                        candidate, kind = self.evolve(parent, evidence)
                    else:
                        # Genuine no-gradient path from the official ablation.
                        candidate, kind = self.evolve_without_gradient(parent, evidence)
                    item = {"index": index, "type": kind, "score": None, "accepted": False}
                    if not structurally_valid(candidate):
                        item["error"] = "invalid_candidate_structure"
                        record["candidates"].append(item)
                        continue
                    candidate.family = getattr(parent, "family", parent.name)
                    candidate.allowed_actions = list(getattr(parent, "allowed_actions", []))
                    item["skill"] = candidate.__dict__.copy()
                    if gate_enabled:
                        new = self.probabilities(candidate, gate_evidence)
                        value, diagnostics = gate_score(gate_evidence, old, new, baselines,
                            mode=self.config.get("gate_mode", "official_code"), epsilon=self.epsilon)
                        item.update(score=value, diagnostics=diagnostics)
                    else:
                        value = 0.0
                        item["gate"] = "disabled_no_logprob_calls"
                    candidates.append((value, candidate, item))
                    record["candidates"].append(item)
                if candidates:
                    best = max(candidates, key=lambda x: x[0])
                    if not gate_enabled:
                        # Paper's no-gate ablation admits generated valid candidates.
                        selected = candidates
                    else:
                        selected = [best] if best[0] > self.config.get("acceptance_margin", 0.0) else []
                    for value, candidate, item in selected:
                        if any(s.name == candidate.name for s in pool.skills):
                            candidate.name += f"_c{item['index']}"
                        candidate.maturity = 0
                        parent.last_evolved_iter = iteration
                        pool.add_skill(candidate)
                        item["accepted"] = True
                        record["accepted"] = True
                current = self.experience_buffer[parent.name]
                self.experience_buffer[parent.name] = [] if record["accepted"] else current[-max(1, self.threshold//2):]
            except Exception as exc:
                # Probability, context and model failures NEVER count as acceptance.
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["accepted"] = False
                for item in record["candidates"]:
                    item["accepted"] = False
                pool.skills, pool._skill_embs = before_skills, before_cache
            record["semantic_gradients"] = self.gradients[start_gradient:]
            logs.append(record)
        if self.config.get("score_maintenance", True):
            if self.config.get('maintenance_mode','official_code')=='paper_score':
                from .paper_algorithm import maintain_paper
                maintenance=maintain_paper(pool)
            else:
                maintenance = pool.maintain()  # actual official-code LCB/semantic pruning
        else:
            maintenance = pool.maintain_fifo()
        return {"evolution": logs, "maintenance": maintenance,
                "seconds": time.perf_counter()-started, "pool_size": len(pool.skills)}

    def snapshot(self):
        return {"buffers": {k: [e.__dict__ for e in v] for k, v in self.experience_buffer.items()},
                "fifo": [e.__dict__ for e in self.fifo.experiences],
                "golden": [e.__dict__ for e in self.golden.get_all()]}

    def restore(self, data):
        self.experience_buffer = defaultdict(list, {k: [Experience(**e) for e in v] for k, v in data.get("buffers", {}).items()})
        for e in data.get("fifo", []): self.fifo.add(Experience(**e))
        for e in data.get("golden", []): self.golden.add(Experience(**e))
