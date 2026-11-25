"""Non-parametric Skill-Pro evolution, preserving official components and modes."""
from collections import Counter, defaultdict
from copy import deepcopy
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import json
import heapq
import time
from pathlib import Path
import hashlib
import numpy as np

from .upstream import (OriginalEvolution, Experience, ExperiencePool, GoldenExperiencePool,
                       build_success_failure_eval_batch, get_priority)
from .skills import structurally_valid
from .gate import gate_score
from .progress import (digest, rng_snapshot, rng_restore, BudgetExceeded, IncompleteOptimization,
                       OptimizerPreflightRequired, optimizer_batch_identity)


def semantic_chunks(experience,config):
    """The single original chunk planner, shared with the cost audit (no model calls)."""
    reuse=config.get('schedule')=='paper_batch'
    width = config.get("semantic_chunk_steps", 8)
    fields = ("step", "agent", "observation_before", "requested_action", "executed_actions", "action_status",
              "observation_after", "joint_reward", "skill_id", "skill_version", "skill_terminated",
              "response", "parse_error", "policy_invoked", "request_applied")
    records = [{k: t.get(k) for k in fields} for t in experience.transitions]
    for transition,record in zip(experience.transitions,records):
        for key in ("behavior_context","behavior_outcome","traffic_zone_interface","requested_traffic_zone_pair"):
            if key in transition:record[key]=deepcopy(transition[key])
    budget = config.get("semantic_chunk_chars", 48000 if reuse else None)
    chunks, pending, chars = [], [], 0
    for transition, record in zip(experience.transitions, records):
        size = len(json.dumps(record, ensure_ascii=False))
        if pending and (len(pending) >= width or budget is not None and chars + size > budget):
            chunks.append(pending); pending, chars = [], 0
        pending.append((transition, record)); chars += size
    if pending:
        chunks.append(pending)
    return chunks


class Evolution(OriginalEvolution):
    def __init__(self, model, config, rollout_validator=None, selection_verifier=None):
        super().__init__(model, threshold=config.get("threshold", 6), epsilon=config.get("clip", 0.2))
        self.config = config
        self.verification_identity=None
        if config.get('evidence_verification') or config.get('action_support_verification') or config.get('selection_shift_verification'):
            self.verification_identity={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ('semantic_evidence.py','diagnosis.py')}
        self.selection_verifier=selection_verifier
        self.gradients = []
        self._gradient_cache = {}
        self.fifo = ExperiencePool(max_size=config.get("experience_capacity", 1000))
        self.golden = GoldenExperiencePool(max_size=config.get("golden_capacity", 20))
        self._golden_counter=0
        self.rollout_validator=rollout_validator
        self.rollout_attempts=Counter()
        self.journal=getattr(getattr(model,'client',None),'journal',None)
        self._iteration=None
        self._loop_key=None
        self._parent=None
        self._candidate_index=None
        self._trace_hash=None

    def _generate_semantic_gradient(self, old_skill, experience):
        cache_key = digest([old_skill.__dict__, experience.__dict__])
        reuse = self.config.get("schedule") == "paper_batch"
        if reuse and cache_key in self._gradient_cache:
            return deepcopy(self._gradient_cache[cache_key])
        # Every temporal chunk is presented; never replace history with only the
        # final reward. Context overflow/error rejects evolution rather than hiding it.
        chunks=semantic_chunks(experience,self.config)
        prepared = []
        for chunk_index,chunk in enumerate(chunks):
            transitions = [t for t, _ in chunk]
            visible = [deepcopy(record) for _, record in chunk]
            if reuse and visible:
                visible[0]["execution_contract"] = {
                    "environment": "CAGE4; only the acting Blue's observations and assigned action targets are available",
                    "primitive_actions": ["Analyse", "Remove", "Restore", "DeployDecoy", "BlockTrafficZone", "AllowTrafficZone", "Sleep"],
                    "asynchronous_actions": "IN_PROGRESS and policy_invoked=false Sleep rows are automatic waiting, not failed attempts or deliberate inaction. Inspect completion before attribution.",
                    "reward": "Unmodified joint Blue reward; other KEEP defenders and Green/Red interactions contribute. Negative reward alone does not prove this skill caused failure.",
                    "generalization": "Refine observable decision rules; never memorize a training seed's hostname, IP, or subnet instance. Use exact assigned targets supplied at decision time.",
                    "action_record_format": "The acting policy's output is primitive-action JSON. Skill names are procedural guidance, not executable environment actions."}
                if any("behavior_context" in record for record in visible):
                    visible[0]["execution_contract"]["behavior_diagnostics"]=(
                        "Case history contains only this Blue's observed PID/file/connection events and own action outcomes. "
                        "Partial connection fields are uncertain investigation leads, not confirmed compromise. "
                        "Repeated analysis, duplicate successful decoys and Sleep with an open case are diagnostic evidence, not shaped rewards. "
                        "A repeated same-host decoy may add a different observed service or replace one after Restore; its host count alone does not prove waste. "
                        "Monitor runs automatically; busy Sleep has no alternative accepted intervention. "
                        "A flagged PID supports timely Remove; observed privilege escalation requires Restore because Remove cannot kill privileged persistence. "
                        "TRUE Remove is not proof of successful cleanup, and unchanged old cmd files alone are not fresh compromise. "
                        "Refine initiation, response sequence and termination using actual outcome transitions rather than copying another agent's action frequencies.")
            local = Experience(reward=experience.reward, skill=experience.skill,
                trajectory=json.dumps(visible, ensure_ascii=False), transitions=transitions, env_name=experience.env_name)
            prepared.append((chunk_index,transitions,visible,local))
        model=self.evolver_model
        reservations=[model.reserve() for _ in prepared] if hasattr(model,'reserve') else [None]*len(prepared)
        self._progress('sg_plan:'+old_skill.name+':'+digest(experience.__dict__),{
            'trace_hash':digest(experience.__dict__),'chunk_steps':[[t['step'] for t in job[1]] for job in prepared],
            'logical_reservations':reservations,'merge_order':list(range(len(prepared)))})
        def execute(job):
            (index,transitions,visible,local),reservation=job
            cache_inputs=[self._loop_key,old_skill.__dict__,experience.__dict__,index,
                self.config.get('evidence_verification'),self.config.get('diagnosis_verification')]
            if self.verification_identity is not None:cache_inputs.append(self.verification_identity)
            key='sg-result:'+digest(cache_inputs)
            cached=self.journal.restore(key) if self.journal else None
            if cached is not None:
                return cached
            callback=(lambda prompt:model.invoke(prompt,reservation)) if reservation is not None else model
            audits=[]
            if self.config.get('evidence_verification',False):
                from .semantic_evidence import EvidenceModel
                callback=EvidenceModel(callback,visible,diagnosis_verification=self.config.get('diagnosis_verification',False),
                    schema=self.config.get('evidence_schema','typed_claims_v8'))
                audits=callback.audits
            local_evolver=OriginalEvolution(callback,threshold=self.threshold,epsilon=self.epsilon)
            with self._scope(stage='sg_chunk',chunk=index,logical_counter=reservation['counter'] if reservation else None):
                gradient,related=OriginalEvolution._generate_semantic_gradient(local_evolver,old_skill,local)
            result={'skill':old_skill.name,'steps':[t['step'] for t in transitions],
                    'gradient':gradient,'related':related,**({'evidence_audits':audits} if audits else {})}
            if self.journal:self.journal.checkpoint(key,result)
            return result
        client=getattr(model,'client',None)
        workers=min(len(prepared),int(getattr(client,'config',{}).get('sg_workers',1))) if prepared else 1
        # Calls are reserved in original serial order; completion order never sets seeds or merge order.
        if workers>1 and reservations and reservations[0] is not None:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                results=list(executor.map(execute,zip(prepared,reservations)))
        else:
            results=[execute(item) for item in zip(prepared,reservations)]
        self.gradients.extend(results)
        gradients=[r['gradient'] for r in results if r['related']]
        combined = {key: "\n".join(g[key] for g in gradients if g.get(key))
                    for key in ("initiation", "execution", "termination")}
        if reuse:
            self._gradient_cache[cache_key] = (combined, bool(gradients))
        return combined, bool(gradients)

    def _run_semantic_update(self,old_skill,gradients_list,evolution_type):
        if self.config.get("evidence_verification",False) and not gradients_list:
            return {}  # No ungrounded NEW fallback after all evidence was rejected.
        if self.config.get("evidence_verification",False):
            original=self.evolver_model
            self._candidate_links=[]
            def constrained(prompt):
                raw=original(prompt+"\nOnly modify components with nonempty evidence-supported gradients. "
                    "Copy every other original initiation/policy/termination component EXACTLY, without paraphrasing. "
                    "Do not include training host instance names, IP addresses or trace citations in the reusable Skill. "
                    "Also return top-level change_links for EVERY changed initiation, policy step, and termination: "
                    "{component: initiation|execution|termination, element: null for initiation/termination or policy step index, "
                    "candidate_text: exact text of that new element, proposal_ids: [IDs from the structured verified gradients]}. "
                    "Keep those IDs only in change_links metadata, never reusable Skill text. "
                    "Links are untested hypotheses, not proof of causal benefit.")
                parsed=OriginalEvolution._parse_json_safely(self,raw) or {}
                self._candidate_links=deepcopy(parsed.get('change_links',[]))
                return raw
            local=OriginalEvolution(constrained,threshold=self.threshold,epsilon=self.epsilon)
            with self._scope(stage='aggregation_candidate',logical_stages=['aggregation','candidate_generation']):
                return OriginalEvolution._run_semantic_update(local,old_skill,gradients_list,evolution_type)
        with self._scope(stage='aggregation_candidate',logical_stages=['aggregation','candidate_generation']):
            return super()._run_semantic_update(old_skill,gradients_list,evolution_type)

    def _scope(self,**context):
        client=getattr(self.evolver_model,'client',None)
        if client is None or not hasattr(client,'context'):
            return nullcontext()
        return client.context(optimizer=True,loop=self._iteration,skill=self._parent,
                              candidate=self._candidate_index,trace_hash=self._trace_hash,**context)

    def _progress(self,key,value):
        if self.journal and self._loop_key:
            self.journal.checkpoint(self._loop_key+':'+key,value)

    def _scored(self,skill,evidence,stage):
        with self._scope(stage=stage,skill_version=skill.version):
            return self.probabilities(skill,evidence)

    def probabilities(self, skill, experiences, reference=False):
        samples = []
        for exp in experiences:
            for transition in exp.transitions:
                if transition.get("parse_error") or not transition.get("token_ids"):
                    raise ValueError("Unparseable/non-generated action is not PPO verification evidence")
                messages = json.loads(transition["state"])
                messages[0]["content"] += "\n# ACTIVE SECURITY SKILL\n" + skill.format_for_llm()
                samples.append({"messages": messages, "token_ids": transition["token_ids"]})
        values = self.evolver_model.score(samples,reference=True,use_cache=False) if reference else self.evolver_model.score(samples)
        if len(values) != len(samples) or not np.isfinite(values).all():
            raise ValueError("Missing/non-finite exact log probabilities; retain previous skill")
        return values

    def verify_post_gate(self,pool,parent,candidate,evidence,gate_evidence,iteration):
        """Optional Proposed checks run after, and never change, the original PPO score."""
        checks={}
        if self.config.get('action_support_verification',False):
            from .semantic_evidence import action_support
            checks['action_support']=action_support(parent,candidate,evidence,gate_evidence)
        if self.config.get('selection_shift_verification',False):
            if self.selection_verifier is None:raise RuntimeError('Selection verification enabled without a paired-state verifier')
            from .rollout import proposed_pool
            proposed,_=proposed_pool(pool,candidate,{'parent':parent.name,'iteration':iteration})
            checks['selection_shift']=self.selection_verifier(pool.snapshot(),proposed.snapshot())
        required=any(v.get('requires_rollout',False) for v in checks.values())
        return {'checks':checks,'requires_rollout':required,
            'decision':'requires_real_rollout' if required else 'within_measured_support',
            'action_banned':False,'ppo_gate_unchanged':True}

    def update(self,pool,experiences,baselines,iteration):
        self._iteration=iteration
        if self.journal is None:
            return self._update_impl(pool,experiences,baselines,iteration)
        if self.config.get('preflight_required'):
            label=getattr(self.evolver_model.client,'base_context',{}).get('arm','unknown')
            pending=self.journal.restore(f'pending_batch:{iteration}:{label}')
            authorization=self.journal.restore(f'preflight:{label}:{iteration}')
            if (not pending or not authorization or authorization.get('status')!='passed'
                    or authorization.get('batch_identity')!=optimizer_batch_identity(pending)):
                raise OptimizerPreflightRequired('Pending batch saved; measured optimizer cost preflight is required')
        model=self.evolver_model
        context=getattr(getattr(model,'client',None),'base_context',{})
        self._loop_key='optimizer:'+digest([context,iteration])
        inputs=[pool.snapshot(),[e.__dict__ for e in experiences],baselines,self.config]
        if self.verification_identity is not None:inputs.append(self.verification_identity)
        input_hash=digest(inputs)
        state=self.journal.restore(self._loop_key)
        if state and state['input_hash']!=input_hash:
            raise ValueError('Optimizer resume inputs, Pool, baseline or algorithm differ')
        if state is None:
            base={'pool':pool.snapshot(),'experience':self.snapshot(),
                  'inputs':[deepcopy(e.__dict__) for e in experiences],
                  'model':model.snapshot(),'rng':rng_snapshot(),'gradients':deepcopy(self.gradients)}
            state={'input_hash':input_hash,'base':base,'status':'partial','attempts':0,
                'context':context,'iteration':iteration}
        base=state['base']
        if state['status'].startswith('completed_'):
            final=state['final'];pool.restore(final['pool']);self.restore(final['experience'])
            model.restore(final['model']);rng_restore(final['rng'])
            return deepcopy(state['result'])
        # Replay deterministic CPU control flow from the same transaction boundary;
        # completed stochastic responses/scores and SG validation are durable cache hits.
        pool.restore(base['pool']);self.restore(base['experience']);model.restore(base['model'])
        self.gradients=deepcopy(base['gradients']);self._gradient_cache.clear();rng_restore(base['rng'])
        experiences=[Experience(**deepcopy(e)) for e in base['inputs']]
        state.update(status='partial',attempts=state['attempts']+1)
        self.journal.checkpoint(self._loop_key,state)
        try:
            result=self._update_impl(pool,experiences,baselines,iteration)
            unfinished=[r for r in self.journal.tasks(('unknown','failed','started','interrupted'))
                        if r['context'].get('loop')==iteration and r['context'].get('optimizer')
                        and all(r['context'].get(k)==v for k,v in context.items())]
            if unfinished or any(r.get('error') for r in result['evolution']):
                raise IncompleteOptimization('Required optimizer calls or judgments are incomplete')
            eligible=[r for r in result['evolution'] if not r.get('deferred') and r.get('valid_behavior_decisions',0)]
            changed=pool.snapshot()!=base['pool']
            status='completed_no_eligible_skill' if not eligible else 'completed_changed' if changed else 'completed_unchanged'
            result.update(optimizer_status=status,pool_changed=changed)
            final={'pool':pool.snapshot(),'experience':self.snapshot(),'model':model.snapshot(),'rng':rng_snapshot()}
            state.update(status=status,result=result,final=final)
            self.journal.checkpoint(self._loop_key,state)
            self._progress('commit',{'status':status,'pool_changed':changed,'result':result,'pool':pool.snapshot()})
            return result
        except BaseException as exc:
            state.update(status='interrupted_budget' if isinstance(exc,BudgetExceeded) else 'failed' if isinstance(exc,Exception) else 'partial',
                         error=f'{type(exc).__name__}: {exc}',
                         partial_pool=pool.snapshot(),partial_experience=self.snapshot(),model=model.snapshot(),rng=rng_snapshot())
            self.journal.checkpoint(self._loop_key,state)
            pool.restore(base['pool'])
            raise

    def _update_impl(self, pool, experiences, baselines, iteration):
        started = time.perf_counter()
        selection_base=deepcopy(pool.snapshot()) if self.config.get('selection_shift_verification') else None
        logs = []
        paper_batch = self.config.get("schedule") == "paper_batch"
        current_batch = defaultdict(list)
        for exp in experiences:
            self.fifo.add(exp)
            # The upstream heap's memory-address tie ID cannot survive restart.
            # Keep its Top-K/reward admission unchanged; use persisted insertion
            # IDs solely for equal-reward ordering. This pool is not read by the
            # paper_batch gate, parent selection or candidate tie-break.
            self._golden_counter+=1
            self.golden.add(exp)
            self.golden.pool=[(reward,self._golden_counter if item is exp else tie,item)
                              for reward,tie,item in self.golden.pool]
            heapq.heapify(self.golden.pool)
            if exp.skill and exp.skill != "None":
                self.experience_buffer[exp.skill].append(exp)
                current_batch[exp.skill].append(exp)
        if paper_batch:
            # Published Algorithm 1: visit every skill invoked in THIS batch.
            # The repository's early cooldown/priority heuristics are not part
            # of the paper and must not choose a different subset here.
            ready = [s for s in pool.skills if current_batch[s.name]]
        else:
            ready = [s for s in pool.skills if len(self.experience_buffer[s.name]) >= self.threshold]
            ready.sort(key=lambda s: get_priority(s, pool.skills, iteration), reverse=True)
            ready = ready[:self.config.get("max_evolutions", 2)]
        rollout_mode=self.config.get("rollout_verification",False)
        if rollout_mode:
            if self.rollout_validator is None:raise RuntimeError("Rollout verification enabled without a validator")
            # Proposed resource policy: calibrate the best-supported parent first.
            # No fixed minimum sample count is silently added to original Skill-Pro.
            ready.sort(key=lambda s:(self.rollout_attempts[getattr(s,'family',s.name)],
                -sum(t.get("policy_invoked",False) and t.get("request_applied",False)
                and not t.get("parse_error") for e in current_batch[s.name] for t in e.transitions)))
        order=[s.name for s in ready]
        buffers_before_candidates=deepcopy(self.experience_buffer) if selection_base is not None else None
        if self.journal:
            registered=self.journal.restore(self._loop_key+':parent_order')
            if registered is not None and registered!=order:
                raise ValueError('Registered parent order changed during resume')
        self._progress('parent_order',order)
        rollout_checks=0
        for parent in ready:
            self._parent=parent.name;self._candidate_index=-1
            if rollout_mode:
                parent=next((s for s in pool.skills if s.name==parent.name),None)
                if parent is None:continue
            before_skills, before_cache = deepcopy(pool.skills), deepcopy(pool._skill_embs)
            evidence = list(current_batch[parent.name]) if paper_batch else build_success_failure_eval_batch(self.experience_buffer[parent.name], self.threshold)
            self._gradient_cache.clear()
            gate_evidence = []
            for exp in evidence:
                valid = [t for t in exp.transitions if t.get("policy_invoked") and t.get("request_applied")
                         and not t.get("parse_error") and t.get("token_ids")]
                if valid:
                    gate_evidence.append(Experience(reward=exp.reward, skill=exp.skill, trajectory=exp.trajectory,
                        transitions=valid, env_name=exp.env_name, step_count=exp.step_count))
            record = {"iteration": iteration, "parent": parent.name, "old_skill": parent.__dict__.copy(),
                      "candidates": [], "accepted": False, "error": None,
                      "evidence_source": "current_training_batch" if paper_batch else "historical_buffer",
                      "evidence_trajectories": len(evidence),
                      "valid_behavior_decisions": sum(len(e.transitions) for e in gate_evidence)}
            from .attribution import support_statistics
            record['experience_support']=support_statistics(gate_evidence)
            required=self.config.get('minimum_independent_episodes',0)
            if record['experience_support']['independent_episode_count'] < required:
                record['deferred']='insufficient_independent_training_episodes'
                record['additional_training_episodes_needed']=required-record['experience_support']['independent_episode_count']
                record['pool_changed']=False
                logs.append(record);self._progress("parent:"+parent.name,record);continue
            self._trace_hash=digest([e.__dict__ for e in evidence])
            start_gradient = len(self.gradients)
            if rollout_mode and rollout_checks>=self.config.get("maximum_rollout_candidates_per_iteration",1):
                record["deferred"]="registered_rollout_budget_exhausted; keep previous skill including rare skills"
                logs.append(record);self._progress("parent:"+parent.name,record);continue
            try:
                gate_enabled = self.config.get("ppo_gate", True)
                # Proposed-only: reject unsupported diagnoses before spending
                # exact-scoring calls. B2's original calculation is unchanged.
                old = self._scored(parent, gate_evidence,'old_score') if gate_enabled and not self.config.get('evidence_verification',False) else None
                candidates = []
                for index in range(self.config.get("best_of_n", 3)):
                    self._candidate_index=index
                    if self.config.get("semantic_gradient", True):
                        candidate, kind = self.evolve(parent, evidence)
                    else:
                        # Genuine no-gradient path from the official ablation.
                        candidate, kind = self.evolve_without_gradient(parent, evidence)
                    item = {"index": index, "type": kind, "score": None, "accepted": False}
                    if not structurally_valid(candidate):
                        item["error"] = "invalid_candidate_structure"
                        record["candidates"].append(item)
                        self._progress('candidate:'+parent.name+':'+str(index),item)
                        continue
                    candidate.family = getattr(parent, "family", parent.name)
                    candidate.allowed_actions = list(getattr(parent, "allowed_actions", []))
                    # The name is part of format_for_llm(), so finalize it before
                    # exact scoring. Never change a verified policy on admission.
                    reserved_names = {s.name for s in pool.skills}
                    if not gate_enabled:
                        # This ablation admits every proposal, so siblings also
                        # need distinct identities before they are logged.
                        reserved_names.update(s.name for _, s, _ in candidates)
                    if candidate.name in reserved_names:
                        base = candidate.name
                        candidate.name = f"{base}_c{index}"
                        suffix = 1
                        while candidate.name in reserved_names:
                            candidate.name = f"{base}_c{index}_{suffix}"
                            suffix += 1
                    item["skill"] = candidate.__dict__.copy()
                    self._progress('candidate:'+parent.name+':'+str(index),{'status':'parsed_before_scoring',**item})
                    if self.config.get("evidence_verification",False):
                        from .semantic_evidence import candidate_supported
                        support=candidate_supported(parent,candidate,self.gradients[start_gradient:],getattr(self,'_candidate_links',[]))
                        item['semantic_support']=support
                        if not support['passed']:
                            item['deferred']='candidate_linkage_or_known_rule_not_supported'
                            record['candidates'].append(item)
                            self._progress('candidate:'+parent.name+':'+str(index),item)
                            continue
                    if gate_enabled:
                        if old is None:old=self._scored(parent,gate_evidence,'old_score')
                        new = self._scored(candidate, gate_evidence,'new_score')
                        value, diagnostics = gate_score(gate_evidence, old, new, baselines,
                            mode=self.config.get("gate_mode", "official_code"), epsilon=self.epsilon)
                        client=getattr(self.evolver_model,'client',None)
                        if getattr(client,'config',{}).get('scoring_backend','incremental_reference')!='incremental_reference':
                            error_bound=getattr(client,'scorer_certificate',{}).get('decision_error_bound',float('inf'))
                            if abs(value-self.config.get('acceptance_margin',0.0))<=error_bound:
                                with self._scope(stage='old_score_reference_boundary'):
                                    exact_old=self.probabilities(parent,gate_evidence,reference=True)
                                with self._scope(stage='new_score_reference_boundary'):
                                    exact_new=self.probabilities(candidate,gate_evidence,reference=True)
                                value,diagnostics=gate_score(gate_evidence,exact_old,exact_new,baselines,
                                    mode=self.config.get('gate_mode','official_code'),epsilon=self.epsilon)
                                item['reference_boundary_recheck']=True
                        item.update(score=value, diagnostics=diagnostics)
                        diagnostic_old=exact_old if item.get('reference_boundary_recheck') else old
                        unchanged_value=gate_score(gate_evidence,diagnostic_old,diagnostic_old,baselines,
                            mode=self.config.get('gate_mode','official_code'),epsilon=self.epsilon)[0]
                        item['surrogate_diagnostics']={'J_candidate':value,'J_old':unchanged_value,
                            'delta_J':value-unchanged_value,'admission_rule_unchanged':True,
                            'scope':'Diagnostics only; original best positive candidate rule is unchanged.'}
                    else:
                        value = 0.0
                        item["gate"] = "disabled_no_logprob_calls"
                    candidates.append((value, candidate, item))
                    record["candidates"].append(item)
                    self._progress('candidate:'+parent.name+':'+str(index),item)
                if candidates:
                    best = max(candidates, key=lambda x: x[0])
                    if not gate_enabled:
                        # Paper's no-gate ablation admits generated valid candidates.
                        selected = candidates
                    else:
                        selected = [best] if best[0] > self.config.get("acceptance_margin", 0.0) else []
                    for value, candidate, item in selected:
                        candidate.maturity = 0
                        admission=self.config.get('admission_mode','legacy_rollout_veto')
                        item['admission_mode']=admission
                        item['efficacy_verified']=False
                        if (item.get('semantic_support') or {}).get('requires_rollout') and not rollout_mode and admission!='factual_ppo':
                            item['ppo_accepted']=True;item['deferred']='candidate_semantics_require_real_rollout'
                            continue
                        if any(self.config.get(k,False) for k in ('action_support_verification','selection_shift_verification')):
                            item['ppo_accepted']=True
                            item['post_gate_verification']=self.verify_post_gate(pool,parent,candidate,evidence,gate_evidence,iteration)
                            if item['post_gate_verification']['requires_rollout'] and not rollout_mode:
                                item['deferred']='additional_real_rollout_required'
                                continue
                        if rollout_mode:
                            item["ppo_accepted"]=True;rollout_checks+=1
                            self.rollout_attempts[getattr(parent,'family',parent.name)]+=1
                            validation=self.rollout_validator(pool,candidate,{"parent":parent.name,"iteration":iteration,
                                "valid_behavior_decisions":record["valid_behavior_decisions"],"evidence_trajectories":len(evidence)})
                            item["rollout_validation"]=validation
                            if not validation.get("accepted",False):continue
                            pool.restore(validation["verified_pool"])
                            item['efficacy_verified']=True
                            item['efficacy_scope']='Registered paired validation only, not independent-test generalization.'
                            item["accepted"]=record["accepted"]=True
                            continue
                        parent.last_evolved_iter = iteration
                        pool.add_skill(candidate)
                        item["accepted"] = True
                        record["accepted"] = True
                current = self.experience_buffer[parent.name]
                self.experience_buffer[parent.name] = [] if record["accepted"] else current[-max(1, self.threshold//2):]
            except (BudgetExceeded,IncompleteOptimization):
                pool.skills,pool._skill_embs=before_skills,before_cache
                raise
            except Exception as exc:
                # Probability, context and model failures NEVER count as acceptance.
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["accepted"] = False
                for item in record["candidates"]:
                    item["accepted"] = False
                pool.skills, pool._skill_embs = before_skills, before_cache
            record["semantic_gradients"] = self.gradients[start_gradient:]
            logs.append(record)
            self._progress('parent:'+parent.name,{'record':record,'pool':pool.snapshot(),
                'experience':self.snapshot(),'model':self.evolver_model.snapshot() if hasattr(self.evolver_model,'snapshot') else None,
                'rng':rng_snapshot()})
        if self.config.get("score_maintenance", True):
            if self.config.get('maintenance_mode','official_code') in ('paper_score', 'paper_algorithm1'):
                from .paper_algorithm import maintain_paper
                maintenance=maintain_paper(pool, capacity_only=self.config.get('maintenance_mode')=='paper_algorithm1')
            else:
                maintenance = pool.maintain()  # actual official-code LCB/semantic pruning
        else:
            maintenance = pool.maintain_fifo()
        final_selection=None
        if selection_base is not None and not rollout_mode and any(r['accepted'] for r in logs):
            from .rollout import pool_digest
            final_snapshot=pool.snapshot()
            final_selection={'scope':'whole_update_after_capacity_maintenance',
                'previous_pool_sha256':pool_digest(selection_base),
                'probed_pool_sha256':pool_digest(final_snapshot),
                'probe':self.selection_verifier(deepcopy(selection_base),deepcopy(final_snapshot))}
            if final_selection['probe'].get('requires_rollout',False):
                pool.restore(selection_base);self.experience_buffer=buffers_before_candidates
                for record in logs:
                    record['accepted']=False
                    for item in record['candidates']:
                        if item.get('accepted'):
                            item.update(accepted=False,deferred='final_pool_selection_requires_real_rollout')
                final_selection['decision']='deferred_previous_pool_retained'
            else:final_selection['decision']='exact_final_pool_selection_screen_passed'
            final_selection['committed_pool_sha256']=pool_digest(pool.snapshot())
            self._progress('final_selection_verification',final_selection)
        self._progress("maintenance",{'maintenance':maintenance,'pool':pool.snapshot()})
        from .attribution import evolution_statistics
        return {"evolution": logs, "maintenance": maintenance,'final_selection_verification':final_selection,
                "skill_experience_statistics":evolution_statistics(experiences,logs),
                "seconds": time.perf_counter()-started, "pool_size": len(pool.skills)}

    def snapshot(self):
        return {"rollout_attempts":dict(self.rollout_attempts),
                "buffers": {k: [e.__dict__ for e in v] for k, v in self.experience_buffer.items()},
                "fifo": [e.__dict__ for e in self.fifo.experiences],
                "golden": [e.__dict__ for e in self.golden.get_all()],
                "golden_order":[item[1] for item in self.golden.pool],
                "golden_counter":self._golden_counter}

    def restore(self, data):
        self.rollout_attempts=Counter(data.get('rollout_attempts',{}))
        self.experience_buffer = defaultdict(list, {k: [Experience(**e) for e in v] for k, v in data.get("buffers", {}).items()})
        self.fifo.experiences=[]
        self.golden.pool=[]
        for e in data.get("fifo", []): self.fifo.add(Experience(**e))
        golden=data.get('golden',[]);order=data.get('golden_order',list(range(1,len(golden)+1)))
        if len(order)!=len(golden) or len(set(order))!=len(order):raise ValueError('Invalid stored Golden heap order')
        self.golden.pool=[(e['reward'],tie,Experience(**e)) for e,tie in zip(golden,order)]
        heapq.heapify(self.golden.pool)
        self._golden_counter=data.get('golden_counter',max(order,default=0))
