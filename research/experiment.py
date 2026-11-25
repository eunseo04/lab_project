"""Resumable, paired CC4 experiments; only completed real episodes become results."""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import time

from acd.runner import seed_everything, load_agents
from CybORG.Simulator.Actions import Sleep
from .adapter import (BLUE_AGENTS, DefensePolicy, make_state, make_environment,
                      executed_actions, plain)
from .client import ModelClient, TextModel
from .config import digest, compatible_digest, split_seeds
from .metrics import JointRewardLedger, aggregate_episodes
from .skills import SecuritySkillPool
from .upstream import configure_encoder, Experience
from .evolution import Evolution
from .gate import returns_to_go
from . import ROOT
from .progress import atomic_json, digest as progress_digest, rng_snapshot, rng_restore, BudgetExceeded, implementation_identity


def policy_roles(composition):
    if composition == "no_blue": return {a: "sleep" for a in BLUE_AGENTS}
    if composition == "all_rl": return {a: "rl" for a in BLUE_AGENTS}
    return {a: "llm" if composition == "all_llm" or a == "blue_agent_4" else "rl" for a in BLUE_AGENTS}


def experiences_from_rows(rows, config, total, steps):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("policy") == "llm" and row.get("skill_id"):
            grouped[(row["agent"],row["skill_id"])].append(row)
    return [Experience(reward=total,skill=name,trajectory=json.dumps(transitions,ensure_ascii=False),
        transitions=transitions,env_name=f"{config['composition']}:{config['red']}",step_count=steps)
        for (agent,name),transitions in sorted(grouped.items(),key=lambda item:BLUE_AGENTS.index(item[0][0]))]


def recover_episode(result, config, client, text_model, directory):
    expected = "mock_test" if getattr(client,"is_mock",False) else "real_cage4"
    if result.get("kind") != expected: raise ValueError("Cannot mix mock and real training records")
    rows=[json.loads(line) for line in Path(result["trace"]).open()]
    if result.get('trace_sha256') and hashlib.sha256(Path(result['trace']).read_bytes()).hexdigest()!=result['trace_sha256']:
        raise ValueError('Completed training trace hash changed')
    if len(rows) != len(BLUE_AGENTS)*result["joint_transitions"] or any(r["seed"] != result["seed"] for r in rows):
        raise ValueError("Incomplete or misaligned archived training trace")
    seed_everything(result["seed"],config["threads"])
    state=result.get("text_model_state",{})
    text_model.seed=state.get("seed",result["seed"])
    text_model.counter=state.get("counter",sum(c.get("purpose") in ("skill_selection","termination") for c in result.get("calls",[])))
    if result.get("python_random_state"):
        s=result["python_random_state"]
        random.setstate((s[0],tuple(s[1]),s[2]))
    atomic_json(directory/"interrupted"/f"recovered-episode-{result['episode_id']:04d}.json",
        {"source":"completed_episode_and_full_trace","episode_id":result["episode_id"],
         "behavior_samples_unchanged":True,"legacy_evolution_rng_reseeded":not bool(result.get("python_random_state")),
         "text_model_counter":text_model.counter})
    return experiences_from_rows(rows,config,result["joint_return"],result["joint_transitions"])


def run_episode(config, seed, episode_id, client, pool, text_model, directory):
    started = time.perf_counter()
    journal=getattr(client,'journal',None)
    episode_task=None
    if journal:
        role=getattr(client,'base_context',{}).get('env_stage',config.get('evaluation_role','collection'))
        episode_task='episode:'+progress_digest([str(directory.resolve()),seed,episode_id,
            pool.snapshot() if pool else None,getattr(client,'identity',None)])
        journal.start(episode_task,role,{'seed':seed,'episode_id':episode_id,
            'arm':config.get('comparison_id',config['arm']),'source':'run_episode','directory':str(directory.resolve())},
            {'config_digest':digest(config),'pool':pool.snapshot() if pool else None})
    seed_everything(seed, config["threads"])
    text_model.seed, text_model.counter = seed, 0
    env = make_environment(config, seed)
    observations, _ = env.reset(seed=seed)
    reward_audit=None
    if config.get("diagnostics",{}).get("reward_events",False):
        from .reward_audit import RewardAudit
        reward_audit=RewardAudit(env,directory/f"episode-{episode_id:04d}-reward-events.jsonl")
    role_map = policy_roles(config["composition"])
    old_config = {**config, "team": "mixed" if config["composition"] == "hybrid" else
        ("all_rl" if config["composition"] == "all_rl" else "all_llm"),
        "communication": "keep", "deterministic_rl": False}
    rl_agents, weights = load_agents(old_config, 192)
    policies = {a: DefensePolicy(a, client, text_model, config, pool if config["arm"] != "E0" else None)
                for a, role in role_map.items() if role == "llm"}
    ledger = JointRewardLedger()
    rows, step_rewards = [], []
    requested, executed, automatic = Counter(), Counter(), Counter()
    previous_rows = {a: [] for a in BLUE_AGENTS}
    progress = directory / f"episode-{episode_id:04d}.partial.jsonl"
    if progress.exists():
        archive = directory/"interrupted"
        archive.mkdir(exist_ok=True)
        os.replace(progress, archive/f"{progress.stem}.{time.time_ns()}.jsonl")
    with progress.open("w") as partial:
        for step in range(config["steps"]):
            if hasattr(client,'_check_budget'):client._check_budget()
            raw = env.raw_observations()
            actions, details = {}, {}
            outgoing = env.paper_messages() if config["communication"] == "paper_projected_keep" else deepcopy(env.msg)
            for agent, role in role_map.items():
                if role == "llm":
                    state = make_state(env, agent, raw[agent], policies[agent].base)
                    state = policies[agent].prepare_state(state,step)
                    busy = "IN_PROGRESS" in str(raw[agent].get("success"))
                    if busy and config["skip_busy_policy"]:
                        active = policies[agent].controller.active if policies[agent].controller else None
                        actions[agent], details[agent] = Sleep(), {"policy_invoked": False,
                            "skill_id": active.name if active else None, "skill_version": active.version if active else None}
                    else:
                        policy_seed = int(hashlib.sha256(f"{seed}:{step}:{agent}:policy".encode()).hexdigest()[:8], 16)
                        from contextlib import nullcontext
                        scope=client.context(seed=seed,episode_id=episode_id,step=step,agent=agent) if hasattr(client,'context') else nullcontext()
                        with scope:
                            actions[agent],details[agent]=policies[agent].decide(state,episode=episode_id,step=step,seed=policy_seed)
                        for event in (details[agent].get("skill_controller") or {}).get("events", []):
                            if event.get("terminated"):
                                for row in reversed(previous_rows[agent]):
                                    if row.get("skill_id") == event["terminated"]:
                                        row["skill_terminated"] = True
                                        break
                elif role == "rl":
                    actions[agent], details[agent] = rl_agents[agent].get_action(observations[agent]), {}
                else:
                    actions[agent], details[agent] = Sleep(), {}
            observations, rewards, terminated, truncated, _ = env.step(actions, messages=outgoing)
            terminal = env.env.environment_controller.determine_done()
            joint, included = ledger.add(rewards, terminal)
            if config["scoring"] == "terminal_inclusive" and terminal:
                ledger.rewards.append(joint); included = True
            step_rewards.append(joint if included else 0.0)
            if reward_audit:reward_audit.capture(step,joint,included)
            after = env.raw_observations()
            for agent, role in role_map.items():
                action = env.action_translator(agent, actions[agent])
                chosen = str(action) if actions[agent] is not None else None
                actual = executed_actions(env, agent)
                if chosen is not None: requested[type(action).__name__] += 1
                for item in actual: executed[item["name"]] += 1
                automatic["Monitor"] += 1
                evidence = {str(k): v for k, v in plain(raw[agent]).items() if k not in ("action", "phase", "message")}
                row = {"type": "action", "episode_id": episode_id, "step": step, "seed": seed,
                    "agent": agent, "policy": role, "mission_phase": raw[agent]["phase"],
                    "comm_vector": plain(raw[agent].get("message", [])),
                    "observation_before": plain(raw[agent]), "observation_after": plain(after[agent]),
                    "requested_action": chosen, "executed_actions": plain(actual),
                    "request_applied": "IN_PROGRESS" not in str(raw[agent].get("success")),
                    "action_status": str(after[agent].get("success", "UNKNOWN")),
                    "joint_reward": joint, "score_included": included, "terminal": terminal,
                    "skill_terminated": False, "automatic_monitor": True,
                    "evidence_hash": hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest(), **details[agent]}
                if role=="llm":policies[agent].observe_transition(row)
                rows.append(row); previous_rows[agent].append(row)
                if role == "llm" and policies[agent].controller:
                    policies[agent].controller.history.append(row)
                partial.write(json.dumps(row, ensure_ascii=False)+"\n")
            if step % 25 == 0:
                partial.flush()
                print(f"{config['arm']} {config['red']} seed={seed} step={step+1}/{config['steps']-1} R={ledger.total:.1f}", flush=True)
            if terminal: break
    if len(step_rewards) != config["steps"]-1:
        raise RuntimeError("Native CAGE4 episode terminated at an unexpected step")
    for agent, policy in policies.items():
        if policy.controller and policy.controller.active:
            for row in reversed(previous_rows[agent]):
                if row.get("skill_id") == policy.controller.active.name:
                    row["skill_terminated"] = True
                    break
    rtgs = returns_to_go(step_rewards, config["evolution"]["gamma"])
    for row in rows: row["return_to_go"] = rtgs[row["step"]]
    trace = directory/f"episode-{episode_id:04d}.jsonl"
    if trace.exists():
        archive=directory/'interrupted';archive.mkdir(exist_ok=True)
        os.replace(trace,archive/f'{trace.stem}.uncommitted-{time.time_ns()}.jsonl')
    with trace.open("w") as stream:
        for row in rows: stream.write(json.dumps(row, ensure_ascii=False)+"\n")
        stream.flush();os.fsync(stream.fileno())
    progress.unlink()
    experiences = experiences_from_rows(rows,config,ledger.total,len(step_rewards))
    result = {"status": "completed", "episode_id": episode_id, "seed": seed,
        "steps_requested": config["steps"], "joint_transitions": len(step_rewards), "scored_transitions": len(ledger.rewards),
        "joint_return": ledger.total, "raw_joint_return": sum(ledger.raw_rewards), "execution_time_s": time.perf_counter()-started,
        "raw_joint_rewards": ledger.raw_rewards, "scored_step_rewards": step_rewards,
        "text_model_state":{"seed":text_model.seed,"counter":text_model.counter},
        "python_random_state":random.getstate(),
        "requested_action_counts": dict(requested), "executed_action_counts": dict(executed),
        "automatic_action_counts": dict(automatic), "invalid_json": sum(bool(r.get("parse_error")) for r in rows),
        "model_identity": client.identity if client else None,
        "kind": "mock_test" if getattr(client, "is_mock", False) else "real_cage4",
        "weights": weights, "trace": str(trace),'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),
        "calls": deepcopy(client.calls) if client else []}
    if reward_audit:result.update(reward_audit.finish(ledger.total))
    if config.get('diagnostics',{}).get('skill_attribution',False):
        from .attribution import auxiliary_attribution
        diagnostic=directory/f'episode-{episode_id:04d}-skill-attribution.json'
        atomic_json(diagnostic,auxiliary_attribution(rows,reward_audit.ticks if reward_audit else []))
        result['auxiliary_attribution']=str(diagnostic)
    atomic_json(directory/f"episode-{episode_id:04d}.json", result)
    if journal:journal.finish(episode_task,result,{'execution_time_s':result['execution_time_s'],
        'joint_transitions':result['joint_transitions'],'trace_sha256':result['trace_sha256']})
    return result, experiences


def prepare_batch(config,pending_results,pending_experiences,pool,baselines):
    task = f"{config['composition']}:{config['red']}"
    average = sum(r["joint_return"] for r in pending_results)/len(pending_results)
    previous_baseline = baselines.get(task, 0.0)
    paper_mode = config["evolution"].get("schedule") == "paper_batch"
    if paper_mode:
        # Both running baselines are estimated once from training's
        # complete joint series, including the first batch. Starting
        # the step baseline at zero would label almost every skill
        # harmful in an environment whose rewards are penalties.
        from .paper_algorithm import update_running_baselines
        update_running_baselines(pending_results, baselines, task, config["evolution"])
    if pool:
        by_name = {s.name: s for s in pool.skills}
        def invocation_count(exp):
            return sum(1 for t in exp.transitions for event in (t.get("skill_controller") or {}).get("events", [])
                       if event.get("activated") == exp.skill)
        def trajectory_key(exp):
            first = exp.transitions[0] if exp.transitions else {}
            return first.get("seed"), first.get("agent")
        total_calls = Counter()
        for exp in pending_experiences:
            total_calls[trajectory_key(exp)] += invocation_count(exp)
        if paper_mode or config['evolution'].get('maintenance_mode','official_code')=='paper_score':
            from .paper_algorithm import update_online_scores
            update_online_scores(pool,pending_experiences,baselines.get(task+':step',0.0))
        else:
            for exp in pending_experiences:
                if exp.skill in by_name:
                    by_name[exp.skill].update_stats(exp.reward, previous_baseline,
                        max(total_calls[trajectory_key(exp)], 1), invocation_count(exp))
        pool.update_pool_status()
    alpha = config["evolution"]["ema_alpha"]
    if paper_mode:
        # Updated above, before online score attribution.
        pass
    elif config['evolution']['gate_mode']=='paper_rtg':
        from .paper_algorithm import update_running_baselines
        update_running_baselines(pending_results, baselines, task, config['evolution'])
    else:
        baselines[task] = average if task not in baselines else (1-alpha)*previous_baseline+alpha*average
    return task,average


def run(config, stage="eval", output=None, resume=False, limit=None, client=None):
    if stage not in ("train","eval"): raise ValueError("Stage must be train or eval")
    if stage=='eval' and config.get('evaluation_disabled'):raise ValueError('No evaluation seeds registered for this probe-only plan')
    if stage == "train" and (config["arm"] not in ("E2","E3") or config["composition"] not in ("hybrid","all_llm")):
        raise ValueError("Skill learning requires E2/E3 with hybrid/all_llm defenders")
    output = Path(output or Path(config["output"])/f"{config['arm']}-{config['composition']}-{config['red']}"/stage)
    output.mkdir(parents=True, exist_ok=True)
    lock_stream = (output/".lock").open("w")
    try: fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError("Another process is writing this experiment")
    try:
        identity_path = output/"manifest.json"
        fingerprint = digest(config)
        implementation=implementation_identity(ROOT)
        if identity_path.exists():
            previous = json.loads(identity_path.read_text())
            if not resume: raise FileExistsError(f"Existing experiment {output}; use --resume")
            if previous["config_digest"] != fingerprint and compatible_digest(previous["config"]) != compatible_digest(config):
                raise ValueError("Resume config does not match this experiment")
            recorded = previous.get("implementation_sha256", {})
            changed = [name for name, value in recorded.items() if implementation.get(name) != value]
            if changed:
                raise ValueError("Implementation changed; use a new experiment directory: " + ", ".join(changed))
        else:
            atomic_json(identity_path, {"config": config, "config_digest": fingerprint, "stage": stage,
                "training_run_id":hashlib.sha256((str(output.resolve())+':'+str(time.time_ns())).encode()).hexdigest() if stage=='train' else None,
                "comparison_run_id":config.get('comparison_run_id',config.get('experiment_id',str(output.parent.resolve()))),
                "sources": json.loads((ROOT/"research_sources.lock.json").read_text()),
                "implementation_sha256": implementation})
        manifest=json.loads(identity_path.read_text())
        training_run_id=manifest.get('training_run_id')
        invocation_id = time.time_ns()
        atomic_json(output/"invocations"/f"{invocation_id}.json", {"config_digest": fingerprint,
            "resume": resume, "limit": limit, "implementation_sha256": implementation,'config':config})
        if resume and (output/"failure.json").exists():
            archive = output/"interrupted"; archive.mkdir(exist_ok=True)
            os.replace(output/"failure.json", archive/f"failure-{invocation_id}.json")
        needs_llm = config["composition"] in ("hybrid", "all_llm")
        if client is None and needs_llm:
            runtime_model={**config['model'],'journal_path':config.get('journal_path',str(output/'journal.sqlite3')),
                'request_context':{'experiment':config.get('experiment_id',str(output.resolve())),
                    'arm':config.get('comparison_id',config['arm']),'red':config['red'],
                    'env_stage':config.get('evaluation_role','frozen_test' if stage=='eval' else 'collection')}}
            client=ModelClient(runtime_model)
        if client is not None: configure_encoder(client.embed)
        text_model = TextModel(client)
        pool = SecuritySkillPool(config["pool_size"],seed_set=config["skill_seed_set"]) if config["arm"] != "E0" and needs_llm else None
        evolution = Evolution(text_model, config["evolution"]) if pool and config["arm"] in ("E2", "E3") else None
        if evolution and stage=="train" and config["evolution"].get("rollout_verification",False):
            from .rollout_execution import EnvironmentVerifier
            evolution.rollout_validator=EnvironmentVerifier(config,output,client.identity)
        checkpoint_path = output/"checkpoint.json"
        checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else None
        training_provenance = None
        inherited = None
        lineage_id=(checkpoint or {}).get('training_lineage_id') or training_run_id
        inherited_seeds = list((checkpoint or {}).get("inherited_training_seeds", []))
        inherited_episodes = (checkpoint or {}).get("inherited_training_episodes", 0)
        if stage == "train" and config.get("pool") and checkpoint is None:
            initial_path = Path(config["pool"])
            inherited = json.loads(initial_path.read_text())
            if inherited.get("stage") != "train" or inherited.get("pending_experiences"):
                raise ValueError("Warm-start requires a committed training snapshot")
            if any(inherited["model_identity"].get(k) != client.identity.get(k) for k in ("model","revision")):
                raise ValueError("Warm-start policy model differs")
            inherited_seeds = inherited.get("training_seeds", [])
            if set(inherited_seeds) & set(split_seeds(config,'final_test')):
                raise ValueError("Warm-start training/evaluation seeds overlap")
            if set(inherited_seeds) & set(split_seeds(config,'validation')):
                raise ValueError("Warm-start training/validation seeds overlap")
            inherited_episodes = inherited.get("total_training_episodes",inherited["episodes_completed"])
            lineage_id=inherited.get('training_lineage_id') or inherited.get('training_run_id') or 'historical:'+hashlib.sha256(initial_path.read_bytes()).hexdigest()
            pool.restore(inherited["pool"])
            if evolution: evolution.restore(inherited.get("experience",{}))
            training_provenance = {"path":str(initial_path.resolve()),
                "sha256":hashlib.sha256(initial_path.read_bytes()).hexdigest(),"total_training_episodes":inherited_episodes,
                'initial_pool_sha256':inherited.get('initial_pool_sha256'),
                'segment_initial_pool_sha256':hashlib.sha256(json.dumps(inherited['pool'],sort_keys=True).encode()).hexdigest()}
            atomic_json(output/"initial_pool.json",{"pool":pool.snapshot(),"training_provenance":training_provenance})
        if checkpoint and checkpoint.get("pool") is not None:
            pool.restore(checkpoint["pool"])
            if evolution: evolution.restore(checkpoint.get("experience", {}))
        elif stage == "eval" and (output/"frozen_pool.json").is_file() and pool:
            pool.restore(json.loads((output/"frozen_pool.json").read_text())["pool"])
        elif stage == "eval" and (config["arm"] in ("E2", "E3") or config["arm"] == "E1" and config.get("pool")):
            pool_path = Path(config.get("pool") or output.parent/"train/checkpoint.json")
            if not pool_path.is_file(): raise FileNotFoundError(f"Train this arm first or specify --pool: {pool_path}")
            trained = json.loads(pool_path.read_text())
            if trained.get("stage") != "train": raise ValueError("Evaluation requires a training snapshot")
            if trained.get("pending_experiences"):
                raise ValueError("Training has an uncommitted evolution batch; resume training before frozen evaluation")
            if any(trained["model_identity"].get(k) != client.identity.get(k) for k in ("model","revision")):
                raise ValueError("Skill pool was trained with another policy model")
            if set(trained.get("training_seeds", [])) & set(split_seeds(config,'final_test')):
                raise ValueError("Training/evaluation seeds overlap with the loaded pool")
            pool.restore(trained["pool"])
            training_provenance = {"path": str(pool_path.resolve()),
                "sha256": hashlib.sha256(pool_path.read_bytes()).hexdigest(),
                "episodes_completed": trained["episodes_completed"],
                "total_training_episodes": trained.get("total_training_episodes",trained["episodes_completed"]),
                "training_seeds": trained.get("training_seeds", []), "algorithm": trained.get("algorithm"),
                "optimizer_status":trained.get('optimizer_status','unknown'),
                "pool_changed":trained.get('pool_changed'),
                'training_run_id':trained.get('training_run_id'),
                'training_lineage_id':trained.get('training_lineage_id'),
                'planned_training_episodes':trained.get('planned_training_episodes'),
                'planned_optimizer_iterations':trained.get('planned_optimizer_iterations'),
                'training_completed':trained.get('training_completed'),
                'final_pool_sha256':hashlib.sha256(json.dumps(trained['pool'],sort_keys=True).encode()).hexdigest(),
                'optimizer_history':[{'iteration':h['iteration'],'optimizer_status':h.get('optimizer_status'),
                    'episodes':h.get('episodes')} for h in trained.get('history',[])],
                'initial_pool_sha256':trained.get('initial_pool_sha256')}
        if pool and pool.seed_set!=config["skill_seed_set"]:
            raise ValueError("Stored skill seed set differs from the declared experiment")
        if stage == "eval" and pool and not (output/"frozen_pool.json").exists():
            atomic_json(output/"frozen_pool.json", {"pool": pool.snapshot(), "model_identity": client.identity,
                "training_provenance": training_provenance})
        if stage == "eval" and pool:
            training_provenance = json.loads((output/"frozen_pool.json").read_text()).get("training_provenance")
        pool_before = pool.snapshot() if pool else None
        initial_pool_hash=(checkpoint or {}).get('initial_pool_sha256')
        if initial_pool_hash is None and pool:
            initial_pool_hash=(training_provenance or {}).get('initial_pool_sha256')
            if initial_pool_hash is None and (stage=='train' or config['arm']=='E1' or not (training_provenance or {}).get('episodes_completed')):
                initial_pool_hash=hashlib.sha256(json.dumps(pool_before,sort_keys=True).encode()).hexdigest()
        completed, counter = [], 0
        planned = config["train_episodes"] if stage == "train" else config["episodes"]
        planned_seeds=split_seeds(config,'train' if stage=='train' else 'final_test')
        boundaries = {(i*planned+config["iterations"]-1)//config["iterations"]
                      for i in range(1,config["iterations"]+1)} if stage == "train" else {planned}
        pending_experiences = [Experience(**e) for e in checkpoint.get("pending_experiences", [])] if checkpoint else []
        pending_results = checkpoint.get("pending_results", []) if checkpoint else []
        history = deepcopy((checkpoint or inherited or {}).get("history", []))
        baselines = deepcopy((checkpoint or inherited or {}).get("baselines", {}))
        committed = checkpoint.get("episodes_completed", 0) if checkpoint else 0
        for episode in range(planned):
            result_path = output/f"episode-{episode:04d}.json"
            if result_path.is_file() and (stage == "eval" or episode < committed):
                result = json.loads(result_path.read_text())
                if result.get("status") == "completed":
                    completed.append(result)
                    continue
            if limit is not None and counter >= limit: break
            if client: client.calls = []
            recovered = stage == "train" and result_path.is_file() and episode >= committed
            if recovered:
                result=json.loads(result_path.read_text())
                if result.get("status") != "completed": raise ValueError("Cannot recover an incomplete episode result")
                experiences=recover_episode(result,config,client,text_model,output)
            else:
                result, experiences = run_episode(config, planned_seeds[episode], episode, client, pool, text_model, output)
            completed.append(result); pending_results.append(result); pending_experiences.extend(experiences)
            counter += int(not recovered)
            if stage == "train" and episode+1 in boundaries:
                iteration=len(history)+1
                journal=getattr(client,'journal',None)
                preparation_key='preparation:'+str(iteration)+':'+config.get('comparison_id',config['arm'])
                prepared=journal.restore(preparation_key) if journal else None
                pending_hash=progress_digest([e.__dict__ for e in pending_experiences])
                if journal:
                    journal.checkpoint('pending_batch:'+str(iteration)+':'+config.get('comparison_id',config['arm']),{
                        'trace_hash':pending_hash,'episodes':pending_results,
                        'experiences':[e.__dict__ for e in pending_experiences],
                        'environment_episodes_completed':len(completed),'committed_optimizer_loops':len(history)})
                if prepared and prepared['pending_hash']!=pending_hash:
                    raise ValueError('Pending training batch changed during optimizer resume')
                if prepared is None and journal:
                    prepared={'pending_hash':pending_hash,'status':'before_preparation','before':{
                        'pool':pool.snapshot() if pool else None,'baselines':deepcopy(baselines),
                        'evolution':evolution.snapshot() if evolution else None,
                        'model':text_model.snapshot(),'rng':rng_snapshot()}}
                    journal.checkpoint(preparation_key,prepared)
                if prepared:
                    restore=prepared['ready'] if prepared['status']=='prepared' else prepared['before']
                    if pool:pool.restore(restore['pool'])
                    baselines=deepcopy(restore['baselines'])
                    if evolution:evolution.restore(restore['evolution'])
                    text_model.restore(restore['model']);rng_restore(restore['rng'])
                if not prepared or prepared['status']!='prepared':
                    task,average=prepare_batch(config,pending_results,pending_experiences,pool,baselines)
                    if journal:
                        prepared['ready']={'pool':pool.snapshot() if pool else None,'baselines':deepcopy(baselines),
                            'evolution':evolution.snapshot() if evolution else None,
                            'model':text_model.snapshot(),'rng':rng_snapshot()}
                        prepared.update(status='prepared',task=task,average=average)
                        journal.checkpoint(preparation_key,prepared)
                else:
                    task,average=prepared['task'],prepared['average']
                ev = {"evolution": [], "maintenance": [], "seconds": 0.0}
                if evolution and config["evolution"].get("enabled", True):
                    if config['evolution'].get('selection_shift_verification',False):
                        from .skills import probe_selection
                        probe_rows=[]
                        for result in pending_results:
                            trace=output/Path(result['trace']).name
                            probe_rows.extend(json.loads(line) for line in trace.open() if line.strip())
                        probe_rule={**config['evolution'].get('selection_shift_rule',{}),
                            'epsilon':config['selector']['epsilon'],
                            'allowed_seeds':split_seeds(config,'train'),
                            'forbidden_seeds':split_seeds(config,'final_test')}
                        def verify_selector(previous,proposed):
                            key='selector_probe:'+progress_digest([previous,proposed,probe_rows,probe_rule,client.identity])
                            cached=journal.restore(key) if journal else None
                            if cached is not None:return cached
                            measured=probe_selection(previous,proposed,probe_rows,client,probe_rule,'training')
                            if journal:journal.checkpoint(key,measured)
                            return measured
                        evolution.selection_verifier=verify_selector
                    text_model.purpose = "evolution"
                    call_offset = len(client.calls) if client else 0
                    try:
                        ev=evolution.update(pool,pending_experiences,baselines,len(history)+1)
                        ev.setdefault('model_calls',deepcopy(client.calls[call_offset:]) if client else [])
                    finally:
                        text_model.purpose='skill_selection'
                validation=[]
                if config.get("validation_episodes",0):
                    validation_dir=output/f"validation-iteration-{len(history)+1:04d}"
                    validation_dir.mkdir(exist_ok=True)
                    before_validation=pool.snapshot() if pool else None
                    for v in range(config["validation_episodes"]):
                        if client: client.calls=[]
                        measured,_=run_episode(config,split_seeds(config,'validation')[v],v,client,pool,text_model,validation_dir)
                        validation.append(measured)
                    if pool and pool.snapshot()!=before_validation: raise RuntimeError("Validation updated the skill pool")
                    ev["validation_reward_mean"]=aggregate_episodes(validation)["mean_joint_reward"]
                    ev["validation_episodes"]=validation
                history.append({"iteration": len(history)+1, "episodes": episode+1,
                    "reward_mean": average, "baseline": baselines[task], **ev, "pool_snapshot": pool.snapshot() if pool else None})
                snapshot = {"stage": "train", "pool": pool.snapshot() if pool else None,
                    'training_run_id':training_run_id,
                    'training_lineage_id':lineage_id,
                    'planned_training_episodes':planned,'planned_optimizer_iterations':config['iterations'],
                    'training_completed':episode+1==planned,
                    "episodes_completed": episode+1,
                    "total_training_episodes": inherited_episodes+episode+1,
                    "inherited_training_seeds": inherited_seeds, "inherited_training_episodes": inherited_episodes,
                    "training_seeds": sorted(set(inherited_seeds)|set(planned_seeds[:episode+1])),
                    "model_identity": client.identity if client else None, "baselines": baselines,
                    "experience": evolution.snapshot() if evolution else {}, "history": history,
                    "config_digest": fingerprint, "algorithm": config["evolution"],
                    'optimizer_status':ev.get('optimizer_status','completed_unchanged'),
                    'initial_pool_sha256':initial_pool_hash,
                    'pool_changed':ev.get('pool_changed',False),
                    'text_model_state':text_model.snapshot(),'rng_state':rng_snapshot()}
                atomic_json(output/f"pool-iteration-{len(history):04d}.json", snapshot)
                atomic_json(checkpoint_path, snapshot)
                if journal:
                    journal.checkpoint('loop_commit:'+str(len(history))+':'+config.get('comparison_id',config['arm']),{
                        'optimizer_status':snapshot['optimizer_status'],'pool_changed':snapshot['pool_changed'],
                        'checkpoint_sha256':hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()})
                pending_experiences, pending_results = [], []
            elif stage == "train":
                # Persist the pending batch too, so a one-episode batch invocation
                # resumes without rerunning or overwriting completed experience.
                atomic_json(checkpoint_path, {"stage":"train", "pool":pool.snapshot() if pool else None,
                    'training_run_id':training_run_id,
                    'training_lineage_id':lineage_id,
                    'planned_training_episodes':planned,'planned_optimizer_iterations':config['iterations'],
                    'training_completed':False,
                    "episodes_completed":episode+1,
                    "total_training_episodes":inherited_episodes+episode+1,
                    "inherited_training_seeds":inherited_seeds,"inherited_training_episodes":inherited_episodes,
                    "training_seeds":sorted(set(inherited_seeds)|set(planned_seeds[:episode+1])),
                    "model_identity":client.identity if client else None, "baselines":baselines,
                    "experience":evolution.snapshot() if evolution else {}, "history":history,
                    "pending_experiences":[e.__dict__ for e in pending_experiences],
                    "pending_results":pending_results, "algorithm":config["evolution"],
                    "initial_pool_sha256":initial_pool_hash})
            atomic_json(output/"summary.json", {"status": "partial", "stage": stage, "config": config,
                "episodes": completed, "metrics": aggregate_episodes(completed),
                "result_kind": "mock_test" if getattr(client, "is_mock", False) else "real_cage4", "history": history, "planned_episodes": planned})
        if stage == "eval" and pool and pool.snapshot() != pool_before:
            raise RuntimeError("Evaluation modified the frozen skill pool")
        summary = {"status": "completed" if len(completed) == planned else "partial", "stage": stage,
            "config": config, "episodes": completed, "metrics": aggregate_episodes(completed),
            "result_kind": "mock_test" if getattr(client, "is_mock", False) else "real_cage4",
            "history": history, "planned_episodes": planned, "pool_frozen": stage == "eval",
            "model_identity": client.identity if client else None,
            "training_provenance": training_provenance,
            'comparison_run_id':manifest.get('comparison_run_id'),
            'comparison_initial_pool_sha256':initial_pool_hash,
            "pool_sha256": hashlib.sha256(json.dumps(pool.snapshot() if pool else None, sort_keys=True).encode()).hexdigest(),
            "initial_pool_sha256": hashlib.sha256(json.dumps(pool_before, sort_keys=True).encode()).hexdigest()}
        atomic_json(output/"summary.json", summary)
        print(json.dumps({"output": str(output), "status": summary["status"], "metrics": summary["metrics"]}), flush=True)
        return summary
    except BaseException as exc:
        active_journal=getattr(client,'journal',None)
        if active_journal:
            for task in active_journal.tasks(('started',)):
                if task['context'].get('source')=='run_episode' and task['context'].get('directory')==str(output.resolve()):
                    active_journal.mark(task['id'],'interrupted' if isinstance(exc,BudgetExceeded) else 'unknown',{'error':str(exc)})
        atomic_json(output/"failure.json", {"status": "interrupted_budget" if isinstance(exc,BudgetExceeded) else "failed", "error": f"{type(exc).__name__}: {exc}",
            "model": config["model"], "no_fallback_model": True})
        raise
    finally:
        fcntl.flock(lock_stream, fcntl.LOCK_UN); lock_stream.close()
