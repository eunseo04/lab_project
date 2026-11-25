from copy import deepcopy
from pathlib import Path
import hashlib
import json
import yaml
from acd.config import LEGACY_ENCODER, validate_communication_encoder
from . import ROOT

ARMS = ("E0", "E1", "E2", "E3")
REDS = ("finite", "aggressive", "stealthy", "impact", "degrade")
VERIFICATION_VARIANTS = {name:{'base':'B4' if name.startswith('B4-') else 'B2',
    'action_support_verification':'ASV' in name,'selection_shift_verification':'SSV' in name,
    'rollout_verification':'-RV' in name or name.startswith('B4-')}
    for name in ('B2-ASV','B2-SSV','B2-ASV-SSV','B2-ASV-SSV-RV',
                 'B2-ASV-SSV-RV-screen','B4-ASV','B4-SSV','B4-ASV-SSV')}
COMPLETED_OPTIMIZER_STATUSES=frozenset(('completed_changed','completed_unchanged','completed_no_eligible_skill'))
CONTROLLER_VARIANTS={'B1-selector-contract':'B1','B3-selector-contract':'B3',
                     'B1-urgent-preemption':'B1','B3-urgent-preemption':'B3'}
ZONE_PAIR_CONTRACT = 'explicit_zone_pair_v1'


def comparison_metadata(config):
    """Shared identity for runtime variants and reports; labels do not prove learning."""
    c={'comparison_id':config} if isinstance(config,str) else config
    name=c.get('comparison_id',c.get('arm','unknown'));variant=VERIFICATION_VARIANTS.get(name,{})
    base=c.get('base_comparison_id',CONTROLLER_VARIANTS.get(name,variant.get('base',name)))
    internal=c.get('arm')
    known=internal in ARMS or base in ('B0','B1','B2','B3-no-support','B3','B4')
    consistent=not (internal in ARMS and base in ('B0','B1','B2','B3-no-support','B3','B4') and
        internal not in ({'E0'} if base=='B0' else {'E1'} if base=='B1' else {'E2','E3'}))
    requires=(internal in ('E2','E3') if internal in ARMS else base in ('B2','B3-no-support','B3','B4')) if known else None
    return {'comparison_id':name,'base_comparison_id':base,'internal_arm':internal,
            'requires_training':requires,'policy_type_known':known,'policy_label_consistent':consistent,**variant}


def split_seeds(config,split):
    names={'train':('train_seeds','train_seed','train_episodes'),
           'final_test':('eval_seeds','eval_seed','episodes'),
           'validation':('validation_seeds','validation_seed','validation_episodes')}
    explicit,start,count=names[split]
    return list(config[explicit]) if explicit in config else list(range(config[start],config[start]+config[count]))


def seed_exposure_history(directory):
    """Read actual episode records, partial traces and registrations without editing them."""
    directory=Path(directory);events=[];registered=[]
    known={s:'historical_development_test' for s in [*range(101000,101100),102000,58000,201000,201001,211000,211001,215000,788380683]}
    known.update({s:'historical_training' for s in (91000,91001)})
    known.update({s:'historical_candidate_validation' for s in (213000,213001)})
    events.extend({'seed':s,'role':role,'state':'exposed','source':'provided research history'} for s,role in known.items())
    for path in sorted(directory.rglob('manifest.json')):
        data=json.loads(path.read_text());cfg=data.get('config') or {}
        role='validation' if cfg.get('evaluation_role')=='rollout' else 'train' if data.get('stage')=='train' else 'final_test'
        for record in sorted(path.parent.glob('episode-????.json')):
            e=json.loads(record.read_text())
            if e.get('kind')!='real_cage4' or e.get('unit_test_fixture_only'):continue
            events.append({'seed':e['seed'],'role':role,'red':cfg.get('red'),'state':'exposed',
                'source':str(record.resolve()),'status':e.get('status')})
        for partial in sorted(path.parent.glob('episode-????.partial.jsonl')):
            with partial.open() as stream:line=next((line for line in stream if line.strip()),None)
            if line:
                row=json.loads(line)
                if type(row.get('seed')) is int:events.append({'seed':row['seed'],'role':role,'state':'exposed_partial','source':str(partial.resolve())})
    for path in sorted(directory.rglob('protocol.json')):
        p=json.loads(path.read_text());plan=p.get('seed_plan')
        if plan:
            for role in ('train','validation','final_test','transfer','reserved_final_test'):
                for seed in plan.get(role,[]):registered.append({'seed':seed,'role':role,'source':str(path.resolve())})
        else:
            for role,key in (('train','training_seeds'),('validation','rollout_validation_seeds'),
                             ('final_test','pilot_test_seeds'),('transfer','generalization_seeds'),
                             ('reserved_final_test','reserved_full_test_seeds')):
                for seed in p.get(key,[]):registered.append({'seed':seed,'role':role,'source':str(path.resolve())})
    exposed={e['seed'] for e in events}
    return {'schema':'seed-exposure-v1','events':events,'registrations':registered,
        'exposed_seeds':sorted(exposed),'reserved_unexposed_seeds':sorted({r['seed'] for r in registered if r['role']=='reserved_final_test'}-exposed),
        'scope':'Observed records and supplied history; unexecuted registrations are not completed samples.'}


def validate_seed_plan(plan,history=None):
    plan=deepcopy(plan)
    if history is None:history=seed_exposure_history(ROOT/'research_outputs')
    if plan.get('purpose') not in ('final_evaluation','development','reproduction'):
        raise ValueError('Seed plan needs explicit purpose: final_evaluation/development/reproduction')
    for role in ('train','validation','final_test','transfer'):
        values=plan.get(role)
        if not isinstance(values,list) or any(type(s) is not int or s<0 for s in values) or len(set(values))!=len(values):
            raise ValueError('Explicit unique integer seed list required: '+role)
    roles=('train','validation','final_test','transfer')
    for i,a in enumerate(roles):
        for b in roles[i+1:]:
            if set(plan[a])&set(plan[b]):raise ValueError('Seed plan splits overlap: '+a+'/'+b)
    exposed=set(history.get('exposed_seeds',[]))
    reused={role:sorted(set(plan[role])&exposed) for role in roles}
    if plan['purpose']=='final_evaluation' and (reused['final_test'] or reused['transfer']):
        raise ValueError('Final holdout/transfer seed already exposed: '+str(reused))
    reserved=plan.get('reserved_final_test',[])
    if not isinstance(reserved,list) or any(type(s) is not int for s in reserved) or len(set(reserved))!=len(reserved):
        raise ValueError('Invalid reserved final-test seed list')
    if set(reserved)&(set(plan['train'])|set(plan['validation'])|set(plan['transfer'])):
        raise ValueError('Reserved final-test seeds overlap learning/validation/transfer')
    plan.update(reused_seed_roles=reused,validation_reuse_is_new_independent_sample=False,
        reserved_unseen_seeds=sorted(set(reserved)-exposed-set(plan['final_test'])),
        reserved_exposed_seeds=sorted(set(reserved)&exposed),
        actual_final_test_is_reserved_unused=False)
    return plan


def apply_seed_plan(config,plan,history=None):
    if history is None:history=seed_exposure_history(ROOT/'research_outputs')
    c=deepcopy(config);plan=validate_seed_plan(plan,history)
    if not plan['train']:raise ValueError('An explicit training/source seed set is required')
    c.update(seed_plan=plan,seed_use=plan['purpose'],train_seeds=plan['train'],
        train_seed=plan['train'][0],train_episodes=len(plan['train']),validation_seeds=[],validation_episodes=0)
    if history is not None:c['seed_exposure_history']=history
    if plan['final_test']:
        c.update(eval_seeds=plan['final_test'],eval_seed=plan['final_test'][0],episodes=len(plan['final_test']),evaluation_disabled=False)
    else:
        c.pop('eval_seeds',None);c['evaluation_disabled']=True
    if c['evolution'].get('rollout_verification'):
        c['rollout_validation']['seeds']=plan['validation']
    return validate(c)


def variant_config(name,seed_plan=None):
    """Named overrides of existing configs; no copied YAML or silent B2/B3/B4 redefinition."""
    if name in CONTROLLER_VARIANTS:
        base=CONTROLLER_VARIANTS[name];c=load(ROOT/f'research_configs/rollout-{base}.yaml')
        if name.endswith('-urgent-preemption'):
            c['selector']['urgent_preemption']=True
            version='cage4-acd-urgent-preemption-v1'
        else:
            c['selector']['contract']='cage4_role_v1'
            version='cage4-acd-selector-contract-v1'
        c.update(comparison_id=name,base_comparison_id=base,experiment_version=version)
        return apply_seed_plan(c,seed_plan) if seed_plan is not None else validate(c)
    if name not in VERIFICATION_VARIANTS:
        c=load(ROOT/f'research_configs/rollout-{name}.yaml')
        if seed_plan is not None:return apply_seed_plan(c,seed_plan)
        c.setdefault('seed_use','legacy_reproduction')
        return validate(c)
    metadata=VERIFICATION_VARIANTS[name];base=metadata['base']
    c=load(ROOT/f'research_configs/rollout-{base}.yaml')
    c['evolution'].update(action_support_verification=metadata['action_support_verification'],
                          selection_shift_verification=metadata['selection_shift_verification'],
                          admission_mode='rollout' if metadata['rollout_verification'] else 'factual_ppo',
                          method_version='cage4-fact-intervention-v1')
    if '-RV' in name:
        c['evolution']['rollout_verification']=True
        c['rollout_validation']=load(ROOT/'research_configs/rollout-B4.yaml')['rollout_validation']
    if name.endswith('-screen'):
        c['rollout_validation'].update(reject_screen_pairs=2,reject_screen_only=True,
            procedure_version='necessary-worst-seed-rejection-v1')
    c.setdefault('seed_use','legacy_reproduction')
    c.update(comparison_id=name,base_comparison_id=base)
    if seed_plan is not None:return apply_seed_plan(c,seed_plan)
    return validate(c)


def load(path):
    config = yaml.safe_load(Path(path).read_text())
    return validate(config)


def execution_config(config, stage):
    """Bind the runtime stage while retaining the registered training method."""
    if stage not in ('train','eval'):
        raise ValueError('Stage must be train or eval')
    if config.get('execution_stage', stage) != stage:
        raise ValueError('Execution stage does not match the registered configuration')
    return validate({**config, 'execution_stage':stage})


def policy_interface(config):
    selector=config.get('selector',{})
    return {'communication':config.get('communication','paper_projected_keep'),
            'communication_encoder':config.get('communication_encoder',LEGACY_ENCODER),
            'action_contract':config.get('action_contract','legacy'),
            'selector_contract':selector.get('contract','upstream'),
            'urgent_preemption':selector.get('urgent_preemption',False)}


def read_policy_manifest(path, *, expected_sha256=None):
    """Read the original run record and verify its recorded configuration digest."""
    path=Path(path)
    raw=path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest()!=expected_sha256:
        raise ValueError('Policy manifest file hash differs from the registered source')
    manifest=json.loads(raw)
    if (not isinstance(manifest,dict) or manifest.get('stage') not in ('train','eval')
            or not isinstance(manifest.get('config'),dict)
            or manifest.get('config_digest')!=digest(manifest['config'])):
        raise ValueError('Policy manifest has no verified original configuration digest')
    if 'policy_interface' in manifest and manifest['policy_interface']!=policy_interface(manifest['config']):
        raise ValueError('Policy manifest interface disagrees with its verified configuration')
    return manifest


def _frozen_training_manifest(path):
    """Recover a campaign's registered source from a byte-identical training copy."""
    if path.parent.name!='frozen':
        return None
    def file_hash(file):
        sha=hashlib.sha256()
        with file.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                sha.update(chunk)
        return sha.hexdigest()
    candidates=[]
    size=path.stat().st_size
    source_hash=None
    for run in (path.parent.parent/'runs').glob(f'{path.stem}-*-train'):
        manifest=run/'manifest.json'
        if not manifest.is_file():
            continue
        for checkpoint in (run/'checkpoint.json',*run.glob('pool-iteration-*.json')):
            if not checkpoint.is_file() or checkpoint.stat().st_size!=size:
                continue
            matches=checkpoint.samefile(path)
            if not matches:
                if source_hash is None:
                    source_hash=file_hash(path)
                matches=file_hash(checkpoint)==source_hash
            if matches:
                candidates.append(manifest)
                break
    if len(candidates)>1:
        raise ValueError('Frozen checkpoint has more than one matching original training manifest')
    return candidates[0] if candidates else None


def read_training_manifest(artifact_path, artifact, *, manifest_ref=None):
    """Verify the available original training record against this checkpoint."""
    artifact_path=Path(artifact_path)
    if manifest_ref is not None:
        if (not isinstance(manifest_ref,dict) or not isinstance(manifest_ref.get('path'),str)
                or not manifest_ref['path'] or not manifest_ref.get('sha256')):
            raise ValueError('Original training manifest reference needs a path and registered SHA256')
        path=artifact_path.parent/manifest_ref['path']
        if not path.is_file():
            raise ValueError('Registered original training manifest file is unavailable')
        manifest=read_policy_manifest(path,expected_sha256=manifest_ref['sha256'])
    else:
        if not artifact.get('config_digest'):
            return None
        path=artifact_path.parent/'manifest.json'
        if not path.is_file():
            path=_frozen_training_manifest(artifact_path)
        if path is None:
            return None
        manifest=read_policy_manifest(path)
    if (manifest['stage']!='train' or artifact.get('config_digest')!=manifest['config_digest']
            or artifact.get('training_run_id')!=manifest.get('training_run_id')):
        raise ValueError('Original training manifest does not identify this checkpoint')
    return manifest


def check_pool_interface(artifact, config, *, static_initial=False, manifest=None):
    """Require a compatible artifact interface or an independently verified run manifest."""
    expected=policy_interface(config)
    unused=(static_initial and artifact.get('episodes_completed')==0
            and artifact.get('optimizer_status') in (None,'not_started') and not any(
                artifact.get(key) for key in ('training_seeds','total_training_episodes','inherited_training_episodes',
                                              'experience','pending_experiences','history','training_completed')))
    if unused and artifact.get('initialization_only') is True:
        initial=json.loads((ROOT/'artifacts/initial_pool.json').read_text())
        if artifact.get('pool')==initial['pool']:
            return True
    recorded=artifact.get('policy_interface')
    if manifest is not None:
        original=policy_interface(manifest['config'])
        if recorded is not None and recorded!=original:
            raise ValueError('Skill Pool policy interface disagrees with its original manifest')
        recorded=original
    encoder=(recorded.get('communication_encoder') if isinstance(recorded,dict)
             else artifact.get('communication_encoder',LEGACY_ENCODER))
    if artifact.get('communication_encoder',encoder)!=encoder or encoder!=expected['communication_encoder']:
        raise ValueError('Skill Pool communication encoder differs from this experiment')
    if 'controller_contract' in artifact and artifact['controller_contract']!=expected['selector_contract']:
        raise ValueError('Skill Pool controller contract differs from this experiment')
    if recorded is None:
        # Explicit user-authored static controls predate interface metadata.
        if unused and config.get('arm')=='E1' and expected==policy_interface({}):
            return False
        raise ValueError('Missing Skill Pool policy interface and verified original manifest')
    if recorded!=expected:
        raise ValueError('Skill Pool policy interface differs from this experiment')
    return False


def validate(config):
    c = deepcopy(config)
    defaults = {"arm": "E0", "composition": "hybrid", "red": "finite", "steps": 500,
        "episodes": 100, "train_episodes": 6, "iterations": 3, "train_seed": 100,
        "eval_seed": 10000, "profile": "controlled-comparison", "scoring": "official",
        "communication": "paper_projected_keep", "threads": 1,
        "weights": "third_party/cage-4-submission/weights", "pool_size": 10,
        "output": "research_outputs", "skip_busy_policy": True, "action_contract": "legacy",
        "behavior_context": "legacy", "skill_seed_set": "legacy", "case_presentation": "json",
        "connection_leads": False}
    for key, value in defaults.items(): c.setdefault(key, value)
    c.setdefault("validation_episodes", 1)
    c.setdefault('seed_use','legacy_reproduction')
    c.setdefault("validation_seed", 5000)
    c.setdefault("model", {"endpoint": "http://127.0.0.1:8018", "model": "Qwen/Qwen3-8B",
        "revision": "b968826d9c46dd6066d109eabc6255188de91218", "temperature": 1.0,
        "max_tokens": 192, "timeout_seconds": 240})
    c.setdefault("selector", {"epsilon": 0.1, "max_skill_age": 8})
    c["selector"].setdefault("verification_history", "legacy_ticks")
    c.setdefault("evolution", {})
    for key, value in {"enabled": True, "ppo_gate": True, "semantic_gradient": True,
        "score_maintenance": True, "gate_mode": "official_code", "threshold": 6,
        "best_of_n": 3, "max_evolutions": 2, "clip": 0.2, "gamma": 0.99,
        "ema_alpha": 0.1, "acceptance_margin": 0.0, "semantic_chunk_steps": 8,
        "maintenance_mode": "official_code", "schedule": "official_code"}.items():
        c["evolution"].setdefault(key, value)
    if c["arm"] not in ARMS or c["red"] not in REDS: raise ValueError("Invalid experiment arm/red policy")
    if c["composition"] not in ("hybrid", "all_llm", "no_blue", "all_rl"):
        raise ValueError("composition must be hybrid/all_llm/no_blue/all_rl")
    if c["profile"] not in ("paper-compatible", "controlled-comparison"):
        raise ValueError("Invalid action/prompt comparison profile")
    if c["scoring"] not in ("official", "terminal_inclusive"): raise ValueError("Invalid scoring profile")
    if c["action_contract"] not in ("legacy", "explicit", ZONE_PAIR_CONTRACT):
        raise ValueError("Unknown action contract")
    if c["behavior_context"] not in ("legacy", "evidence_v1"):
        raise ValueError("Invalid observable behavior context")
    if c["case_presentation"] not in ("json","readable"):
        raise ValueError("Invalid case history presentation")
    if type(c["connection_leads"]) is not bool:
        raise ValueError("connection_leads must be a boolean")
    if c["connection_leads"] and c["behavior_context"] != "evidence_v1":
        raise ValueError("Connection lead history requires evidence_v1")
    if c["selector"]["verification_history"] not in ("legacy_ticks", "accepted_decisions"):
        raise ValueError("Invalid verification history mode")
    if c["skill_seed_set"] not in ("legacy", "response_v1"):
        raise ValueError("Invalid versioned skill seed set")
    if c["skill_seed_set"]=="response_v1" and c["behavior_context"]!="evidence_v1":
        raise ValueError("response_v1 skills require observable evidence_v1 context")
    if c["profile"]=="paper-compatible" and (c["behavior_context"]!="legacy" or c["skill_seed_set"]!="legacy"):
        raise ValueError("Unmodified paper-compatible mode requires legacy state and skill seeds")
    if c["profile"] == "paper-compatible" and c["action_contract"] != "legacy":
        raise ValueError("Unmodified ACD paper-compatible prompts require the legacy action contract")
    if c["communication"] not in ("paper_projected_keep", "keep_native"): raise ValueError("Invalid communication profile")
    c['communication_encoder']=validate_communication_encoder(c['communication'],c.get('communication_encoder',LEGACY_ENCODER))
    if c.get('execution_stage') not in (None,'train','eval'):
        raise ValueError('Invalid execution stage')
    rollout_eval=c.get('evaluation_role')=='rollout'
    if rollout_eval and (c.get('execution_stage')!='eval' or not c['evolution'].get('rollout_verification')):
        raise ValueError('Candidate rollout needs a registered rollout evaluation configuration')
    for key in ("steps", "episodes", "train_episodes", "iterations", "threads", "pool_size"):
        if type(c[key]) is not int or c[key] < 1: raise ValueError(f"{key} must be a positive integer")
    if c["steps"] < 3: raise ValueError("CC4 requires at least 3 steps")
    if c["iterations"] > c["train_episodes"]: raise ValueError("iterations cannot exceed training episodes")
    if type(c["validation_episodes"]) is not int or c["validation_episodes"] < 0:
        raise ValueError("validation_episodes must be a nonnegative integer")
    e = c["evolution"]
    mode=e.get('admission_mode','legacy_rollout_veto')
    if mode not in ('legacy_rollout_veto','factual_ppo','rollout'):raise ValueError('Invalid admission mode')
    if mode!='legacy_rollout_veto' and not e.get('method_version'):raise ValueError('Changed admission policy requires method_version')
    if mode=='rollout' and not e.get('rollout_verification'):raise ValueError('Rollout admission requires registered rollout verification')
    if mode=='factual_ppo' and e.get('rollout_verification'):raise ValueError('Factual PPO and rollout admission contracts conflict')
    if e.get('evidence_schema','typed_claims_v8') not in ('typed_claims_v8','fact_interventions_v1'):raise ValueError('Invalid evidence schema')
    if e.get('evidence_schema')=='fact_interventions_v1' and not (e.get('diagnosis_verification') and e.get('method_version')):raise ValueError('Fact interventions require a versioned diagnosis method')
    if c['selector'].get('contract','upstream') not in ('upstream','cage4_role_v1'):raise ValueError('Unknown selector contract')
    if c['selector'].get('contract')=='cage4_role_v1' and c.get('comparison_id') not in ('B1-selector-contract','B3-selector-contract'):raise ValueError('Selector change requires its own controller ablation')
    if type(c['selector'].get('urgent_preemption',False)) is not bool:
        raise ValueError('Urgent preemption must be a boolean')
    if c['selector'].get('urgent_preemption') and (c.get('comparison_id') not in ('B1-urgent-preemption','B3-urgent-preemption') or
            c['selector'].get('contract','upstream')!='upstream' or c['behavior_context']!='evidence_v1' or not c['skip_busy_policy']):
        raise ValueError('Urgent preemption needs its observable-case ablation with the upstream selector')
    if e.get('diagnosis_verification',False) and not e.get('evidence_verification',False):
        raise ValueError('CAGE4 diagnosis verification requires trace evidence verification')
    minimum=e.get('minimum_independent_episodes',0)
    if type(minimum) is not int or minimum<0:
        raise ValueError('minimum_independent_episodes must be a nonnegative integer')
    if minimum and not e.get('evidence_verification',False):
        raise ValueError('Experience support gating is a Proposed ablation, not original B2')
    for key in ('action_support_verification','selection_shift_verification'):
        if key in e and type(e[key]) is not bool:raise ValueError(key+' must be boolean')
    if any(e.get(k,False) for k in ('action_support_verification','selection_shift_verification')):
        if not c.get('comparison_id') or c.get('comparison_id') in ('B0','B1','B2','B3','B4','B3-no-support'):
            raise ValueError('New verification needs a distinct ablation ID; preserve original arm definitions')
        if not c.get('experiment_version'):raise ValueError('New verification requires an explicit experiment version')
    shift=e.get('selection_shift_rule',{})
    if type(shift.get('maximum_states',24)) is not int or shift.get('maximum_states',24)<1:
        raise ValueError('Positive selector probe state count required')
    for key,default in (('maximum_disagreement',.25),('maximum_total_variation',.2),('maximum_share_increase',.2)):
        if not 0<=shift.get(key,default)<=1:raise ValueError('Invalid selector shift threshold')
    if e["gate_mode"] not in ("official_code", "paper_rtg"): raise ValueError("Invalid PPO gate mode")
    if e["maintenance_mode"] not in ("official_code", "paper_score", "paper_algorithm1"):
        raise ValueError("Invalid maintenance mode")
    if e["schedule"] not in ("official_code", "paper_batch"):
        raise ValueError("Invalid Skill-Pro evolution schedule")
    if e["schedule"] == "paper_batch" and (e["gate_mode"] != "paper_rtg" or e["maintenance_mode"] != "paper_algorithm1"):
        raise ValueError("paper_batch requires paper_rtg and paper_algorithm1; do not mix algorithm variants")
    for key in ("threshold", "best_of_n", "max_evolutions", "semantic_chunk_steps"):
        if type(e[key]) is not int or e[key] < 1: raise ValueError(f"evolution.{key} must be positive")
    if not 0 < e["gamma"] <= 1 or not 0 < e["ema_alpha"] <= 1 or not 0 < e["clip"] < 1:
        raise ValueError("Invalid discount, EMA rate or PPO clipping interval")
    if 'seed_plan' in c:
        if 'seed_exposure_history' not in c:c['seed_exposure_history']=seed_exposure_history(ROOT/'research_outputs')
        c['seed_plan']=validate_seed_plan(c['seed_plan'],c['seed_exposure_history'])
        if set(split_seeds(c,'train'))!=set(c['seed_plan']['train']):raise ValueError('Config training seeds differ from registered plan')
        if not c.get('evaluation_disabled'):
            role='validation' if c.get('evaluation_role')=='rollout' else 'transfer' if c.get('evaluation_role')=='transfer_test' else 'final_test'
            if not set(split_seeds(c,'final_test'))<=set(c['seed_plan'][role]):raise ValueError('Evaluation seeds differ from registered role: '+role)
    for split,count in (('train','train_episodes'),('final_test','episodes'),('validation','validation_episodes')):
        values=split_seeds(c,split)
        if len(values)!=c[count] or len(set(values))!=len(values) or any(type(s) is not int or s<0 for s in values):
            raise ValueError('Explicit seed list/count mismatch: '+split)
    train_seeds = set(split_seeds(c,'train'))
    eval_seeds = set(split_seeds(c,'final_test'))
    if train_seeds & eval_seeds: raise ValueError("Training and evaluation seed sets overlap")
    validation_seeds=set(split_seeds(c,'validation'))
    if validation_seeds & (train_seeds | eval_seeds): raise ValueError("Validation seeds overlap with training/evaluation")
    if c["profile"] == "paper-compatible" and c["skip_busy_policy"]:
        raise ValueError("Paper-compatible mode requires skip_busy_policy: false")
    if c["model"].get("temperature", 1.0) <= 0: raise ValueError("Positive policy temperature required")
    for key in ('sg_workers','score_batch_size'):
        if key in c['model'] and (type(c['model'][key]) is not int or c['model'][key]<1):
            raise ValueError(key+' must be a positive integer')
    if c['selector'].get('domain_verification_revision','v4') not in ('v4','v5'):
        raise ValueError('Invalid domain verification revision')
    if e.get('rollout_verification',False):
        if not e.get('ppo_gate',True) or e['schedule']!='paper_batch':
            raise ValueError('Proposed rollout verification requires the unchanged paper PPO gate')
        rule=c.get('rollout_validation',{})
        seeds=rule.get('seeds',[])
        if not seeds or len(set(seeds))!=len(seeds) or any(type(s) is not int for s in seeds):
            raise ValueError('Explicit unique rollout validation seeds required')
        from .rollout import approval_seeds
        approval_seeds(rule)
        if rollout_eval and not eval_seeds <= set(seeds):
            raise ValueError('Candidate evaluation seeds must belong to the registered rollout validation set')
        if set(seeds)&(train_seeds|validation_seeds|(set() if rollout_eval else eval_seeds)):
            raise ValueError('Rollout validation seeds overlap other splits')
        required=('minimum_mean_gain','minimum_win_rate','minimum_median_gain','maximum_seed_loss',
            'maximum_mean_lwf_loss','maximum_mean_asf_loss','maximum_std_ratio','maximum_std_increase','ports')
        if any(k not in rule for k in required):raise ValueError('Incomplete registered rollout approval rule')
        count=rule.get('reject_screen_pairs',0)
        if type(count) is not int or not 0<=count<=len(seeds):raise ValueError('Invalid predeclared reject-screen prefix')
        if count:
            if not rule.get('procedure_version') or c.get('comparison_id')=='B4':
                raise ValueError('Early rejection needs a separate procedure version/ablation')
            if c.get('comparison_id') in ('B0','B1','B2','B3'):raise ValueError('Reject-screen cannot relabel a baseline')
    if not c["model"].get("model") or not c["model"].get("revision"):
        raise ValueError("Explicit model ID and pinned revision are required; no model fallback")
    for key in ("weights", "output", "pool"):
        if c.get(key):
            p = Path(c[key]).expanduser()
            c[key] = str(p if p.is_absolute() else ROOT/p)
    return c


def digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def compatible_digest(config):
    """Normalize newly explicit defaults without changing a recorded manifest."""
    c = validate(config)
    c.pop('execution_stage', None)  # run() checks stage separately, including old manifests.
    c.pop("maintenance_mode", None)  # former unused top-level default
    for key in ('endpoint','scoring_endpoints','generation_endpoints','sg_workers','score_batch_size','timeout_seconds'):
        c['model'].pop(key,None)
    if 'rollout_validation' in c:c['rollout_validation'].pop('ports',None)
    return digest(c)
