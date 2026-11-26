"""Compare new completed runs without historical results or model calls."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys
import hashlib
import io
import tarfile
import tempfile
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.recompute_public_metrics import recompute,paired_episodes,episode_group,GROUP_FIELDS
from research.progress import Journal,atomic_json,file_sha256
from research.config import (comparison_metadata,VERIFICATION_VARIANTS,CONTROLLER_VARIANTS,digest as config_digest,
    COMPLETED_OPTIMIZER_STATUSES,check_pool_interface,policy_interface,read_policy_manifest,read_training_manifest,
    verify_completed_update,training_boundary,split_seeds)
from acd.config import LEGACY_ENCODER


def load(path):
    return json.loads(path.read_text())


def training_evidence(config,summary,run,cache,*,fixed_source=None,checkpoint=None,resolve=None):
    """Classify a frozen policy using a completed, hash-matched training artifact."""
    metadata=comparison_metadata(config)
    if not metadata['policy_type_known'] or not metadata['policy_label_consistent']:
        return {'state':'unknown_or_inconsistent_policy_type','verified':False,'requires_training':None,
                'training_run_id':None,'training_identity':None}
    if not metadata['requires_training']:
        return {'state':'not_applicable','verified':True,'requires_training':False,
                'training_run_id':None,'training_identity':None}
    p=summary.get('training_provenance') or {}
    result={'state':'unknown_provenance','verified':False,'requires_training':True,
            'training_run_id':None,'training_identity':None}
    if config.get('checkpoint_evaluation') != checkpoint:
        result['state']='unregistered_checkpoint_evaluation';return result
    if checkpoint and (p.get('checkpoint_evaluation')!=checkpoint
            or p.get('path')!=checkpoint['path'] or p.get('sha256')!=checkpoint['checkpoint_sha256']
            or p.get('training_manifest')!=checkpoint['training_manifest']):
        result['state']='checkpoint_evaluation_identity_mismatch';return result
    if not p.get('episodes_completed'):
        result['state']='initial_pool' if p.get('episodes_completed')==0 else 'unknown_provenance'
        return result
    if p.get('optimizer_status') not in COMPLETED_OPTIMIZER_STATUSES:
        result['state']='incomplete_or_unknown_optimizer';return result
    path=Path(p.get('path',''))
    if not path.is_absolute():path=run/path
    if resolve:path=resolve(path)
    if not path.is_file() or not p.get('sha256'):
        result['state']='missing_training_artifact';return result
    key=(str(path.resolve()),p['sha256'])
    if key not in cache:
        sha=file_sha256(path)
        snapshot=load(path) if sha==p['sha256'] else {}
        for field in ('experience','baselines','rng_state','text_model_state'):
            snapshot.pop(field,None)
        if 'history' in snapshot:
            snapshot['history']=[{k:h.get(k) for k in ('iteration','episodes','optimizer_status')} for h in snapshot['history']]
        cache[key]=sha,snapshot
    actual,trained=cache[key]
    if actual!=p['sha256']:
        result['state']='training_artifact_hash_mismatch';return result
    try:
        manifest_ref=p.get('training_manifest')
        if resolve and manifest_ref:
            manifest_ref={**manifest_ref,'path':str(resolve(Path(manifest_ref['path'])))}
        original=read_training_manifest(path,trained,manifest_ref=manifest_ref)
        if original is None and trained.get('policy_interface') is None and fixed_source:
            # A fixed-update study archived an exact copy of a previously trained Pool.
            directory,source=fixed_source
            registered_checkpoint=source.get('checkpoint')
            registered_hashes=source.get('files_sha256') or {}
            if registered_checkpoint and registered_hashes.get(registered_checkpoint)==actual:
                original_path=directory/registered_checkpoint
                if original_path.is_file() and hashlib.sha256(original_path.read_bytes()).hexdigest()==actual:
                    name=source.get('training_manifest')
                    manifest_ref=({'path':str(directory/name),'sha256':registered_hashes.get(name)} if name else None)
                    original=read_training_manifest(original_path,trained,manifest_ref=manifest_ref)
        check_pool_interface(trained,config,manifest=original)
    except (ValueError,OSError):
        result['state']='training_policy_interface_mismatch';return result
    algorithm=trained.get('algorithm');declared=p.get('algorithm')
    if not isinstance(algorithm,dict) or declared!=algorithm:
        result['state']='unknown_or_inconsistent_training_method';return result
    base=metadata['base_comparison_id'];e=config.get('evolution') or {}
    proposed=base in ('B3','B3-no-support','B4')
    defaults={'evidence_verification':proposed,'diagnosis_verification':proposed,
        'minimum_independent_episodes':2 if base in ('B3','B4') else 0,
        'action_support_verification':metadata.get('action_support_verification',False),
        'selection_shift_verification':metadata.get('selection_shift_verification',False)}
    expected={k:e.get(k,v) for k,v in defaults.items()}
    expected['rollout_verification']=base=='B4' or metadata.get('rollout_verification',False)
    for k in ('gate_mode','schedule','maintenance_mode'):
        if k in e:expected[k]=e[k]
    for k in ('method_version','admission_mode','evidence_schema'):
        if k in e:expected[k]=e[k]
    if any(algorithm.get(k,False if isinstance(value,bool) else 0)!=value for k,value in expected.items()):
        result['state']='training_method_mismatch';return result
    history=trained.get('history',[]);status=trained.get('optimizer_status')
    if (trained.get('stage')!='train' or trained.get('pending_experiences') or trained.get('pending_results') or
            status not in COMPLETED_OPTIMIZER_STATUSES or status!=p['optimizer_status'] or not history or
            any(h.get('optimizer_status') not in COMPLETED_OPTIMIZER_STATUSES for h in history)):
        result['state']='incomplete_training_commit';return result
    if checkpoint:
        try:
            proof=verify_completed_update(trained,checkpoint,config,original)
        except (KeyError,ValueError):
            result['state']='invalid_completed_update';return result
        if (any(checkpoint.get(k)!=v for k,v in proof.items())
                or checkpoint['training_run_id']!=trained.get('training_run_id')
                or trained.get('model_identity')!=summary.get('model_identity')):
            result['state']='checkpoint_training_provenance_mismatch';return result
    else:
        completed=trained.get('segment_completed_iterations',len(history))
        if (trained.get('training_completed') is False or
                trained.get('episodes_completed',0)<trained.get('planned_training_episodes',0) or
                completed<trained.get('planned_optimizer_iterations',0)):
            result['state']='incomplete_training_plan';return result
        if (config.get('train_episodes',0)>trained.get('episodes_completed',0) or
                config.get('iterations',0)>completed):
            result['state']='training_plan_does_not_match_evaluation';return result
    if not trained.get('planned_training_episodes') and not config.get('train_episodes'):
        result['state']='unknown_training_plan';return result
    seeds=trained.get('training_seeds');initial=trained.get('initial_pool_sha256')
    final=hashlib.sha256(json.dumps(trained.get('pool'),sort_keys=True).encode()).hexdigest()
    frozen=run/'frozen_pool.json'
    if frozen.is_file():
        frozen_record=load(frozen)
        try:
            manifest_path=run/'manifest.json'
            original=(read_policy_manifest(manifest_path)
                      if 'policy_interface' not in frozen_record and manifest_path.is_file() else None)
            check_pool_interface(frozen_record,config,manifest=original)
        except (ValueError,OSError):
            result['state']='frozen_policy_interface_mismatch';return result
        frozen_hash=hashlib.sha256(json.dumps(frozen_record.get('pool'),sort_keys=True).encode()).hexdigest()
    else:frozen_hash=None
    if (not seeds or not initial or p.get('initial_pool_sha256')!=initial or
            p.get('training_seeds')!=seeds or p.get('episodes_completed')!=trained.get('episodes_completed') or
            summary.get('pool_sha256')!=final or (frozen_hash is not None and frozen_hash!=final) or
            (p.get('final_pool_sha256') is not None and p['final_pool_sha256']!=final)):
        result['state']='unverified_pool_or_training_identity';return result
    identity=trained.get('training_lineage_id') or trained.get('training_run_id') or 'historical:'+actual
    return {**result,'state':'verified_completed_update' if checkpoint else 'verified_completed_training','verified':True,
        'training_run_id':trained.get('training_run_id') or 'historical:'+actual,'training_identity':identity,
        'training_seeds':seeds,'initial_pool_sha256':initial,'final_pool_sha256':final,
        'optimizer_status':status,'committed_iterations':len(history),
        'training_episodes_completed':trained['episodes_completed'],
        'origin':'recorded_training_run' if trained.get('training_run_id') else 'verified_historical_update',
        **({'checkpoint_iteration':checkpoint['iteration'],'checkpoint_sha256':actual,**proof} if checkpoint else {})}


def training_summary(items):
    if any(e.get('requires_training',True) is None for e in items):return {'count':None,'status':'unknown'}
    learned=[e for e in items if e.get('requires_training',comparison_metadata(e['label'])['requires_training'])]
    if not learned:return {'count':None,'status':'not_applicable'}
    if any(not e.get('training_identity') for e in learned):return {'count':None,'status':'unknown'}
    return {'count':len({e['training_identity'] for e in learned}),'status':'verified_artifacts',
            'training_run_ids':sorted({e['training_run_id'] for e in learned}),
            'note':'Environment seeds and repeated evaluation of one artifact are not independent training repetitions.'}


def campaign_file(directory, protocol, path):
    """Use immutable archived dependencies when reporting a restored bundle."""
    path=Path(path)
    embedded=directory/'bundle_manifest.json'
    if not embedded.is_file():return path
    mapping=load(embedded).get('historical_dependencies',{})
    if str(path) in mapping:return directory/mapping[str(path)]['relative_path']
    if path.is_absolute() and protocol.get('experiment_id'):
        original=Path(protocol['experiment_id'])
        if original in path.parents:return directory/path.relative_to(original)
    return path


def collect(directory, include_trace=False, *, include_trace_paths=False):
    directory = Path(directory).resolve()
    if directory == ROOT or not directory.is_dir():
        raise ValueError('Pass a specific experiment output directory')
    runs = sorted({p.parent for name in ('summary.json', 'manifest.json') for p in directory.rglob(name)
                   if not {'report','rollout-verification','historical_inputs','checkout'}.intersection(p.relative_to(directory).parts)})
    episodes, statuses, signatures, policy_signatures, initial_pools = [], [], {}, {}, {}
    provenance_cache={};protocol_signatures={};controller_signatures={}
    protocol_path=directory/'protocol.json'
    fixed_source=(directory,load(protocol_path).get('fixed_update_source',{})) if protocol_path.is_file() else None
    campaigns={}
    for run in runs:
        summary = load(run/'summary.json') if (run/'summary.json').exists() else {}
        manifest = load(run/'manifest.json') if (run/'manifest.json').exists() else {}
        recorded_config=manifest.get('config') or {}
        config = summary.get('config') or recorded_config
        if not config and not summary.get('result_kind'):
            continue
        stage = summary.get('stage', manifest.get('stage', 'unknown'))
        label = config.get('comparison_id', config.get('arm', 'unknown'))
        stage={'evaluate':'eval'}.get(stage,stage)
        split = 'train' if stage == 'train' else 'test' if stage=='eval' else 'untrained_control'
        registered_role=recorded_config.get('evaluation_role')
        role = config.get('evaluation_role')
        if {'rollout','validation'}.intersection((role,registered_role)):
            continue
        registered_stage={'evaluate':'eval'}.get(manifest.get('stage'),manifest.get('stage'))
        if registered_stage is not None and stage!=registered_stage:
            raise ValueError('Run stage differs from its manifest')
        if role is not None and registered_role is not None and role!=registered_role:
            raise ValueError('Evaluation role differs from its manifest')
        role=role or registered_role
        registered_seed_use=recorded_config.get('seed_use') or (recorded_config.get('seed_plan') or {}).get('purpose')
        seed_use=config.get('seed_use') or (config.get('seed_plan') or {}).get('purpose')
        if seed_use and registered_seed_use and seed_use!=registered_seed_use:
            raise ValueError('Seed use differs from its manifest')
        seed_use=seed_use or registered_seed_use
        if stage=='eval' and role=='transfer_test':
            split='transfer'
        elif stage=='eval' and seed_use in ('development','reproduction','legacy_reproduction'):
            split='development' if seed_use=='development' else 'reproduction'
        evaluation_scope=split
        campaign=run.parent.parent if run.parent.name=='runs' else directory
        if campaign not in campaigns:
            registration=load(campaign/'protocol.json') if (campaign/'protocol.json').is_file() else {}
            bindings_path=campaign/'frozen/checkpoints.json'
            campaigns[campaign]=(registration,load(bindings_path) if bindings_path.is_file() else {})
        registration,bindings=campaigns[campaign]
        curve=registration.get('checkpoint_evaluations',{})
        checkpoint=None
        if stage=='eval' and curve and (label in ('B0','B1') or label in curve):
            evaluation_scope='completed_update_curve'
            if (config.get('protocol_digest')!=registration['comparison_protocol_digest']
                    or recorded_config.get('protocol_digest')!=config.get('protocol_digest')
                    or not set(config.get('eval_seeds',[]))<=set(registration['pilot_test_seeds'])):
                raise ValueError('Learning-curve evaluation differs from its registered protocol')
            if label in curve:
                checkpoint=bindings.get(label)
                if (not checkpoint or any(checkpoint.get(k)!=v for k,v in curve[label].items())
                        or recorded_config.get('checkpoint_evaluation')!=checkpoint
                        or config.get('checkpoint_evaluation')!=checkpoint):
                    raise ValueError('Learning-curve evaluation differs from its locked checkpoint')
                read_policy_manifest(run/'manifest.json')
        if stage=='eval' and label.startswith('B2@') and checkpoint is None:
            raise ValueError('Checkpoint label requires a registered evaluation binding')
        resolve=lambda path:campaign_file(campaign,registration,path)
        provenance=training_evidence(config,summary,run,provenance_cache,fixed_source=fixed_source,
                                     checkpoint=checkpoint,resolve=resolve if checkpoint else None) if stage=='eval' else {}
        if stage=='eval' and not provenance['verified']:split='untrained_control'
        implicit_run='.' if run==directory else str(run.parent.relative_to(directory))
        comparison_run=summary.get('comparison_run_id') or config.get('comparison_run_id') or config.get('experiment_id') or implicit_run
        version = config.get('experiment_version', 'unversioned')+':'+config.get('composition', 'unknown')
        result_kind = summary.get('result_kind')
        found = {}
        if config and result_kind in (None, 'real_cage4') and not summary.get('unit_test_fixture_only'):
            found = {(e['seed'], e.get('episode_id', 0)): e for e in summary.get('episodes', []) if 'seed' in e}
            for path in sorted(run.glob('episode-????.json')):
                episode = load(path); key = (episode['seed'], episode.get('episode_id', 0))
                if key in found:
                    if found[key].get('joint_return') != episode.get('joint_return'):
                        raise ValueError(f'Episode summary disagrees with its completed record: {path}')
                    if found[key].get('unit_test_fixture_only'):
                        episode = {**episode, 'unit_test_fixture_only': True}
                found[key] = episode
        completed = 0
        interface=policy_interface(config);encoder=interface['communication_encoder']
        if ((recorded_config and policy_interface(recorded_config)!=interface)
                or (manifest.get('policy_interface') is not None and manifest['policy_interface']!=interface)):
            raise ValueError('Run policy interface differs from its manifest')
        for episode in found.values():
            if (episode.get('status') != 'completed' or episode.get('kind') != 'real_cage4'
                    or episode.get('unit_test_fixture_only')):
                continue
            if config.get('steps') == 500 and (episode.get('joint_transitions'), episode.get('scored_transitions')) != (499, 498):
                raise ValueError('Invalid official 500-step episode evidence')
            if episode.get('communication_encoder',LEGACY_ENCODER)!=encoder:
                raise ValueError('Episode communication encoder differs from its config')
            if episode.get('policy_interface') is not None and episode['policy_interface'] != interface:
                raise ValueError('Episode policy interface differs from its config')
            if evaluation_scope=='completed_update_curve' and episode['seed'] not in config.get('eval_seeds',[]):
                raise ValueError('Learning-curve result has an unregistered seed')
            if checkpoint and episode.get('input_pool_sha256')!=checkpoint['pool_sha256']:
                raise ValueError('Learning-curve episode did not use the registered Pool')
            identity = episode.get('model_identity') or summary.get('model_identity') or {}
            signature = json.dumps({
                'conditions': {k: config.get(k) for k in ('steps', 'composition', 'profile', 'scoring',
                    'communication', 'behavior_context', 'skip_busy_policy', 'case_presentation', 'connection_leads')},
                'model': {k: config.get('model', {}).get(k) for k in ('model', 'revision', 'temperature', 'max_tokens')},
                'identity': {k: identity.get(k) for k in ('model', 'revision', 'dtype', 'chat_template_sha256')},
                'keep_weights': {k: v.get('sha256') for k, v in (episode.get('weights') or {}).items()},
                'implementation':manifest.get('implementation_sha256'),
            }, sort_keys=True)
            comparison_conditions={'version':version,'red':config['red'],'split':split,
                'communication_encoder':encoder,'action_contract':interface['action_contract'],
                'evaluation_scope':evaluation_scope}
            group=episode_group(comparison_conditions)
            if group in signatures and signatures[group] != signature:
                raise ValueError('Comparison conditions differ within the same version/red/split')
            signatures[group] = signature
            controller_key = group + (label,)
            controller = {'contract':'upstream','urgent_preemption':False,**(config.get('selector') or {})}
            if controller_signatures.setdefault(controller_key, controller) != controller:
                raise ValueError('Controller conditions differ within the same comparison arm')
            protocol_key=group+(comparison_run,)
            if protocol_key in protocol_signatures and protocol_signatures[protocol_key]!=config.get('protocol_digest'):
                raise ValueError('Protocol conditions differ within a comparison run')
            protocol_signatures[protocol_key]=config.get('protocol_digest')
            shared_initial=summary.get('comparison_initial_pool_sha256')
            if label!='B0' and shared_initial:
                if group in initial_pools and initial_pools[group]!=shared_initial:
                    raise ValueError('Comparison initial Pools differ')
                initial_pools[group]=shared_initial
            policy_key=group+(comparison_run,label,provenance.get('training_run_id'))
            policy_value=summary.get('pool_sha256')
            if policy_key in policy_signatures and policy_signatures[policy_key]!=policy_value:
                raise ValueError('Different frozen Pools cannot be merged within one arm')
            policy_signatures[policy_key]=policy_value
            item = {k: episode[k] for k in ('seed', 'status', 'kind', 'joint_return', 'execution_time_s')}
            item.update(comparison_conditions,label=label,policy_interface=interface,
                        comparison_run_id=comparison_run,training_run_id=provenance.get('training_run_id'),
                        training_identity=provenance.get('training_identity'),
                        requires_training=provenance.get('requires_training',comparison_metadata(config)['requires_training']),
                        training_provenance_state=provenance.get('state'),training_origin=provenance.get('origin'),
                        holdout_role=seed_use or 'historical_or_unverified_holdout',
                        reward_components=episode.get('reward_components'),
                        scored_step_rewards=episode.get('scored_step_rewards'), model_identity=identity,
                        source=str(run.relative_to(directory)), episode_id=episode.get('episode_id', 0))
            if 'calls' in episode:
                item['inference_tokens']={field:sum(c.get(field,0) for c in episode['calls'])
                                          for field in ('input_tokens','output_tokens','actual_model_calls')}
            if stage=='train' and curve:
                offset=(config.get('continuation') or {}).get('iteration',0)
                local=next(k for k in range(1,config['iterations']+1) if
                           item['episode_id']<training_boundary(config['train_episodes'],config['iterations'],k))
                item.update(training_collection_iteration=offset+local,input_pool_sha256=episode.get('input_pool_sha256'))
            if checkpoint:
                item.update(base_arm='B2',checkpoint_iteration=checkpoint['iteration'],
                    checkpoint_sha256=checkpoint['checkpoint_sha256'],method_sha256=checkpoint['method_sha256'],
                    training_lineage_id=checkpoint['training_lineage_id'],pool_sha256=checkpoint['pool_sha256'])
            if (include_trace or include_trace_paths) and episode.get('trace'):
                path = Path(episode['trace'])
                if not path.is_absolute():
                    path = run/path
                path=resolve(path)
                if include_trace_paths:
                    item['trace'] = str(path)
                counts, flags, busy = Counter(), Counter(), 0
                if include_trace and path.is_file():
                    if checkpoint and file_sha256(path)!=episode.get('trace_sha256'):
                        raise ValueError('Checkpoint evaluation trace differs from its completed record')
                    visible_rows=[]
                    selection_counts,termination_counts=Counter(),Counter()
                    offered_stats={}
                    for line in path.open():
                        row = json.loads(line)
                        if row.get('policy') != 'llm':
                            continue
                        visible_rows.append(row)
                        if row.get('policy_invoked') and not row.get('parse_error'):
                            counts[(row.get('requested_action') or 'unknown').split()[0]] += 1
                        elif not row.get('policy_invoked'):
                            busy += 1
                        for key in ('repeated_analysis_same_evidence', 'duplicate_successful_decoy', 'sleep_with_open_case'):
                            flags[key] += bool((row.get('behavior_outcome') or {}).get(key))
                        controller=row.get('skill_controller') or {}
                        selection=controller.get('selection_audit')
                        if selection is not None:
                            selection_counts['decisions']+=1
                            selection_counts['gain_filter_veto']+=bool(selection.get('gain_filter_veto'))
                            selection_counts['none_selected']+=not selection.get('selected')
                            selection_counts['epsilon_exploration']+=selection.get('source')=='epsilon_exploration'
                            if selection.get('selected'):selection_counts['selected:'+selection['selected']]+=1
                            calls=selection.get('model_calls') or []
                            if calls:
                                choice=re.search(r'<choice>\s*(.*?)\s*</choice>',calls[-1].get('response',''),re.I|re.S)
                                if choice:selection_counts['nominated:'+choice[1].strip()]+=1
                            for skill in selection.get('offered_skills',[]):
                                offered_stats[skill['name']]={k:skill[k] for k in ('avg_gain','maturity') if k in skill}
                        termination=controller.get('termination_audit')
                        if termination:termination_counts[termination.get('status','unknown')]+=1
                        termination_counts['maximum_age']+=sum(e.get('reason')=='maximum_age' for e in controller.get('events',[]))
                    from research.attribution import response_opportunity_audit
                    audit=response_opportunity_audit(visible_rows)
                    item['response_opportunities']={k:v for k,v in audit.items() if k!='cases'}
                    item['llm_behavior'] = {'free_actions': dict(counts), 'automatic_busy_ticks': busy, 'observable_flags': dict(flags)}
                    item['llm_behavior'].update(selection=dict(selection_counts),termination=dict(termination_counts),offered_skill_stats=offered_stats)
            episodes.append(item); completed += 1
        statuses.append({'directory': str(run.relative_to(directory)), 'label': label, 'red': config.get('red'),
                         'stage': stage, 'status': summary.get('status', 'not_committed'),
                         'split': split, 'evaluation_role': role, 'result_kind': result_kind,
                         'completed_environment_episodes': completed, 'committed_iterations': len(summary.get('history', [])),
                         'optimizer_states':[h.get('optimizer_status','unknown') for h in summary.get('history',[])],
                         'pool_changes':sum(h.get('pool_changed',False) for h in summary.get('history',[])),
                         'failure':load(run/'failure.json') if (run/'failure.json').exists() else None})
        statuses[-1].update(comparison_run_id=comparison_run,training_evidence=provenance)
        for field in ('inherited_optimizer_iterations','segment_completed_iterations','total_committed_iterations',
                      'inherited_training_episodes','segment_training_episodes','total_training_episodes'):
            if field in summary:statuses[-1][field]=summary[field]
    return episodes, statuses



COMPARISONS=(('B1','B0'),('B2','B1'),('B2','B0'),('B4','B2'),('B4','B0'),
             ('B3-no-support','B2'),('B3','B3-no-support'),('B3','B2'),('B4','B3'),
             ('E1','E0'),('E2','E1'),('E2','E0'),('E3','E2'),('E3','E0'),
             *[(name,metadata['base']) for name,metadata in VERIFICATION_VARIANTS.items()],
             *[(name,'B1' if base=='B1' else 'B1'+name[2:]) for name,base in CONTROLLER_VARIANTS.items()])


def paired_comparisons(episodes, checkpoint_labels=()):
    groups=defaultdict(lambda:defaultdict(list))
    for e in episodes:groups[episode_group(e)][e['label']].append(e)
    output=[]
    for group,arms in sorted(groups.items()):
        curve=group[GROUP_FIELDS.index('evaluation_scope')]=='completed_update_curve'
        if curve:
            labels=checkpoint_labels or sorted((name for name in arms if name.startswith('B2@')),key=lambda x:int(x[3:]))
            comparisons=[('B1','B0')]
            for index,label in enumerate(labels):
                comparisons.extend((label,previous) for previous in ('B0','B1',*labels[:index]))
        else:
            comparisons=list(dict.fromkeys([*COMPARISONS,*[(name,'B0') for name in arms if name!='B0']]))
        for new,old in comparisons:
            keys,before,after=paired_episodes(arms.get(old,[]),arms.get(new,[]))
            if curve and new.startswith('B2@') and old.startswith('B2@') and any(
                    a.get('training_lineage_id')!=b.get('training_lineage_id') for a,b in zip(before,after)):
                raise ValueError('Learning-curve pairs must share their training lineage')
            seeds=[k[1] for k in keys];delta=[b['joint_return']-a['joint_return'] for a,b in zip(before,after)]
            reps={label:training_summary(items) for label,items in ((old,before),(new,after))}
            learned=[v for v in reps.values() if v['status']!='not_applicable']
            repetitions=min(v['count'] for v in learned) if learned and all(v['count'] is not None for v in learned) else None
            row={'comparison':new+'-'+old,**dict(zip(GROUP_FIELDS,group)),'paired_n':len(seeds),
                'seeds':seeds,'differences':delta,'mean_difference':statistics.mean(delta) if delta else None,
                'median_difference':statistics.median(delta) if delta else None,
                'win_tie_loss':[sum(d>0 for d in delta),sum(d==0 for d in delta),sum(d<0 for d in delta)] if delta else None,
                'paired_bootstrap_95ci':None,'reward_component_mean_differences':None,
                'paired_run_seeds':[{'comparison_run_id':k[0],'seed':k[1]} for k in keys],
                'training_repetitions':repetitions,'training_repetitions_by_arm':reps,
                'training_repetitions_status':'not_applicable' if not learned else 'unknown' if repetitions is None else 'verified_artifacts',
                'interpretation':'Paired environment seeds do not identify individual-action causation or independent training-run uncertainty.'}
            if len(delta)>1:
                clusters=defaultdict(list)
                for key,d in zip(keys,delta):clusters[key[0]].append(d)
                rng=random.Random(20261125)
                if len(clusters)>1:
                    blocks=list(clusters.values());means=sorted(statistics.mean(d for block in rng.choices(blocks,k=len(blocks)) for d in block) for _ in range(10000))
                    row['bootstrap_unit']='comparison_run_cluster'
                else:
                    means=sorted(statistics.mean(rng.choices(delta,k=len(delta))) for _ in range(10000));row['bootstrap_unit']='environment_seed_conditional_on_one_training_artifact'
                row['paired_bootstrap_95ci']=[means[249],means[9749]]
            if delta and all(e.get('reward_components') is not None for e in before+after):
                component_keys=set().union(*(e['reward_components'] for e in before+after))
                row['reward_component_mean_differences']={k:statistics.mean(b['reward_components'].get(k,0)-a['reward_components'].get(k,0) for a,b in zip(before,after)) for k in component_keys}
            output.append(row)
    if not groups:
        output=[{'comparison':new+'-'+old,'paired_n':0,'mean_difference':None,'median_difference':None,
                 'win_tie_loss':None,'paired_bootstrap_95ci':None,'reason':'No completed paired real test episodes'} for new,old in COMPARISONS]
    return output


def rollout_progress(validation, admission_mode=None):
    """Count measured pairs, including old results with checked reward differences."""
    if not validation:
        return {'rollout_pending':admission_mode=='rollout' or None,'rollout_completed':None,
                'rollout_status':'unrecorded' if admission_mode=='rollout' else 'not_applicable',
                'rollout_completed_pairs':None,'rollout_required_pairs':None}

    def valid_seed_list(values):
        if not isinstance(values,list) or any(type(seed) is not int or seed<0 for seed in values):
            return []
        return values if len(set(values))==len(values) else []

    registered=valid_seed_list(validation.get('registered_seeds',validation.get('seeds')))
    measured_field=next((field for field in ('completed_paired_seeds','paired_seeds') if field in validation),None)
    measured=valid_seed_list(validation.get(measured_field))
    required=validation.get('required_approval_pairs',len(registered) or None)
    if type(required) is not int or required<20:
        required=None
    differences=validation.get('differences')
    reason=validation.get('reason')
    full_result=reason in ('passed','failed_registered_criteria')
    if isinstance(differences,dict):
        recorded_deltas={str(k):value for k,value in differences.items()}
        if len(recorded_deltas)!=len(differences):
            recorded_deltas={}
    elif full_result and isinstance(differences,list) and len(differences)==len(registered):
        # Legacy compare_rollouts emits one delta per registered seed only after checking both episodes.
        if measured_field is None:
            measured=registered
        recorded_deltas=dict(zip(map(str,registered),differences))
    else:
        recorded_deltas={}
    complete_pairs={s for s in measured if (not registered or s in registered)
                    and type(recorded_deltas.get(str(s))) in (int,float)
                    and math.isfinite(recorded_deltas[str(s)])}
    completed=(full_result and validation.get('validation_complete',True) is True and
               required is not None and required==len(registered)==len(complete_pairs) and
               set(registered)==set(measured)==complete_pairs and set(recorded_deltas)==set(map(str,registered)))
    prefix=reason in ('failed_registered_worst_seed_early','reject_screen_inconclusive')
    stopped=reason=='failed_registered_worst_seed_early' or validation.get('early_approval_allowed') is False
    state='complete' if completed else 'reject_only' if prefix else 'partial' if complete_pairs else 'planned' if registered else 'unrecorded'
    return {'rollout_pending':not (completed or stopped),'rollout_completed':completed,
            'rollout_status':state,'rollout_completed_pairs':len(complete_pairs),
            'rollout_required_pairs':required}


def recorded_updates(directory, protocol, journal):
    previous=protocol.get('continuation_source')
    if previous:
        path=campaign_file(directory,protocol,previous['path'])
        if file_sha256(path)!=previous['checkpoint_sha256']:
            raise ValueError('Inherited optimizer history differs from the registered source')
        for entry in load(path)['history']:
            yield f"inherited:B2:{entry['iteration']}", {
                'context':{'arm':'B2'},'iteration':entry['iteration'],
                'status':entry['optimizer_status'],'result':entry}, True
    if journal:
        keys=[r[0] for r in journal.db.execute("SELECT key FROM checkpoints WHERE key LIKE 'optimizer:%'").fetchall() if r[0].count(':')==1]
        for key in keys:
            yield key,journal.restore(key),False


def next_batch_evidence(directory, statuses, change):
    matched=[run for run in statuses if run['stage']=='train' and run['label']==change['arm']]
    verified=[];active=Counter();requests=Counter();completions=Counter();complete=bool(matched)
    for run in matched:
        path=directory/run['directory'];manifest=path/'manifest.json'
        if not manifest.is_file():complete=False;continue
        cfg=load(manifest)['config'];seeds=split_seeds(cfg,'train');loops=cfg['iterations']
        local=change['iteration']-(cfg.get('continuation') or {}).get('iteration',0)
        if not 0<=local<loops:complete=False;continue
        first=training_boundary(len(seeds),loops,local)
        last=training_boundary(len(seeds),loops,local+1)
        for episode in range(first,last):
            trace=path/f'episode-{episode:04d}.jsonl';record_path=trace.with_suffix('.json')
            if not trace.is_file() or not record_path.is_file():complete=False;continue
            record=load(record_path)
            input_hash=record.get('input_pool_sha256')
            expected_pool=change.get('post_update_pool_sha256')
            if (record.get('status')!='completed' or record.get('kind')!='real_cage4'
                    or record.get('seed')!=seeds[episode] or file_sha256(trace)!=record.get('trace_sha256')
                    or (expected_pool and (input_hash is not None or cfg.get('continuation')) and input_hash!=expected_pool)):
                complete=False;continue
            verified.append({'seed':seeds[episode],'trace_sha256':record['trace_sha256'],'input_pool_sha256':input_hash})
            starts=set();completed=set()
            for line in trace.open():
                row=json.loads(line)
                if row.get('policy')!='llm':continue
                child=change['after']
                if row.get('skill_id')==child['name'] and row.get('skill_version')==child['version']:
                    for event in (row.get('skill_controller') or {}).get('events',[]):
                        active['activations']+=event.get('activated')==child['name'] and event.get('version')==child['version']
                    if row.get('policy_invoked') and row.get('request_applied') and not row.get('parse_error'):
                        active['free_decisions']+=1
                        action=(row.get('requested_action') or '').split()
                        if action:requests[action[0]]+=1;starts.add((row['agent'],row['step'],action[0]))
                outcome=(row.get('behavior_outcome') or {}).get('completed_action') or {}
                operation=(row['agent'],outcome.get('started_at'),outcome.get('name'))
                if operation in starts and operation not in completed and outcome.get('status') in ('TRUE','FALSE'):
                    completed.add(operation);completions[outcome['status']]+=1
    return {'verified_episodes':verified,'complete':complete,'activations':active['activations'],
            'free_decisions':active['free_decisions'],'requested_primitives':dict(requests),
            'observed_completion_status':dict(completions),
            'used':True if sum(active.values()) else False if complete else None}


def optimizer_report(directory,output,statuses):
    states=[];changes=[];diagnoses=[];tasks=[];usage=None;candidate_stages=[]
    protocol=load(directory/'protocol.json') if (directory/'protocol.json').is_file() else {}
    db=directory/'journal.sqlite3';journal=None
    if db.exists():
        journal=Journal(db);tasks=journal.tasks();usage=journal.budget_usage()
    for key,state,inherited in recorded_updates(directory,protocol,journal):
        result=state.get('result') or {}
        pool=result.get('pool_snapshot') or (state.get('final') or {}).get('pool')
        pool_hash=config_digest(pool) if pool else None
        arm=state.get('context',{}).get('arm')
        iteration=state.get('iteration')
        loop_candidates=[]
        for record in result.get('evolution',[]):
            for sg in record.get('semantic_gradients',[]):
                diagnoses.extend(sg.get('evidence_audits',[]))
            for candidate in record.get('candidates',[]):
                scored=candidate.get('score') is not None
                details={'loop_id':key,'arm':arm,'iteration':iteration,
                    'name':(candidate.get('skill') or {}).get('name'),
                    'planned':True,'collected':True,'grounded':(candidate.get('semantic_support') or {}).get('passed'),
                    'candidates_generated':bool(candidate.get('skill')),'scored':scored,
                    'ppo_passed':candidate['score']>0 if scored else None,
                    **rollout_progress(candidate.get('rollout_validation'),candidate.get('admission_mode')),
                    'committed_changed':bool(candidate.get('accepted')) and state['status'] in COMPLETED_OPTIMIZER_STATUSES,
                    'used_next_batch':None,'frozen_evaluated':None,
                    'admission_mode':candidate.get('admission_mode'),'efficacy_verified':candidate.get('efficacy_verified'),
                    'surrogate_diagnostics':candidate.get('surrogate_diagnostics'),
                    'unknown_is_not_zero':True}
                candidate_stages.append(details);loop_candidates.append(details)
                if details['committed_changed']:
                    changes.append({'loop_id':key,'arm':arm,'iteration':iteration,
                        'parent':record['old_skill'],'after':candidate['skill'],
                        'semantic_gradients':record.get('semantic_gradients',[]),'gate':candidate.get('diagnostics'),
                        'surrogate_diagnostics':candidate.get('surrogate_diagnostics'),
                        'post_update_pool_sha256':pool_hash,'inherited':inherited,
                        'rollout':candidate.get('rollout_validation'),'next_loop_used':None})
        states.append({'loop_id':key,'arm':arm,'iteration':iteration,
                       'status':state['status'],'pool_changed':result.get('pool_changed'),
                       'attempts':state.get('attempts'),'error':state.get('error'),'inherited':inherited,
                       'optimizer_seconds':result.get('seconds'),'pool_size':result.get('pool_size'),
                       'pool_sha256':pool_hash,
                       'candidate_counts':{field:sum(bool(c[field]) for c in loop_candidates) for field in
                                           ('candidates_generated','scored','ppo_passed','committed_changed')} if result else None})
    for change in changes:
        evidence=next_batch_evidence(directory,statuses,change)
        change['next_loop_used']=evidence['used']
        change['next_batch']=evidence
        change['next_loop_note']='Verified next-batch seeds, trace hashes and child version; absent experience is unknown.'
        for item in candidate_stages:
            if item['loop_id']==change['loop_id'] and item['name']==change['after']['name']:item['used_next_batch']=change['next_loop_used']
    atomic_json(output/'optimizer_progress.json',{'loops':states,'environment_runs':statuses,'tasks':tasks,
        'candidate_stages':candidate_stages,
        'resource_phases':journal.restore('resource_phases') if db.exists() else None,
        'budget':usage,'unknown_is_not_zero':True,
        'completed_optimizer_loops':sum(s['status'].startswith('completed_') for s in states),
        'inherited_optimizer_loops':sum(s['inherited'] for s in states),
        'segment_completed_optimizer_loops':sum(not s['inherited'] and s['status'].startswith('completed_') for s in states),
        'pool_change_loops':sum(s['pool_changed'] is True for s in states),
        'phase_costs':'See bottleneck_report; GPU intervals include startup/idle/cleanup, test episode seconds remain separate.'})
    atomic_json(output/'skill_changes.json',{'committed_changes':changes,'accepted_count':len(changes) if states and all(s['status'].startswith('completed_') for s in states) else None})
    path=output/'diagnosis_evaluation.json'
    if not path.exists():
        atomic_json(path,{'real_sg_audits':diagnoses,'labeled_real_cases':None,'false_accept_rate':None,'false_reject_rate':None,
            'reason':'No labeled real records available for rate estimation; offline contract tests are not real-Qwen diagnosis accuracy.',
            'limitation':'Restricted typed facts/proposal links are checked; general language truth, causation and defense improvement are not established.'})
    resume=output/'resume_verification.json'
    if not resume.exists():atomic_json(resume,{'actual_qwen_interruption_matrix':'not_run','offline_contract_verification':'See recorded test results',
        'reusable':'Completed exact requests, SG validation, candidates, per-sample scores, completed rollout seeds and committed Pool/baseline state',
        'replayed':'CPU control flow from the registered loop base; RNG/counter restored. Unknown requests may need an explicitly recorded retry.',
        'prior_training_recovery':'Unavailable without original full traces and complete training checkpoint; initial_pool is static.'})


def reproduction_bundle(directory,output):
    """Archive exact inputs, including hash-addressed historical dependencies."""
    selected=[p for p in directory.rglob('*') if p.is_file() and output not in p.parents
              and p.suffix in ('.json','.jsonl','.log','.yaml','.sqlite3')]
    # WAL-backed SQLite requires a consistent backup, not a blind copy of its main file.
    selected += [p for p in output.rglob('*') if p.is_file() and p.suffix in ('.json','.md','.xml')
                 and p.name!='reproducibility_manifest.json']
    archive=output/'reproduction_bundle.tar.gz';temporary=output/'reproduction_bundle.tmp.gz';manifest=[];by_hash={};by_name={};dependencies={}
    protocol=load(directory/'protocol.json') if (directory/'protocol.json').exists() else {}
    continuation=protocol.get('continuation_source')
    source_info=continuation or protocol.get('fixed_update_source') or {}
    required=source_info.get('files_sha256',{})
    def dependency_path(value):
        p=Path(value);return p if p.is_absolute() else directory/p
    if continuation:
        referenced=[continuation['path'],continuation['source_protocol'],continuation['training_manifest']['path']]
    else:
        referenced=[source_info[k] for k in ('checkpoint','journal','initial_pool','training_manifest') if source_info.get(k)]+source_info.get('training_traces',[])
    if any(str(p) not in required for p in referenced):raise ValueError('Historical dependency lacks a registered file hash')
    for path,expected in required.items():
        if not dependency_path(path).is_file() or file_sha256(dependency_path(path))!=expected:
            raise ValueError('Historical dependency missing or changed: '+path)
    with tempfile.TemporaryDirectory(prefix='lab-bundle-') as temp, tarfile.open(temporary,'w:gz') as bundle:
        def add(name,raw,kind,**metadata):
            sha=hashlib.sha256(raw).hexdigest();info=tarfile.TarInfo(name)
            if name in by_name:
                if by_name[name]!=sha:raise ValueError('Bundle path collision: '+name)
                return sha
            by_name[name]=sha
            if sha in by_hash:info.type=tarfile.LNKTYPE;info.linkname=by_hash[sha];bundle.addfile(info)
            else:
                info.size=len(raw);bundle.addfile(info,io.BytesIO(raw));by_hash[sha]=name
            manifest.append({'path':name,'sha256':sha,'bytes':len(raw),'kind':kind,**metadata})
            return sha
        def contents(path,name):
            if path.suffix!='.sqlite3':return path.read_bytes()
            snapshot=Path(temp)/(hashlib.sha256(name.encode()).hexdigest()+'.sqlite3')
            with sqlite3.connect('file:'+str(path.resolve())+'?mode=ro',uri=True) as db,sqlite3.connect(snapshot) as backup:
                db.backup(backup)
            return snapshot.read_bytes()
        for path in sorted(selected):
            name=str(path.relative_to(directory))
            add(name,contents(path,name),'full_trace' if path.suffix=='.jsonl' else 'durable_journal' if path.suffix=='.sqlite3' else 'checkpoint_or_metadata')
        source_paths=sorted(list((ROOT/'research').glob('*.py'))+list((ROOT/'acd').glob('*.py'))+
            list((ROOT/'scripts').glob('*.py'))+list((ROOT/'scripts').glob('*.sh'))+
            list((ROOT/'research_configs').glob('*.yaml'))+list(ROOT.glob('*lock*'))+
            [ROOT/'research.sh',ROOT/'artifacts/initial_pool.json'])
        for source in source_paths:
            if not source.is_file():continue
            name='checkout/'+str(source.relative_to(ROOT));raw=source.read_bytes()
            add(name,raw,'reproduction_source_or_lock')
        for old,expected in required.items():
            path=dependency_path(old);name='historical_inputs/'+expected+'/'+path.name
            raw=contents(path,name);sha=add(name,raw,'historical_frozen_input',source_path=old,source_file_sha256=expected)
            dependencies[old]={'relative_path':name,'bundle_sha256':sha,'source_file_sha256':expected,
                'sqlite_consistent_backup':path.suffix=='.sqlite3'}
        embedded={'schema':'reproduction-bundle-v2','members':deepcopy(manifest),'historical_dependencies':dependencies,
            'own_manifest_hash_excluded':True,'weights_included':False}
        add('bundle_manifest.json',json.dumps(embedded,ensure_ascii=False,indent=2).encode(),'embedded_manifest')
    import os
    os.replace(temporary,archive)
    atomic_json(output/'reproducibility_manifest.json',{'bundle':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'members':manifest,'historical_dependencies':dependencies,'schema':'reproduction-bundle-v2',
        'compact_metrics':'episodes.json does not contain exact actor prompts, full raw observations, RNG or optimizer responses.',
        'source_model_locks':'Use protocol.json hashes and the pinned root checkout; weights are excluded. No external upload or Git action performed.'})


def restore_reproduction_bundle(archive,destination):
    """Restore into an empty directory; original absolute paths are provenance only."""
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    if any(destination.iterdir()):raise ValueError('Restore destination must be empty')
    with tarfile.open(archive,'r:gz') as bundle:
        members=bundle.getmembers();names={m.name for m in members}
        def safe(name):return bool(name) and not Path(name).is_absolute() and '..' not in Path(name).parts
        if len(names)!=len(members) or any(not safe(m.name) or not (m.isfile() or m.isdir() or m.islnk()) or
            (m.islnk() and (not safe(m.linkname) or m.linkname not in names)) for m in members):
            raise ValueError('Unsafe or duplicate bundle member')
        with bundle.extractfile('bundle_manifest.json') as stream:manifest=json.load(stream)
        bundle.extractall(destination)
    for item in manifest['members']:
        path=destination/item['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Restored file hash mismatch: '+item['path'])
    path=destination/'protocol.json';protocol=load(path) if path.exists() else {}
    original=protocol.get('fixed_update_source')
    if original:
        mapping=manifest['historical_dependencies']
        def remap(value):
            if isinstance(value,str):return mapping[value]['relative_path'] if value in mapping else value
            if isinstance(value,list):return [remap(v) for v in value]
            if isinstance(value,dict):return {k:remap(v) for k,v in value.items() if k!='files_sha256'}
            return value
        source=remap(original);source['files_sha256']={v['relative_path']:v['bundle_sha256'] for v in mapping.values()}
        protocol.update(original_fixed_update_source=original,fixed_update_source=source,restored_relative_paths=True)
    atomic_json(destination/'restored_protocol.json',protocol)
    return protocol

def report_experiment(directory, output=None, include_trace=False, include_bundle=True, *, include_legacy_tables=False):
    directory = Path(directory).resolve()
    output = Path(output).resolve() if output else directory/'report'
    protocol=load(directory/'protocol.json') if (directory/'protocol.json').exists() else {}
    checkpoint_labels=list(protocol.get('checkpoint_evaluations',{}))
    episodes, statuses = collect(directory, include_trace, include_trace_paths=include_legacy_tables)
    metrics = recompute(episodes)
    metrics['paired_comparisons']=paired_comparisons(episodes,checkpoint_labels)
    for pair in metrics['paired_vs_B0']:
        checked=next(p for p in metrics['paired_comparisons'] if p['comparison']==pair['label']+'-B0' and
            episode_group(p)==episode_group(pair))
        for key in ('median_difference','paired_bootstrap_95ci','bootstrap_unit'):
            pair[key]=checked.get(key)
    for group in metrics['groups']:
        group['mean_execution_time_s']=group['total_execution_time_s']/group['n']
        items=[e for e in episodes if e['label']==group['label'] and episode_group(e)==episode_group(group)]
        group['training_repetitions']=training_summary(items)
        if all('inference_tokens' in e for e in items):
            group['inference_tokens']={field:sum(e['inference_tokens'][field] for e in items)
                                       for field in ('input_tokens','output_tokens','actual_model_calls')}
        if all(e.get('reward_components') is not None for e in items):
            components=set().union(*(e['reward_components'] for e in items))
            group['mean_reward_components']={k:statistics.mean(e['reward_components'].get(k,0) for e in items) for k in components}
        else:group['mean_reward_components']=None
    decisions = []
    for path in sorted(directory.rglob('decision.json')):
        if 'rollout-verification' in path.relative_to(directory).parts:
            decisions.append({'source': str(path.relative_to(directory)),
                              **{k: v for k, v in load(path).items() if k != 'verified_pool'}})
    for decision in decisions:
        decision.update(rollout_progress(decision))
    output.mkdir(parents=True, exist_ok=True)
    compact_episodes = [{k:v for k,v in episode.items() if k!='trace'} for episode in episodes] if include_legacy_tables else episodes
    (output/'episodes.json').write_text(json.dumps({'schema': 'compact-episode-metrics-v1', 'episodes': compact_episodes}, ensure_ascii=False, indent=2)+'\n')
    (output/'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2)+'\n')
    (output/'run_status.json').write_text(json.dumps(statuses, ensure_ascii=False, indent=2)+'\n')
    (output/'rollout_decisions.json').write_text(json.dumps(decisions, ensure_ascii=False, indent=2)+'\n')
    fmt = lambda value: 'N/A' if value is None else f'{value:.2f}' if isinstance(value, float) else str(value)
    lines = ['# Experiment comparison', '',
             'Completed real CAGE4 episodes are grouped by train, test, transfer, development, reproduction and untrained controls. Candidate-validation episodes are excluded from final-test metrics.', '',
             '| Version | Red | Split | Communication encoder | Action contract | Evaluation scope | Arm | N | Mean reward | Sample SD | Total seconds | Paired gain vs B0 | Win rate | Median gain | 95% bootstrap CI |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    cases = sorted({episode_group(g) for g in metrics['groups']})
    for case in cases:
        groups = {g['label']: g for g in metrics['groups'] if episode_group(g)==case}
        paired = {g['label']: g for g in metrics['paired_vs_B0'] if episode_group(g)==case}
        displayed=('B0','B1',*checkpoint_labels) if checkpoint_labels else ('B0','B1','B2','B3','B4')
        for label in sorted(set(groups)|set(displayed)):
            group, pair = groups.get(label, {}), paired.get(label, {})
            row = [*case, label, group.get('n', 0), group.get('mean'), group.get('sample_std'), group.get('total_execution_time_s'),
                   pair.get('mean_difference'), pair.get('win_rate'), pair.get('median_difference'), pair.get('paired_bootstrap_95ci')]
            lines.append('| '+' | '.join(fmt(v).replace('|', '/') for v in row)+' |')
    if not cases:
        for label in (('B0','B1',*checkpoint_labels) if checkpoint_labels else ('B0','B1','B2','B3-no-support','B3','B4')):
            lines.append('| registered | finite | test | N/A | N/A | test | '+label+' | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |')
        lines += ['', 'No completed real episodes. Mock or partial runs are not performance samples.']
    lines += ['', 'Small samples and bootstrap intervals do not establish a general performance gain. Environment completion and committed optimizer iterations are reported separately in run_status.json.', '']
    lines += ['', '| Comparison | Version | Red | Split | Communication encoder | Action contract | Evaluation scope | Paired N | Mean gain | Median | Win/tie/loss | 95% CI |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for pair in metrics['paired_comparisons']:
        lines.append('| '+' | '.join(fmt(pair.get(k)).replace('|','/') for k in
            ('comparison',*GROUP_FIELDS,'paired_n','mean_difference','median_difference','win_tie_loss','paired_bootstrap_95ci'))+' |')
    reference=protocol.get('historical_performance_reference')
    if reference:
        raw=Path(reference['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=reference['sha256']:raise ValueError('Historical performance reference changed')
        previous=json.loads(raw)
        atomic_json(output/'historical_metrics_reference.json',{'reference':reference,'metrics':previous})
        lines+=['','## Historical context (not new test samples)','',reference['role'],
            'Frozen performance source: '+str(reference.get('source_commit')),
            '| Red | Arm | N | Mean reward | Sample SD | Mean episode seconds |',
            '|---|---|---|---|---|---|']
        for g in previous.get('groups',[]):
            lines.append('| '+' | '.join(fmt(g.get(k)) for k in ('red','label','n','mean','sample_std','mean_execution_time_s'))+' |')
    if protocol.get('protocol_kind')=='historical-diagnosis-probes-v1':
        probe=load(output/'sg_boundary_profile.json') if (output/'sg_boundary_profile.json').exists() else {}
        lines+=['','## Real Qwen diagnosis probes on historical training','',
            'Complete original SG chunks, frozen BF16 weights. No new CAGE4 episode, optimizer loop, candidate generation, PPO approval or rollout admission is claimed.',
            'Feature-selected outputs are not a population estimate of natural diagnosis accuracy or defense performance. Correct abstention is separate from a grounded proposal.','',
            '| Stage | Observed count |','|---|---|']
        for stage,value in probe.get('stage_counts',{}).items():lines.append('| '+stage+' | '+fmt(value)+' |')
        lines+=['','[Selection plan](diagnosis_probe_plan.json), [raw responses and checks](sg_boundary_profile.json).']
    if protocol.get('protocol_kind')=='fixed-committed-update-verification-v1':
        lines+=['','## Fixed committed-update verification','',
            'B2 uses the exact historical completed update, evaluated on freshly registered test seeds. No new training loop is claimed. B3/B4 full training and 20-pair approval are N/A.',
            'The ASV/SSV ablations below screen one fixed historical whole update; they are not trained B3/B4 arms. A retained initial pool can reuse the matching B1 policy measurement, with zero additional episodes.','']
        for filename in ('action_support.json','selector_probe.json','fixed_update_rollout.json','fixed_update_ablations.json',
                         'scorer_parity.json','sg_boundary_profile.json','guard_policy_reuse.json','historical_forensics.json'):
            if (output/filename).exists():lines.append(f'- [{filename}]({filename})')
        screen=load(output/'fixed_update_rollout.json') if (output/'fixed_update_rollout.json').exists() else {}
        if screen:lines+=['',f"Reject-only validation: {screen.get('reason')}; paired seeds={screen.get('paired_seeds')}; differences={screen.get('differences')}. Partial validation never approves."]
    path=output/'performance.md';marker='<!-- generated-comparison:start -->'
    previous=path.read_text() if path.exists() else ''
    if marker in previous:preface=previous.split(marker,1)[0]
    elif '# Experiment comparison\n' in previous:preface=previous.split('# Experiment comparison\n',1)[0]
    else:preface=previous+'\n\n' if previous else ''
    path.write_text(preface+marker+'\n'+'\n'.join(lines))
    optimizer_report(directory,output,statuses)
    from scripts.audit_training_cost import audit
    audit(directory)
    bundle_action='not_requested'
    if include_bundle:
        mismatch=any(not (ROOT/name).is_file() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected
            for name,expected in protocol.get('source_sha256',{}).items())
        if mismatch:
            archive=output/'reproduction_bundle.tar.gz';manifest=output/'reproducibility_manifest.json'
            if not archive.is_file() or not manifest.is_file() or hashlib.sha256(archive.read_bytes()).hexdigest()!=load(manifest).get('sha256'):
                raise ValueError('Registered source differs; an exact immutable source bundle is required, never snapshot the current source as historical evidence')
            bundle_action='preserved_registered_source_snapshot'
        else:
            reproduction_bundle(directory,output);bundle_action='created_from_registered_source'
    if include_legacy_tables:
        from research.reporting import report
        report(directory,collected=(episodes,statuses),output=output)
    return {'episodes': len(episodes), 'runs': len(statuses), 'rollout_decisions': len(decisions), 'output': str(output),'bundle_action':bundle_action}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path,nargs='?')
    parser.add_argument('--restore-bundle',type=Path,help='Restore a v2 bundle into an empty --destination')
    parser.add_argument('--destination',type=Path)
    parser.add_argument('--summary-only',action='store_true',help='Export compact metrics/status without copying raw inputs or weights')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--reasoning', action='store_true', help='Also read available raw traces for Qwen-only behavior counts')
    args = parser.parse_args()
    if args.restore_bundle:
        if not args.destination:parser.error('--restore-bundle requires --destination')
        restore_reproduction_bundle(args.restore_bundle,args.destination)
        print(json.dumps({'restored':str(args.destination)}))
    else:
        if args.directory is None:parser.error('directory is required')
        print(json.dumps(report_experiment(args.directory, args.output, args.reasoning,not args.summary_only)))
