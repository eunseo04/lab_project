"""Audit real records/journal; measured costs and planned requests stay distinct."""
from collections import Counter,defaultdict
from copy import deepcopy
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.progress import Journal,atomic_json,digest


def distribution(values):
    values=sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:return {'n':0,'p50':None,'p95':None,'minimum':None,'maximum':None}
    # Empirical nearest-rank percentile, useful even when the sample is too small for inference.
    return {'n':len(values),'p50':statistics.median(values),'p95':values[math.ceil(.95*len(values))-1],
            'minimum':values[0],'maximum':values[-1]}


def render_sg(config,pool_snapshot,experiences):
    """Use the actual Evolution renderer, without any LLM call or changed chunk boundary."""
    from research.evolution import Evolution
    from research.skills import SecuritySkillPool
    from research.attribution import support_statistics
    from research.upstream import Experience
    class Capture:
        def __init__(self):self.calls=[];self.evidence_rows=[]
        def capture_evidence_rows(self,rows):self.evidence_rows.append(deepcopy(rows))
        def __call__(self,prompt):
            self.calls.append(prompt)
            return '{"is_related":false,"semantic_gradient":{"initiation":"","policy":"","termination":""}}'
    model=Capture();evolution=Evolution(model,config['evolution'])
    pool=SecuritySkillPool(config['pool_size'],seed_set=config['skill_seed_set']);pool.restore(pool_snapshot)
    by_name={s.name:s for s in pool.skills};output=[]
    grouped=defaultdict(list)
    for exp in experiences:grouped[exp.skill].append(exp)
    for parent in pool.skills:
        supported=[]
        for exp in grouped[parent.name]:
            valid=[t for t in exp.transitions if t.get('policy_invoked') and t.get('request_applied') and not t.get('parse_error') and t.get('token_ids')]
            if valid:supported.append(Experience(**{**exp.__dict__,'transitions':valid}))
        eligible=support_statistics(supported)['independent_episode_count']>=config['evolution'].get('minimum_independent_episodes',0)
        for exp in grouped[parent.name]:
            offset=len(model.calls);gradient_offset=len(evolution.gradients)
            evolution._generate_semantic_gradient(parent,exp)
            for index,prompt in enumerate(model.calls[offset:]):
                steps=evolution.gradients[gradient_offset+index]['steps']
                output.append({'skill':parent.name,'chunk':index,'prompt':prompt,'eligible':eligible,
                    'evidence_rows':model.evidence_rows[offset+index] if config['evolution'].get('evidence_verification') else [t for t in exp.transitions if t['step'] in steps],
                    'trace_sha256':digest(exp.__dict__),'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()})
    return output,by_name


def journal_cost(journal):
    tasks=journal.tasks();physical=[t for t in tasks if t['id'].startswith('server:')]
    stages={}
    for stage in dict.fromkeys(['acting','selection','termination','sg_chunk','aggregation_candidate','old_score','new_score','rollout','frozen_test']+[t['kind'] for t in physical]):
        rows=[t for t in physical if t['kind']==stage]
        if stage in ('rollout','frozen_test'):
            rows=[t for t in tasks if t['kind']==stage and not t['id'].startswith('server:')]
        complete=[t for t in rows if t['status']=='completed']
        stages[stage]={'states':dict(Counter(t['status'] for t in rows)),
            'completed_physical_operations':len(complete) if tasks else None,
            'input_tokens':sum((t['metrics'] or {}).get('input_tokens',0) for t in complete) if complete else None,
            'output_tokens':sum((t['metrics'] or {}).get('output_tokens',0) for t in complete) if complete else None,
            'target_tokens':sum((t['metrics'] or {}).get('target_tokens',0) for t in complete) if complete else None,
            'processing_seconds':distribution([(t['metrics'] or {}).get('processing_seconds') for t in complete]),
            'queue_wait_seconds':distribution([(t['metrics'] or {}).get('queue_wait_seconds') for t in complete]),
            'peak_allocated_bytes':distribution([(t['metrics'] or {}).get('peak_allocated_bytes') for t in complete])}
    events=journal.events()
    phases={phase:sum(t['status']=='completed' for t in physical if t['context'].get('env_stage')==phase)
            for phase in ('collection','profile','rollout','frozen_test')}
    return {'stages':stages,'completed_operations_by_environment_phase':phases if physical else None,
        'actual_model_call_count':sum(t['status']=='completed' for t in physical) if physical else None,
        'incomplete_physical_requests':dict(Counter(t['status'] for t in physical if t['status']!='completed')),
        'cache_hit_events':sum(bool((e['metrics'] or {}).get('cache_hit')) for e in events),
        'aggregation_and_candidate_generation':'Two logical stages in one original Skill-Pro request; never double count.',
        'physical_operation_definition':'One completed server generation or reference-scored sample. Forward calls are a separate timing metric; unknown requests are not fabricated.'}


def audit(directory):
    directory=Path(directory).resolve();records=[];field_sizes=Counter();field_seen=defaultdict(Counter)
    for run in sorted(directory.glob('runs/*-train')):
        config_path=run/'config.json'
        if not config_path.exists():continue
        cfg=json.loads(config_path.read_text());label=cfg.get('comparison_id',cfg['arm'])
        for trace in sorted(run.glob('episode-????.jsonl')):
            if not trace.with_suffix('.json').exists():continue
            grouped=defaultdict(list)
            for line in trace.open():
                row=json.loads(line)
                if row.get('policy')!='llm' or not row.get('skill_id'):continue
                grouped[(row['agent'],row['skill_id'])].append(row)
                for key,value in row.items():
                    raw=json.dumps(value,ensure_ascii=False);field_sizes[key]+=len(raw);field_seen[key][hashlib.sha256(raw.encode()).hexdigest()]+=1
            from research.evolution import semantic_chunks
            from research.upstream import Experience
            for (agent,skill),rows in grouped.items():
                experience=Experience(reward=0,skill=skill,trajectory='',transitions=rows,env_name=cfg['composition']+':'+cfg['red'])
                planned=semantic_chunks(experience,cfg['evolution'])
                chunks=len(planned);oversized=[];characters=0
                for chunk in planned:
                    for row,record in chunk:
                        size=len(json.dumps(record,ensure_ascii=False));characters+=size
                        if size>cfg['evolution']['semantic_chunk_chars']:oversized.append({'step':row['step'],'characters':size})
                records.append({'label':label,'agent':agent,'skill':skill,'seed':rows[0]['seed'],'trace':str(trace),
                    'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),'record_characters':characters,
                    'planned_semantic_gradient_chunks':chunks,'single_records_over_character_budget':oversized,
                    'valid_policy_decisions':sum(bool(t.get('policy_invoked') and t.get('request_applied') and not t.get('parse_error') and t.get('token_ids')) for t in rows)})
    db=directory/'journal.sqlite3'
    measured=journal_cost(Journal(db)) if db.exists() else {'actual_model_call_count':None,'stages':{},'cache_hit_events':None}
    report={'directory':str(directory),'records':records,**measured,
        'field_characters':dict(field_sizes),'field_repeated_value_fraction':{k:1-len(c)/sum(c.values()) for k,c in field_seen.items()},
        'characters_are_not_tokens':True,'context_overflow':'Only a recorded tokenizer+reservation check or error establishes actual overflow; character sizes alone do not.',
        'complexity':'Repeated values are measured; no quadratic-complexity conclusion follows from this audit.',
        'baseline_comparison':{'prior_raw_records_available':False,'before_after_throughput':'N/A without comparable baseline measurement'},
        'new_performance_samples':0}
    atomic_json(directory/'report/bottleneck_report.json',report)
    return report


def profile_diagnosis(directory,protocol,client,pending,preparation,time_available,maximum_probes=3,cover_features=False,required=False):
    """Real SG probes on complete original chunks; never a shortened optimizer loop."""
    from research.upstream import Experience,OriginalEvolution
    from research.semantic_evidence import EvidenceModel
    from research.diagnosis import canonical_operations
    from research.client import stable_seed
    directory=Path(directory);deadline=time.monotonic()+min(protocol.get('maximum_profile_seconds',900),time_available()-90)
    rendered,_=render_sg(protocol['configs']['B3'],preparation['ready']['pool'],[Experience(**e) for e in pending['experiences']])
    eligible=sorted((r for r in rendered if r['eligible']),key=lambda r:(len(r['prompt']),r['prompt_sha256']))
    features=[]
    for item in eligible:
        kinds=set()
        for row in item['evidence_rows']:
            ops=canonical_operations(row,item['evidence_rows']);decision=ops['requested_decision'];op=ops['completed_or_pending_operation']
            if decision['name']=='Sleep' and decision['request_applied']:kinds.add('current_sleep')
            if decision['target']:kinds.add('host_decision')
            if op.get('name')=='Restore' and op.get('completed_at') is not None:kinds.add('previous_restore')
            if op.get('name')=='Remove' and op.get('completed_at') is not None:kinds.add('previous_remove')
            if not decision['policy_invoked'] or op.get('status')=='IN_PROGRESS':kinds.add('busy')
            if op['restore_epoch_after']>=0:kinds.add('restore_epoch')
            for body in (row.get('observation_after') or {}).values():
                if isinstance(body,dict) and any(isinstance(p,dict) and 'PID' in p and 'username' not in p for p in body.get('Processes',[])):kinds.add('observed_pid')
        features.append(kinds)
    chosen=[];covered=set();used_skills=set()
    if cover_features:
        for _ in range(min(maximum_probes,len(eligible))):
            remaining=[i for i in range(len(eligible)) if i not in chosen]
            index=max(remaining,key=lambda i:(len(features[i]-covered),eligible[i]['skill'] not in used_skills,-len(eligible[i]['prompt']),eligible[i]['prompt_sha256']))
            chosen.append(index);covered.update(features[index]);used_skills.add(eligible[index]['skill'])
    elif eligible:chosen=sorted({0,len(eligible)//2,len(eligible)-1})[:maximum_probes]
    plan={'scope':'Historical training only; full original chunks; deterministic feature coverage or length cost sampling',
        'registered_before_generation':True,'expected_probes':len(chosen),'feature_coverage':sorted(set().union(*(features[i] for i in chosen)) if chosen else set()),
        'selected':[{'index':i,'skill':eligible[i]['skill'],'prompt_sha256':eligible[i]['prompt_sha256'],'features':sorted(features[i])} for i in chosen],
        'model_identity':client.identity,'new_optimizer_iterations':0,'natural_accuracy_population_estimate':None}
    atomic_json(directory/'report/diagnosis_probe_plan.json',plan)
    result={'schema':'diagnosis-boundary-probes-v2','training_data_only':True,'optimizer_ran':False,'profiles':[],'status':'not_started',
        'source_sha256':protocol.get('source_sha256'),
        'expected_probes':len(chosen),'sampling':plan['scope'],'model_identity':client.identity,
        'candidate_generation':None,'ppo_approval':None,'whole_pool_rollout_admission':None,
        'limitation':'Observation/format/link checks on selected SG outputs are not general diagnosis accuracy, causation or defense reward.'}
    failure=None
    try:
        for i in chosen:
            if time.monotonic()>=deadline:raise TimeoutError('Registered probe allowance exhausted')
            item=eligible[i];messages=[{'role':'user','content':item['prompt']}];tokens=client._tokens(messages)
            if tokens['prompt_tokens']+768>tokens['max_context']:raise ValueError('Actual complete SG input exceeds context; no truncation')
            began=time.perf_counter()
            with client.context(stage='diagnosis_boundary_probe',skill=item['skill'],prompt_sha256=item['prompt_sha256']):
                response=client.generate(messages,seed=stable_seed(protocol.get('experiment_id'),i,item['prompt_sha256']),purpose='evolution',max_tokens=768)
            boundary=EvidenceModel(lambda _:response['text'],item['evidence_rows'],diagnosis_verification=True, schema=protocol['configs']['B3']['evolution'].get('evidence_schema','typed_claims_v8'))
            validation_error=None
            try:
                canonical=json.loads(boundary('Replay only this exact newly generated response for deterministic checks'))
                audits=boundary.audits;audit=audits[-1]
            except (TypeError,ValueError,KeyError,AttributeError) as exc:
                validation_error=f'{type(exc).__name__}: {exc}'
                canonical={'is_related':False,'semantic_gradient':{}}
                audit={'passed':False,'raw_response':response['text'],'validation_error':validation_error};audits=[audit]
            parsed=OriginalEvolution._parse_json_safely(None,response['text'])
            gradient=parsed.get('semantic_gradient') if isinstance(parsed,dict) else None
            abstained=isinstance(parsed,dict) and parsed.get('is_related') is False and isinstance(gradient,dict) and not any(gradient.values()) and not parsed.get('proposals') and not parsed.get('claims') and not validation_error
            result['profiles'].append({'skill':item['skill'],'chunk':item['chunk'],'prompt_sha256':item['prompt_sha256'],
                'source_trace_sha256':item['trace_sha256'],'features':sorted(features[i]),'input_tokens':tokens['prompt_tokens'],
                'output_tokens':response['output_tokens'],'seconds':time.perf_counter()-began,'canonical':canonical,
                'audits':audits,'format_valid':isinstance(parsed,dict),'validation_tool_error':validation_error,
                'references_valid':bool(audit.get('valid_references')) and not audit.get('errors'),
                'observation_claims_consistent':audit.get('diagnosis_validation',{}).get('passed'),
                'proposal_links_valid':audit.get('structured_proposals',{}).get('passed'),
                'decision':'grounded_proposal' if audit['passed'] else 'abstained' if abstained else 'blocked_unverified_diagnosis',
                'raw_unverified_prose_used_for_generation':False})
            atomic_json(directory/'report/sg_boundary_profile.json',result)
        result['status']='measured'
    except Exception as exc:
        failure=exc;result.update(status='incomplete',error=f'{type(exc).__name__}: {exc}')
    result['stage_counts']={'generated':len(result['profiles']),
        'format_valid':sum(r['format_valid'] for r in result['profiles']),
        'validation_tool_errors':sum(r['validation_tool_error'] is not None for r in result['profiles']),
        'references_valid':sum(r['references_valid'] for r in result['profiles']),
        'observation_claims_consistent':sum(bool(r['observation_claims_consistent']) for r in result['profiles']),
        'proposal_links_valid':sum(bool(r['proposal_links_valid']) for r in result['profiles']),
        'grounded_proposals':sum(r['decision']=='grounded_proposal' for r in result['profiles']),
        'abstentions':sum(r['decision']=='abstained' for r in result['profiles']),
        'candidate_generation':None,'ppo_approval':None,'rollout_admission':None}
    atomic_json(directory/'report/sg_boundary_profile.json',result)
    if required and result['status']!='measured':raise failure or RuntimeError('Required diagnosis probe incomplete')
    return result


def profile_fixed_update(directory,protocol,client,trained,pending,preparation,time_available,diagnosis_only=False):
    """Measure existing exact scorers and repaired SG boundary on historical TRAINING only.

    Historical per-token reference results may be reused only for identical full
    messages/targets and a matching frozen model/scorer fingerprint. Fresh reference
    samples check that reuse. Neither this cost probe nor partial SGs train a pool.
    """
    import sqlite3
    from research.progress import _unpack
    from research.gate import gate_score
    from research.upstream import Experience,Skill
    from research.client import stable_seed
    from research.semantic_evidence import EvidenceModel
    directory=Path(directory);deadline=time.monotonic()+min(900.,time_available()-90)
    if diagnosis_only:
        return profile_diagnosis(directory,protocol,client,pending,preparation,time_available,
            protocol.get('diagnosis_probe_count',6),cover_features=True,required=True)
    report={'backend_adopted':False,'complete_matrix':False,'status':'not_measured',
        'tolerances':{'token_absolute':1e-4,'sequence_absolute':1e-3,'ppo_objective_absolute':1e-6},
        'production_backend':'incremental_reference','training_data_only':True,'measurements':[]}
    path=directory/'report/scorer_parity.json'
    def persist():atomic_json(path,report)
    def check():
        if time.monotonic()>=deadline:raise TimeoutError('Registered numerical/SG probe allowance exhausted')
    fields=set(Skill.__dataclass_fields__)
    def skill(row):return Skill(**{k:v for k,v in row.items() if k in fields})
    parent_record=next(r for r in trained['history'][0]['evolution'] if r['parent']=='NetworkContainment')
    exps=[]
    for row in pending['experiences']:
        if row['skill']!='NetworkContainment':continue
        exp=Experience(**row);exp.transitions=[t for t in exp.transitions if t.get('policy_invoked') and
            t.get('request_applied') and not t.get('parse_error') and t.get('token_ids')]
        if exp.transitions:exps.append(exp)
    options=[skill(parent_record['old_skill'])]+[skill(c['skill']) for c in parent_record['candidates']]
    samples=[]
    for option in options:
        for exp in exps:
            for t in exp.transitions:
                messages=json.loads(t['state']);messages[0]['content']+='\n# ACTIVE SECURITY SKILL\n'+option.format_for_llm()
                samples.append({'messages':messages,'token_ids':t['token_ids']})
    fingerprint=client._fingerprint()
    if any(trained['model_identity'].get(k)!=v for k,v in fingerprint.items()):
        report.update(status='historical_reference_identity_mismatch');persist();return report
    reference_by_sample={}
    with sqlite3.connect('file:'+protocol['fixed_update_source']['journal']+'?mode=ro',uri=True) as db:
        for request,response in db.execute("SELECT request,response FROM tasks WHERE id LIKE 'server:%' AND status='completed' AND kind IN ('old_score','new_score')"):
            a,b=_unpack(request),_unpack(response)
            if isinstance(b,dict) and b.get('token_logprobs') is not None:reference_by_sample[digest(a)]=b
    reference=[reference_by_sample.get(digest(s)) for s in samples]
    report.update(historical_exact_reference_samples=sum(r is not None for r in reference),model_identity=client.identity,
        reference_reuse_key='Full messages, target IDs, model and tokenizer hashes, dtype, scorer source, torch/transformers, temperature',
        reference_temperature=protocol['configs']['B2']['model']['temperature'])
    def score(batch,backend,label):
        check();context={**client.base_context,'stage':'numerical_'+label}
        ids=[digest([context,backend,s]) for s in batch]
        began=time.perf_counter()
        response=client._request('score',{'samples':batch,'backend':backend,'temperature':1.,
            'request_ids':ids,'request_context':context})
        client._validate_scorer(response['identity'])
        result=response['scores']
        if len(result)!=len(batch) or any(len(a['token_ids'])!=len(b['token_logprobs']) for a,b in zip(batch,result)):
            raise ValueError('Exact token alignment changed')
        report['measurements'].append({'stage':label,'backend':backend,'samples':len(batch),
            'target_tokens':sum(len(s['token_ids']) for s in batch),'seconds':time.perf_counter()-began,
            'timings':[r.get('timing') for r in result]});persist()
        return result
    def errors(a,b):
        return {'token_max_absolute_error':max(abs(x-y) for r,s in zip(a,b) for x,y in zip(r['token_logprobs'],s['token_logprobs'])),
            'sequence_max_absolute_error':max(abs(r['sum']-s['sum']) for r,s in zip(a,b))}
    n=sum(len(e.transitions) for e in exps)
    def gates(scored):
        return [gate_score(exps,[r['sum'] for r in scored[:n]],[r['sum'] for r in scored[i*n:(i+1)*n]],
            preparation['ready']['baselines'],mode='paper_rtg',epsilon=.2)[0] for i in range(1,len(options))]
    try:
        # Acting generation primes the old-policy cache with an exact sequence
        # sum, not per-token scores. Fill only missing token references using the
        # unchanged TeacherForce backend; never label absence as a parity result.
        report['fresh_missing_per_token_reference_samples']=sum(r is None for r in reference)
        for i,item in enumerate(reference):
            if item is None:reference[i]=score([samples[i]],'incremental_reference','missing_per_token_reference')[0]
        indices=sorted({0,n-1,n,len(samples)//2,len(samples)-1})
        fresh=[score([samples[i]],'incremental_reference','fresh_reference')[0] for i in indices]
        report['fresh_reference_vs_available_reference']=errors([reference[i] for i in indices],fresh)
        ref_values=gates(reference);report['reference_ppo_values']=ref_values
        report['saved_ppo_values']=[c['score'] for c in parent_record['candidates']]
        report['reference_gate_reproduced']=max(abs(a-b) for a,b in zip(ref_values,report['saved_ppo_values']))<=1e-6
        report['microbatches']={}
        if protocol.get('profile_full_sequence'):
            for size in (1,2):
                scored=[]
                for i in range(0,len(samples),size):scored.extend(score(samples[i:i+size],'full_sequence_experimental',f'full_sequence_batch_{size}'))
                values=gates(scored);err=errors(reference,scored)
                objective=max(abs(a-b) for a,b in zip(ref_values,values))
                same=[a>0 for a in ref_values]==[a>0 for a in values] and max(range(len(values)),key=values.__getitem__)==max(range(len(ref_values)),key=ref_values.__getitem__)
                report['microbatches'][str(size)]={**err,'ppo_values':values,'ppo_max_absolute_error':objective,'ppo_decisions_match':same,
                    'passed':err['token_max_absolute_error']<=1e-4 and err['sequence_max_absolute_error']<=1e-3 and objective<=1e-6 and same}
                persist()
            report['status']='partial_numerical_check_passed' if all(r['passed'] for r in report['microbatches'].values()) else 'failed'
        else:report['status']='not_run_resource_minimized'
    except Exception as exc:report.update(status='failed_or_incomplete',error=f'{type(exc).__name__}: {exc}')
    report['adoption_reason']='Default remains reference. All actual Network candidates are checked, but this pilot is not a complete near-boundary/EOS/context-length certificate.'
    persist()
    profile_diagnosis(directory,protocol,client,pending,preparation,lambda:min(time_available(),deadline-time.monotonic()+90))
    return report


def profile_optimizer(directory,label,pending,port):
    """Small real-Qwen cost sample. Profile requests are separate from optimizer RNG."""
    from research.client import ModelClient,stable_seed
    from research.upstream import Experience,OriginalEvolution,Skill,configure_encoder
    from research.skills import SecuritySkillPool
    directory=Path(directory);protocol=json.loads((directory/'protocol.json').read_text())
    cfg=deepcopy(protocol['configs'][label]);journal=Journal(directory/'journal.sqlite3')
    cached=journal.restore('profile:'+label)
    if cached:
        audit_path=directory/f'report/{label}-prompt_audit.json'
        if audit_path.exists() and 'planned_sg_input_tokens' not in cached:
            cached={**cached,'planned_sg_input_tokens':[r['input_tokens'] for r in json.loads(audit_path.read_text()) if r.get('eligible')]}
        return cached
    runtime={**cfg['model'],'endpoint':f'http://127.0.0.1:{port}','journal_path':str(journal.path),
        'request_context':{'experiment':str(directory),'arm':label,'env_stage':'profile'}}
    client=ModelClient(runtime);configure_encoder(client.embed)
    preparation=journal.restore('preparation:1:'+label) or journal.restore('preparation:1')
    snapshot=preparation['ready']['pool'] if preparation else json.loads((ROOT/'artifacts/initial_pool.json').read_text())['pool']
    experiences=[Experience(**e) for e in pending['experiences']]
    rendered,parents=render_sg(cfg,snapshot,experiences)
    eligible=[r for r in rendered if r['eligible']]
    tokens=[]
    for r in rendered:
        t=client._tokens([{'role':'user','content':r['prompt']}])
        tokens.append({k:v for k,v in r.items() if k not in ('prompt','evidence_rows')}|{'input_tokens':t['prompt_tokens'],
            'reserved_output_tokens':768,'max_context':t['max_context'],'fits':t['prompt_tokens']+768<=t['max_context']})
    atomic_json(directory/f'report/{label}-prompt_audit.json',tokens)
    if any(not t['fits'] for t in tokens if t['eligible']):
        return {'status':'context_overflow','prompt_audit':tokens,'measured_stage_costs':{},'planned_sg_chunks':len(eligible)}
    sampled=[];grads=[];new_data=None
    ordered=sorted(eligible,key=lambda r:len(r['prompt']))
    indices=sorted({0,len(ordered)//2,len(ordered)-1}) if ordered else []
    for index in indices:
        r=ordered[index]
        with client.context(stage='profile_sg',skill=r['skill'],chunk=r['chunk'],trace_hash=r['trace_sha256']):
            response=client.generate([{'role':'user','content':r['prompt']}],seed=stable_seed(label,index,'cost-profile'),purpose='evolution',max_tokens=768)
        parsed=OriginalEvolution._parse_json_safely(None,response['text']) or {}
        if cfg['evolution'].get('evidence_verification'):
            from research.semantic_evidence import EvidenceModel
            boundary=EvidenceModel(lambda _:response['text'],r['evidence_rows'],
                diagnosis_verification=cfg['evolution'].get('diagnosis_verification',False), schema=cfg['evolution'].get('evidence_schema','typed_claims_v8'))
            parsed=json.loads(boundary('Profile uses the same canonical boundary as the optimizer'))
        gradient=parsed.get('semantic_gradient',{})
        if parsed.get('is_related'):
            grads.append({'initiation':str(gradient.get('initiation','')),'execution':str(gradient.get('policy',gradient.get('execution',''))),'termination':str(gradient.get('termination',''))})
        sampled.append({'stage':'sg_chunk','input_tokens':response['prompt_tokens'],'output_tokens':response['output_tokens'],
                        'seconds':response['seconds'],'timing':response.get('timing',{})})
    if ordered:
        # The final aggregate does not exist until all SG chunks finish. This small
        # partial-gradient profile is explicitly a representative cost probe.
        def call(prompt):
            with client.context(stage='profile_aggregation_candidate',skill=ordered[0]['skill']):
                response=client.generate([{'role':'user','content':prompt}],seed=stable_seed(label,'aggregate-profile'),purpose='evolution',max_tokens=768)
            sampled.append({'stage':'aggregation_candidate','input_tokens':response['prompt_tokens'],'output_tokens':response['output_tokens'],
                'seconds':response['seconds'],'timing':response.get('timing',{}),'representative_partial_gradient_prompt':True})
            return response['text']
        from research.evolution import Evolution
        new_data=Evolution(call,cfg['evolution'])._run_semantic_update(parents[ordered[0]['skill']],grads,'REFINE')
    score_samples=[];score_parents=[];old_missing=0;valid_count=0
    for exp in experiences:
        parent=parents.get(exp.skill)
        if parent is None:continue
        valid=[t for t in exp.transitions if t.get('policy_invoked') and t.get('request_applied') and not t.get('parse_error') and t.get('token_ids')]
        valid_count+=len(valid)
        for transition in valid:
            messages=json.loads(transition['state']);messages[0]['content']+='\n# ACTIVE SECURITY SKILL\n'+parent.format_for_llm()
            key=client._score_key({'messages':messages,'token_ids':transition['token_ids']})
            old_missing+=journal.cache_get('score:'+key) is None
        for t in (valid[:1]+valid[-1:] if valid else []):
            messages=json.loads(t['state']);messages[0]['content']+='\n# ACTIVE SECURITY SKILL\n'+parent.format_for_llm()
            score_samples.append({'messages':messages,'token_ids':t['token_ids']});score_parents.append(parent.name)
    score_samples=score_samples[:12];score_parents=score_parents[:12]
    def measured_score(samples,stage,backend='incremental_reference'):
        context={**client.base_context,'stage':'profile_'+stage}
        ids=[digest([context,backend,s]) for s in samples]
        if backend=='incremental_reference':
            result=[]
            for sample,key in zip(samples,ids):
                response=client._request('score',{'samples':[sample],'temperature':cfg['model']['temperature'],
                    'backend':backend,'request_ids':[key],'request_context':context})
                result.extend(response['scores'])
        else:
            result=client._request('score',{'samples':samples,'temperature':cfg['model']['temperature'],
                'backend':backend,'request_ids':ids,'request_context':context})['scores']
        return result
    reference=[]
    if score_samples:
        with client.context(stage='profile_old_score'):
            began=time.perf_counter();reference=measured_score(score_samples,'old_score')
        sampled.append({'stage':'old_score','samples':len(score_samples),'seconds':time.perf_counter()-began,
                        'target_tokens':sum(len(s['token_ids']) for s in score_samples)})
    new_score_measured=False
    if isinstance(new_data,dict) and new_data and ordered:
        from research.skills import structurally_valid
        parent=parents[ordered[0]['skill']]
        candidate=Skill(name=re.sub(r'_v\d+$','',parent.name)+f'_v{parent.version+1}',
            initiation=new_data.get('initiation',''),policy=new_data.get('policy',[]),termination=new_data.get('termination',''),
            parent_id=parent.name,version=parent.version+1)
        if structurally_valid(candidate):
            changed=[]
            for sample,name in zip(score_samples,score_parents):
                if name!=parent.name:continue
                messages=deepcopy(sample['messages'])
                messages[0]['content']=messages[0]['content'].rsplit('\n# ACTIVE SECURITY SKILL\n',1)[0]+'\n# ACTIVE SECURITY SKILL\n'+candidate.format_for_llm()
                changed.append({'messages':messages,'token_ids':sample['token_ids']})
            if changed:
                began=time.perf_counter();new_reference=measured_score(changed,'new_score')
                sampled.append({'stage':'new_score','samples':len(changed),'seconds':time.perf_counter()-began,
                    'target_tokens':sum(len(s['token_ids']) for s in changed),'representative_partial_gradient_candidate':True})
                score_samples.extend(changed);reference.extend(new_reference);new_score_measured=True
    parity={'status':'not_measured','backend_adopted':False,'tolerances':{'token_absolute':1e-4,'sequence_absolute':1e-3,'ppo_objective_absolute':1e-6,'decisions_must_match':True},
        'scope':'Numerical scorer check only; no diagnosis or defense-performance claim.'}
    if reference and protocol.get('profile_full_sequence',True):
        try:
            began=time.perf_counter()
            batch=measured_score(score_samples,'parity','full_sequence_experimental')
            token_error=max(abs(a-b) for x,y in zip(reference,batch) for a,b in zip(x['token_logprobs'],y['token_logprobs']))
            sequence_error=max(abs(x['sum']-y['sum']) for x,y in zip(reference,batch))
            parity.update(status='partial_numerical_check_passed' if token_error<=1e-4 and sequence_error<=1e-3 else 'failed',
                token_max_absolute_error=token_error,sequence_max_absolute_error=sequence_error,
                reference_seconds=sampled[-1]['seconds'],full_sequence_seconds=time.perf_counter()-began,
                samples=len(score_samples),ppo_objective_comparison=None,
                adoption_reason='Full matrix including candidate scores, EOS/mixed lengths and near-boundary PPO decisions is required. Default remains reference.')
        except Exception as exc:parity.update(status='failed',error=str(exc))
    elif reference:
        parity.update(status='not_run_resource_minimized',reason='Experimental batch scorer probe disabled; production remains incremental_reference')
    atomic_json(directory/'report/scorer_parity.json',parity)
    physical=journal.tasks()
    for sample in sampled:
        if sample['stage'] in ('old_score','new_score'):
            measured=[(t['metrics'] or {}).get('processing_seconds') for t in physical
                if t['id'].startswith('server:') and t['kind']=='profile_'+sample['stage'] and t['status']=='completed'
                and t['context'].get('arm')==label]
            measured=[v for v in measured if v is not None]
            if measured:sample['seconds']=sum(measured)
    times=[e['execution_time_s'] for e in pending['episodes']]
    episode_upper=max(times)*1.5 if times else None
    if episode_upper:journal.checkpoint('episode_cost_upper_seconds',episode_upper)
    result={'status':'measured','arm':label,'model_identity':client.identity,'samples':sampled,'episode_seconds':distribution(times),
        'planned_old_score_cache_misses':old_missing,'old_score_samples':valid_count,
        'episode_upper_seconds':episode_upper,'planned_sg_chunks':len(eligible),'planned_parents':len({r['skill'] for r in eligible}),
        'planned_valid_decisions':sum(bool(t.get('policy_invoked') and t.get('request_applied') and not t.get('parse_error') and t.get('token_ids')) for e in experiences for t in e.transitions),
        'best_of_n':cfg['evolution']['best_of_n'],'prompt_input_tokens':distribution([t['input_tokens'] for t in tokens]),
        'planned_sg_input_tokens':[t['input_tokens'] for t in tokens if t['eligible']],
        'context_overflows_measured':sum(not t['fits'] for t in tokens),'aggregate_final_prompt_measured':False,
        'aggregate_context_risk':'Final actual aggregate must be tokenized on submission; partial-gradient profile does not prove it fits.',
        'new_score_measured':new_score_measured,'uncertainty':'At most three SG samples and a partial-gradient aggregate. Maximum observed cost scaled by context and reserved output, plus 50% margin. Not a p95 confidence bound.'}
    journal.checkpoint('profile:'+label,result);atomic_json(directory/f'report/{label}-cost_profile.json',result)
    return result


def generation_cost_upper(samples,maximum_prompt,maximum_output=768):
    """Use a nearby measured context; do not extrapolate the shortest probe over all work.

    Preserve full generation limits. This forecasts time only; the shared ledger
    enforces the hard cap when real output lengths or CUDA timing differ.
    """
    if not samples:return None
    timed=[s for s in samples if s.get('timing',{}).get('prefill_gpu_seconds') is not None
           and s.get('timing',{}).get('decode_gpu_seconds',0)>0 and s.get('output_tokens',0)>1]
    if timed:
        adequate=[s for s in timed if s['input_tokens']>=maximum_prompt]
        selected_length=min(s['input_tokens'] for s in adequate) if adequate else max(s['input_tokens'] for s in timed)
        samples=[s for s in timed if s['input_tokens']==selected_length]
    estimates=[]
    for sample in samples:
        timing=sample.get('timing',{})
        prefill=timing.get('prefill_gpu_seconds');decode=timing.get('decode_gpu_seconds')
        length=max(1,sample['input_tokens']);output=max(1,sample['output_tokens'])
        ratio=max(1,maximum_prompt/length)
        if prefill is not None and decode is not None and output>1 and decode>0:
            overhead=max(0,sample['seconds']-prefill-decode)*max(1,maximum_output/output)
            context=max(1,(max(length,maximum_prompt)+maximum_output)/(length+output))
            estimate=prefill*ratio*ratio+decode/(output-1)*(maximum_output-1)*context+overhead
        else:estimate=sample['seconds']*ratio*max(1,maximum_output/output)
        estimates.append(estimate*1.5)
    return max(estimates)


def preflight(protocol,measurement,usage,gpu_count,label,include_collection=False,evaluation_labels=None):
    if measurement.get('status')!='measured' or measurement.get('episode_upper_seconds') is None:
        return {'status':'deferred','reason':'Actual end-to-end cost samples are missing','measurement':measurement}
    stages=defaultdict(list)
    for s in measurement['samples']:stages[s['stage']].append(s)
    if not stages['sg_chunk'] or not stages['old_score']:
        return {'status':'deferred','reason':'SG/scorer cost measurement missing'}
    # Forecast every actual SG prompt length, retaining the 768-token capacity
    # separately from an empirical, padded output-length forecast. Neither value
    # changes the real max_new_tokens or scientific admission rule.
    kernels=stages['sg_chunk']+stages['aggregation_candidate']
    sg_outputs=min(768,math.ceil(max(s['output_tokens'] for s in stages['sg_chunk'])*1.5))
    aggregate_outputs=min(768,math.ceil(max([s['output_tokens'] for s in stages['aggregation_candidate']] or [768])*1.5))
    lengths=measurement.get('planned_sg_input_tokens') or [measurement['prompt_input_tokens']['maximum'] or 1]*measurement['planned_sg_chunks']
    sg=sum(generation_cost_upper(kernels,length,sg_outputs) for length in lengths)
    sg_capacity=sum(generation_cost_upper(kernels,length,768) for length in lengths)
    aggregate_count=measurement['planned_parents']*measurement['best_of_n']
    aggregate=generation_cost_upper(kernels,24576-768,aggregate_outputs)*aggregate_count
    aggregate_capacity=generation_cost_upper(kernels,24576-768,768)*aggregate_count
    score_upper=max(s['seconds']/s['samples'] for s in stages['old_score']+stages['new_score'])*1.5
    ep=measurement['episode_upper_seconds'];n=protocol['statistics']['pilot_n']
    cfg=protocol.get('configs',{}).get(label,{});loops=cfg.get('iterations',1)
    rollout=cfg.get('evolution',{}).get('rollout_verification',label=='B4')
    attempts=cfg.get('evolution',{}).get('maximum_rollout_candidates_per_iteration',1)*loops
    planned_rollouts=40*attempts if rollout else 0
    reused=measurement.get('completed_identical_rollout_episodes',0) if measurement.get('rollout_reuse_identity_verified') else 0
    if type(reused) is not int or not 0<=reused<=planned_rollouts:raise ValueError('Invalid verified rollout reuse count')
    labels=evaluation_labels if evaluation_labels is not None else protocol.get('core_evaluation_labels',['B0','B1','B2'])
    collected=measurement.get('collected_training_episodes',measurement.get('episode_seconds',{}).get('n',2))
    collection=cfg.get('train_episodes',2) if include_collection else max(0,cfg.get('train_episodes',2)-collected)
    parts={'collection':collection*ep,
        'sg':sg*loops,
        'aggregation_best_of_n':aggregate*loops,
        'old_new_score':(measurement.get('planned_old_score_cache_misses',measurement['planned_valid_decisions'])
                         +measurement['planned_valid_decisions']*measurement['best_of_n'])*score_upper*loops,
        'paired_candidate_validation':(planned_rollouts-reused)*ep/gpu_count,
        'frozen_paired_test':len(labels)*n*ep/gpu_count,
        'checkpoint_shutdown_margin':120}
    wall=sum(parts.values());remaining_gpu=usage['gpu_hour_cap']-usage['gpu_hours'];remaining_wall=usage['wall_second_cap']-usage['wall_seconds']
    capacity=wall-sg*loops-aggregate*loops+(sg_capacity+aggregate_capacity)*loops
    return {'status':'passed' if wall<=remaining_wall and wall*gpu_count/3600<=remaining_gpu else 'deferred',
        'arm':label,'projected_upper_wall_seconds':wall,'projected_occupied_gpu_hours':wall*gpu_count/3600,
        'remaining_gpu_hours':remaining_gpu,'remaining_wall_seconds':remaining_wall,'parts_seconds':parts,
        'estimated_not_measured':True,'uncertainty':measurement['uncertainty'],
        'generation_cost_model':'nearest adequate measured context, actual prompt-length grid; observed maximum output plus 50% and timing plus 50%, with full-capacity forecast separately retained',
        'forecast_output_tokens':{'sg':sg_outputs,'aggregation_candidate':aggregate_outputs},
        'actual_generation_max_tokens_unchanged':768,'full_768_token_capacity_wall_seconds':capacity,
        'full_768_token_capacity_gpu_hours':capacity*gpu_count/3600,
        'full_capacity_fits':capacity<=remaining_wall and capacity*gpu_count/3600<=remaining_gpu,
        'forecast_is_not_completion_guarantee':True,
        'b4_pairs':20 if rollout else 0,'rollout_candidate_attempts_reserved':attempts if rollout else 0,
        'planned_validation_episodes':planned_rollouts,'reused_validation_episodes':reused,
        'training_plan':{'episodes':cfg.get('train_episodes',2),'iterations':loops},
        'evaluation_labels_reserved':labels,'thresholds_unchanged':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',type=Path,required=True)
    parser.add_argument('--profile-arm',choices=['B2','B3-no-support','B3','B4'])
    parser.add_argument('--port',type=int,default=8418)
    args=parser.parse_args()
    if args.profile_arm:
        journal=Journal(args.pilot/'journal.sqlite3');pending=journal.restore('pending_batch:1:'+args.profile_arm)
        if not pending:raise ValueError('A complete pending batch is required')
        profile_optimizer(args.pilot,args.profile_arm,pending,args.port)
    print(json.dumps(audit(args.pilot),ensure_ascii=False))


if __name__=='__main__':main()
