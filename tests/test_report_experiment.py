"""Synthetic arithmetic fixtures in temporary directories, never performance data."""
import csv
import importlib.util
import json
import hashlib
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('report_experiment',ROOT/'scripts/report_experiment.py')
reporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(reporter)


def write_run(root,label,stage,rewards,*,model='fixture',trained=0,optimizer_status='completed_changed',train_seed=90000):
    directory=root/f'{label}-{stage}';directory.mkdir()
    config={'comparison_id':label,'arm':'E0' if label=='B0' else 'E1' if label=='B1' else 'E2','red':'finite',
            'steps':500,'composition':'hybrid','experiment_version':'unit_fixture',
            'profile':'controlled-comparison','scoring':'official','communication':'paper_projected_keep',
            'skip_busy_policy':True,'selector':{'epsilon':0.1,'max_skill_age':16},
            'model':{'model':model,'revision':'unit_fixture'}}
    metadata=reporter.comparison_metadata(config);base=metadata['base_comparison_id']
    proposed=base in ('B3','B3-no-support','B4')
    config['evolution']={'gate_mode':'paper_rtg','schedule':'paper_batch','maintenance_mode':'paper_algorithm1',
        'evidence_verification':proposed,'diagnosis_verification':proposed,
        'minimum_independent_episodes':2 if base in ('B3','B4') else 0,
        'action_support_verification':metadata.get('action_support_verification',False),
        'selection_shift_verification':metadata.get('selection_shift_verification',False),
        'rollout_verification':base=='B4' or metadata.get('rollout_verification',False)}
    config.update(train_episodes=trained or 2,iterations=1)
    episodes=[]
    for i,reward in enumerate(rewards):
        episode={'status':'completed','kind':'real_cage4','seed':100+i,'episode_id':i,
                 'joint_return':reward,'execution_time_s':1.,'joint_transitions':499,
                 'scored_transitions':498}
        (directory/f'episode-{i:04d}.json').write_text(json.dumps(episode))
        episodes.append(episode)
    summary={'config':config,'stage':stage,'status':'partial' if stage=='train' else 'completed','result_kind':'real_cage4',
             'episodes':episodes[:1] if stage=='train' else episodes,'history':[],
             'training_provenance':{'episodes_completed':trained}}
    if trained:
        pool={'skills':[{'fixture_only':label}],'max_size':1};initial='0'*64
        checkpoint={'stage':'train','pool':pool,'episodes_completed':trained,'total_training_episodes':trained,
            'planned_training_episodes':trained,'planned_optimizer_iterations':1,'training_completed':True,
            'training_run_id':'fixture-'+label+'-'+root.name,'training_seeds':list(range(train_seed,train_seed+trained)),
            'optimizer_status':optimizer_status,'initial_pool_sha256':initial,
            'history':[{'iteration':1,'episodes':trained,'optimizer_status':optimizer_status}],
            'algorithm':config['evolution'],'policy_interface':reporter.policy_interface(config),
            'communication_encoder':'legacy_local_v1'}
        source=directory/'training-checkpoint.json';source.write_text(json.dumps(checkpoint))
        summary.update(pool_sha256=hashlib.sha256(json.dumps(pool,sort_keys=True).encode()).hexdigest(),pool_frozen=True,
            comparison_initial_pool_sha256=initial,training_provenance={'path':str(source.resolve()),
                'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'episodes_completed':trained,
                'training_seeds':checkpoint['training_seeds'],'optimizer_status':optimizer_status,'initial_pool_sha256':initial,
                'algorithm':checkpoint['algorithm']})
        (directory/'frozen_pool.json').write_text(json.dumps({'pool':pool,
            'policy_interface':checkpoint['policy_interface'],'communication_encoder':'legacy_local_v1'}))
    (directory/'summary.json').write_text(json.dumps(summary))
    return directory


def test_uncommitted_completed_training_is_not_a_trained_test(tmp_path):
    write_run(tmp_path,'B4','train',[-10,-20])
    write_run(tmp_path,'B4','eval',[-5],trained=0)
    result=reporter.report_experiment(tmp_path)
    metrics=json.loads((Path(result['output'])/'metrics.json').read_text())
    assert {g['split']:g['n'] for g in metrics['groups']}=={'train':2,'untrained_control':1}
    status=json.loads((Path(result['output'])/'run_status.json').read_text())
    train=next(r for r in status if r['stage']=='train')
    assert train['completed_environment_episodes']==2 and train['committed_iterations']==0


def test_empty_curve_reports_only_registered_checkpoint_comparisons():
    rows = reporter.paired_comparisons([], ['B2@1', 'B2@2', 'B2@3'])

    assert [row['comparison'] for row in rows] == [
        'B1-B0',
        'B2@1-B0', 'B2@1-B1',
        'B2@2-B0', 'B2@2-B1', 'B2@2-B2@1',
        'B2@3-B0', 'B2@3-B1', 'B2@3-B2@1', 'B2@3-B2@2',
    ]
    for row in rows:
        assert row['paired_n'] == 0
        assert row['mean_difference'] is None
        assert row['median_difference'] is None
        assert row['win_tie_loss'] is None
        assert row['paired_bootstrap_95ci'] is None


@pytest.mark.parametrize('inherited',[0,1])
def test_registered_intermediate_checkpoints_keep_identical_eval_seeds_separate(tmp_path,inherited):
    from research.config import digest,validate
    from scripts.run_campaign import bind_checkpoints

    train=tmp_path/'runs/B2-finite-train';train.mkdir(parents=True)
    initial='0'*64;lineage='fixture-lineage';model={'model':'fixture'}
    training_episodes=6-2*inherited;iterations=3-inherited
    cfg=validate({'arm':'E2','comparison_id':'B2','train_episodes':training_episodes,'iterations':iterations,
                  'validation_episodes':0,'eval_seed':701,'episodes':1,
                  'seed_use':'final_evaluation',
                  'protocol_digest':'fixture-curve','execution_stage':'train'})
    if inherited:cfg['continuation']={'iteration':inherited,'total_training_episodes':2*inherited}
    identity=reporter.policy_interface(cfg)
    manifest={'stage':'train','config':cfg,'config_digest':digest(cfg),'training_run_id':'fixture-train',
              'policy_interface':identity}
    reporter.atomic_json(train/'manifest.json',manifest)
    journal=reporter.Journal(tmp_path/'journal.sqlite3');spec={}
    labels=(f'B2@{inherited+1}',f'B2@{inherited+2}')
    for k in (inherited+1,inherited+2):
        segment=2*(k-inherited)
        seeds=list(range(90,90+2*inherited))+list(range(cfg['train_seed'],cfg['train_seed']+segment))
        path=train/f'pool-iteration-{k:04d}.json'
        snapshot={'stage':'train','pool':{'max_size':1,'skills':[{'fixture_version':k}]},
                  'algorithm':cfg['evolution'],'model_identity':model,'config_digest':digest(cfg),
                  'training_run_id':'fixture-train','training_lineage_id':lineage,
                  'training_completed':segment==training_episodes,'planned_training_episodes':training_episodes,
                  'planned_optimizer_iterations':iterations,
                  'episodes_completed':segment,'training_seeds':seeds,'initial_pool_sha256':initial,
                  'optimizer_status':'completed_changed','policy_interface':identity,
                  'history':[{'iteration':j,'optimizer_status':'completed_changed'} for j in range(1,k+1)]}
        reporter.atomic_json(path,snapshot)
        journal.checkpoint(f'loop_commit:{k}:B2',{'checkpoint_sha256':reporter.file_sha256(path)})
        spec[f'B2@{k}']={'path':str(path),'iteration':k,'training_seeds':seeds,
                        'total_training_episodes':2*k,'initial_pool_sha256':initial}
    protocol={'configs':{'B2':cfg},'comparison_protocol_digest':'fixture-curve',
              'checkpoint_evaluations':spec,'pilot_test_seeds':[701]}
    reporter.atomic_json(tmp_path/'protocol.json',protocol)
    bindings=bind_checkpoints(protocol,tmp_path,journal)
    assert len({b['checkpoint_sha256'] for b in bindings.values()})==2
    for label,binding in bindings.items():
        run=tmp_path/'runs'/f"{label}-{binding['checkpoint_sha256'][:16]}-finite-evaluate-701"
        run.mkdir()
        evaluation={**cfg,'execution_stage':'eval','comparison_id':label,'base_comparison_id':'B2',
                    'evaluation_role':'frozen_test','checkpoint_evaluation':binding,'eval_seeds':[701]}
        reporter.atomic_json(run/'manifest.json',{'stage':'eval','config':evaluation,
                             'config_digest':digest(evaluation),'policy_interface':identity})
        record={'status':'completed','kind':'real_cage4','seed':701,'joint_return':float(binding['iteration']),
                'episode_id':0,'execution_time_s':1.,'joint_transitions':499,'scored_transitions':498,
                'input_pool_sha256':binding['pool_sha256'],'communication_encoder':identity['communication_encoder']}
        proof={'path':binding['path'],'sha256':binding['checkpoint_sha256'],'checkpoint_evaluation':binding,
               'training_manifest':binding['training_manifest'],'episodes_completed':binding['segment_training_episodes'],
               'training_seeds':binding['training_seeds'],'initial_pool_sha256':initial,
               'algorithm':cfg['evolution'],'optimizer_status':'completed_changed'}
        reporter.atomic_json(run/'summary.json',{'status':'completed','stage':'eval','config':evaluation,
            'model_identity':model,'result_kind':'real_cage4','pool_sha256':binding['pool_sha256'],
            'comparison_run_id':'fixture-curve','training_provenance':proof,'episodes':[record]})
    episodes,statuses=reporter.collect(tmp_path)
    assert {(r['label'],r['seed'],r['split']) for r in episodes}=={(label,701,'test') for label in labels}
    assert {r['training_provenance_state'] for r in episodes}=={'verified_completed_update'}
    assert reporter.training_summary(episodes)['count']==1
    pair=next(r for r in reporter.paired_comparisons(episodes) if r['comparison']==f'{labels[1]}-{labels[0]}')
    assert pair['paired_n']==1 and pair['mean_difference']==1.
    path=Path(spec[labels[1]]['path']);original=reporter.load(path)
    reporter.atomic_json(path,{**original,'pool':{'fixture_replacement':True}})
    journal.checkpoint(f'loop_commit:{inherited+2}:B2',{'checkpoint_sha256':reporter.file_sha256(path)})
    with pytest.raises(ValueError,match='different checkpoint hash'):
        bind_checkpoints(protocol,tmp_path,journal)
    reporter.atomic_json(path,{**original,'pending_results':[{'uncommitted':True}]})
    with pytest.raises(ValueError,match='complete optimizer commit prefix'):
        bind_checkpoints(protocol,tmp_path,journal)


def test_paired_test_ignores_candidate_validation_and_mock_samples(tmp_path):
    write_run(tmp_path,'B0','eval',[-10,-30])
    write_run(tmp_path,'B1','eval',[-5,-25])
    candidate=tmp_path/'rollout-verification';candidate.mkdir()
    write_run(candidate,'B1','eval',[999])
    mock=write_run(tmp_path,'B3','eval',[998],trained=2)
    for path in [mock/'summary.json',mock/'episode-0000.json']:
        data=json.loads(path.read_text())
        if 'episodes' in data:data['episodes'][0]['kind']='mock_test'
        else:data['kind']='mock_test'
        path.write_text(json.dumps(data))
    result=reporter.report_experiment(tmp_path)
    metrics=json.loads((Path(result['output'])/'metrics.json').read_text())
    assert len(metrics['groups'])==2
    pair=metrics['paired_vs_B0'][0]
    assert pair['paired_n']==2 and pair['mean_difference']==5 and pair['win_rate']==1


def test_legacy_tables_share_canonical_admission_with_boundary_roles(tmp_path):
    for name, label, reward, trained, role in (
            ('primary', 'B2', -10, 2, 'frozen_test'), ('candidate', 'B2', 997, 0, 'rollout'),
            ('manifest_candidate', 'B2', 996, 0, 'frozen_test'), ('untrained', 'B2', 995, 0, 'frozen_test'),
            ('transfer', 'B2', -50, 2, 'transfer_test'), ('fixture', 'B1', 994, 0, 'frozen_test')):
        root=tmp_path/name;root.mkdir()
        run=write_run(root,label,'eval',[reward],trained=trained)
        path=run/'summary.json';summary=json.loads(path.read_text())
        summary['config']['evaluation_role']=role
        if name=='manifest_candidate':
            (run/'manifest.json').write_text(json.dumps({'stage':'eval',
                'config':{**summary['config'],'evaluation_role':'rollout'}}))
        if name=='fixture':
            # A marker in the summary must not be erased by a separate episode file.
            summary['episodes'][0]['unit_test_fixture_only']=True
        path.write_text(json.dumps(summary))
    result=reporter.report_experiment(tmp_path,include_bundle=False,include_legacy_tables=True)
    out=Path(result['output'])
    episodes=json.loads((out/'episodes.json').read_text())['episodes']
    assert {(e['split'],e['joint_return']) for e in episodes}=={
        ('test',-10),('transfer',-50),('untrained_control',995)}
    with (out/'table4_ablation.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    assert {(r['Split'],r['Mean Reward']) for r in rows if r['Configuration']=='Full Skill-Pro'}=={
        ('test','-10.000'),('transfer','-50.000')}
    assert len(list(out.glob('group-*/completed_cases.csv')))==2
    assert '995.000' in (out/'untrained_controls.csv').read_text()


def test_registered_training_stage_cannot_be_relabeled_as_final_eval(tmp_path):
    run=write_run(tmp_path,'B0','eval',[-1])
    (run/'manifest.json').write_text(json.dumps({'stage':'train'}))
    with pytest.raises(ValueError,match='stage differs from its manifest'):
        reporter.collect(tmp_path)


def test_legacy_internal_arm_collision_preserves_distinct_canonical_labels(tmp_path):
    write_run(tmp_path,'B2','eval',[-10],trained=2)
    write_run(tmp_path,'B3','eval',[-20],trained=2)
    result=reporter.report_experiment(tmp_path,include_bundle=False,include_legacy_tables=True)
    out=Path(result['output'])
    assert not list(out.glob('group-*/completed_cases.csv'))
    manifest=json.loads((out/'report_manifest.json').read_text())
    assert len(manifest['omitted_ambiguous_legacy_groups'])==1
    with (out/'table4_ablation.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    assert {(r['Configuration'],r['Arm'],r['Mean Reward']) for r in rows if r['N']=='1'}=={
        ('Full Skill-Pro','B2','-10.000'),('B3','B3','-20.000')}


def test_different_model_conditions_do_not_share_a_comparison(tmp_path):
    write_run(tmp_path,'B0','eval',[-10])
    write_run(tmp_path,'B1','eval',[-5],model='different_fixture')
    with pytest.raises(ValueError,match='conditions differ'):
        reporter.collect(tmp_path)


def test_core_and_ablation_pairs_report_unchanged_policy_without_claiming_gain(tmp_path):
    write_run(tmp_path,'B0','eval',[-10,-30])
    write_run(tmp_path,'B1','eval',[-5,-25])
    write_run(tmp_path,'B2','eval',[-5,-25],trained=2,optimizer_status='completed_unchanged')
    write_run(tmp_path,'B3-no-support','eval',[-4,-24],trained=2)
    result=reporter.report_experiment(tmp_path)
    metrics=json.loads((Path(result['output'])/'metrics.json').read_text())
    pairs={p['comparison']:p for p in metrics['paired_comparisons']}
    assert pairs['B2-B1']['mean_difference']==0 and pairs['B2-B1']['win_tie_loss']==[0,2,0]
    assert pairs['B3-no-support-B2']['mean_difference']==1
    assert pairs['B4-B2']['mean_difference'] is None and pairs['B4-B2']['paired_n']==0
    assert pairs['B1-B0']['training_repetitions'] is None
    assert pairs['B2-B1']['training_repetitions']==1


@pytest.mark.parametrize('label',list(reporter.VERIFICATION_VARIANTS))
@pytest.mark.parametrize('status',['partial','interrupted_budget','failed','unknown',
    'completed_changed','completed_unchanged','completed_no_eligible_skill'])
def test_every_variant_requires_a_verified_completed_training_artifact(tmp_path,label,status):
    write_run(tmp_path,label,'eval',[-10],trained=2,optimizer_status=status)
    rows,_=reporter.collect(tmp_path)
    assert rows[0]['split']==('test' if status in reporter.COMPLETED_OPTIMIZER_STATUSES else 'untrained_control')


def test_unknown_or_initial_pool_is_not_a_trained_variant(tmp_path):
    for label in reporter.VERIFICATION_VARIANTS:write_run(tmp_path,label,'eval',[-10],trained=0)
    rows,_=reporter.collect(tmp_path)
    assert all(e['split']=='untrained_control' and e['training_identity'] is None for e in rows)


def test_same_eval_seed_from_two_training_runs_is_not_overwritten(tmp_path):
    for number in (1,2):
        root=tmp_path/f'repetition-{number}';root.mkdir()
        write_run(root,'B0','eval',[-10,-20]);write_run(root,'B1','eval',[-9,-19])
        write_run(root,'B2','eval',[-8,-18],trained=2,train_seed=90000+100*number)
    rows,_=reporter.collect(tmp_path);metrics=reporter.recompute(rows)
    b2=next(g for g in metrics['groups'] if g['label']=='B2')
    assert b2['n']==4 and b2['unique_eval_seed_count']==2 and b2['rewards_by_seed'] is None
    pair=next(p for p in reporter.paired_comparisons(rows) if p['comparison']=='B2-B1')
    assert pair['paired_n']==4 and pair['training_repetitions']==2
    assert pair['bootstrap_unit']=='comparison_run_cluster'


def test_frozen_pool_hash_mismatch_is_an_unverified_control(tmp_path):
    run=write_run(tmp_path,'B2-ASV','eval',[-10],trained=2)
    (run/'frozen_pool.json').write_text(json.dumps({'pool':{'wrong':'fixture'}}))
    rows,_=reporter.collect(tmp_path)
    assert rows[0]['split']=='untrained_control'


def test_completed_original_update_cannot_be_relabeled_as_trained_proposed(tmp_path):
    run=write_run(tmp_path,'B2','eval',[-10],trained=2)
    summary=json.loads((run/'summary.json').read_text());summary['config']['comparison_id']='B4-ASV-SSV'
    summary['config'].pop('evolution');(run/'summary.json').write_text(json.dumps(summary))
    rows,_=reporter.collect(tmp_path)
    assert rows[0]['split']=='untrained_control' and rows[0]['training_provenance_state']=='training_method_mismatch'


@pytest.mark.parametrize('with_training_manifest',[False,True])
def test_historical_inputs_restore_without_original_absolute_paths(tmp_path,with_training_manifest):
    from research.config import digest, policy_interface
    from research.progress import Journal
    from scripts.run_campaign import historical_batch
    original=tmp_path/'original';original.mkdir();current=tmp_path/'current';current.mkdir();report=current/'report';report.mkdir()
    config={'communication':'paper_projected_keep','action_contract':'explicit'}
    checkpoint=original/'B2.json';checkpoint.write_text(json.dumps({'stage':'train','history':[{'iteration':1}],
        'pool':{'fixture':1},'config_digest':digest(config)}))
    initial=original/'initial_pool.json';initial.write_text(json.dumps({'pool':{'fixture':0}}))
    traces=[]
    for i in range(2):
        path=original/f'episode-{i:04d}.jsonl';path.write_text(json.dumps({'seed':7+i,'unit_test_fixture_only':True})+'\n');traces.append(path)
    journal=Journal(original/'journal.sqlite3');journal.checkpoint('pending_batch:1:B2',{'experiences':['recorded']})
    journal.checkpoint('preparation:1:B2',{'ready':{'pool':'original'}});journal.db.close()
    paths=[checkpoint,initial,original/'journal.sqlite3',*traces]
    protocol={'fixed_update_source':{'checkpoint':str(checkpoint),'initial_pool':str(initial),
        'journal':str(original/'journal.sqlite3'),'training_traces':list(map(str,traces)),
        'files_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}}
    if with_training_manifest:
        training_manifest=original/'manifest.json'
        training_manifest.write_text(json.dumps({'stage':'train','config':config,'config_digest':digest(config)}))
        source=protocol['fixed_update_source']
        source.update(training_manifest=str(training_manifest),policy_interface=policy_interface(config))
        source['files_sha256'][str(training_manifest)]=hashlib.sha256(training_manifest.read_bytes()).hexdigest()
        protocol['configs']={'B2':config}
    (current/'protocol.json').write_text(json.dumps(protocol));(current/'frozen.json').write_bytes(checkpoint.read_bytes())
    reporter.reproduction_bundle(current,report)
    manifest=json.loads((report/'reproducibility_manifest.json').read_text())
    assert len(manifest['historical_dependencies'])==5+int(with_training_manifest)
    original.rename(tmp_path/'original_unavailable')
    restored=tmp_path/'isolated';mapped=reporter.restore_reproduction_bundle(report/'reproduction_bundle.tar.gz',restored)
    assert not any(Path(p).is_absolute() for p in mapped['fixed_update_source']['files_sha256'])
    trained,pending,prep=historical_batch(mapped,root=restored,require_manifest=with_training_manifest)
    assert trained['history']==[{'iteration':1}] and pending=={'experiences':['recorded']}
    assert prep=={'ready':{'pool':'original'}}
    assert [json.loads((restored/p).read_text())['seed'] for p in mapped['fixed_update_source']['training_traces']]==[7,8]
    assert mapped['original_fixed_update_source']==protocol['fixed_update_source']
    if with_training_manifest:
        mapped['configs']['B2']['communication_encoder']='acd_original_v1'
        with pytest.raises(ValueError,match='preserve the historical policy interface'):
            historical_batch(mapped,root=restored)
    else:
        with pytest.raises(ValueError,match='available for inspection only'):
            historical_batch(mapped,root=restored,require_manifest=True)


def test_source_or_frozen_pool_mismatch_is_not_merged(tmp_path):
    first=write_run(tmp_path,'B0','eval',[-10]);second=write_run(tmp_path,'B1','eval',[-5])
    for run,source in ((first,'source_a'),(second,'source_b')):
        (run/'manifest.json').write_text(json.dumps({'implementation_sha256':{'adapter.py':source}}))
    with pytest.raises(ValueError,match='conditions differ'):reporter.collect(tmp_path)


def test_regenerating_automatic_report_preserves_manual_research_findings(tmp_path):
    write_run(tmp_path,'B0','eval',[-10]);reporter.report_experiment(tmp_path,include_bundle=False)
    path=tmp_path/'report/performance.md';manual='Research findings: no gain claimed.\n\n'
    path.write_text(manual+path.read_text())
    reporter.report_experiment(tmp_path,include_bundle=False)
    reporter.report_experiment(tmp_path,include_bundle=False)
    assert path.read_text().startswith(manual)
    assert path.read_text().count('<!-- generated-comparison:start -->')==1
    assert path.read_text().count('# Experiment comparison')==1


def test_regeneration_after_source_change_preserves_existing_immutable_bundle(tmp_path):
    write_run(tmp_path,'B0','eval',[-10]);out=tmp_path/'report';out.mkdir()
    (tmp_path/'protocol.json').write_text(json.dumps({'source_sha256':{'research/gate.py':'historical source differs'}}))
    archive=out/'reproduction_bundle.tar.gz';archive.write_bytes(b'previous immutable archive fixture')
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest=out/'reproducibility_manifest.json';manifest.write_text(json.dumps({'sha256':digest}))
    before=manifest.read_bytes();result=reporter.report_experiment(tmp_path)
    assert result['bundle_action']=='preserved_registered_source_snapshot'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==digest and manifest.read_bytes()==before


def test_encoder_groups_split_full_reports_and_keep_implicit_legacy_compatible(tmp_path):
    for directory,label in ((tmp_path/'legacy','B0'),(tmp_path/'original','B0'),(tmp_path/'original','B1')):
        directory.mkdir(exist_ok=True)
        run=write_run(directory,label,'eval',[-4])
        if directory.name=='legacy':continue
        summary_path=run/'summary.json';episode_path=run/'episode-0000.json'
        summary=json.loads(summary_path.read_text());summary['config']['communication_encoder']='acd_original_v1'
        record=summary['episodes'][0];record['communication_encoder']='acd_original_v1'
        episode_path.write_text(json.dumps(record));summary_path.write_text(json.dumps(summary))
    explicit=tmp_path/'legacy-explicit';explicit.mkdir()
    run=write_run(explicit,'B1','eval',[-7]);path=run/'summary.json';value=json.loads(path.read_text())
    # The two legacy arms share a comparison run even though one recorded the default explicitly.
    value['config'].update(communication_encoder='legacy_local_v1',comparison_run_id='legacy')
    path.write_text(json.dumps(value))
    out=reporter.report_experiment(tmp_path,include_bundle=False)['output']
    metrics=json.loads((Path(out)/'metrics.json').read_text())
    assert {(g['label'],g['communication_encoder']) for g in metrics['groups']}=={
        (label,encoder) for label in ('B0','B1') for encoder in ('acd_original_v1','legacy_local_v1')}
    assert {p['communication_encoder']:p['mean_difference'] for p in metrics['paired_vs_B0']}=={
        'acd_original_v1':0,'legacy_local_v1':-3}


def test_optimizer_report_distinguishes_registered_seeds_from_completed_pairs(tmp_path):
    seeds=list(range(20));journal=reporter.Journal(tmp_path/'journal.sqlite3')
    candidates=[
        ('planned',{'seeds':seeds}),
        ('partial',{'reason':'incomplete_validation','registered_seeds':seeds,
                    'completed_paired_seeds':[0,1],'differences':{'0':1.0,'1':2.0},'validation_complete':False}),
        ('early',{'reason':'failed_registered_worst_seed_early','paired_seeds':[0],
                  'differences':{'0':-51.0},'required_approval_pairs':20,'validation_complete':False}),
        ('legacy',{'reason':'failed_registered_criteria','seeds':seeds,'differences':[0.0]*20}),
        ('complete',{'reason':'passed','registered_seeds':seeds,'completed_paired_seeds':seeds,
                     'seeds':seeds,'differences':[1.0]*20,'validation_complete':True}),
        ('claimed',{'reason':'passed','seeds':seeds,'validation_complete':True}),
        ('explicit_empty',{'reason':'passed','registered_seeds':seeds,'completed_paired_seeds':[],
                           'differences':[1.0]*20,'validation_complete':True}),
        ('duplicate',{'reason':'passed','registered_seeds':seeds,'completed_paired_seeds':[0]*20,
                      'differences':[1.0]*20,'validation_complete':True}),
    ]
    journal.checkpoint('optimizer:loop',{'status':'completed_unchanged','attempts':1,'result':{'evolution':[
        {'candidates':[{'skill':{'name':name},'admission_mode':'rollout','rollout_validation':validation}
                       for name,validation in candidates]}]}})
    journal.db.close()
    out=tmp_path/'report';out.mkdir();reporter.optimizer_report(tmp_path,out,[])
    report=json.loads((out/'optimizer_progress.json').read_text())['candidate_stages']
    actual={item['name']:(item['rollout_status'],item['rollout_completed_pairs'],item['rollout_completed']) for item in report}
    assert actual=={'planned':('planned',0,False),'partial':('partial',2,False),'early':('reject_only',1,False),
                    'legacy':('complete',20,True),'complete':('complete',20,True),'claimed':('planned',0,False),
                    'explicit_empty':('planned',0,False),'duplicate':('planned',0,False)}
    nonfinite=reporter.rollout_progress({'reason':'passed','registered_seeds':seeds,'completed_paired_seeds':seeds,
        'differences':[1.0]*19+[float('inf')],'validation_complete':True})
    assert (nonfinite['rollout_status'],nonfinite['rollout_completed_pairs'],nonfinite['rollout_completed'])==('partial',19,False)
