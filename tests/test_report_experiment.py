"""Synthetic arithmetic fixtures in temporary directories, never performance data."""
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
                 'scored_transitions':498,'unit_test_fixture_only':True}
        (directory/f'episode-{i:04d}.json').write_text(json.dumps(episode))
        episodes.append(episode)
    summary={'config':config,'stage':stage,'status':'partial' if stage=='train' else 'completed',
             'episodes':episodes[:1] if stage=='train' else episodes,'history':[],
             'training_provenance':{'episodes_completed':trained}}
    if trained:
        pool={'skills':[{'fixture_only':label}],'max_size':1};initial='0'*64
        checkpoint={'stage':'train','pool':pool,'episodes_completed':trained,'total_training_episodes':trained,
            'planned_training_episodes':trained,'planned_optimizer_iterations':1,'training_completed':True,
            'training_run_id':'fixture-'+label+'-'+root.name,'training_seeds':list(range(train_seed,train_seed+trained)),
            'optimizer_status':optimizer_status,'initial_pool_sha256':initial,
            'history':[{'iteration':1,'episodes':trained,'optimizer_status':optimizer_status}],
            'algorithm':config['evolution']}
        source=directory/'training-checkpoint.json';source.write_text(json.dumps(checkpoint))
        summary.update(pool_sha256=hashlib.sha256(json.dumps(pool,sort_keys=True).encode()).hexdigest(),pool_frozen=True,
            comparison_initial_pool_sha256=initial,training_provenance={'path':str(source.resolve()),
                'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'episodes_completed':trained,
                'training_seeds':checkpoint['training_seeds'],'optimizer_status':optimizer_status,'initial_pool_sha256':initial,
                'algorithm':checkpoint['algorithm']})
        (directory/'frozen_pool.json').write_text(json.dumps({'pool':pool}))
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


def test_historical_inputs_restore_without_original_absolute_paths(tmp_path):
    from research.progress import Journal
    from scripts.run_campaign import historical_batch
    original=tmp_path/'original';original.mkdir();current=tmp_path/'current';current.mkdir();report=current/'report';report.mkdir()
    checkpoint=original/'B2.json';checkpoint.write_text(json.dumps({'stage':'train','history':[{'iteration':1}],'pool':{'fixture':1}}))
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
    (current/'protocol.json').write_text(json.dumps(protocol));(current/'frozen.json').write_bytes(checkpoint.read_bytes())
    reporter.reproduction_bundle(current,report)
    manifest=json.loads((report/'reproducibility_manifest.json').read_text())
    assert len(manifest['historical_dependencies'])==5
    original.rename(tmp_path/'original_unavailable')
    restored=tmp_path/'isolated';mapped=reporter.restore_reproduction_bundle(report/'reproduction_bundle.tar.gz',restored)
    assert not any(Path(p).is_absolute() for p in mapped['fixed_update_source']['files_sha256'])
    trained,pending,prep=historical_batch(mapped,root=restored)
    assert trained['history']==[{'iteration':1}] and pending=={'experiences':['recorded']}
    assert prep=={'ready':{'pool':'original'}}
    assert [json.loads((restored/p).read_text())['seed'] for p in mapped['fixed_update_source']['training_traces']]==[7,8]
    assert mapped['original_fixed_update_source']==protocol['fixed_update_source']


def test_source_or_frozen_pool_mismatch_is_not_merged(tmp_path):
    first=write_run(tmp_path,'B0','eval',[-10]);second=write_run(tmp_path,'B1','eval',[-5])
    for run,source in ((first,'source_a'),(second,'source_b')):
        (run/'manifest.json').write_text(json.dumps({'implementation_sha256':{'adapter.py':source}}))
    with pytest.raises(ValueError,match='conditions differ'):reporter.collect(tmp_path)
