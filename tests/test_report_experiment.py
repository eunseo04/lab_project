"""Synthetic arithmetic fixtures in temporary directories, never performance data."""
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('report_experiment',ROOT/'scripts/report_experiment.py')
reporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(reporter)


def write_run(root,label,stage,rewards,*,model='fixture',trained=0):
    directory=root/f'{label}-{stage}';directory.mkdir()
    config={'comparison_id':label,'arm':'E0' if label=='B0' else 'E2','red':'finite',
            'steps':500,'composition':'hybrid','experiment_version':'unit_fixture',
            'model':{'model':model,'revision':'unit_fixture'}}
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
