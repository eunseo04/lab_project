from pathlib import Path
import hashlib
import json
import yaml

ROOT=Path(__file__).resolve().parents[1]


def configs():
    return {label:yaml.safe_load((ROOT/f'research_configs/rollout-{label}.yaml').read_text()) for label in ['B0','B1','B2','B3','B4']}


def test_public_labels_isolate_semantic_and_rollout_verification():
    c=configs()
    assert {k:v['arm'] for k,v in c.items()}=={'B0':'E0','B1':'E1','B2':'E2','B3':'E2','B4':'E2'}
    assert not c['B2']['evolution']['evidence_verification']
    assert c['B3']['evolution']['diagnosis_verification'] and not c['B3']['evolution']['rollout_verification']
    assert c['B4']['evolution']['rollout_verification']
    for key in ['composition','model','steps','communication','profile','action_contract','scoring',
                'behavior_context','connection_leads','case_presentation','weights','selector','eval_seed']:
        assert all(value[key]==c['B0'][key] for value in c.values()),key
    assert len({c[label]['pool'] for label in ['B1','B2','B3','B4']})==1


def test_reward_gate_and_token_probability_implementation_are_unchanged():
    reference=json.loads((ROOT/'tests/fixtures/reference_integrity.json').read_text())
    for name in ['gate.py','metrics.py','paper_algorithm.py','adapter.py','model_server.py','client.py','skills.py']:
        assert hashlib.sha256((ROOT/'research'/name).read_bytes()).hexdigest()==reference['core_sha256'][name],name


def test_rollout_approval_thresholds_are_not_relaxed():
    current=configs()['B4']['rollout_validation']
    old=json.loads((ROOT/'tests/fixtures/reference_integrity.json').read_text())['rollout_thresholds']
    for key in ['minimum_mean_gain','minimum_win_rate','minimum_median_gain','maximum_seed_loss',
                'maximum_mean_lwf_loss','maximum_mean_asf_loss','maximum_std_ratio','maximum_std_increase']:
        assert current[key]==old[key],key
    assert len(current['seeds'])==20
