from pathlib import Path
import hashlib
import ast
import json
import yaml

ROOT=Path(__file__).resolve().parents[1]


def normalized_ast(node):
    for item in ast.walk(node):
        if getattr(item,'type_params',None)==[]:
            del item.type_params
    return ast.dump(node,include_attributes=False)


def configs():
    return {label:yaml.safe_load((ROOT/f'research_configs/rollout-{label}.yaml').read_text()) for label in ['B0','B1','B2','B3','B4']}


def test_public_labels_isolate_semantic_and_rollout_verification():
    c=configs()
    assert {k:v['arm'] for k,v in c.items()}=={'B0':'E0','B1':'E1','B2':'E2','B3':'E2','B4':'E2'}
    assert not c['B2']['evolution']['evidence_verification']
    assert c['B3']['evolution']['diagnosis_verification'] and not c['B3']['evolution']['rollout_verification']
    assert c['B4']['evolution']['rollout_verification']
    for key in ['composition','model','steps','communication','communication_encoder','profile','action_contract','scoring',
                'behavior_context','connection_leads','case_presentation','weights','selector','eval_seed']:
        assert all(value[key]==c['B0'][key] for value in c.values()),key
    assert len({c[label]['pool'] for label in ['B1','B2','B3','B4']})==1
    assert c['B0']['communication_encoder']=='acd_original_v1'
    assert c['B0']['action_contract']=='explicit_zone_pair_v1'


def test_reward_gate_token_backend_and_default_skill_components_are_protected():
    reference=json.loads((ROOT/'tests/fixtures/reference_integrity.json').read_text())
    source=ast.parse((ROOT/'research/skills.py').read_text());parts={}
    for node in source.body:
        name=getattr(node,'name',None)
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SEEDS' for t in node.targets):name='SEEDS'
        if name in reference['skills_protected_ast_sha256']:
            parts[name]=hashlib.sha256(normalized_ast(node).encode()).hexdigest()
    assert parts==reference['skills_protected_ast_sha256']
    for name in ['gate.py','metrics.py','paper_algorithm.py','adapter.py','skills.py']:
        assert hashlib.sha256((ROOT/'research'/name).read_bytes()).hexdigest()==reference['core_sha256'][name],name
    for file, protected in reference['unchanged_interface_ast_sha256'].items():
        tree=ast.parse((ROOT/file).read_text());current={}
        for node in tree.body:
            definitions=[(node.name,node)] if isinstance(node,ast.FunctionDef) else []
            if isinstance(node,ast.ClassDef):
                definitions.extend((node.name+'.'+item.name,item) for item in node.body if isinstance(item,ast.FunctionDef))
            for name,definition in definitions:
                if name in protected:current[name]=hashlib.sha256(normalized_ast(definition).encode()).hexdigest()
        assert current==protected,file


def test_rollout_approval_thresholds_are_not_relaxed():
    current=configs()['B4']['rollout_validation']
    old=json.loads((ROOT/'tests/fixtures/reference_integrity.json').read_text())['rollout_thresholds']
    for key in ['minimum_mean_gain','minimum_win_rate','minimum_median_gain','maximum_seed_loss',
                'maximum_mean_lwf_loss','maximum_mean_asf_loss','maximum_std_ratio','maximum_std_increase']:
        assert current[key]==old[key],key
    assert len(current['seeds'])==20


def test_instrumented_server_preserves_reference_probability_expressions():
    reference=json.loads((ROOT/'tests/fixtures/reference_integrity.json').read_text())
    tree=ast.parse((ROOT/'research/model_server.py').read_text());selected=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.ClassDef) and node.name=='TeacherForce':selected.append(normalized_ast(node))
        if isinstance(node,ast.FunctionDef) and node.name=='prompt_ids':selected.append(normalized_ast(node))
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='model' and node.func.attr in ('generate','compute_transition_scores'):
            selected.append(normalized_ast(node))
    assert hashlib.sha256(json.dumps(sorted(selected)).encode()).hexdigest()==reference['reference_probability_ast_sha256']


def test_ablation_and_new_heldout_seeds_are_explicit():
    c=configs();ablation=yaml.safe_load((ROOT/'research_configs/rollout-B3-no-support.yaml').read_text())
    assert ablation['comparison_id']=='B3-no-support'
    assert ablation['experiment_version']==c['B3']['experiment_version']
    assert ablation['evolution']['minimum_independent_episodes']==0
    assert c['B3']['evolution']['minimum_independent_episodes']==2
    assert {v['eval_seed'] for v in c.values()}=={201000}
    assert not set(range(201000,201100))&set(range(101000,101100))
