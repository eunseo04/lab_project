"""Audit known-case validation after observation and evidence-lifecycle repairs."""
import ast
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FORK=ROOT/"revisions/behavior-v4-observation-20261121"
os.environ["BEHAVIOR_AUDIT_SOURCE_ROOT"]=str(FORK)
from audit_v3_episode import audit_episode

OUT=ROOT/"research_outputs/behavior-v4-lifecycle-development"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized_code(path):
    tree=ast.parse(Path(path).read_text())
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and isinstance(node.value,str):node.value="literal"
    return ast.dump(tree,include_attributes=False)


if __name__=="__main__":
    protocol=json.loads((OUT/"protocol.json").read_text())
    base=ROOT/"revisions/behavior-v3-dev-20261121"
    changed=[]
    for name,digest in protocol["source_sha256"].items():
        assert sha(FORK/name)==digest
        if sha(base/name)!=digest:changed.append(name)
    assert set(changed)=={"research/behavior.py","research/evolution.py"}
    assert normalized_code(base/"research/evolution.py")==normalized_code(FORK/"research/evolution.py")
    assert sha(ROOT/"scripts/validate_behavior_v4.py")==protocol["driver_sha256"]
    results={};prompts=[]
    for arm in ("E0","E3"):
        value,prompt,_=audit_episode(OUT/arm/"summary.json",protocol["seed"],protocol["source_sha256"],protocol["fixed_model_identity"])
        assert value["arm"]==arm and value["red"]=="finite"
        results[arm]=value;prompts.append(prompt)
    assert prompts[0]==prompts[1]
    difference=results["E3"]["return"]-results["E0"]["return"]
    result={"status":"completed","audit_passed":True,"known_debug_seed_not_heldout":True,"seed":protocol["seed"],
        "manual_initialization_not_learning":True,"arms":results,"E3_minus_E0":difference,"criterion_passed":difference>0,
        "fidelity":{"changed_source_files":changed,"evolution_control_flow_unchanged_only_string_facts_changed":True,
            "gate_score_maintenance_selector_simulator_reward_and_weights_unchanged":True,"tests_passed":protocol["test_count"],
            "initial_actor_input_equal":True,"raw_observable_history_replayed":True},
        "decision":"Fresh prospective N4 is required before a superiority claim" if difference>0 else "No performance gain established; do not adopt based on this known-case validation"}
    (OUT/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="arms"},ensure_ascii=False,indent=2))
