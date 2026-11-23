"""Audit the registered behavior campaign without filling unrun performance."""
import argparse
import ast
from collections import Counter,defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from behavior_metrics import episode_metrics,aggregate_behavior
from research.gate import returns_to_go

OUT=ROOT/"research_outputs/behavior-v2"
REPORT=OUT/"report"


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def mean(values):return statistics.mean(values) if values else None
def std(values):return statistics.stdev(values) if len(values)>1 else None


def write_csv(name,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (REPORT/name).open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def audit_sources(directory,protocol):
    manifest=read(directory/"manifest.json")
    assert manifest["implementation_sha256"]==protocol["implementation_sha256"]
    for name,digest in manifest["implementation_sha256"].items():
        assert sha(ROOT/protocol["source_snapshot"]/name)==digest
        assert sha(ROOT/name)==digest,"The registered implementation changed: "+name


def action_menu_audit():
    path=ROOT/"runtime/CybORG/Agents/Wrappers/CybermonicWrappers/globals.py"
    lists={}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.List):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ("NODE_ACTIONS","EDGE_ACTIONS","GLOBAL_ACTIONS"):
                    lists[target.id]=[item.id for item in node.value.elts]
    keep=lists["NODE_ACTIONS"]+lists["EDGE_ACTIONS"]+lists["GLOBAL_ACTIONS"]
    assert "Sleep" not in keep and "Monitor" in keep
    llm=["Analyse","Remove","Restore","DeployDecoy","BlockTrafficZone","AllowTrafficZone","Sleep"]
    return {"llm_controlled_explicit_actions":llm,"keep_explicit_actions":keep,
        "automatic_monitor_for_both":True,"llm_and_proposed_action_sets_identical":True,
        "same_action_set_for_keep_and_llm":False,
        "passive_analysis_category":["Sleep","Monitor"],
        "passive_category_is_descriptive_not_a_claim_of_identical_MDP":True,
        "sources":{str(p.relative_to(ROOT)):sha(p) for p in (path,ROOT/"runtime/CybORG/Agents/Wrappers/CybermonicWrappers/graph_wrapper.py",ROOT/"acd/llm_agent.py")}}


def audit_episode(path,stage,split,protocol):
    data=read(path);config=data["config"]
    assert data["status"]=="completed" and data["result_kind"]=="real_cage4"
    assert data["pool_frozen"] and data["initial_pool_sha256"]==data["pool_sha256"]
    assert config["steps"]==500 and config["scoring"]=="official"
    assert config["profile"]=="controlled-comparison" and config["action_contract"]=="explicit"
    assert config["communication"]=="paper_projected_keep" and config["skip_busy_policy"]
    assert config["pool_size"]==24
    if stage=="baseline_raw":assert config["behavior_context"]=="legacy"
    elif config["composition"]=="hybrid":
        assert config["behavior_context"]=="evidence_v1" and config["case_presentation"]=="readable"
    audit_sources(path.parent,protocol)
    assert len(data["episodes"])==1
    episode=data["episodes"][0]
    if config["composition"]=="hybrid":
        identity=episode["model_identity"]
        assert identity==protocol["fixed_model_identity"]
        assert identity["weights_frozen"] and identity["trainable_parameters"]==0
    rows=[json.loads(line) for line in Path(episode["trace"]).open()]
    assert len(rows)==2495 and episode["joint_transitions"]==499 and episode["scored_transitions"]==498
    ticks=defaultdict(list)
    for row in rows:ticks[row["step"]].append(row)
    total=0.0;phases=Counter()
    for tick,group in ticks.items():
        assert {r["agent"] for r in group}=={f"blue_agent_{i}" for i in range(5)}
        assert len({(r["joint_reward"],r["score_included"],r["terminal"]) for r in group})==1
        if group[0]["score_included"]:
            total+=group[0]["joint_reward"];phases[str(group[0]["mission_phase"])]+=group[0]["joint_reward"]
    assert total==episode["joint_return"]
    provenance=data.get("training_provenance")
    if config["arm"]=="E3":
        assert provenance and sha(provenance["path"])==provenance["sha256"]
        pool=read(provenance["path"])
        assert pool["pool"]["seed_set"]=="response_v1"
        if stage in ("static","validation_initial"):
            assert pool["initialization_only"] and pool["episodes_completed"]==0
        else:
            iteration=1 if stage in ("loop1","validation_loop1") else 2
            assert pool["history"][-1]["iteration"]==iteration
    agents=episode_metrics(episode,rows)
    meta={"selection_xml_errors":0,"termination_xml_errors":0,"unknown_selection_names":0}
    available={s["name"].lower() for s in read(provenance["path"])["pool"]["skills"]} if provenance else set()
    for call in episode["calls"]:
        if call["purpose"]=="skill_selection":
            match=re.search(r"<choice>\s*(.*?)\s*</choice>",call.get("response",""),re.I|re.S)
            meta["selection_xml_errors"]+=not bool(match)
            if match and match[1].lower()!="none" and match[1].lower() not in available:meta["unknown_selection_names"]+=1
        if call["purpose"]=="termination":
            meta["termination_xml_errors"]+=not bool(re.search(r"<status>\s*(DONE|CONTINUE)\s*</status>",call.get("response",""),re.I))
    first=next(r["state"] for r in rows if r["agent"]=="blue_agent_4" and r.get("policy_invoked")) if config["composition"]=="hybrid" else None
    return {"stage":stage,"split":split,"red":config["red"],"composition":config["composition"],"seed":episode["seed"],
        "joint_return":total,"phase_returns":dict(phases),"seconds":episode["execution_time_s"],
        "agents":agents,"meta":meta,"source":str(path.relative_to(ROOT)),"first_base_prompt":first}


def train_audit(protocol):
    path=OUT/"E3-hybrid-finite/train/summary.json"
    if not path.exists():return [],[],False
    summary=read(path);audit_sources(path.parent,protocol)
    config=summary["config"];assert config["validation_episodes"]==0
    assert config["evolution"]["schedule"]=="paper_batch"
    assert config["evolution"]["gate_mode"]=="paper_rtg" and config["evolution"]["maintenance_mode"]=="paper_algorithm1"
    loops=[];evolution=[];offset=0
    for h in summary.get("history",[]):
        batch=summary["episodes"][offset:h["episodes"]];offset=h["episodes"]
        assert mean([e["joint_return"] for e in batch])==h["reward_mean"]
        baseline=h["baseline"];behavior={};rtg={};visited=set()
        for episode in batch:
            assert episode["kind"]=="real_cage4" and episode["model_identity"]==protocol["fixed_model_identity"]
            assert episode["joint_transitions"]==499 and episode["scored_transitions"]==498
            rows=[json.loads(line) for line in Path(episode["trace"]).open()]
            assert len(rows)==2495
            ticks=defaultdict(list)
            for row in rows:ticks[row["step"]].append(row)
            rewards=[]
            for tick,items in sorted(ticks.items()):
                assert {r["agent"] for r in items}=={f"blue_agent_{i}" for i in range(5)}
                assert len({(r["joint_reward"],r["score_included"],r["terminal"]) for r in items})==1
                rewards.append(items[0]["joint_reward"] if items[0]["score_included"] else 0.)
            assert rewards==episode["scored_step_rewards"] and sum(rewards)==episode["joint_return"]
            assert episode_metrics(episode,rows)["blue_agent_4"]["stored_context_replay_verified"]
            for step,value in enumerate(returns_to_go(episode["scored_step_rewards"],config["evolution"]["gamma"])):
                rtg[(episode["seed"],step)]=value
            for row in rows:
                if row["agent"]=="blue_agent_4":
                    behavior[(row["seed"],row["agent"],row["step"])]=row
                    if row.get("skill_id"):visited.add(row["skill_id"])
        assert {e["parent"] for e in h["evolution"]}==visited
        accepted=[]
        for event in h["evolution"]:
            assert event["evidence_source"]=="current_training_batch"
            candidate_scores=[];parent_reference=None
            for candidate in event["candidates"]:
                if candidate.get("score") is None:continue
                trajectory_means=[];references=[]
                for diagnostic in candidate["diagnostics"]:
                    values=[]
                    for transition,old,new,adv in zip(diagnostic["transitions"],diagnostic["old_logprobs"],diagnostic["new_logprobs"],diagnostic["advantages"]):
                        row=behavior[(transition["seed"],transition["agent"],transition["step"])]
                        assert row["policy_invoked"] and row["request_applied"] and not row.get("parse_error")
                        assert row["skill_id"]==event["parent"] and row["skill_version"]==event["old_skill"]["version"]
                        assert abs(old-row["generation_logprob"])<1e-4
                        assert math.isclose(adv,rtg[(row["seed"],row["step"])]-baseline,abs_tol=1e-9,rel_tol=1e-9)
                        ratio=math.exp(new-old)
                        values.append(min(ratio*adv,min(1.2,max(.8,ratio))*adv))
                    assert len(values)==diagnostic["n"]
                    trajectory_means.append(mean(values));references.append(mean(diagnostic["advantages"]))
                assert math.isclose(mean(trajectory_means),candidate["score"],rel_tol=1e-9,abs_tol=1e-9)
                candidate_scores.append(candidate["score"]);parent_reference=mean(references)
                if candidate["accepted"]:
                    accepted.append(candidate["skill"]["name"])
            if not event["error"] and candidate_scores:
                chosen=[c for c in event["candidates"] if c["accepted"]]
                assert len(chosen)==int(max(candidate_scores)>0)
                if chosen:assert chosen[0]["score"]==max(candidate_scores)
            evolution.append({"iteration":h["iteration"],"parent":event["parent"],"scores":candidate_scores,
                "parent_reference_surrogate":parent_reference,"accepted":[c["skill"]["name"] for c in event["candidates"] if c["accepted"]],
                "valid_policy_samples":event["valid_behavior_decisions"],"semantic_chunks":len(event["semantic_gradients"]),"error":event["error"]})
        loops.append({"iteration":h["iteration"],"train_seeds":[e["seed"] for e in batch],"train_returns":[e["joint_return"] for e in batch],
            "train_mean":h["reward_mean"],"baseline":baseline,"parents":len(h["evolution"]),"accepted":accepted,
            "pool_size":h["pool_size"],"maintenance":h["maintenance"],"evolution_seconds":h["seconds"],
            "evolution_errors":sum(bool(e["error"]) for e in h["evolution"])})
    return loops,evolution,summary["status"]=="completed"


def plot(metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    finite=[m for m in metrics if m["stage"] in ("baseline_raw","baseline_context","static","loop1","loop2") and m["n"]]
    if finite:
        fig,ax=plt.subplots(figsize=(9,4.8));xs=list(range(len(finite)))
        for seed in range(22000,22004):ax.plot(xs,[m["returns_by_seed"].get(str(seed),float("nan")) for m in finite],"o--",alpha=.35,linewidth=1,label=f"seed {seed}")
        ax.errorbar(xs,[m["mean"] for m in finite],yerr=[m["sample_std"] or 0 for m in finite],fmt="ko-",capsize=4,label="Mean ± sample SD")
        ax.set_xticks(xs,[f"{m['stage']}\nN={m['n']}" for m in finite]);ax.set_ylabel("Official joint return (higher is better)")
        ax.set_title("Qwen3-8B / Hybrid / 500 steps / paired initial states")
        ax.grid(axis="y",alpha=.2);ax.legend(fontsize=8,ncol=2);fig.tight_layout();fig.savefig(REPORT/"paired_finite.png",dpi=180);plt.close(fig)
    actions=("Analyse","DeployDecoy","Remove","Restore","Sleep","Monitor","BlockTrafficZone","AllowTrafficZone")
    behavior=[m for m in metrics if m["stage"] in ("baseline_raw","baseline_context","static","loop1","loop2","all_rl_finite") and m["n"]]
    if behavior:
        fig,ax=plt.subplots(figsize=(10,5.5));bottom=[0.]*len(behavior)
        for action in actions:
            values=[m["behavior"]["free_choice_percent"].get(action,0) for m in behavior]
            ax.bar(range(len(behavior)),values,bottom=bottom,label=action);bottom=[a+b for a,b in zip(bottom,values)]
        ax.set_xticks(range(len(behavior)),[f"{m['stage']}\nN={m['n']}, free={m['behavior']['free_opportunities']}" for m in behavior],rotation=15)
        ax.set_ylabel("Applied requests on free ticks (%)");ax.set_ylim(0,100)
        ax.set_title("Blue 4 action requests: busy requests excluded")
        ax.legend(fontsize=8,ncol=4,loc="upper center",bbox_to_anchor=(.5,-.19))
        fig.text(.02,.01,"Automatic Monitor runs every tick. KEEP: explicit Monitor, no Sleep. LLM: explicit Sleep, no Monitor. Bars describe different menus.",fontsize=8)
        fig.tight_layout(rect=(0,.09,1,1));fig.savefig(REPORT/"action_distribution.png",dpi=180);plt.close(fig)
    transfer=[m for m in metrics if m["stage"] in ("transfer_baseline","transfer_loop2","all_rl_aggressive") and m["n"]]
    if transfer:
        fig,ax=plt.subplots(figsize=(7,4.5))
        ax.errorbar(range(len(transfer)),[m["mean"] for m in transfer],yerr=[m["sample_std"] or 0 for m in transfer],fmt="ko",capsize=5)
        for i,m in enumerate(transfer):ax.scatter([i]*m["n"],list(m["returns_by_seed"].values()),alpha=.5)
        ax.set_xticks(range(len(transfer)),[f"{m['stage']}\nN={m['n']}" for m in transfer]);ax.set_ylabel("Official joint return (higher is better)")
        ax.set_title("AggressiveFSM: frozen FiniteState-trained pool")
        ax.grid(axis="y",alpha=.2);fig.tight_layout();fig.savefig(REPORT/"transfer.png",dpi=180);plt.close(fig)


def main(require_complete=False):
    REPORT.mkdir(parents=True,exist_ok=True)
    protocol=read(OUT/"protocol.json");queue=read(OUT/"queue.json")
    menu=action_menu_audit()
    (REPORT/"action_menu_audit.json").write_text(json.dumps(menu,indent=2)+"\n")
    episodes=[];missing=[]
    for task in queue:
        if task["status"]=="completed":episodes.append(audit_episode(Path(task["summary"]),task["stage"],task["split"],protocol))
        else:missing.append(task["id"])
    for red in ("finite","aggressive"):
        for seed in protocol["heldout_seeds"]:
            path=OUT/"rl-controls"/f"{red}-seed-{seed}/summary.json"
            if path.exists() and read(path).get("status")=="completed":episodes.append(audit_episode(path,"all_rl_"+red,"control",protocol))
            else:missing.append(f"all_rl_{red}-seed-{seed}")
    groups=defaultdict(list)
    for episode in episodes:groups[episode["stage"]].append(episode)
    prompt_comparisons=0
    for seed in protocol["heldout_seeds"]:
        prompts=[e["first_base_prompt"] for e in episodes if e["seed"]==seed and e["stage"] in ("baseline_context","static","loop1","loop2")]
        assert len(set(prompts))<=1
        prompt_comparisons+=max(0,len(prompts)-1)
    for episode in episodes:episode.pop("first_base_prompt")
    metrics=[]
    for stage,items in groups.items():
        values=[e["joint_return"] for e in items]
        metrics.append({"stage":stage,"split":items[0]["split"],"n":len(items),"mean":mean(values),"sample_std":std(values),
            "returns_by_seed":{str(e["seed"]):e["joint_return"] for e in items},"behavior":aggregate_behavior(items),
            "mean_phase_returns":{str(p):mean([e["phase_returns"].get(str(p),0) for e in items]) for p in (0,1,2)},
            "meta":dict(sum((Counter(e["meta"]) for e in items),Counter()))})
    pairs=[]
    for target,base in [("baseline_context","baseline_raw"),("static","baseline_context"),("loop1","static"),("loop2","static"),
                        ("loop2","baseline_context"),("loop2","baseline_raw"),("loop2","loop1"),("transfer_loop2","transfer_baseline")]:
        a={e["seed"]:e for e in groups[target]};b={e["seed"]:e for e in groups[base]};common=sorted(a.keys()&b.keys())
        differences=[a[s]["joint_return"]-b[s]["joint_return"] for s in common]
        pairs.append({"comparison":target+" - "+base,"n":len(common),"seeds":common,"differences":differences,
            "mean_difference":mean(differences),"wins":sum(v>0 for v in differences)})
    loops,evolution,training_complete=train_audit(protocol)
    for loop in loops:
        validation=groups[f"validation_loop{loop['iteration']}"]
        loop["validation_return"]=validation[0]["joint_return"] if validation else None
    for name,digest in protocol["preserved_previous_results_sha256"].items():assert sha(ROOT/name)==digest
    preexisting=read(ROOT/"docs/preexisting_files_sha256.json")
    assert all(sha(ROOT/p)==digest for p,digest in preexisting.items())
    tests=ET.parse(ROOT/protocol["test_results"]).getroot().find("testsuite")
    assert int(tests.attrib["errors"])==int(tests.attrib["failures"])==0
    complete=not missing and training_complete and len(loops)==2
    result={"status":"completed" if complete else "partial","metric":"original official joint reward, higher is better",
        "metrics":metrics,"paired_differences":pairs,"closed_loops":loops,"evolution":evolution,"episodes":episodes,"missing":missing,"action_menu_audit":menu,
        "audit":{"passed":True,"heldout_completed":sum(e["split"]=="heldout" for e in episodes),
            "validation_completed":sum(e["split"]=="validation" for e in episodes),"all_rl_completed":sum(e["split"]=="control" for e in episodes),
            "training_complete":training_complete,"tests_passed":int(tests.attrib["tests"]),"previous_raw_files_preserved":len(protocol["preserved_previous_results_sha256"]),
            "training_worlds_and_observable_histories_audited":sum(len(h["train_seeds"]) for h in loops),
            "original_files_preserved":len(preexisting),"common_first_actor_prompt_equal":True if prompt_comparisons else None,
            "common_first_actor_prompt_comparisons":prompt_comparisons,
            "exact_gate_math_recomputed":True if evolution else None,"gate_candidate_scores_checked":sum(len(e["scores"]) for e in evolution),
            "observable_history_replayed":sum(bool(e["agents"]["blue_agent_4"]["stored_context_replay_verified"]) for e in episodes),
            "model_weights_frozen":True,"algorithm_changed_after_registration":False}}
    (REPORT/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    flat=[{**{k:v for k,v in m.items() if k!="behavior"},**m["behavior"]} for m in metrics]
    write_csv("performance.csv",flat);write_csv("paired_differences.csv",pairs)
    write_csv("closed_loops.csv",loops);write_csv("evolution.csv",evolution)
    write_csv("episodes.csv",[{k:v for k,v in e.items() if k!="agents"} for e in episodes])
    agents=[]
    for e in episodes:
        for agent,m in e["agents"].items():agents.append({"stage":e["stage"],"seed":e["seed"],"agent":agent,**m})
    write_csv("agent_behavior.csv",agents)
    plot(metrics)
    lines=["# 행동 이력·대응 Skill 수정 후 실제 성능","",f"상태: {result['status']}. 500-step / frozen Qwen3-8B / Hybrid / heldout {protocol['heldout_seeds']}.",
        "","| 방식 | 구분 | N | 평균 공동 보상 | 표본 SD | 시드별 보상 |","|---|---|---:|---:|---:|---|"]
    for m in metrics:lines.append(f"| {m['stage']} | {m['split']} | {m['n']} | {m['mean']} | {m['sample_std']} | {m['returns_by_seed']} |")
    lines += ["","## 대응 시드 차이","","| 비교 | N | 평균 차이 | 개선 시드 수 | 시드별 차이 |","|---|---:|---:|---:|---|"]
    for p in pairs:
        lines.append(f"| {p['comparison']} | {p['n']} | {p['mean_difference']} | {p['wins']} | {p['differences']} |")
    lines += ["","## Closed-loop 훈련·별도 검증","",
        "훈련 평균은 갱신 전 Pool의 서로 다른 훈련 시드 성능이다. 학습 효과는 위 동일 평가 시드의 동결 Pool 결과로 판단한다.","",
        "| Loop | 훈련 시드 | 훈련 보상 | 훈련 평균 | 검증 보상 | 승인 Skill | Pool |",
        "|---|---|---|---:|---:|---|---:|"]
    for h in loops:
        lines.append(f"| {h['iteration']} | {h['train_seeds']} | {h['train_returns']} | {h['train_mean']} | {h['validation_return']} | {', '.join(h['accepted'])} | {h['pool_size']} |")
    lines += ["","## Blue 4의 free tick 요청","",
        "아래는 각 조건 에피소드의 합계다. busy 중 요청·지속 행동의 tick·자동 Monitor를 신규 모델 선택으로 합산하지 않는다.","",
        "| 방식 | N | free / busy | Analyse | DeployDecoy | Remove | Restore | Sleep | 명시적 Monitor | Block / Allow |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|"]
    for m in metrics:
        if m["stage"] not in ("baseline_raw","baseline_context","static","loop1","loop2","all_rl_finite"):continue
        b=m["behavior"];a=b["free_choice_actions"]
        lines.append(f"| {m['stage']} | {m['n']} | {b['free_opportunities']} / {b['busy_ticks']} | {a.get('Analyse',0)} | {a.get('DeployDecoy',0)} | {a.get('Remove',0)} | {a.get('Restore',0)} | {a.get('Sleep',0)} | {a.get('Monitor',0)} | {a.get('BlockTrafficZone',0)} / {a.get('AllowTrafficZone',0)} |")
    lines += ["","## Gate 계산과 실제 개선의 구분","",
        "부모 r=1 기준은 같은 저장 행동·Advantage에 대한 surrogate이며 별도 부모 재실행 보상이 아니다. 승인 조건은 논문의 Best-of-N, J>0을 유지했다.","",
        "| Loop | Parent | 실제 정책 표본 | 부모 r=1 surrogate | 후보 J | 승인 |",
        "|---|---|---:|---:|---|---|"]
    for e in evolution:
        lines.append(f"| {e['iteration']} | {e['parent']} | {e['valid_policy_samples']} | {e['parent_reference_surrogate']} | {e['scores']} | {', '.join(e['accepted'])} |")
    lines += ["","공통 이력 효과는 baseline_context−baseline_raw, 정적 Skill 효과는 static−baseline_context, 학습 효과는 각 loop−static으로 분리한다.",
        "현재 KEEP 메뉴에는 explicit Monitor가 있고 Sleep이 없다. LLM 메뉴에는 Sleep이 있고 explicit Monitor가 없다. 자동 Monitor는 양쪽 모두 매 tick 수행한다. Sleep/Monitor는 보조적으로 수동적 선택 범주로 묶되 실제 메뉴와 개별 행동 횟수도 함께 보존한다.",
        "행동 분포는 busy tick을 제외한 실제 선택 기회를 분모로 한다. 이 공개 KEEP와 논문의 별도 학습된 KEEP를 동일한 정책으로 취급하지 않는다.",
        "free_choice_actions는 free tick에 적용된 요청이며 파싱 실패 시 fallback Sleep을 포함한다. 순수한 유효 모델 선택과 실패 fallback은 valid_generated_choice_actions·invalid_policy_fallback_actions로 별도 제공한다.",
        "environment_last_action_ticks는 환경이 매 tick 반환한 실행 행동 기록이다. 지속 행동의 여러 tick과 자동 Monitor를 포함할 수 있어 신규 결정 수와 더하거나 동일시하지 않는다. LLM의 완료 TRUE/FALSE도 agent_behavior.csv에 별도 보존한다.",
        "All RL은 동일 시드·Blue 4 위치의 대조이며 모든 Blue의 정책 구성이 Hybrid와 다르다. 한 에피소드 행동 빈도로 평균 방어 성능을 대신하지 않는다.",
        "N=4이며 표본 SD는 신뢰구간이 아니다. 관측 사례의 반응 지연·반복 지표는 진단이며 보상 함수를 대체하지 않는다.",
        "duplicate_successful_decoy는 같은 host의 과거 성공 후 재요청이다. 다른 서비스 추가나 Restore 후 재배치가 가능하므로 불필요한 행동으로 단정하지 않는다. 관측된 서비스별 재배치는 decoy_deployment_diagnostics에서 별도로 제공한다.",
        "실행 시간은 병행 GPU 추론의 대기 시간도 포함한다. 이 캠페인의 조건별 시간을 추론 효율의 비교 지표로 사용하지 않는다.",""]
    for file in ("paired_finite.png","action_distribution.png","transfer.png","reward_components.png"):
        if (REPORT/file).exists():lines.append(f"![{file}]({file})")
    (REPORT/"performance.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":result["status"],"audit":result["audit"],"missing":len(missing)},ensure_ascii=False,indent=2))
    if require_complete and not complete:raise SystemExit("Registered experiment is incomplete; do not claim final performance")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--require-complete",action="store_true")
    main(parser.parse_args().require_complete)
