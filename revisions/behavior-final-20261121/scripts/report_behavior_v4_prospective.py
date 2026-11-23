"""Audit prospective v4 results and exact replays under the registered stop rule."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics

import os
os.environ["BEHAVIOR_AUDIT_SOURCE_ROOT"]=str(Path(__file__).resolve().parents[1]/"revisions/behavior-v4-observation-20261121")
from audit_v3_episode import ROOT,FORK,audit_episode
from behavior_metrics import aggregate_behavior
from research.config import validate

OUT=ROOT/"research_outputs/behavior-v4-prospective"


def read(path):return json.loads(Path(path).read_text())


def summarize_reward_replays(queue):
    """Check recorded inputs before aggregating the unchanged CAGE4 reward."""
    groups=defaultdict(list);missing=[]
    for task in queue:
        if task["status"]=="skipped_by_prespecified_rule":continue
        path=OUT/"reward-decomposition"/(task["id"]+".json")
        if task["status"]!="completed" or not path.exists():
            missing.append(task["id"]);continue
        audit=read(path);summary=Path(task["summary"]);data=read(summary)
        assert audit["passed"] and audit["all_blue_observations_actions_rewards_matched"]
        assert audit["recorded_llm_outputs_replayed"] and not audit["new_performance_sample"]
        assert not audit["decomposition_supplied_to_actor_or_optimizer"]
        assert (ROOT/audit["source"]).resolve()==summary.resolve()
        assert Path(audit["replay_source_root"]).resolve()==FORK.resolve()
        assert hashlib.sha256(summary.read_bytes()).hexdigest()==audit["source_summary_sha256"]
        episode=data["episodes"][0]
        trace=summary.parent/episode["trace"]
        assert hashlib.sha256(trace.read_bytes()).hexdigest()==audit["source_trace_sha256"]
        assert audit["seed"]==task["seed"]==episode["seed"]
        assert audit["joint_return"]==episode["joint_return"]==sum(audit["components"].values())
        groups[(task["red"],task["arm"])].append(audit)
    metrics=[]
    for (red,arm),items in sorted(groups.items()):
        metrics.append({"red":red,"arm":arm,"n":len(items),
            **{kind:statistics.mean(x["components"].get(kind,0) for x in items)
               for kind in ("action_cost","RIA","LWF","ASF")},
            "joint_return":statistics.mean(x["joint_return"] for x in items)})
    return {"status":"completed" if not missing else "partial","metrics":metrics,
        "exact_replays":sum(map(len,groups.values())),"missing":missing,"new_performance_samples":0,
        "limitation":"Exact recorded-trajectory attribution, not a causal action ablation; never supplied to actor or optimizer."}


def main(require_complete=False):
    protocol=read(OUT/"protocol.json");queue=read(OUT/"queue.json")
    for name,digest in protocol["source_sha256"].items():
        assert hashlib.sha256((FORK/name).read_bytes()).hexdigest()==digest
    assert hashlib.sha256((OUT/"initial_pool.json").read_bytes()).hexdigest()==protocol["pool_sha256"]
    assert hashlib.sha256((ROOT/"scripts/behavior_v4_prospective.py").read_bytes()).hexdigest()==protocol["driver_sha256"]
    registered=datetime.fromisoformat(protocol["registered_at_utc"]).timestamp()
    base=read(ROOT/"research_outputs/behavior-v4-lifecycle-development/E3-config.json")
    base.pop("development_validation",None);base.pop("known_debug_seed",None)
    used=set(range(base["train_seed"],base["train_seed"]+base["train_episodes"]))|{protocol["development_seed"]}
    finite_seeds=set(protocol["finite_seeds"]);transfer_seeds=set(protocol["conditional_transfer_seeds"])
    assert not (finite_seeds&transfer_seeds or (finite_seeds|transfer_seeds)&used)
    episodes=[];prompts={};missing=[];skipped=[]
    for task in queue:
        if task["status"]=="skipped_by_prespecified_rule":skipped.append(task["id"]);continue
        if task["status"]!="completed":missing.append(task["id"]);continue
        assert task["started_unix"]>=registered
        expected=json.loads(json.dumps(base))
        expected.update(arm=task["arm"],red=task["red"],eval_seed=task["seed"],episodes=1,validation_episodes=0,
            pool=str(OUT/"initial_pool.json") if task["arm"]=="E3" else None,output=str(OUT))
        expected["model"].update(endpoint=f"http://127.0.0.1:{task['port']}",scoring_endpoints=[])
        data=read(task["summary"])
        assert data["config"]==validate(expected)
        if task["arm"]=="E3":assert data["training_provenance"]["sha256"]==protocol["pool_sha256"]
        result,prompt,_=audit_episode(task["summary"],task["seed"],protocol["source_sha256"],protocol["fixed_model_identity"])
        assert result["arm"]==task["arm"] and result["red"]==task["red"]
        episodes.append(result);prompts[(task["red"],task["seed"],task["arm"])]=prompt
    groups=defaultdict(list)
    for episode in episodes:groups[(episode["red"],episode["arm"])].append(episode)
    metrics=[];pairs=[];prompt_pairs=0
    for red,seeds in (("finite",protocol["finite_seeds"]),("aggressive",protocol["conditional_transfer_seeds"])):
        for arm in ("E0","E3"):
            items=groups[(red,arm)];values=[e["return"] for e in items]
            if not items:continue
            behavior=aggregate_behavior([{"agents":{"blue_agent_4":e["behavior"]}} for e in items])
            metrics.append({"red":red,"arm":arm,"n":len(items),"mean":statistics.mean(values),
                "sample_std":statistics.stdev(values) if len(values)>1 else None,
                "returns_by_seed":{str(e["seed"]):e["return"] for e in items},"behavior":behavior})
        a={e["seed"]:e for e in groups[(red,"E3")]};b={e["seed"]:e for e in groups[(red,"E0")]}
        common=sorted(a.keys()&b.keys());differences=[a[s]["return"]-b[s]["return"] for s in common]
        for seed in common:
            assert prompts[(red,seed,"E0")]==prompts[(red,seed,"E3")];prompt_pairs+=1
        if common:
            complete=len(common)==4;value=statistics.mean(differences);wins=sum(d>0 for d in differences)
            pairs.append({"red":red,"n":len(common),"seeds":common,"differences":differences,
                "mean_difference":value,"wins":wins,"criterion_passed":value>0 and wins>=3 if complete else None})
    finite=next((p for p in pairs if p["red"]=="finite"),None)
    decision=read(OUT/"finite_decision.json") if (OUT/"finite_decision.json").exists() else None
    if skipped:
        assert len(skipped)==8 and decision and decision["passed"] is False
        assert finite and finite["criterion_passed"] is False
        assert all(t["red"]=="aggressive" for t in queue if t["id"] in skipped)
    if decision is not None:
        assert finite["differences"]==decision["paired_differences"] and finite["criterion_passed"]==decision["passed"]
    complete=not missing and len(episodes) in (8,16)
    if complete:
        assert finite and finite["n"]==4 and decision is not None
        assert (len(episodes)==16 and not skipped and decision["passed"]) or (len(episodes)==8 and len(skipped)==8 and not decision["passed"])
    reward_replays=summarize_reward_replays(queue)
    result={"status":"completed" if complete else "partial","manual_initialization_not_learning":True,
        "metrics":metrics,"paired_differences":pairs,"episodes":episodes,"missing":missing,"skipped_by_prespecified_rule":skipped,
        "transfer_status":"not_run_finite_criterion_failed" if skipped else "completed" if len(episodes)==16 else "pending",
        "audit":{"passed":True,"worlds_audited":len(episodes),"common_initial_actor_prompt_pairs":prompt_pairs,
            "weights_frozen":True,"pool_frozen":True,"learned_skill_decisions":0,"source_unchanged":True,
            "test_seeds_disjoint_from_development_and_training":True,"config_template_equal_except_declared_case_fields":True},
        "reward_replays":reward_replays,
        "limitation":"N=4 per policy, descriptive paired results; no learned Skill-Pro or universal superiority claim"}
    report=OUT/"report";report.mkdir(exist_ok=True)
    (report/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    (report/"reward_components.json").write_text(json.dumps(reward_replays,indent=2)+"\n")
    for name,rows in (("performance.csv",[{k:v for k,v in m.items() if k!="behavior"} for m in metrics]),("paired_differences.csv",pairs),
                      ("episodes.csv",[{k:v for k,v in e.items() if k!="behavior"} for e in episodes]),
                      ("reward_components.csv",reward_replays["metrics"])):
        if not rows:continue
        with (report/name).open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader()
            writer.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
    lines=["# v4 수동 Skill 보완의 전향적 평가","",f"상태: {result['status']}. 같은 v4 사건·연결 이력과 Sleep 선택지를 공유한다. Qwen3-8B/Hybrid/500-step, Pool 동결.",
        "","수동 7개 초기 Skill과 관찰 기반 검증의 평가다. closed-loop 학습 Skill의 개선 수치로 합산하지 않는다.","",
        "| Red | 방식 | N | 평균 공동 보상 | 표본 SD | 시드별 보상 |","|---|---|---:|---:|---:|---|"]
    for m in metrics:lines.append(f"| {m['red']} | {m['arm']} | {m['n']} | {m['mean']} | {m['sample_std']} | {m['returns_by_seed']} |")
    lines += ["","| Red | 대응 N | E3−E0 평균 | 개선 시드 | 시드별 차이 | 사전 기준 통과 |","|---|---:|---:|---:|---|---|"]
    for p in pairs:lines.append(f"| {p['red']} | {p['n']} | {p['mean_difference']} | {p['wins']} | {p['differences']} | {p['criterion_passed']} |")
    lines += ["","사전 기준은 평균 차이 양수·4개 시드 중 3개 이상 개선이다. FiniteState에서 통과하지 않으면 AggressiveFSM은 사전 조건으로 미실시한다. 미실시를 0점이나 성공으로 표시하지 않는다.",
        f"전이 평가 상태: {result['transfer_status']}. N=4의 표본 SD는 신뢰구간이 아니다. 오류 분석에 사용한 시드 24001을 최종 표본으로 재사용하지 않는다.",""]
    lines += ["오류 분석 시드 24001, 새 평가 29000–29003, 조건부 전이 30000–30003을 고정 프로토콜로 구분한다. 기존 개발 태그를 새 평가 설정에서 제거했으며 어느 평가도 Pool을 학습시키지 않는다.",
              "미완료 표의 조건별 N이 다르면 평균 차이를 최종 대응 효과로 해석하지 않는다.",""]
    lines += ["## 기록된 궤적의 원본 보상 분해","",
              "모든 Blue의 관측·행동·보상이 원본과 일치한 재생만 집계했다. 입력 summary와 trace의 hash도 확인했다. 새 성능 표본이나 단일 행동의 인과 분석이 아니다.","",
              "| Red | 방식 | N | 팀 행동 비용 | Red Impact | 로컬 작업 실패 | 서비스 접근 실패 | 합계 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for m in reward_replays["metrics"]:
        lines.append(f"| {m['red']} | {m['arm']} | {m['n']} | {m['action_cost']} | {m['RIA']} | {m['LWF']} | {m['ASF']} | {m['joint_return']} |")
    lines.append("")
    if metrics:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        reds=[red for red in ("finite","aggressive") if any(m["red"]==red for m in metrics)]
        fig,axes=plt.subplots(1,len(reds),figsize=(6*len(reds),4.8),squeeze=False)
        for ax,red in zip(axes[0],reds):
            a={e["seed"]:e for e in groups[(red,"E0")]};b={e["seed"]:e for e in groups[(red,"E3")]}
            for seed in sorted(a.keys()&b.keys()):ax.plot([0,1],[a[seed]["return"],b[seed]["return"]],"o--",alpha=.4,label=f"seed {seed}")
            for x,arm in enumerate(("E0","E3")):
                m=next((m for m in metrics if m["red"]==red and m["arm"]==arm),None)
                if m:ax.errorbar([x],[m["mean"]],yerr=[m["sample_std"] or 0],fmt="ko",capsize=4)
            labels=[]
            for arm,label in (("E0","ACD + same v4 evidence"),("E3","Manual v4 skills")):
                labels.append(f"{label}\nN={len(groups[(red,arm)])}")
            ax.set_xticks([0,1],labels);ax.set_title(red)
            ax.set_ylabel("Official joint return (higher is better)");ax.grid(axis="y",alpha=.2);ax.legend(fontsize=8)
        fig.suptitle("Prospective frozen manual skills / Qwen3-8B / Mean ± sample SD")
        fig.tight_layout();fig.savefig(report/"paired.png",dpi=180);plt.close(fig)
        lines.append("![Prospective paired scores](paired.png)")
    (report/"performance.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":result["status"],"episodes":len(episodes),"missing":len(missing),"skipped":len(skipped),"paired":pairs},indent=2))
    if require_complete and not complete:raise SystemExit("Prospective experiment is incomplete")
    if require_complete and reward_replays["missing"]:raise SystemExit("Exact reward replays are incomplete")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--require-complete",action="store_true")
    main(parser.parse_args().require_complete)
