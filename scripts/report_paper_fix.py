"""Audit and report predeclared, real paper-conformant CAGE4 experiments."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import re
from pathlib import Path
import statistics
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"research_outputs/paper-fix"
REPORT=OUT/"report"
STAGES=("baseline","static_verified","loop1","loop2","transfer_baseline","transfer_loop2")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_file(name,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (REPORT/name).open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)


def mean(values):return statistics.mean(values) if values else None
def sd(values):return statistics.stdev(values) if len(values)>1 else None


def audit_implementation(manifest):
    # Long-run source is immutable even when a later, separately tested fix is
    # applied. Check the recorded bytes, never relabel an old run as new code.
    archives=(ROOT/"revisions/paper-fix-long-run-20261120",ROOT/"revisions/paper-fix-name-before-ablation-20261120",
              ROOT/"revisions/paper-fix-final-20261120")
    for name,checksum in manifest["implementation_sha256"].items():
        candidates=[ROOT/name]+[archive/name for archive in archives]
        assert any(p.is_file() and sha(p)==checksum for p in candidates),name


def admitted_name(candidate,history):
    # Candidate logs precede the existing collision suffix. Resolve the actual
    # admitted pool name from its text and lineage without editing raw evidence.
    fields=("initiation","policy","termination","version","parent_id")
    matches=[s["name"] for s in history["pool_snapshot"]["skills"]
             if all(s.get(k)==candidate["skill"].get(k) for k in fields)]
    if matches:
        assert len(matches)==1
        return matches[0],True
    name=candidate["skill"]["name"]
    assert any(m["name"]==name for m in history["maintenance"])
    return name,False


def audit_episode(summary_path,stage):
    summary=read(summary_path);config=summary["config"]
    assert summary["status"]=="completed" and summary["result_kind"]=="real_cage4"
    assert summary["pool_frozen"] and summary["initial_pool_sha256"]==summary["pool_sha256"]
    expected={"arm":"E0" if stage in ("baseline","transfer_baseline") else "E3",
        "red":"aggressive" if stage.startswith("transfer") else "finite",
        "composition":"hybrid","steps":500,"scoring":"official","action_contract":"explicit",
        "communication":"paper_projected_keep"}
    assert all(config[k]==v for k,v in expected.items())
    assert len(summary["episodes"])==1
    ep=summary["episodes"][0];identity=ep["model_identity"]
    assert identity["model"]=="Qwen/Qwen3-8B" and identity["weights_frozen"]
    assert identity["revision"]=="b968826d9c46dd6066d109eabc6255188de91218"
    assert ep["joint_transitions"]==499 and ep["scored_transitions"]==498
    rows=[json.loads(x) for x in Path(ep["trace"]).open()]
    ticks=defaultdict(list)
    for row in rows:ticks[row["step"]].append(row)
    assert len(ticks)==499
    total=0
    for group in ticks.values():
        assert {r["agent"] for r in group}=={f"blue_agent_{i}" for i in range(5)}
        assert len({(r["joint_reward"],r["score_included"],r["terminal"]) for r in group})==1
        if group[0]["score_included"]:total+=group[0]["joint_reward"]
    assert total==ep["joint_return"]
    assert len(rows)==2495
    blue=[r for r in rows if r["agent"]=="blue_agent_4"]
    phase_returns=Counter()
    for row in blue:
        if row["score_included"]:phase_returns[str(row["mission_phase"])]+=row["joint_reward"]
    decisions=[r for r in blue if r.get("policy_invoked")]
    evolved=[r for r in decisions if (r.get("skill_version") or 0)>1]
    terms=Counter(e["reason"] for r in blue for e in (r.get("skill_controller") or {}).get("events",[]) if e.get("terminated"))
    for i,row in enumerate(blue):
        for event in (row.get("skill_controller") or {}).get("events",[]):
            if event.get("reason")=="repeated_action_without_new_evidence":
                assert i>=2 and all(r.get("policy_invoked") and r.get("request_applied") and r["requested_action"]!="Sleep" for r in blue[i-2:i])
    provenance=summary.get("training_provenance")
    valid_names={"none"}
    if stage not in ("baseline","transfer_baseline"):
        assert provenance and sha(provenance["path"])==provenance["sha256"]
        pool=read(provenance["path"])
        valid_names.update(s["name"].lower() for s in pool["pool"]["skills"])
        if stage=="static_verified":assert pool.get("initialization_only") and pool["episodes_completed"]==0
        else:
            expected_iteration=1 if stage=="loop1" else 2
            assert pool["history"][-1]["iteration"]==expected_iteration
    manifest=read(summary_path.parent/"manifest.json")
    audit_implementation(manifest)
    unknown=[]
    for call in ep["calls"]:
        if call["purpose"]!="skill_selection":continue
        match=re.search(r"<choice>\s*(.*?)\s*</choice>",call.get("response",""),re.I|re.S)
        if match and match[1].lower() not in valid_names:unknown.append(match[1])
    return {"stage":stage,"red":config["red"],"arm":config["arm"],"seed":ep["seed"],
        "joint_return":total,"wall_s":ep["execution_time_s"],"policy_decisions":len(decisions),
        "policy_errors":sum(bool(r.get("parse_error")) for r in decisions),
        "evolved_policy_decisions":len(evolved),"evolved_policy_fraction":len(evolved)/max(1,len(decisions)),
        "action_counts":dict(Counter(r["requested_action"].split()[0] for r in blue)),
        "policy_action_counts":dict(Counter(r["requested_action"].split()[0] for r in decisions)),
        "automatic_wait_ticks":sum(not r.get("policy_invoked") and r["requested_action"]=="Sleep" for r in blue),
        "skill_action_counts":{skill:dict(Counter(r["requested_action"].split()[0] for r in decisions
                                                   if (r.get("skill_id") or "NONE")==skill))
                               for skill in sorted({r.get("skill_id") or "NONE" for r in decisions})},
        "phase_returns":dict(phase_returns),
        "completed_action_counts":dict(Counter(a["name"] for r in blue for a in r["executed_actions"])),
        "skill_usage":dict(Counter(r.get("skill_id") or "NONE" for r in decisions)),
        "terminations":dict(terms),"model_calls":len(ep["calls"]),"source":str(summary_path.relative_to(ROOT)),
        "termination_format_errors":sum(c["purpose"]=="termination" and not re.search(r"<status>\s*(DONE|CONTINUE)\s*</status>",c.get("response",""),re.I) for c in ep["calls"]),
        "selection_noncanonical_responses":sum(c["purpose"]=="skill_selection" and not re.search(r"<choice>.*?</choice>",c.get("response",""),re.I|re.S) for c in ep["calls"]),
        "selection_unknown_skill_responses":len(unknown),"selection_unknown_skill_names":dict(Counter(unknown)),
        "initial_base_prompt":blue[0]["state"],"pool_sha256":summary["pool_sha256"]}


def training_history():
    path=OUT/"E3-hybrid-finite/train/summary.json"
    if not path.exists():return [],[],False
    summary=read(path);loops=[];evolutions=[];offset=0
    audit_implementation(read(path.parent/"manifest.json"))
    identity_check=read(OUT/"candidate-identity-audit.json")
    for h in summary.get("history",[]):
        episodes=summary["episodes"][offset:h["episodes"]];offset=h["episodes"]
        assert mean([e["joint_return"] for e in episodes])==h["reward_mean"]
        seeds={e["seed"] for e in episodes}
        behavior={};rtg={}
        gamma=summary["config"]["evolution"]["gamma"]
        for episode in episodes:
            accumulated=0.0
            for step in range(len(episode["scored_step_rewards"])-1,-1,-1):
                accumulated=episode["scored_step_rewards"][step]+gamma*accumulated
                rtg[(episode["seed"],step)]=accumulated
            for line in Path(episode["trace"]).open():
                row=json.loads(line)
                if row["agent"]=="blue_agent_4":behavior[(row["seed"],row["agent"],row["step"])]=row
        accepted=[];retained=[]
        for ev in h["evolution"]:
            assert ev["evidence_source"]=="current_training_batch"
            scores=[]
            for c in ev["candidates"]:
                if c.get("score") is None:continue
                traj_scores=[]
                for d in c["diagnostics"]:
                    assert {t["seed"] for t in d["transitions"]}<=seeds
                    assert all(t["skill_id"]==ev["parent"] for t in d["transitions"])
                    for transition,old,adv in zip(d["transitions"],d["old_logprobs"],d["advantages"]):
                        row=behavior[(transition["seed"],transition["agent"],transition["step"])]
                        assert row["policy_invoked"] and row["request_applied"] and not row.get("parse_error")
                        assert row["skill_id"]==ev["parent"] and row["skill_version"]==ev["old_skill"]["version"]
                        assert abs(row["generation_logprob"]-old)<1e-4
                        expected_adv=rtg[(row["seed"],row["step"])]-h["baseline"]
                        assert math.isclose(expected_adv,adv,rel_tol=1e-9,abs_tol=1e-9)
                    terms=[]
                    for old,new,adv in zip(d["old_logprobs"],d["new_logprobs"],d["advantages"]):
                        ratio=math.exp(new-old)
                        terms.append(min(ratio*adv,min(1.2,max(.8,ratio))*adv))
                    assert len(terms)==d["n"]
                    traj_scores.append(mean(terms))
                assert math.isclose(mean(traj_scores),c["score"],rel_tol=1e-9,abs_tol=1e-9)
                scores.append(c["score"])
                if c["accepted"]:
                    assert c["score"]>0
                    name,kept=admitted_name(c,h)
                    accepted.append(name)
                    if kept:retained.append(name)
            if ev["error"] is None and scores:
                chosen=[c for c in ev["candidates"] if c["accepted"]]
                assert len(chosen)==int(max(scores)>0)
                if chosen:assert chosen[0]["score"]==max(scores)
            verified_scores=list(scores)
            if h["iteration"]==2 and ev["parent"]=="WaitVerification":
                verified_scores=[]
                for original,reverified in zip(ev["candidates"],identity_check["candidates"]):
                    values=[]
                    assert original["index"]==reverified["index"]
                    for before,after in zip(original["diagnostics"],reverified["diagnostics"]):
                        for key in ("old_logprobs","advantages","transitions"):
                            assert before[key]==after[key]
                        terms=[]
                        for old,new,adv in zip(after["old_logprobs"],after["new_logprobs"],after["advantages"]):
                            ratio=math.exp(new-old)
                            terms.append(min(ratio*adv,min(1.2,max(.8,ratio))*adv))
                        values.append(mean(terms))
                    assert math.isclose(mean(values),reverified["verified_score"],rel_tol=1e-9,abs_tol=1e-9)
                    verified_scores.append(reverified["verified_score"])
                original_best=next(c["index"] for c in ev["candidates"] if c["accepted"])
                assert max(verified_scores)>0 and verified_scores.index(max(verified_scores))==original_best
            first=next((c for c in ev["candidates"] if c.get("score") is not None),None)
            parent_reference=mean([mean(d["advantages"]) for d in first["diagnostics"]]) if first else None
            evolutions.append({"iteration":h["iteration"],"parent":ev["parent"],"frequency":ev["old_skill"]["frequency"],
                "valid_behavior_decisions":ev["valid_behavior_decisions"],"candidate_scores":scores,
                "final_name_verified_scores":verified_scores,"parent_ratio1_surrogate":parent_reference,
                "best_minus_parent_surrogate":max(verified_scores)-parent_reference if verified_scores else None,
                "accepted":[admitted_name(c,h)[0] for c in ev["candidates"] if c["accepted"]],
                "retained_after_maintenance":[admitted_name(c,h)[0] for c in ev["candidates"]
                                              if c["accepted"] and admitted_name(c,h)[1]],"error":ev["error"]})
        loops.append({"iteration":h["iteration"],"train_seeds":sorted(seeds),
            "train_returns":[e["joint_return"] for e in episodes],"train_mean":h["reward_mean"],
            "rtg_baseline":h["baseline"],"validation_seed":5000,"validation_return":h["validation_reward_mean"],
            "validation_episode_count":len(h["validation_episodes"]),
            "parents_visited":len(h["evolution"]),"accepted_skills":accepted,"retained_new_skills":retained,
            "maintenance":h["maintenance"],"pool_size":h["pool_size"],
            "evolution_errors":sum(e["error"] is not None for e in h["evolution"]),"evolution_wall_s":h["seconds"],
            "semantic_chunks":sum(len(e["semantic_gradients"]) for e in h["evolution"])})
    return loops,evolutions,summary["status"]=="completed"


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    finite=[r for r in rows if r["stage"] in ("baseline","static_verified","loop1","loop2") and r["n"]]
    if not finite:return
    fig,ax=plt.subplots(figsize=(8,4.5));xs=list(range(len(finite)))
    for seed in (12000,12001,12002,12003):
        ax.plot(xs,[r["returns_by_seed"].get(str(seed),float("nan")) for r in finite],"o--",alpha=.35,linewidth=1,label=f"seed {seed}")
    ax.errorbar(xs,[r["mean"] for r in finite],yerr=[r["sample_std"] or 0 for r in finite],fmt="ko-",capsize=5,linewidth=2,label="Mean ± sample SD")
    ax.set_xticks(xs,[r["stage"] for r in finite]);ax.set_ylabel("Official Blue joint return (higher is better)")
    ax.set_title("Paper-conformant Skill-Pro / Qwen3-8B / Hybrid / 500 steps")
    ax.grid(axis="y",alpha=.2);ax.legend(fontsize=8,ncol=2);fig.tight_layout()
    fig.savefig(REPORT/"finite_learning.png",dpi=180);plt.close(fig)
    transfer=[r for r in rows if r["stage"] in ("transfer_baseline","transfer_loop2") and r["n"]]
    if transfer:
        fig,ax=plt.subplots(figsize=(6.5,4.5));xs=list(range(len(transfer)))
        for seed in (12000,12001,12002,12003):
            ax.plot(xs,[r["returns_by_seed"].get(str(seed),float("nan")) for r in transfer],"o--",alpha=.35,linewidth=1,label=f"seed {seed}")
        ax.errorbar(xs,[r["mean"] for r in transfer],yerr=[r["sample_std"] or 0 for r in transfer],fmt="ko-",capsize=5,linewidth=2,label="Mean ± sample SD")
        ax.set_xticks(xs,[r["stage"] for r in transfer]);ax.set_ylabel("Official Blue joint return (higher is better)")
        ax.set_title("AggressiveFSM transfer / FiniteState-trained frozen pool")
        ax.grid(axis="y",alpha=.2);ax.legend(fontsize=8,ncol=2);fig.tight_layout()
        fig.savefig(REPORT/"aggressive_transfer.png",dpi=180);plt.close(fig)


def main(require_complete=False):
    REPORT.mkdir(parents=True,exist_ok=True)
    protocol=read(OUT/"protocol.json");seeds=protocol["heldout_seeds"]
    episodes=[];missing=[]
    for stage in STAGES:
        for seed in seeds:
            path=OUT/f"paired/{stage}-seed-{seed}/summary.json"
            if path.exists() and read(path).get("status")=="completed":episodes.append(audit_episode(path,stage))
            else:missing.append({"stage":stage,"seed":seed})
    by_stage={stage:[e for e in episodes if e["stage"]==stage] for stage in STAGES}
    metrics=[]
    for stage,items in by_stage.items():
        values=[e["joint_return"] for e in items]
        metrics.append({"stage":stage,"n":len(items),"planned_n":len(seeds),"mean":mean(values),"sample_std":sd(values),
            "returns_by_seed":{str(e["seed"]):e["joint_return"] for e in items},
            "policy_decisions":sum(e["policy_decisions"] for e in items),"policy_errors":sum(e["policy_errors"] for e in items),
            "termination_format_errors":sum(e["termination_format_errors"] for e in items),
            "selection_noncanonical_responses":sum(e["selection_noncanonical_responses"] for e in items),
            "selection_unknown_skill_responses":sum(e["selection_unknown_skill_responses"] for e in items),
            "evolved_policy_decisions":sum(e["evolved_policy_decisions"] for e in items)})
        metrics[-1]["mean_phase_returns"]={str(phase):mean([e["phase_returns"].get(str(phase),0) for e in items]) for phase in (0,1,2)}
        metrics[-1]["policy_action_counts"]=dict(sum((Counter(e["policy_action_counts"]) for e in items),Counter()))
        metrics[-1]["automatic_wait_ticks"]=sum(e["automatic_wait_ticks"] for e in items)
    pairs=[]
    for target,base in [("loop1","baseline"),("loop2","baseline"),("loop1","static_verified"),
                        ("loop2","static_verified"),("loop2","loop1"),("transfer_loop2","transfer_baseline")]:
        a={e["seed"]:e for e in by_stage[target]};b={e["seed"]:e for e in by_stage[base]};common=sorted(a.keys()&b.keys())
        differences=[a[s]["joint_return"]-b[s]["joint_return"] for s in common]
        pairs.append({"comparison":target+" - "+base,"n":len(common),"seeds":common,
            "differences":differences,"mean_difference":mean(differences),"wins":sum(d>0 for d in differences)})
    # The common action prompt is identical before adding a Skill, including
    # exactly the same first observation and agent-specific target catalog.
    for seed in seeds:
        same=[e["initial_base_prompt"] for e in episodes if e["seed"]==seed and not e["stage"].startswith("transfer")]
        assert len(set(same))<=1
    for e in episodes:e.pop("initial_base_prompt")
    loops,evolutions,training_complete=training_history()
    changed=[p for p,checksum in protocol["preserved_raw_sha256"].items() if sha(ROOT/p)!=checksum]
    assert not changed,changed
    preexisting=read(ROOT/"docs/preexisting_files_sha256.json")
    assert not [p for p,checksum in preexisting.items() if sha(ROOT/p)!=checksum]
    complete=not missing and training_complete and len(loops)==2
    tests_path=ROOT/"research_outputs/paper-fix-final-tests.xml"
    if not tests_path.exists():tests_path=ROOT/"research_outputs/paper-fix-tests.xml"
    tests=ET.parse(tests_path).getroot().find("testsuite")
    assert int(tests.attrib["failures"])==int(tests.attrib["errors"])==0
    identity_audit=read(OUT/"candidate-identity-audit.json")
    assert identity_audit["passed"] and identity_audit["accepted_index_unchanged"] and identity_audit["persisted_pool_unchanged"]
    result={"status":"completed" if complete else "partial","metric":"official joint reward; higher is better",
        "metrics":metrics,"paired_differences":pairs,"closed_loops":loops,"evolution":evolutions,
        "episodes":episodes,"missing":missing,"audit":{"passed":True,"real_eval_episodes":len(episodes),
        "real_training_episodes":sum(len(h["train_returns"]) for h in loops),
        "real_validation_episodes":sum(h["validation_episode_count"] for h in loops),
        "preexisting_files_preserved":len(preexisting),
        "old_raw_files_preserved":len(protocol["preserved_raw_sha256"]),"common_first_prompt_equal":True,
        "test_count":int(tests.attrib["tests"]),"test_failures":int(tests.attrib["failures"]),
        "scoring_parity":read(OUT/"scoring-parity.json")["passed"],
        "candidate_final_name_gate_reverified":True,
        "historical_implementation_bytes_preserved":True}}
    (REPORT/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    for name,data in [("performance.csv",metrics),("paired_differences.csv",pairs),("episodes.csv",episodes),
        ("closed_loops.csv",loops),("evolution.csv",evolutions)]:csv_file(name,data)
    text=["# 수정 후 실제 Qwen 성능", "", f"상태: {result['status']}. 500-step / Hybrid / 동일 Qwen3-8B / seeds {seeds}.","",
        "| 방식 | N | 평균 보상 | 표본 SD | seed별 보상 |", "|---|---:|---:|---:|---|"]
    for r in metrics:
        text.append(f"| {r['stage']} | {r['n']} | {r['mean']} | {r['sample_std']} | {r['returns_by_seed']} |")
    text += ["", "| Loop | 훈련 보상 | 훈련 평균 | 검증 보상 | 방문 Parent | 승인 Skill | Pool |", "|---|---|---:|---:|---:|---|---:|"]
    for r in loops:text.append(f"| {r['iteration']} | {r['train_returns']} | {r['train_mean']} | {r['validation_return']} | {r['parents_visited']} | {r['accepted_skills']} | {r['pool_size']} |")
    text += ["", "학습 전후 Skill 효과는 loop1/loop2와 static_verified의 대응 seed 차이로 판단한다.",
        "공통 행동 인터페이스가 바뀌었으므로, 과거 다른 seed의 표와 차이를 계산해 학습 효과라고 해석하지 않는다.",
        "N=4의 관측된 평균 차이는 광범위한 통계적 우위의 증명이 아니다.", "", "![같은 seed의 수정 후 비교](finite_learning.png)"]
    if any(r["stage"]=="transfer_loop2" and r["n"] for r in metrics):
        text += ["", "![학습하지 않은 Red 정책에 대한 전이](aggressive_transfer.png)"]
    (REPORT/"performance.md").write_text("\n".join(text)+"\n")
    plot(metrics)
    print(json.dumps({"status":result["status"],"real_eval_episodes":len(episodes),"missing":len(missing),
        "closed_loops":len(loops),"pairs":pairs,"old_raw_files_preserved":len(protocol["preserved_raw_sha256"])},ensure_ascii=False,indent=2))
    if require_complete and not complete:raise SystemExit("Predeclared experiment is incomplete; do not claim completed performance")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--require-complete",action="store_true")
    main(parser.parse_args().require_complete)
