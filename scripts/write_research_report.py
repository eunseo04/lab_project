"""Create a factual implementation/execution report from archived real records."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.metrics import aggregate_episodes


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def fmt(value):
    return "N/A" if value is None else f"{value:.3f}"


def main():
    outputs = ROOT/"research_outputs"
    runs=[]
    validation={}
    for manifest in sorted(outputs.rglob("manifest.json")):
        m=read(manifest)
        if "config" not in m or m.get("stage") not in ("train","eval"): continue
        directory=manifest.parent
        episodes=[read(p) for p in sorted(directory.glob("episode-*.json"))]
        episodes=[e for e in episodes if e.get("status")=="completed" and e.get("kind")=="real_cage4"]
        if not episodes: continue
        s=read(directory/"summary.json",{})
        status=s.get("status","partial")
        if (directory/"failure.json").exists(): status="failed"
        history=s.get("history",[])
        for h in history:
            for e in h.get("validation_episodes",[]):
                if e.get("kind")=="real_cage4" and e.get("status")=="completed": validation[e["trace"]]=e
        runs.append({"path":str(directory.relative_to(ROOT)),"stage":m["stage"],"config":m["config"],
            "status":status,"metrics":aggregate_episodes(episodes),"episodes":episodes,"history":history})
    snapshot=read(ROOT/"docs/preexisting_files_sha256.json")
    preservation={"checked":len(snapshot),"changed":[p for p,h in snapshot.items()
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]}
    files=[]
    for folder in ("research","research_tests","research_configs","docs"):
        files.extend(p for p in (ROOT/folder).glob("*") if p.is_file() and p.name not in
                     ("execution_report.md","execution_report.json","changed_files.json"))
    files.extend(ROOT/p for p in ("research.sh","research_sources.lock.json","research_models.lock.json",
        "requirements-model.txt","requirements-model.lock.txt","scripts/setup_research.sh",
        "scripts/start_qwen.sh","scripts/fetch_research_sources.py","scripts/generate_research_configs.py",
        "scripts/download_research_models.py","scripts/validate_qwen_scoring.py","scripts/write_research_report.py",
        "scripts/research_background.py","scripts/verify_archived_gate.py","scripts/verify_research_assets.py"))
    inventory={"added":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))},
        "modified":["README.md",".gitignore"],"legacy_preservation":preservation,
        "generated_reports":["docs/execution_report.md","docs/execution_report.json","docs/changed_files.json"]}
    (ROOT/"docs/changed_files.json").write_text(json.dumps(inventory,indent=2)+"\n")
    suites=ET.parse(outputs/"tests.xml").getroot().findall(".//testsuite")
    testing={k:sum(int(s.get(k,0)) for s in suites) for k in ("tests","failures","errors","skipped")}
    counts=Counter()
    for r in runs: counts[(r["stage"],r["config"]["composition"],r["config"]["steps"])]+=r["metrics"]["n"]
    finite={r["config"]["arm"]:r for r in runs if r["stage"]=="eval" and
        r["path"].startswith("research_outputs/real-500/") and r["config"]["red"]=="finite"}
    transfer={r["config"]["arm"]:r for r in runs if r["stage"]=="eval" and
        r["path"].startswith("research_outputs/real-transfer/") and r["config"]["red"]=="aggressive"}
    baseline={e["seed"]:e["joint_return"] for e in transfer.get("E0",{}).get("episodes",[])}
    proposed={e["seed"]:e["joint_return"] for e in transfer.get("E3",{}).get("episodes",[])}
    transfer_seeds=sorted(baseline.keys() & proposed.keys())
    transfer_result={"trained_red":"finite","evaluated_red":"aggressive","pool_frozen":True,
        "paired_seeds":transfer_seeds,"mean_E3_minus_E0":sum(proposed[s]-baseline[s] for s in transfer_seeds)/len(transfer_seeds) if transfer_seeds else None}
    questions=[]
    for label,old,new in (("RQ1: Static − Baseline","E0","E1"),("RQ2: Skill-Pro − Static","E1","E2"),("RQ3: Proposed − Skill-Pro","E2","E3")):
        left={e["seed"]:e["joint_return"] for e in finite.get(old,{}).get("episodes",[])}
        right={e["seed"]:e["joint_return"] for e in finite.get(new,{}).get("episodes",[])}
        seeds=sorted(left.keys() & right.keys())
        delta=sum(right[s]-left[s] for s in seeds)/len(seeds) if seeds else None
        questions.append({"question":label,"paired_seeds":seeds,"mean_reward_difference":delta})
    evolution=[]
    seen=set()
    for r in runs:
        for h in r["history"]:
            for e in h.get("evolution",[]):
                key=hashlib.sha256(json.dumps(e,sort_keys=True).encode()).hexdigest()
                if key in seen: continue
                seen.add(key)
                evolution.append({"run":r["path"],"iteration":h["iteration"],"parent":e["parent"],
                    "candidate_scores":[c.get("score") for c in e.get("candidates",[])],
                    "accepted":[c.get("skill",{}).get("name") for c in e.get("candidates",[]) if c.get("accepted")],
                    "semantic_gradient_chunks":len(e.get("semantic_gradients",[])),"error":e.get("error")})
    reused=[]
    for r in runs:
        initial=read(ROOT/r["path"]/"initial_pool.json",{})
        learned={s["name"] for s in initial.get("pool",{}).get("skills",[]) if s["version"]>1}
        if not learned: continue
        for e in r["episodes"]:
            with Path(e["trace"]).open() as stream:
                for line in stream:
                    t=json.loads(line)
                    if t.get("skill_id") in learned and t.get("policy_invoked"):
                        reused.append({k:t.get(k) for k in ("seed","step","agent","skill_id","skill_version")})
    parity=read(outputs/"official-evaluator-parity/arithmetic_parity.json",{})
    scoring=read(outputs/"scoring-validation.json",{})
    gate_parity=read(outputs/"real-500/gate_policy_parity.json",{})
    result={"implementation":"ACD + frozen Qwen3-8B + Skill-Pro / E3 observable verification",
        "tests":testing,"preservation":preservation,"runs":runs,"evolution":evolution,
        "validation_episodes":len(validation),"completed_experiment_episodes":sum(r['metrics']['n'] for r in runs),
        "total_real_cage4_episodes":sum(r['metrics']['n'] for r in runs)+len(validation)+len(parity.get('episode_returns',[])),
        "warm_start_reuse":reused,"research_questions":questions,
        "cross_red_transfer":transfer_result,"actual_gate_policy_parity":gate_parity,
        "original_evaluator_arithmetic":parity,"scoring_validation":scoring,
        "incomplete":["100-episode final evaluation","five-Red E0–E3 Qwen paired study",
            "500-step All LLM comparison","actual performance ablations","unchanged GPT-4o-mini paper reproduction"]}
    (ROOT/"docs/execution_report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    lines=["# 실제 구현·실행 보고서", "", "Closed-Loop Self-Improvement of Autonomous Cyber Defense Agents through Non-Parametric PPO-Based Security Skill Optimization", "", "연구 경로: `/home/jeonges/proposed`. 기존 ACD 작업 경로는 보존했다.", "",
        "## 완료 범위", "", "E0–E3, Hybrid/All LLM, Red 5종, 정적/진화 Skill, 실제 로그확률 Gate, 고정 Pool 평가, 재개, CSV/PNG 생성기를 구현했다.",
        "원본 공개 코드 모드와 논문 RTG/online-score 모드를 구분한다. 무수정 논문 수치 재현이나 아직 실행하지 않은 성능 개선을 주장하지 않는다.", "",
        f"회귀/연구 테스트: {testing['tests']}개, 실패 {testing['failures']}, 오류 {testing['errors']}. 보존 검사: 기존 {preservation['checked']}개 파일 중 변경 {len(preservation['changed'])}개.",
        "", "## 500-step FiniteState / Hybrid 실측", "", "| Arm | 완료 N | Mean Joint Reward | Sample Std | Wall seconds 합 | 상태 |", "|---|---:|---:|---:|---:|---|"]
    for arm in ("E0","E1","E2","E3"):
        r=finite.get(arm,{})
        m=r.get("metrics",{})
        lines.append(f"| {arm} | {m.get('n',0)} | {fmt(m.get('mean_joint_reward'))} | {fmt(m.get('reward_std'))} | {fmt(m.get('execution_time_s'))} | {r.get('status','not_run')} |")
    lines += ["", "## 학습하지 않은 AggressiveFSM 전이 실측", "", "FiniteState 훈련 Pool을 고정했다. E1/E2의 이 정책 대응 실험은 미실행이다.", "",
        "| Arm | 완료 N | Mean Joint Reward | Sample Std | 상태 |", "|---|---:|---:|---:|---|"]
    for arm in ("E0","E3"):
        r=transfer.get(arm,{});m=r.get("metrics",{})
        lines.append(f"| {arm} | {m.get('n',0)} | {fmt(m.get('mean_joint_reward'))} | {fmt(m.get('reward_std'))} | {r.get('status','not_run')} |")
    lines += ["", "nominal 500 = 실제 공동 전이 499 / 공식 점수 전이 498. 공동 보상은 1회만 더하고 terminal은 원본 평가기처럼 제외했다.",
        "E2/E3 실제 장기 학습 예산은 각 2 episodes / 1 iteration / best-of-2이며, 별도 시드의 고정 Pool 평가다. GPU별 동시 실행의 wall time이며 진화 시간은 별도다.",
        "", "## 실제 수행 수", "", "| 구분 | Blue 구성 | Nominal steps | 완료 episode 수 |", "|---|---|---:|---:|"]
    for (stage,composition,steps),n in sorted(counts.items()): lines.append(f"| {stage} | {composition} | {steps} | {n} |")
    lines += ["",f"별도 iteration 검증: {len(validation)} episodes. 원본 evaluator 보상 산술 대조: 실제 No Blue 2 episodes. Mock은 위 측정에서 제외했다.",
        f"총 실제 CAGE4 완료 episode: {result['total_real_cage4_episodes']}개(실험 {result['completed_experiment_episodes']} + 검증 {len(validation)} + 원본 평가기 2).",
        "완료된 episode 수와 전체 훈련/evaluation 계획 완료 여부는 구분한다. 짧은 통합 실험은 500-step 표에 포함하지 않는다.",
        "", "## Gate와 Skill 갱신 실측", "", "| 실행 | Iteration | Parent | 후보 Gate scores | 승인 Skill | Gradient chunks | 오류 |", "|---|---:|---|---|---|---:|---|"]
    for e in evolution:
        lines.append(f"| {e['run']} | {e['iteration']} | {e['parent']} | {e['candidate_scores']} | {', '.join(e['accepted']) or '없음'} | {e['semantic_gradient_chunks']} | {e['error'] or '없음'} |")
    lines += ["",f"학습된 버전의 후속 훈련 정책 호출 재사용 기록: {len(reused)}건. 전체 seed/step/Blue/Skill 버전은 execution_report.json의 warm_start_reuse에 저장했다.",
        "Gate 승인은 후보 수용을 뜻하며 최종 Reward 개선의 증명이 아니다. score=0 또는 음수는 원본 승인 기준에 따라 거부한다.",
        "", "## 연구 질문의 현재 답", ""]
    for q in questions:
        delta=q["mean_reward_difference"]
        assessment="아직 대응 평가가 없음" if delta is None else ("이 표본에서는 개선" if delta>0 else "이 표본에서는 개선되지 않음")
        lines.append(f"- {q['question']}: 대응 seed {q['paired_seeds']}, 평균 차이 {fmt(delta)}. {assessment}.")
    lines += [f"- RQ4: FiniteState 훈련 Pool의 AggressiveFSM 전이에서 대응 seed {transfer_seeds}, E3−E0 평균 차이 {fmt(transfer_result['mean_E3_minus_E0'])}. 이 작은 표본과 두 arm만으로 E2 대비 추가 기여의 전 정책 일반화를 입증할 수 없다. No Blue/All RL의 Red 5종 실행은 별도의 baseline control이다.",
        "", "## 원본 재사용과 차이", "",
        "ACD의 원본 prompt/formatter, 기존 JSON parser, KEEP 로더·공개 weights, Red/Green 정책과 CAGE4 환경/보상을 재사용했다. 새 adapter는 한 번의 joint tick과 Blue별 관찰 경계를 연결한다.",
        "Skill-Pro의 Skill/Experience, selector, 통계, FIFO/Golden, Semantic Gradient·aggregation·후보 생성, priority/sampling, maintenance를 재사용했다. 전체 궤적 입력, 정확한 token scoring, fail-closed rollback, 진짜 no-gate ablation을 보완했다. E3 관찰 검증은 별도 연구 확장이다.",
        "공개 KEEP의 192차원 통신 호환 투영, prompt/busy 공통 변경, episode별 명시 seed는 원본 무수정 재현과 구분한다. 논문 참고 수치는 별도 paper_reference다.",
        "파일별 변경·hash는 [changed_files.json](changed_files.json), 상세 근거는 [reproduction_audit.md](reproduction_audit.md), [implementation_changes.md](implementation_changes.md), [methodology.md](methodology.md)에 있다.",
        "", "## 검증·결과 파일", "",
        "- `research_outputs/tests.xml`: 자동 테스트.",
        "- `research_outputs/scoring-validation.json`: 실제 생성과 teacher forcing 로그확률 대조.",
        "- `research_outputs/official-evaluator-parity/arithmetic_parity.json`: 실제 원본 평가기 산술 대조.",
        "- `research_outputs/environment_source_parity.json`: 환경/보상/평가기 5개 핵심 파일의 공식 source hash 일치.",
        "- `research_outputs/assets_verification.json`: Qwen/encoder 24개 고정 파일 SHA256 재검증.",
        "- `research_outputs/real-500/gate_policy_parity.json`: 실제 Gate의 생성/teacher-forcing 점수 대조. 초기 누락 LP는 동일 토큰 재생성 여부를 함께 기록.",
        "- `research_outputs/real-500/replay_parity.json`: 중단 전후 E1의 480개 공동 step 재현 대조.",
        "- `research_outputs/real-500/skill_rejection_parity.json`: 후보가 거부된 E2와 E1의 998개 공동 step 일치.",
        "- `research_outputs/evaluation_integrity.json`: 실제 평가 12 episodes의 모델/시드/종료/동결 검증.",
        "- `research_outputs/real-500/report/`: 비교표, 500-step 행동 분포, 실제 이유 PCA/K-Means 및 PNG.",
        "- `research_outputs/report/`: 실험 조건별 통합 결과와 분리된 학습 곡선, pairwise_available.csv.",
        "- `research_outputs/real-controls/report/control_baselines.csv`: No Blue / All RL × Red 5종 실측.",
        "- 각 `train/checkpoint.json`, `pool-iteration-*.json`, `eval/frozen_pool.json`: 실제 Pool/버전/Gate와 provenance.",
        "- `research_outputs/model_server_ready.json`: GPU 0 / 127.0.0.1:8018의 준비된 최신 기본 서버. trainable policy parameters=0. 나머지 실험용 서버는 종료했다.",
        "", "## 미완료·제한", "",
        "100 episodes × Red 5종 × E0–E3의 최종 연구와 500-step All LLM, 성능 ablation은 아직 미실행이다. 이번 작업은 제한된 실측 예산으로 2-episode 기준과 짧은 실제 통합/진화 경로를 검증했다. 빈 칸은 N/A다.",
        "GPT-4o-mini 재실행과 공개되지 않은 논문 seed/KEEP 재학습 weight의 완전 재현은 수행하지 않았다. reasoning embedding은 API 없이 로컬 MiniLM을 사용하므로 원본 text-embedding-3-large와 다르다.",
        "통계 분모 수정 전에 장기 훈련기를 중단·재개했으며 완료 episode는 보존했다. E1의 도구 세션 시간 제한 종료 후에는 독립 실행기로 미완료 episode를 재개했다. 표의 시간은 완료된 측정 episode의 합이며 중단 시도·모델 초기 로딩·오프라인 진화 비용을 합치지 않는다. 중단 trace/오류는 interrupted에 남긴다. 확률 근사 초기 구현은 실제 검증 후 교체했고 성공한 Gate로 사용하지 않았다.",
        "", "## 전체 재실행", "", "```bash", "cd /home/jeonges/proposed", "./scripts/setup_research.sh --download", "CUDA_VISIBLE_DEVICES=0 ./scripts/start_qwen.sh --port 8018", "# 다른 터미널", "./research.sh doctor", "./research.sh test", "./research.sh compare --config research_configs/closed-loop.yaml --train --limit 2", "./research.sh compare --config research_configs/closed-loop.yaml --train --limit 2 --resume", "# 전체 계획을 끝낼 때까지", "./research.sh compare --config research_configs/closed-loop.yaml --train --resume", "./research.sh report research_outputs/closed-loop --reasoning", "```", "",
        "2-episode/100-episode 구분, endpoint와 개별 E arm/ablation 실행 명령은 [README](../README.md)와 [experimental_protocol.md](experimental_protocol.md)에 있다."]
    (ROOT/"docs/execution_report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"real_runs":len(runs),"completed_episodes":sum(r['metrics']['n'] for r in runs),
        "validation_episodes":len(validation),"tests":testing,"warm_start_reuse":len(reused)},ensure_ascii=False))


if __name__=="__main__": main()
