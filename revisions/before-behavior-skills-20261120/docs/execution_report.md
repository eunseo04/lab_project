# 실제 구현·실행 보고서

Closed-Loop Self-Improvement of Autonomous Cyber Defense Agents through Non-Parametric PPO-Based Security Skill Optimization

연구 경로: `/home/jeonges/proposed`. 기존 ACD 작업 경로는 보존했다.

최신 수정: [Skill-Pro 출판 논문 방식으로 수정한 코드·검증·성능 실험](skillpro_paper_fix.md).
아래 42개 테스트와 500-step 수치는 수정 전 snapshot이다. 최신 테스트는 59개이며 새 실험 결과는 별도 디렉터리에 보존한다.

최신 FiniteState / Hybrid / 동일 평가 seed 4개의 평균 보상은 ACD −287.75, 학습 전 E3 −303.50, loop 1 −319.50, loop 2 −320.25다. 이번 표본에서 개선은 확인하지 못했다. 훈련은 4 episodes / 2 loops이며, 후보 승인과 최종 보상 개선은 구분한다.

새 AggressiveFSM 전이 평가도 완료했다. 동일 4개 시드에서 ACD −261.75, 최종 loop 2 E3 −338.50으로 평균 차이는 −76.75다. 수정 실험은 훈련 4 + 검증 2 + 평가 24 = 30개 500-step episodes가 모두 완료됐고, 최종 원시 로그 감사가 통과했다. 세부 시드별 보상·후보 점수·부진 원인·CSV·그래프는 위 최신 보고서에 있다.

2026-11-20 추가 분석: [Qwen 성능 비교·closed-loop별 수치·부진 원인](performance_analysis.md).
아래 원실행 snapshot의 57 episodes와 별도로 동일 검증 seed의 12-step Qwen 대조 2 episodes를 추가했다(수정 전 집계 59). 500-step 원시 결과는 보존했다.

## 완료 범위

E0–E3, Hybrid/All LLM, Red 5종, 정적/진화 Skill, 실제 로그확률 Gate, 고정 Pool 평가, 재개, CSV/PNG 생성기를 구현했다.
원본 공개 코드 모드와 논문 RTG/online-score 모드를 구분한다. 무수정 논문 수치 재현이나 아직 실행하지 않은 성능 개선을 주장하지 않는다.

회귀/연구 테스트: 42개, 실패 0, 오류 0. 보존 검사: 기존 20개 파일 중 변경 0개.

## 500-step FiniteState / Hybrid 실측

| Arm | 완료 N | Mean Joint Reward | Sample Std | Wall seconds 합 | 상태 |
|---|---:|---:|---:|---:|---|
| E0 | 2 | -267.500 | 75.660 | 1102.076 | completed |
| E1 | 2 | -355.000 | 156.978 | 1733.956 | completed |
| E2 | 2 | -355.000 | 156.978 | 1719.161 | completed |
| E3 | 2 | -269.500 | 7.778 | 1408.709 | completed |

## 학습하지 않은 AggressiveFSM 전이 실측

FiniteState 훈련 Pool을 고정했다. E1/E2의 이 정책 대응 실험은 미실행이다.

| Arm | 완료 N | Mean Joint Reward | Sample Std | 상태 |
|---|---:|---:|---:|---|
| E0 | 2 | -257.000 | 69.296 | completed |
| E3 | 2 | -450.000 | 97.581 | completed |

nominal 500 = 실제 공동 전이 499 / 공식 점수 전이 498. 공동 보상은 1회만 더하고 terminal은 원본 평가기처럼 제외했다.
E2/E3 실제 장기 학습 예산은 각 2 episodes / 1 iteration / best-of-2이며, 별도 시드의 고정 Pool 평가다. GPU별 동시 실행의 wall time이며 진화 시간은 별도다.

## 실제 수행 수

| 구분 | Blue 구성 | Nominal steps | 완료 episode 수 |
|---|---|---:|---:|
| eval | all_llm | 12 | 2 |
| eval | all_rl | 500 | 10 |
| eval | hybrid | 12 | 2 |
| eval | hybrid | 500 | 12 |
| eval | no_blue | 12 | 1 |
| eval | no_blue | 500 | 10 |
| train | hybrid | 12 | 6 |
| train | hybrid | 20 | 2 |
| train | hybrid | 500 | 4 |

별도 iteration 검증: 6 episodes. 원본 evaluator 보상 산술 대조: 실제 No Blue 2 episodes. Mock은 위 측정에서 제외했다.
총 실제 CAGE4 완료 episode: 57개(실험 49 + 검증 6 + 원본 평가기 2).
완료된 episode 수와 전체 훈련/evaluation 계획 완료 여부는 구분한다. 짧은 통합 실험은 500-step 표에 포함하지 않는다.

## Gate와 Skill 갱신 실측

| 실행 | Iteration | Parent | 후보 Gate scores | 승인 Skill | Gradient chunks | 오류 |
|---|---:|---|---|---|---:|---|
| research_outputs/real-500/E2-hybrid-finite/train | 1 | TrafficRecovery | [-1.6202084052494743, -1.9280286893533574] | 없음 | 8 | 없음 |
| research_outputs/real-500/E3-hybrid-finite/train | 1 | HostRecovery | [1.2452790199886898, 0.6172098539267559] | HostRecovery_v2 | 8 | 없음 |
| research_outputs/real-closed-loop/E2-hybrid-finite/train | 1 | NetworkContainment | [0.0, 0.0] | 없음 | 4 | 없음 |
| research_outputs/real-closed-loop/E2-hybrid-finite/train | 2 | NetworkContainment | [0.020456996775524882, 0.02242588383021581] | NetworkContainment_v2 | 4 | 없음 |
| research_outputs/real-continued/E2-hybrid-finite/train | 3 | HostInvestigation | [-4.1486972948237675, -4.348714983841035] | 없음 | 2 | 없음 |
| research_outputs/real-paper-rtg/E2-hybrid-finite/train | 1 | NetworkContainment | [-1.3113025476768583, -1.304459800067675] | 없음 | 14 | 없음 |

학습된 버전의 후속 훈련 정책 호출 재사용 기록: 12건. 전체 seed/step/Blue/Skill 버전은 execution_report.json의 warm_start_reuse에 저장했다.
Gate 승인은 후보 수용을 뜻하며 최종 Reward 개선의 증명이 아니다. score=0 또는 음수는 원본 승인 기준에 따라 거부한다.

## 연구 질문의 현재 답

- RQ1: Static − Baseline: 대응 seed [10000, 10001], 평균 차이 -87.500. 이 표본에서는 개선되지 않음.
- RQ2: Skill-Pro − Static: 대응 seed [10000, 10001], 평균 차이 0.000. 이 표본에서는 개선되지 않음.
- RQ3: Proposed − Skill-Pro: 대응 seed [10000, 10001], 평균 차이 85.500. 이 표본에서는 개선.
- RQ4: FiniteState 훈련 Pool의 AggressiveFSM 전이에서 대응 seed [10000, 10001], E3−E0 평균 차이 -193.000. 이 작은 표본과 두 arm만으로 E2 대비 추가 기여의 전 정책 일반화를 입증할 수 없다. No Blue/All RL의 Red 5종 실행은 별도의 baseline control이다.

## 원본 재사용과 차이

ACD의 원본 prompt/formatter, 기존 JSON parser, KEEP 로더·공개 weights, Red/Green 정책과 CAGE4 환경/보상을 재사용했다. 새 adapter는 한 번의 joint tick과 Blue별 관찰 경계를 연결한다.
Skill-Pro의 Skill/Experience, selector, 통계, FIFO/Golden, Semantic Gradient·aggregation·후보 생성, priority/sampling, maintenance를 재사용했다. 전체 궤적 입력, 정확한 token scoring, fail-closed rollback, 진짜 no-gate ablation을 보완했다. E3 관찰 검증은 별도 연구 확장이다.
공개 KEEP의 192차원 통신 호환 투영, prompt/busy 공통 변경, episode별 명시 seed는 원본 무수정 재현과 구분한다. 논문 참고 수치는 별도 paper_reference다.
파일별 변경·hash는 [changed_files.json](changed_files.json), 상세 근거는 [reproduction_audit.md](reproduction_audit.md), [implementation_changes.md](implementation_changes.md), [methodology.md](methodology.md)에 있다.

## 검증·결과 파일

- `research_outputs/tests.xml`: 자동 테스트.
- `research_outputs/scoring-validation.json`: 실제 생성과 teacher forcing 로그확률 대조.
- `research_outputs/official-evaluator-parity/arithmetic_parity.json`: 실제 원본 평가기 산술 대조.
- `research_outputs/environment_source_parity.json`: 환경/보상/평가기 5개 핵심 파일의 공식 source hash 일치.
- `research_outputs/assets_verification.json`: Qwen/encoder 24개 고정 파일 SHA256 재검증.
- `research_outputs/real-500/gate_policy_parity.json`: 실제 Gate의 생성/teacher-forcing 점수 대조. 초기 누락 LP는 동일 토큰 재생성 여부를 함께 기록.
- `research_outputs/real-500/replay_parity.json`: 중단 전후 E1의 480개 공동 step 재현 대조.
- `research_outputs/real-500/skill_rejection_parity.json`: 후보가 거부된 E2와 E1의 998개 공동 step 일치.
- `research_outputs/evaluation_integrity.json`: 실제 평가 12 episodes의 모델/시드/종료/동결 검증.
- `research_outputs/real-500/report/`: 비교표, 500-step 행동 분포, 실제 이유 PCA/K-Means 및 PNG.
- `research_outputs/report/`: 실험 조건별 통합 결과와 분리된 학습 곡선, pairwise_available.csv.
- `research_outputs/real-controls/report/control_baselines.csv`: No Blue / All RL × Red 5종 실측.
- 각 `train/checkpoint.json`, `pool-iteration-*.json`, `eval/frozen_pool.json`: 실제 Pool/버전/Gate와 provenance.
- `research_outputs/model_server_ready.json`: GPU 0 / 127.0.0.1:8018의 준비된 최신 기본 서버. trainable policy parameters=0. 나머지 실험용 서버는 종료했다.

## 미완료·제한

100 episodes × Red 5종 × E0–E3의 최종 연구와 500-step All LLM, 성능 ablation은 아직 미실행이다. 이번 작업은 제한된 실측 예산으로 2-episode 기준과 짧은 실제 통합/진화 경로를 검증했다. 빈 칸은 N/A다.
GPT-4o-mini 재실행과 공개되지 않은 논문 seed/KEEP 재학습 weight의 완전 재현은 수행하지 않았다. reasoning embedding은 API 없이 로컬 MiniLM을 사용하므로 원본 text-embedding-3-large와 다르다.
통계 분모 수정 전에 장기 훈련기를 중단·재개했으며 완료 episode는 보존했다. E1의 도구 세션 시간 제한 종료 후에는 독립 실행기로 미완료 episode를 재개했다. 표의 시간은 완료된 측정 episode의 합이며 중단 시도·모델 초기 로딩·오프라인 진화 비용을 합치지 않는다. 중단 trace/오류는 interrupted에 남긴다. 확률 근사 초기 구현은 실제 검증 후 교체했고 성공한 Gate로 사용하지 않았다.

## 전체 재실행

```bash
cd /home/jeonges/proposed
./scripts/setup_research.sh --download
CUDA_VISIBLE_DEVICES=0 ./scripts/start_qwen.sh --port 8018
# 다른 터미널
./research.sh doctor
./research.sh test
./research.sh compare --config research_configs/closed-loop.yaml --train --limit 2
./research.sh compare --config research_configs/closed-loop.yaml --train --limit 2 --resume
# 전체 계획을 끝낼 때까지
./research.sh compare --config research_configs/closed-loop.yaml --train --resume
./research.sh report research_outputs/closed-loop --reasoning
```

2-episode/100-episode 구분, endpoint와 개별 E arm/ablation 실행 명령은 [README](../README.md)와 [experimental_protocol.md](experimental_protocol.md)에 있다.
