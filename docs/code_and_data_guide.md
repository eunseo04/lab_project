# 코드 및 데이터 구조

현재 구현 기준은 2026-11-25의 정리된 체크아웃이다. 전체 파일별 기능·주요 함수/클래스·JSON 필드는 [파일 목록 CSV](code_and_data_inventory.csv)에 있다. 과거 결과·원시 실행 로그·별도 revision 복사본은 포함하지 않는다.

## 실행 흐름

```text
scripts/run_research.sh B0|B1|B2|B3|B4 train|evaluate
  → scripts/prepare_checkout.py → .local-configs/rollout-B*.yaml
  → research.sh → research/__main__.py → experiment.run()
  → CAGE4 부분 관찰 → 사건 이력 → Skill 선택 → Qwen primitive 행동
  → 5개 Blue의 joint step → 원래 공동 reward와 trace 저장
  → 훈련 batch → Semantic Gradient → 후보 생성 → PPO Gate
  → B4: 이전/후보 Pool의 실제 대응 검증 → 채택/rollback
```

| 위치 | 역할 |
|---|---|
| `research/` | 현재 연구 모듈 23개. 단일 실행 구현 |
| `acd/` | 공통 ACD 프롬프트·행동 JSON·통신·CybORG/KEEP 연동 |
| `research_configs/` | B0–B4와 최소 경험 지원 no-support ablation |
| `artifacts/initial_pool.json` | B1–B4 공통의 정적 초기 Skill. 학습 결과가 아님 |
| `scripts/` | 설치·실행·추가 수집 설정·새 결과 비교·자원 검사 |
| `tests/`, `research_tests/` | 공통/현재 회귀와 통합 검사 |
| `configs/` | 원래 ACD/KEEP 실행·훈련 템플릿 |
| `docs/`, `references/` | 현재 코드/데이터 설명과 출처 |
| `models/`, `third_party/`, `runtime/` | 고정 모델·외부 원본·조립 환경. 로컬 자원이며 Git 제외 |
| `.venv/`, `.model-venv/` | 시뮬레이터/분석과 모델 서버의 분리 환경. Git 제외 |
| `.local-configs/` | 현 checkout 경로로 생성한 설정. Git 제외 |
| `research_outputs/` | 새 실험이 생성하는 결과·raw trace·checkpoint. Git 제외 |

공개 B0=E0, B1=E1, B2/B3/B4=E2다. E3에는 별도 실행 중 검증이 있으므로 새 B4와 동일하지 않다. 기본 hybrid 구성은 Qwen `blue_agent_4`와 고정 공개 KEEP `blue_agent_0`~`blue_agent_3`이다. all_llm/all_rl/no_blue 대조 구성도 지원한다.

## 현재 데이터의 원천

주된 데이터는 CybORG CAGE4 EnterpriseScenarioGenerator와 정책의 상호작용으로 생성하는 관찰·행동·보상 궤적이다. 별도 침입탐지 CSV의 정상/공격 label로 Qwen을 supervised training하는 구조가 아니며, Skill-Pro 원본의 TextArena/ALFWorld 데이터를 이 실험에 투입하지 않는다.

환경에는 router/server/user host, restricted/operational A·B, contractor, public access, admin, office, internet의 subnet enum이 있다. host 수와 초기 상태는 scenario에 따라 달라진다. Green은 정상 업무, Red는 공격, Blue는 방어를 수행하고 mission phase는 0/1/2다. Red 정책은 finite/aggressive/stealthy/impact/degrade를 지원한다.

Qwen은 Blue의 부분 관찰과 사건 이력을 텍스트로 읽는다. KEEP는 node features·edge index·action mask 등 그래프 표현을 사용한다. paper_projected_keep는 ACD 8-bit 통신을 고정 KEEP 입력과 호환되는 형태로 투영하는 공통 설정이다. 숨은 Red/session 전체 상태는 Qwen 정책 입력에 넣지 않는다.

| 모델 자원 | 역할 |
|---|---|
| Qwen3-8B | BF16 고정 정책/Skill 선택/SG/후보 생성 및 exact token scoring |
| all-MiniLM-L6-v2 | Skill embedding 검색 |
| 공개 KEEP 가중치 | GNN/PPO 방어 정책. hybrid는 4개 사용 |

모델 revision과 파일 SHA256은 `research_models.lock.json`, 네 원본 commit과 KEEP 가중치 hash는 `research_sources.lock.json`에 있다. 외부 API 예시는 `acd/backends.py`에 주석 처리돼 있고 현재 연구는 로컬 Qwen 서버 경로를 사용한다. `acd/training.py`의 KEEP 신경망 학습은 Skill 텍스트 진화와 별도 기능이다.

## 생성하는 실행 파일

```text
research_outputs/<새 실험>/
├── protocol.json                 campaign의 사전 계획·hash·seed·예산
└── <개별 실행>/
    ├── manifest.json             실제 설정과 구현 hash
    ├── invocations/*.json        호출별 resume/limit 정보
    ├── initial_pool.json         해당 훈련의 시작 Pool
    ├── episode-0000.partial.jsonl
    ├── episode-0000.jsonl        완료한 원시 trace
    ├── episode-0000.json         완료 결과·보상·시간·모델 정보
    ├── episode-0000-reward-events.jsonl
    ├── episode-0000-skill-attribution.json
    ├── checkpoint.json           재개 상태
    ├── pool-iteration-0001.json  optimizer 반복이 완료될 때 생성
    ├── frozen_pool.json          평가용 동결 Pool
    └── summary.json              마지막 저장된 상태·결과
```

이것은 새 실행에서 생성하는 형식이다. 훈련/평가 stage와 진단 설정에 따라 일부 파일만 존재한다. 모든 파일이나 학습 반복이 이미 완료됐다는 뜻은 아니다. 남은 partial 기록은 완료 성능 표본이 아니다.

## 원시 trace JSONL

한 줄은 `(episode, step, Blue agent)` 하나다. 500 nominal-step 설정의 완료 episode는 joint transition 499개 × Blue 5개 = 2,495줄이다. 2,495개의 독립 episode가 아니다. 공식 채점에 포함되는 transition은 498개다.

| 필드 | 의미 |
|---|---|
| `seed`, `episode_id`, `step`, `agent`, `policy` | 실행 초기 상태·episode·tick·Blue·담당 정책 식별 |
| `mission_phase`, `comm_vector` | phase와 통신 |
| `observation_before`, `observation_after` | 해당 Blue의 전후 부분 관찰 dict |
| `requested_action`, `executed_actions` | 요청 행동과 환경에서 관찰한 실제 행동 name/text/target |
| `request_applied`, `action_status` | 새 요청 적용 여부와 TRUE/FALSE/IN_PROGRESS/UNKNOWN |
| `joint_reward`, `score_included`, `terminal` | 공동 reward와 공식 점수 포함/종료 여부 |
| `automatic_monitor`, `policy_invoked` | 자동 Monitor·busy 대기를 실제 Qwen 자유 선택과 구분 |
| `skill_id`, `skill_version`, `skill_terminated`, `skill_controller` | 활성 Skill·버전·종료/선택 사건 |
| `state`, `user_prompt` | LLM용 기본 system/user messages의 JSON 문자열과 관찰 prompt. 숨은 전체 simulator state가 아님 |
| `response`, `reason`, `parse_error` | 실제 모델 출력과 출력된 이유·해석 오류. reason은 원인 분석 정답 label이 아님 |
| `token_ids`, `generation_logprob` | 실제 생성 token과 sequence logprob. 같은 token을 부모/후보 Skill 조건에서 재평가 |
| `prompt_sha256`, `model_identity`, token 수 | 입력·모델 provenance와 호출량 |
| `behavior_context`, `behavior_outcome`, `evidence_hash` | 관찰된 case 이력·행동 완료·반복 진단·증거 버전 |
| `return_to_go` | 공식 reward 시계열에서 계산한 RTG |

Qwen의 primitive 선택지는 Analyse/DeployDecoy/Remove/Restore/Sleep/BlockTrafficZone/AllowTrafficZone이며 Monitor는 자동이다. TRUE는 호출 완료 성공으로서 위협 제거의 보증이 아니고 IN_PROGRESS는 실패가 아니다. `policy == llm`과 `policy_invoked`로 Qwen의 자유 선택을 분리해야 한다. episode 전체 action counter는 KEEP도 포함한다.

## Episode, 보상, 경험과 Pool

`episode-*.json`은 완료 상태·seed·transition 수·공동 return·시간·행동 횟수·모델 정보·원시 trace 경로를 저장한다. scored_step_rewards는 공식 제외 tick을 0으로 둔 길이 499 배열이므로 scored transition 수 498과 구분한다. 다섯 Blue 행에 반복 저장한 공동 reward는 tick마다 한 번만 합산한다.

| 보상 항목 | 의미 |
|---|---|
| LWF | Green LocalWork 실패 |
| ASF | Green AccessService 실패 |
| RIA | Red Impact 피해 |
| action_cost | 방어 행동 비용 |

RewardAudit는 공식 reward를 변경하지 않고 사건별 합계를 대조한다. attribution sidecar는 같은 host 사건·동시 KEEP 행동·전후 관찰을 연결하는 offline 진단이며 새로운 보상이나 숨은 정책 입력이 아니다.

훈련 trace의 LLM 행을 `(agent, skill_id)`로 묶어 `Experience`를 만든다. reward는 episode 공동 return, skill은 Skill 이름, trajectory는 해당 구간의 직렬화 기록, transitions는 행 목록, env_name은 구성:Red, step_count는 episode 길이다. 유효한 실제 정책 결정만 PPO scoring 증거로 사용한다. 한 episode의 반복 행동을 독립 경험 여러 개로 세지 않는다.

Pool은 max_size/seed_set/skills를 가진다. Skill은 name/initiation/policy/termination, version/family/parent_id, frequency/gain/maturity 등의 통계와 allowed_actions 메타데이터를 갖는다. 초기 보안 절차 7개는 RapidProcessResponse, EscalationDiagnosis, PrivilegedRecovery, SelectiveDeception, NetworkContainment, TrafficRecovery, RiskBasedInvestigation이다.

Checkpoint는 Pool 외에 완료 수·훈련 seed·모델 정보·baseline·FIFO/Golden/Skill별 buffer·history·pending batch를 보관한다. 완료 환경 episode가 optimizer 커밋보다 먼저 저장될 수 있으므로 summary의 완료 수와 개별 episode 파일을 구분한다. 중단된 SG/PPO 내부 계산은 재실행이 필요할 수 있다. 초기 Pool 파일만으로 학습 완료나 재개 가능성을 주장하지 않는다.

## 검증과 평가 분리

| 용도 | 기본 full 설정 |
|---|---|
| 훈련 | seed 91000–91005, 6 episode / 3 iteration |
| 후보 Rollout validation | 97000–97019 |
| 보조 validation 예약 | 99000, 기본 사용 수 0 |
| 독립 test | 101000–101099 |
| 다른 Red pilot | 102000 |

후보 하나는 이전 Pool 20회와 후보 Pool 20회의 대응 결과를 비교한다. 독립 seed 쌍은 20개다. 모든 지정 seed 완료, 평균 개선·승률·중앙값·최악 손실·Green 피해·분산 기준을 유지하고 승인 시 검증한 정확한 Pool을 사용한다. 거부 시 이전 Pool을 유지한다. validation을 test에 섞지 않는다. 작은 campaign pilot은 별도 version과 더 작은 수집 계획으로 기록된다.

## 새 결과 비교 도구

`./research.sh report <실험폴더> --reasoning` 또는 `scripts/report_experiment.py`는 삭제한 과거 결과를 요구하지 않는다. 출력은 해당 실험의 `report/`다.

| 파일 | 의미 |
|---|---|
| episodes.json | 완료한 real CAGE4 episode의 compact 자료 |
| metrics.json | version/Red/split/label별 평균·표본 SD·대응 차이·승률·중앙값·bootstrap 구간 |
| run_status.json | 실제 환경 완료 수와 커밋한 optimizer 반복 수 |
| rollout_decisions.json | 후보별 승인/거부 결과 |
| performance.md | B0–B4와 누락 N/A를 표시하는 비교 표 |

train/test/untrained_control은 분리한다. 학습 완료 provenance가 없는 B2–B4 평가는 untrained_control로 표시한다. mock/미완료 episode와 후보 validation을 최종 test에 넣지 않는다. 동일 version/Red/split의 모델·공통 관찰/채점·KEEP 가중치 조건이 다르면 합치지 않는다. `--reasoning`은 존재하는 raw trace에서 Qwen만의 자유 행동과 busy 대기·진단 플래그를 읽는다.

공개 compact 파일은 `python3 scripts/recompute_public_metrics.py --episodes <episodes.json>`으로 다시 계산할 수 있다. 전체 prompt·token scoring·환경 replay·학습 재개에는 raw 기록과 고정 환경/모델이 별도로 필요하다. 현재 저장소에는 과거 측정 데이터가 없다.

## 검사와 한계

현재 회귀 검사와 고정 core/초기 Pool/검증 임계값 기준은 tests 및 research_tests에 있다. 기준 hash fixture는 source 복사본이 아니라 변경 방지용 작은 고정 값이다. 정리 후 175개 검사가 통과했다.

진단 검증은 구조화된 관찰 일관성을 검사하지만 자유 서술의 모순·인과 진단 전체를 보증하지 못한다. 공동 reward만으로 특정 Skill의 인과 기여를 알 수 없으며, PPO Gate 양수나 훈련 reward를 독립 test 개선으로 해석하지 않는다. 가상 기업망 결과를 실제 기업망 성능으로 바로 일반화하지 않는다.

## 파일별 핵심 기능

| research/ 파일 | 기능 |
|---|---|
| [__init__.py](../research/__init__.py) | 연구 package와 해당 revision ROOT 경로 정의 |
| [__main__.py](../research/__main__.py) | 연구 CLI: 훈련·평가·비교·공식 평가·보고·검사 연결 |
| [adapter.py](../research/adapter.py) | CAGE4 관찰·통신을 Qwen/KEEP 입력과 primitive 행동으로 연결 |
| [attribution.py](../research/attribution.py) | 독립 경험·Skill 성과 통계와 공동 보상 귀속 보조 진단 |
| [behavior.py](../research/behavior.py) | 관찰 기반 사건 이력과 비동기 행동·증거 버전 추적 |
| [behavior_skills.py](../research/behavior_skills.py) | 보안 초기 절차와 관찰 기반 Skill initiation 조건 |
| [client.py](../research/client.py) | 동일 로컬 모델의 생성·exact scoring·embedding 요청과 식별 검증 |
| [config.py](../research/config.py) | 설정·알고리즘 모드·seed 분리·모델 revision 검사 및 digest |
| [diagnosis.py](../research/diagnosis.py) | 구조화된 행동·대상·시점·PID·Restore epoch 진단 검사 |
| [evolution.py](../research/evolution.py) | SG·집계·후보 생성·PPO Gate·Pool 갱신과 검증 예산 관리 |
| [experiment.py](../research/experiment.py) | 공동 episode 실행·trace/결과/체크포인트·학습/동결 평가·복구 |
| [gate.py](../research/gate.py) | RTG·Advantage·importance ratio·clipped surrogate 계산 |
| [metrics.py](../research/metrics.py) | 공동 보상 중복 방지와 공식 채점·episode 통계 |
| [model_server.py](../research/model_server.py) | 고정 Qwen BF16 로컬 서버 및 exact teacher-forced token scoring |
| [official.py](../research/official.py) | 원래 CAGE4 evaluator 실행과 계산 대조 |
| [paper_algorithm.py](../research/paper_algorithm.py) | 논문식 running baseline·온라인 Skill score·Pool 유지 |
| [reporting.py](../research/reporting.py) | 완료된 실행에서 성능 표·그림·reason 분석 생성 |
| [reward_audit.py](../research/reward_audit.py) | 공식 보상 원값을 유지한 사건별 LWF/ASF/RIA/행동 비용 검사 |
| [rollout.py](../research/rollout.py) | 대응 후보 검증의 승인 계산과 transactional Pool 유지/rollback |
| [rollout_execution.py](../research/rollout_execution.py) | 실제 CAGE4 이전/후보 Pool 평가 worker와 결정 JSON |
| [semantic_evidence.py](../research/semantic_evidence.py) | 실제 기록 인용과 변경 성분 지원·훈련 host/IP 암기 검사 |
| [skills.py](../research/skills.py) | Skill Pool·선택·활성 상태·종료와 E3용 실행 검증 |
| [upstream.py](../research/upstream.py) | 고정한 Skill-Pro 원본 자료형/진화 클래스와 encoder 연결 |

## 유지한 실행·분석 도구

| scripts/ 파일 | 기능 |
|---|---|
| [audit_training_cost.py](../scripts/audit_training_cost.py) | 실제 훈련 trace의 문자량·예정 SG chunk 호출 수 계산 |
| [audit_training_coverage.py](../scripts/audit_training_coverage.py) | 완료 trace에서 미활성 Skill 포함 독립 episode/행동/phase/결과 복구 |
| [check_publication.py](../scripts/check_publication.py) | Git index blob의 범위·비밀 값·대형 파일·외부 symlink 반출 검사 |
| [download_research_models.py](../scripts/download_research_models.py) | 모델 lock의 공개 Qwen/encoder 파일을 revision 고정 다운로드 |
| [fetch_research_sources.py](../scripts/fetch_research_sources.py) | CAGE4/ACD/KEEP/Skill-Pro 네 원본 commit 확보 |
| [fetch_sources.py](../scripts/fetch_sources.py) | 초기 ACD/CAGE4/KEEP 고정 원본 확보 |
| [prepare_checkout.py](../scripts/prepare_checkout.py) | 원본 설정을 보존하고 현 checkout 경로용 .local-configs 생성 |
| [prepare_sources.py](../scripts/prepare_sources.py) | 고정 CAGE4·ACD·KEEP 확장을 실행용 runtime으로 조립 |
| [prepare_training_collection.py](../scripts/prepare_training_collection.py) | 추가 독립 훈련 seed·episode/iteration의 별도 version 설정 생성 |
| [recompute_public_metrics.py](../scripts/recompute_public_metrics.py) | 공개 compact episode만으로 분리된 평균·분산·대응 차이 계산 |
| [report_experiment.py](../scripts/report_experiment.py) | 새 완료 결과를 label/version/Red/split별로 분리하고 미학습 대조군·후보 validation을 최종 test와 구분 |
| [run_campaign.py](../scripts/run_campaign.py) | 현재 코드로 예산 내 Qwen 훈련·동결·비교와 task-owned 서버 관리 |
| [run_research.sh](../scripts/run_research.sh) | B0–B4 설정을 현재 v6 launcher로 전달하는 공개 실행기 |
| [setup_research.sh](../scripts/setup_research.sh) | CPU/모델 환경 분리 설치와 네 고정 원본·선택적 모델 다운로드 |
| [start_qwen.sh](../scripts/start_qwen.sh) | 고정된 로컬 Qwen 모델 서버 실행 |
| [verify_research_assets.py](../scripts/verify_research_assets.py) | 고정 모델/encoder 파일 크기·SHA256의 offline 검사 |

전체 관리 파일 100개 경로는 CSV에 있다. 모델·가상환경·생성 결과를 코드 파일 수에 합치지 않는다.
