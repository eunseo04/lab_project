# CAGE4 + ACD + Skill-Pro

고정 Qwen3-8B와 KEEP 4개가 CAGE4 가상 기업망을 방어하고, 실행 경험으로 외부 Security Skill을 개선하는 연구 코드다. Qwen 가중치는 업데이트하지 않는다.

현재 실행 코드는 루트 `research/` 하나다. 공개 Git에는 과거 원시 결과·학습 완료 Pool이 포함되지 않는다. 로컬의 `research_outputs/`에는 실험별 원시 trace·journal·checkpoint를 보존한다. `artifacts/initial_pool.json`은 훈련 전 정적 절차다. 모델·가상환경·원본 소스·runtime은 로컬 실행 자원이며 Git에서 제외한다.

[파일별 기능·데이터 구조](docs/code_and_data_guide.md), [전체 파일 목록 CSV](docs/code_and_data_inventory.csv), [논문·원본 출처](references/README.md)를 제공한다.

## 비교 실험군

| 군 | 구성 | 내부 arm |
|---|---|---|
| B0 | Qwen ACD + 공통 관찰/사건 이력 | E0 |
| B1 | B0 + 고정 초기 Security Skill | E1 |
| B2 | Skill-Pro 논문식 SG·batch·RTG·PPO Gate | E2 |
| B3-no-support | B2 + evidence/diagnosis 검증, 최소 episode 제한 없음 | E2 |
| B3 | B2 + fact-ID/intervention 검증·최소 독립 episode 2개·원본 PPO 채택 | E2 |
| B4 | B3 + 실제 대응 CAGE4 Rollout 검증·rollback | E2 |

B3/B4도 B2와 같은 actor/selector를 사용한다. B4를 내부 E3와 혼용하지 않는다. `artifacts/initial_pool.json`은 B1–B4가 공유하는 정적 초기 절차이며 학습 결과가 아니다.

## 설치와 실행

기록된 환경은 Python 3.10이다. 고정 source/model revision과 의존성 lock을 사용한다.

```bash
./scripts/setup_research.sh --download
```

모델 서버를 실행하는 터미널:

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/start_qwen.sh --port 8418
```

실험 터미널의 직접 CLI 예시는 기존 YAML 재현(`legacy_reproduction`)이다. 새 holdout과 GPU 예산을 관리하는 실행은 아래의 명시적 seed plan 캠페인을 사용한다.

```bash
# 동일 사건 이력 ACD / 고정 Skill
./scripts/run_research.sh B0 evaluate --episodes 4 --output "$PWD/research_outputs/new/B0"
./scripts/run_research.sh B1 evaluate --episodes 4 --output "$PWD/research_outputs/new/B1"

# Proposed 훈련: 기본 6 episode / 3 loop, 후보마다 20개 대응 검증 seed
./scripts/run_research.sh B4 train --output "$PWD/research_outputs/new/B4/train"

# 훈련이 완료된 Pool을 명시해 동결 평가
./scripts/run_research.sh B4 evaluate --episodes 4 \
  --pool "$PWD/research_outputs/new/B4/train/pool-iteration-0003.json" \
  --output "$PWD/research_outputs/new/B4/eval"

# 별도 ablation ID로 같은 설정의 군 비교
./research.sh compare --variants B2 B3-no-support B3 --reds finite --train \
  --output "$PWD/research_outputs/ablation-new"

# B0–B4 label을 구분하고 train/test/미학습 대조군을 분리한 비교
./research.sh report research_outputs/new --reasoning
python3 scripts/recompute_public_metrics.py --episodes research_outputs/new/report/episodes.json
```

B2–B4의 학습 후 성능은 완료한 훈련 Pool로 평가한다. 초기 Pool을 명시해 평가한 경우 새 비교 도구는 `untrained_control`로 분리한다. 기존 출력 경로를 덮어쓰지 말고 새 실험 경로를 지정한다. 생성한 결과는 Git에서 제외한다.

예산 관리자는 필수군의 수집·비용 측정·optimizer·후보 검증·동결 평가·정리를 GPU 시작 전에 예약한다. `--require-all-arms`이면 요청한 모든 군과 등록된 전이 평가를 포함한다. 선택 확장이 예산에 맞지 않으면 해당 군은 `deferred_budget`이고 B0/B1/B2 결과는 남는다. 기본 GPU는 1개다. GPU-hour는 기동·idle·정리까지 합산하며 wall cap과 독립이다.

별도 승인 파일이 없으면 기존 `cage4-research`의 누적 8 GPU-hour가 적용된다. `--authorization`은 실제 승인된 `scope`, 누적 `gpu_hour_cap`, `approval_reference`를 가진 JSON을 받는다. 파일에 `ledger_sources`를 넣거나 `--usage-ledger`로 기존 completion을 연결한다. 같은 scope를 연장하면 이미 쓴 시간도 포함한다. `--gpu-hours`는 이번 실행 예약이며 `--wall-hours`는 별도 제한이다. 계획과 journal에 승인 파일 해시·물리 세션·carry-in을 기록하고 재개에서 같은 제한을 사용한다.

이전 실행기의 4-GPU `--gpu-hours 8`은 8 wall-hour가 아니라 전체 120분이었다. 학습 마감은 85분, 후보 검증 마감은 83.5분이고 기동도 그 안에 포함됐다. 현 실행기는 이 고정 35분 분할 대신 실제 수집·SG·집계/Best-of-N·score 표본으로 남은 경로를 검사한다. B4는 이전 Pool 20회 + 후보 Pool 20회와 최종 비교 비용을 확보해야 한다.

```bash
# SEED_PLAN은 아래 스키마의 명시적 JSON, PRIOR_COMPLETION은 기존 사용 ledger다.
# GPU 없이 B0–B3 1-update 계획 저장. GPU cap은 승인 장부의 잔여 안에서 지정한다.
.venv/bin/python scripts/run_campaign.py --output "$PWD/research_outputs/pilot-new" \
  --seed-plan "$SEED_PLAN" --usage-ledger "$PRIOR_COMPLETION" \
  --gpu-hours "$CAMPAIGN_GPU_HOURS" --wall-hours "$CAMPAIGN_WALL_HOURS" \
  --max-gpus 1 --arms B2 B3 --require-all-arms --iterations 1 --test-episodes 2 --plan-only
# 같은 계획 실행/재개. 여유 GPU가 없으면 다른 사용자 작업을 건드리지 않고 기록 후 종료
.venv/bin/python scripts/run_campaign.py --output "$PWD/research_outputs/pilot-new" \
  --gpu-hours "$CAMPAIGN_GPU_HOURS" --wall-hours "$CAMPAIGN_WALL_HOURS" --resume
# 확장하려면 별도 계획을 사전등록: --arms B2 B3-no-support B3 B4
# 기존 캠페인의 소유 PID만 확인해 정리: 위 실행기에 --cleanup-only --resume
./research.sh report research_outputs/pilot-new --reasoning
```

seed plan은 `purpose`(`final_evaluation`/`development`/`reproduction`), `train`, `validation`, `final_test`, `transfer`의 정수 목록을 필수로 가진다. `reserved_final_test`는 선택 사항이다. 새 final/transfer 시드는 실제 완료·partial 기록과 제공된 과거 노출 이력을 대조한다. 알려진 시드의 재현/개발 재사용은 가능하지만 새 holdout으로 부르지 않는다. validation 재사용도 독립 표본 증가가 아니다. reserved 미사용 수에서는 이미 노출됐거나 pilot에 배정된 시드를 뺀다. 기존 YAML은 재현 설정이며 새 holdout을 자동 배정하지 않는다.

훈련 episode 수는 명시적 `train` 목록 길이다. `--train-episodes`를 주면 그 길이와 일치해야 한다. 작은 계획은 2 episode/1 iteration, full 6/3은 train 시드 6개와 `--iterations 3`으로 구분한다. 모든 계획은 호출 전에 저장한다. B4 예산은 iteration별 최대 검증 후보 수 × 20쌍/40episode와 평가를 포함한다.

GPU 절약용 pilot은 한 GPU에서 요청을 순차 처리한다. KEEP·MiniLM·시뮬레이터는 CPU다. 실험용 batch scorer는 기본 비활성화이며 실제 수치 동등성 실패 시 reference 경로를 유지한다. BF16, 500 nominal-step, 원래 입력·chunk·Best-of-N·보상·승인 기준을 줄이지 않는다. 2-seed pilot은 full 평가를 대신하지 않는다.

full 설정과 작은 pilot 계획을 같은 실험으로 합산하지 않는다. 학습·후보 검증·최종 test seed는 분리돼 있다. 실험 설정은 `research_configs/`, 현재 경로용 생성 설정은 `.local-configs/`에 있다.

## 검사

```bash
.venv/bin/python -m pytest -q
./scripts/run_research.sh B4 evaluate --help
```

회귀 검사는 실제 Qwen의 방어 성능 검증을 대신하지 않는다. SG/후보/score/부분 rollout 중단 후 재개, baseline 중복 반영 방지, 원래 순차 seed를 보존하는 SG 병렬화, 선택 감사 전후 prompt·RNG 일치, 행동 지원 및 조기 거부를 검사한다. 공식 reward·PPO 수식·고정 모델 및 KEEP lock은 유지하며 테스트의 기준 hash는 `tests/fixtures/reference_integrity.json`에 있다.

완료한 응답·후보·sample score·SG 검증과 loop/Pool 상태는 `journal.sqlite3`에 원자 저장한다. 기존 LRU와 batch SG cache를 영속화한 것이며 새 캐시 기능으로 혼동하지 않는다. payload의 JSON 순서도 보존해 재개 시 SG 입력이 달라지지 않게 한다. FIFO/Golden 복원은 기존 내용을 추가하지 않고 대체한다. Golden의 메모리 주소 동점 ID는 저장 가능한 삽입 ID로 바꿨으며, paper_batch의 부모 선택·PPO 계산·후보 tie-break에는 쓰이지 않는다.

loop 상태는 `completed_changed`, `completed_unchanged`, `completed_no_eligible_skill`과 `interrupted_budget`/`failed`/`partial`로 구분한다. 변경 없는 완료 정책도 동결 평가할 수 있지만 자동 성능 개선을 뜻하지 않는다. 미완료 Pool은 `untrained_control`이다.

기본 scorer는 기존 incremental reference다. full-sequence 경로는 실험용이며 실제 Qwen의 token/sequence/PPO/판정 동등성 인증 없이는 채택할 수 없다. 입력 요약·chunk 축소·Best-of-N 축소·모델/dtype/승인 기준 변경은 하지 않는다. 현재 실제 값은 temperature 1.0, actor 최대 192 token, SG/후보 최대 768 token, chunk 32 step/48,000문자, Best-of-N 2다. 입력 token은 최종 chat template 렌더링 후 확인하며 문자를 token으로 계산하지 않는다.

새 B3/B4 방법은 fact-interventions-v1이다. 코드가 만든 canonical fact ID를 LLM이 선택하고, preconditions/proposed_procedure/success_observation/termination/failure_or_retry_rule을 미검증 가설로 전달한다. action/host/time/epoch를 모델이 반복 복사하지 않는다. 잘못 선택한 ID를 다른 사실로 고치지 않으며 원문과 해석된 사실을 보관한다. 원래 B2와 chunk·입력 기록은 유지한다. 검증된 관찰 사실과 연결된 구조화 가설만 후보 입력으로 넘긴다. 원문 diagnosis/free gradient는 감사 기록으로 격리한다. 성분 범위/관찰 관계 검사는 일반 자연어 의미·인과관계·제안 절차의 효과를 보증하지 않는다. 공동 reward를 개별 Skill의 직접 기여로 해석하지 않으며, PPO Gate 양수만으로 방어 성능 개선을 주장하지 않는다.

기존 typed claim의 후속 schema v9은 `requested_decision`과 `completed_or_pending_operation`을 분리한다. 현재 Sleep/null과 이전 Restore/h1이 같은 행에 있어도 각 대상의 as_of/Restore epoch를 표시하고 같은 canonical builder로 검사한다. `recorded_decision`은 실제 호출·적용·파싱된 정책 결정만 확인하며 UNKNOWN을 완료/실패/진행 중으로 바꾸지 않는다. PID/파일/서비스는 별도 observation_target과 해당 호스트의 epoch로 검사하므로 Sleep이나 다른 호스트의 행동 중에도 실제 관측을 사용할 수 있다. 완료·실패·pending은 여전히 실제 작업 대상에 연결한다. 제안·사실 ID는 검증 메타데이터에만 두며 reusable Skill에 복사하지 않는다.

후보 변경은 요소별 정확한 텍스트와 검증된 proposal/fact ID를 연결한다. 같은 component라는 사실만으로 자연어 의미가 검증되지 않는다. Remove TRUE를 privileged persistence 제거의 증거로 확대하는 명확한 모순을 검사하고 나머지 의미는 unknown이다. 사실/출처, candidate_links_valid, known_contradiction, efficacy_verified와 admission_mode를 구분한다. 새 B3의 factual_ppo는 사실·후보 연결·알려진 모순 검사와 원본 양수 PPO를 통과하면 효과가 미검증인 후보도 채택할 수 있다. B4의 rollout 모드는 같은 검사 뒤 실제 20쌍 승인을 요구한다. ASV-only는 미실행 행동·미관측 조건·미분류 변경에 추가 검증을 요구하지만 텍스트가 바뀌었다는 이유만으로 전부 금지하지 않는다. 과거 legacy_rollout_veto protocol은 보존하며 새 방법으로 소급 해석하지 않는다. SSV-only는 여러 parent 채택·capacity 정리 후 최종 Pool을 다시 검사한다. 원본 B2의 양수 J 승인 규칙은 유지하고 J_old/Delta J는 진단값으로만 기록한다.

201000/201001, 211000/211001, 215000은 이미 결과를 보았고 213000/213001은 validation에 사용했다. 101000–101099·102000 등 제공된 이전 개발 시드도 노출 이력에 포함한다. 보고기는 내부 정책 유형·정확한 optimizer 완료 상태·checkpoint 및 동결 Pool 해시·실제 학습 방법으로 train/test/control을 분류한다. 원래 B2 Pool을 완료된 Proposed 학습으로 다시 이름 붙일 수 없다. 독립 학습 수는 run/lineage provenance로 계산하며 B0/B1은 해당 없음, 누락은 미상이다. 반복 run의 같은 eval seed를 덮어쓰지 않고, 여러 run의 CI는 comparison-run cluster를 보존한다.

## 확인된 성능과 진행 범위

| 과거 동결 평가 | N | B0 ACD | B1 Static | B2 Skill-Pro | B2−B1 |
|---|---:|---:|---:|---:|---:|
| FiniteState 211000/211001 | 2 | −164.0 | −193.0 | −201.5 | −8.5 |
| Aggressive 215000 | 1 | −277 | −184 | −262 | −78 |

FiniteState의 B1−B0는 −29, B2−B0는 −37.5다. SD는 B0 28.28/B1 11.31/B2 30.41, 평균 평가 시간은 674.49/934.15/957.54초다. 이는 동결 source 39eaae4의 결과다. 201000/201001은 별도 historical/development이며 이 평균에 합치지 않는다. B2는 2훈련 episode·최적화 1회·채택 1개를 완료했고 fixed-update study의 새 loop는 0이다. B3/B4 전체 학습 후 성능은 N/A다. validation은 2/20쌍, 차이 +15/−35로 reject_screen_inconclusive(보류)이며 승인/보상 거부가 아니다.

검토 기준 6c1d229 이후의 CPU 수정과 실제 SG probe는 이 동결 보상과 다른 증거다. 이번 작업 전 누적 GPU는 6.717276/8시간, 잔여는 1.282724시간이었다. 추가 사용은 새 output의 실제 ledger로 더하며 새 8시간으로 해석하지 않는다. 40회 validation의 약 11.69 GPU-hour는 예상이며 SG/최종 평가 비용은 별도다. 이 비용을 현재 잔여 예산으로 완료했다고 주장하지 않는다.

9cd80a3까지의 경계 수정에서는 CPU 검사 314개가 통과했다. 실제 Qwen SG는 원래 전체 chunk에서 사전 선택한 6건씩 두 번 측정했다. canonical-v2는 JSON 6건·선택된 관찰 claim 검사 1건·proposal 연결 0건, 필드 명세를 맞춘 v3는 각각 6·1·1건이었다. 두 실행은 prompt와 생성 seed가 달라 통과 수 차이를 수정의 인과 효과로 해석하지 않는다. 연결된 제안 1건도 후보 생성·PPO·rollout 채택을 뜻하지 않으며 해당 SG-only 작업의 새 loop와 CAGE4 평가는 0이다. 실제 추가 GPU는 0.165175시간, 누적 6.882450/8시간, 잔여 1.117550시간이다. 등록한 서버 2개는 종료됐고 8418–8421 포트가 닫힌 것을 재확인했다.

로컬 상세 진행 보고는 `research_outputs/audit-fields-ece94dbe/report/performance.md`에 있다. 이전 모델 원문은 `audit-boundaries-20261125-9e1ec3c0`에 그대로 보존했다. 두 원시 실험 경로는 Git 제외 대상이다. 실제 과거 checkpoint·SQLite backup·두 trace·초기 Pool을 묶음에서 격리 복원했고, 원래 프로젝트 데이터 접근을 막은 상태에서 `historical_batch`를 읽었다. weights와 독립 CAGE4 재실행은 이 복원 검사에 포함하지 않는다.

v2 재현 묶음은 현재 output 밖의 과거 checkpoint/journal/훈련 trace/초기 Pool도 포함하고 같은 내용은 hash로 참조한다. SQLite는 read-only 일관 backup이다. `scripts/report_experiment.py --restore-bundle <archive> --destination <empty-dir>`로 상대 경로를 복원한다. 원래 절대 경로는 provenance로 남긴다. source/model/KEEP lock, compact summary, raw bundle의 범위는 다르며 weights는 제외한다. `--summary-only`는 작은 지표만 준비하며 자동 업로드나 Git 작업을 하지 않는다.

보고기의 `bottleneck_report.json`, `optimizer_progress.json`, `resume_verification.json`, `diagnosis_evaluation.json`, `performance.md`/`metrics.json`, `skill_changes.json`, `reproducibility_manifest.json`과 압축 bundle을 확인한다. compact episode 지표만으로는 optimizer나 원시 prompt를 복원할 수 없다. 캠페인은 성공·예산 종료·예외·TERM 모두에서 기록된 자기 PID/자식 worker만 종료하고 GPU PID·메모리·포트를 확인한다. 이 작업은 Git commit/push나 외부 업로드를 자동 실행하지 않는다.

비용 예측은 배치마다 해당 arm의 실제 입력 길이, SG·후보 표본, exact-token scoring, 수집 시간을 사용한다. 같은 모델·prompt·target token의 exact score는 저장한 값을 재사용한다. 비용 측정용 생성은 별도 seed를 쓰며 optimizer 입력에서 분리한다. 첫 예약에서 미측정 optimizer는 과거 계수인 추정치로 표시하고, 배치별 측정 뒤 모든 필수군의 잔여 경로를 다시 계산한다. 예상이 빗나가면 동일 GPU/wall cap에서 journal과 checkpoint를 보존한다.

새 검증은 `--variant B2-ASV`, `B2-SSV`, `B2-ASV-SSV`, `B2-ASV-SSV-RV` 및 `B4-ASV`, `B4-SSV`, `B4-ASV-SSV`로 선택한다. 기존 B2/B3/B4의 정의와 설정 파일은 복제하지 않는다. ASV는 PPO가 평가한 행동, 실제 시작·완료, 관찰 조건을 구분하고 변경된 절차의 지원 범위 밖 행동에는 실제 rollout을 요구한다. SSV는 같은 훈련 관찰에서 기존 upstream selector를 호출하고 선택 변화·집중·사라진 Skill을 기록한다. 원래 Python RNG와 live 모델 counter를 보존하며 재생 선택을 환경 보상으로 해석하지 않는다.

`B2-ASV-SSV-RV-screen`은 별도 버전의 거부 전용 절차다. 사전 등록된 첫 2쌍에서 기존 worst-seed 손실 기준을 위반하면 조기 거부할 수 있다. 위반이 없으면 보류이며, 승인은 여전히 20쌍·40 episode와 기존 모든 기준이 필요하다. 보류를 방어 보상 실패나 rollout 거부율로 세지 않는다.

이미 자동 생성·채택된 update의 검증만 비교할 때는 같은 캠페인 실행기에 `--fixed-update-from`을 사용한다. 이는 전체 B3/B4 재학습이나 새 closed-loop 완료가 아니다. 아래 cap은 실제 연결 ledger의 잔여 한도 이하여야 하며, seed plan은 과거 훈련 시드와 평가 목적을 명시해야 한다.

```bash
.venv/bin/python scripts/run_campaign.py --output "$PWD/research_outputs/support-shift-new" \
  --fixed-update-from "$PWD/research_outputs/one-gpu-resume-20261125-ea06ba64/frozen/B2.json" \
  --seed-plan "$SEED_PLAN" --usage-ledger "$PRIOR_COMPLETION" \
  --gpu-hours "$REMAINING_GPU_HOURS" --wall-hours "$WALL_HOURS" \
  --max-gpus 1 --minimum-free-mib 24000 --generalization --plan-only
# 위 계획에서 --plan-only를 빼고 --resume으로 실행한다.
```

이 모드는 기존 훈련 91000·91001만 검증 입력으로 읽고 평가·validation·transfer는 명시적 seed plan을 따른다. 이미 사용한 211000·211001·213000·213001·215000을 새 독립 holdout으로 재사용할 수 없다. 초기 Pool과 완료한 B2 Pool을 정확히 고정한다. 같은 최종 Pool 결과를 재사용하면 원본 평가와 해시·추론 조건을 명시하고 추가 episode로 세지 않는다. GPU에 기존 compute PID가 있으면 여유 메모리와 관계없이 선택하지 않는다. `report/`의 `action_support.json`, `selector_probe.json`, `historical_forensics.json`, `fixed_update_rollout.json`, `scorer_parity.json`, `sg_boundary_profile.json`에 검증과 수치 정확도·비용을 남긴다. full-sequence scorer는 완전한 인증 전까지 생산 경로에 적용하지 않는다.

## 후속 정합성 수정과 controller 실험

B3의 영구 보류와 cross-host observation 거부를 수정했다. 기존 SG 12건은 같은 원문·관측 행 hash로 재검사했고 새로 해제된 거부는 0건이다. 최종 1/6을 대표 진단 정확도나 방어 성공률로 부르지 않는다. 새 fact-ID schema의 137개 원래 chunk(eligible 124)는 tokenizer 검사에서 최대 19,392 input + 768 output 예약으로 context 24,576 안에 들었다. 실제 새 SG 생성을 측정한 수치는 아니다.

기본 B1/B2/B3/B4의 initiation·allowed_actions는 자연어 지침이며 강제 action mask·eligibility·urgent preemption을 보장하지 않는다. runtime CAGEVerifier는 E3 전용이다. 별도 B1-selector-contract와 B3-selector-contract는 selector 역할 지침 하나만 추가한 정적/최적화 대조군이다. primitive actor·termination prompt·epsilon·행동 공간·maturity/gain 필터는 그대로다. 원본 llm_model의 maturity>=3, avg_gain>-0.015 필터와 min_gain 전달 누락은 감사 로그에 남기며 변경하지 않는다.

기존 `run_campaign.py --frozen-policy-pilot --pilot-controller B0`는 공통 조건의 B0/B1 기준점을 사전 등록한 개발 seed 하나에서 측정한다. 기본 `B1-selector-contract` 선택은 동일 초기 Pool의 두 정적 selector를 비교한다. 각 쌍에 500 nominal-step episode 두 개와 기동·정리 비용을 먼저 예약한다. `--plan-only`는 예약 비용과 현재 예산 충족 여부를 기록한다. 이전 실제 selector 비교는 `research_outputs/selector-pilot-23071375`에 있다.

새 공통 비교 `cage4-acd-parity-v1`의 B0–B4는 고정한 원본 ACD 함수로 송신 메시지와 자기 관찰 벡터를 생성한다. 이전 결과에서 encoder를 명시하지 않은 설정은 `legacy_local_v1`로 읽는다. 원래 조건을 재현하려면 해당 결과의 등록된 config를 사용한다. 연구 CLI와 ACD CLI는 `--communication-encoder`로 모드를 선택할 수 있다. 원본의 severity 변화는 공개 KEEP의 기존 8→3 projection 입력에도 영향을 주므로 새 B0부터 평가한다. KEEP 가중치는 공개 CAGE4 제출본을 사용한다.

새 `explicit_zone_pair_v1` 행동 계약은 각 Blue의 원본 고정 출발 구역과 목적지 한 개를 명시한다. HQ의 출발 구역은 `public_access_zone_subnet`이다. trace에서 요청한 구역 쌍, 실행된 구역 쌍, 관찰된 호출 완료를 각각 확인할 수 있다. 학습된 Pool과 경험은 기록된 encoder 조건에서 재개한다. B4 동결 평가와 후보 검증은 훈련의 등록 승인 규칙을 보존하고 optimizer 없이 실행된다. 최종 평가의 seed는 후보 검증용 seed와 분리한다.

`B1-urgent-preemption`과 `B3-urgent-preemption`은 같은 controller를 쓰는 정적·학습 비교군이다. 현재 관찰로 대응 가능한 새 긴급 단서가 생기고 다른 종류의 Skill이 활성화되어 있으면 원본 selector에 한 번 다시 선택을 요청한다. `--frozen-policy-pilot --pilot-controller B1-urgent-preemption`은 정적 한 쌍을 예약하며, 훈련 캠페인에 `--arms B3-urgent-preemption`을 포함하면 대응 정적 대조군도 함께 평가한다. 각 실행의 전체 비용 예약을 함께 적용한다.

후속 실제 정적 paired pilot(seed 788380683, development)은 B1 -121 / B1-selector-contract -122, 차이 -1이었다. N=1의 SD/CI는 N/A이며 개선이나 자동 Skill 최적화 효과를 입증하지 못했다. LWF/ASF/RIA 합계는 같고 공동 action cost만 -1이었다. 모델 호출은 698→733회였고 전체 평가 시간은 각각 906.69/899.15초다. 이 차이를 처리속도 개선이나 특정 Qwen 행동의 인과 효과로 해석하지 않는다. 다음 최종 holdout에는 이 seed를 쓰지 않는다.

위 정적 실측 직후의 CPU 회귀는 329개였다. 최신 원본 정렬 작업에서는 실제 CybORG와 테스트 모델로 B4 동결 1 episode를 CPU에서 마쳤다. 같은 보고서 맨 위에 최신 검사와 과거 실측을 구분해 기록했다. 기존 승인 누적 사용량은 7.388473/8 GPU-hour이며 잔여는 0.611527 GPU-hour다. 새 공통 조건의 Qwen 점수와 학습 후 B3/B4 점수는 N/A다.

공개 개발 seed `788380683`의 노출 이력은 원시 출력이 없는 checkout에서도 확인된다. 새 final/transfer 등록에서는 거부하고, development/reproduction에서는 노출된 재사용으로 기록한다.

학습된 Pool에 인터페이스가 기록되지 않은 경우 원래 학습 manifest와 체크포인트의 조건을 확인한다. 따로 보관한 과거 Pool에는 원본 manifest 경로와 SHA256을 등록할 수 있다. 후보 검증과 final/transfer, 학습 완료 근거가 없는 대조 실행은 공통 보고 경로에서 구분한다. rollout 완료 수는 실제로 짝이 맞고 보상 차이가 유한한 seed만 센다.
