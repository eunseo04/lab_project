# CAGE4 + ACD + Skill-Pro

고정 Qwen3-8B와 KEEP 4개가 CAGE4 가상 기업망을 방어하고, 실행 경험으로 외부 Security Skill을 개선하는 연구 코드다. Qwen 가중치는 업데이트하지 않는다.

현재 실행 코드는 루트 `research/` 하나다. 과거 실행 결과·원시 로그·학습 완료 Pool은 현재 체크아웃에서 복구할 수 없다. `artifacts/initial_pool.json`은 훈련 전 정적 절차다. 모델·가상환경·원본 소스·runtime은 실행에 필요한 로컬 자원으로 유지하며 Git에는 포함하지 않는다. 과거 성능 수치나 학습 완료 Pool을 제공하는 저장소는 아니다.

[파일별 기능·데이터 구조](docs/code_and_data_guide.md), [전체 파일 목록 CSV](docs/code_and_data_inventory.csv), [논문·원본 출처](references/README.md)를 제공한다.

## 비교 실험군

| 군 | 구성 | 내부 arm |
|---|---|---|
| B0 | Qwen ACD + 공통 관찰/사건 이력 | E0 |
| B1 | B0 + 고정 초기 Security Skill | E1 |
| B2 | Skill-Pro 논문식 SG·batch·RTG·PPO Gate | E2 |
| B3-no-support | B2 + evidence/diagnosis 검증, 최소 episode 제한 없음 | E2 |
| B3 | B2 + evidence/diagnosis 검증·최소 독립 episode 2개 | E2 |
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

실험 터미널:

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

예산을 관리하는 정식 pilot 실행기는 B2 한 update를 먼저 완료하고 비용이 허용할 때 다음 군을 실행한다. 기본은 GPU 1개이며, `--max-gpus 4`는 동일 모델 replica의 SG/scoring·평가 큐를 최대 4개로 확장한다. 처리량 개선은 실제 측정 전에는 주장하지 않는다. 모델 서버의 기동·idle·종료까지 합산한 **신규 8 GPU-hour**와 **wall-clock cap**을 독립 관리하며, 재개해도 누적 사용량을 초기화하지 않는다. wall cap은 최초 GPU 캠페인 시작부터의 경과 시간이라 중단 기간도 포함한다.

이전 실행기의 4-GPU `--gpu-hours 8`은 8 wall-hour가 아니라 전체 120분이었다. 학습 마감은 85분, 후보 검증 마감은 83.5분이고 기동도 그 안에 포함됐다. 현 실행기는 이 고정 35분 분할 대신 실제 수집·SG·집계/Best-of-N·score 표본으로 남은 경로를 검사한다. B4는 이전 Pool 20회 + 후보 Pool 20회와 최종 비교 비용을 확보해야 한다.

```bash
# 계획만 저장: GPU 사용 없음
.venv/bin/python scripts/run_campaign.py --output "$PWD/research_outputs/pilot-new" \
  --gpu-hours 8 --wall-hours 8 --max-gpus 1 --arms B2 --plan-only
# 같은 계획 실행/재개. 여유 GPU가 없으면 다른 사용자 작업을 건드리지 않고 기록 후 종료
.venv/bin/python scripts/run_campaign.py --output "$PWD/research_outputs/pilot-new" \
  --gpu-hours 8 --wall-hours 8 --resume
# 확장하려면 별도 계획을 사전등록: --arms B2 B3-no-support B3 B4
# 기존 캠페인의 소유 PID만 확인해 정리: 위 실행기에 --cleanup-only --resume
./research.sh report research_outputs/pilot-new --reasoning
```

현재처럼 GPU 사용을 줄일 때는 `--max-gpus 1 --test-episodes 2`로 별도의 실행 확인용 pilot을 등록한다. 생성과 reference scoring은 순차 실행하며 KEEP·MiniLM·시뮬레이터는 CPU를 사용한다. 메모리가 큰 실험용 batch scorer 검사는 기본으로 실행하지 않고 `--profile-full-sequence`를 지정한 경우에만 수행한다. Qwen BF16, 500 nominal-step, 원래 입력·chunk·Best-of-N·PPO/rollout 기준은 유지한다. 2-seed 결과는 불확실성이 큰 pilot이며 full 평가를 대신하지 않는다.

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

B3/B4는 typed claim → canonical 관찰 사실 → 연결된 구조화 가설만 후보 입력으로 넘긴다. 원문 diagnosis/free gradient는 감사 기록으로 격리한다. 성분 범위/관찰 관계 검사는 일반 자연어 의미·인과관계·제안 절차의 효과를 보증하지 않는다. 공동 reward를 개별 Skill의 직접 기여로 해석하지 않으며, PPO Gate 양수만으로 방어 성능 개선을 주장하지 않는다.

새 캠페인은 훈련 **212000·212001**, 후보 검증 **213000–213019**, test **211000부터**, 다른 Red **215000**을 호출 전에 등록한다. 이미 결과를 확인한 201000·201001은 개발 분석 자료다. 기존 YAML의 과거 시드는 기존 결과 재현용이며 새 독립 성능 입증에 재사용하지 않는다. 보고기는 B1−B0, B2−B1, B3−B2, B4−B3 등 완료 seed 쌍, 표본 SD, 평균/중앙값, win/tie/loss, 사전 고정 bootstrap CI를 낸다. 미실행은 N/A로 남긴다. 동일 source/model/protocol에서 한 군의 서로 다른 동결 Pool 결과를 합치지 않는다.

보고기의 `bottleneck_report.json`, `optimizer_progress.json`, `resume_verification.json`, `diagnosis_evaluation.json`, `performance.md`/`metrics.json`, `skill_changes.json`, `reproducibility_manifest.json`과 압축 bundle을 확인한다. compact episode 지표만으로는 optimizer나 원시 prompt를 복원할 수 없다. 캠페인은 성공·예산 종료·예외·TERM 모두에서 기록된 자기 PID/자식 worker만 종료하고 GPU PID·메모리·포트를 확인한다. 이 작업은 Git commit/push나 외부 업로드를 자동 실행하지 않는다.

비용 예측은 실측 길이가 가까운 prompt 표본과 전체 SG token 길이 목록을 사용한다. 실측 최대 출력에 여유를 더한 보수적 예상과 모든 호출이 768 token을 채우는 용량 추정을 별도로 기록하며, 실제 생성 상한은 유지한다. 예상이 빗나가면 동일 누적 GPU/wall cap에서 저장 후 중단한다.

새 검증은 `--variant B2-ASV`, `B2-SSV`, `B2-ASV-SSV`, `B2-ASV-SSV-RV` 및 `B4-ASV`, `B4-SSV`, `B4-ASV-SSV`로 선택한다. 기존 B2/B3/B4의 정의와 설정 파일은 복제하지 않는다. ASV는 PPO가 평가한 행동, 실제 시작·완료, 관찰 조건을 구분하고 변경된 절차의 지원 범위 밖 행동에는 실제 rollout을 요구한다. SSV는 같은 훈련 관찰에서 기존 upstream selector를 호출하고 선택 변화·집중·사라진 Skill을 기록한다. 원래 Python RNG와 live 모델 counter를 보존하며 재생 선택을 환경 보상으로 해석하지 않는다.

`B2-ASV-SSV-RV-screen`은 별도 버전의 거부 전용 절차다. 사전 등록된 첫 2쌍에서 기존 worst-seed 손실 기준을 위반하면 조기 거부할 수 있다. 위반이 없으면 보류이며, 승인은 여전히 20쌍·40 episode와 기존 모든 기준이 필요하다. 보류를 방어 보상 실패나 rollout 거부율로 세지 않는다.

이미 자동 생성·채택된 update의 검증만 비교할 때는 같은 캠페인 실행기에 `--fixed-update-from`을 사용한다. 이는 전체 B3/B4 재학습이나 새 closed-loop 완료가 아니다. 예를 들어 기존 8 GPU-hour 중 3.523766시간을 썼다면:

```bash
.venv/bin/python scripts/run_campaign.py --output "$PWD/research_outputs/support-shift-new" \
  --fixed-update-from "$PWD/research_outputs/one-gpu-resume-20261125-ea06ba64/frozen/B2.json" \
  --prior-gpu-hours 3.5237660427888233 --gpu-hours 4.476233957211177 \
  --wall-hours 4.476233957211177 --max-gpus 1 --minimum-free-mib 24000 \
  --test-episodes 2 --profile-full-sequence --generalization --plan-only
# 위 계획에서 --plan-only를 빼고 --resume으로 실행한다.
```

이 모드는 기존 훈련 91000·91001만 검증 입력으로 읽고 새 test는 211000·211001, 후보 검증은 213000부터 사용한다. 초기 Pool과 완료한 B2 Pool을 정확히 고정한다. 같은 최종 Pool 결과를 재사용하면 원본 평가와 해시·추론 조건을 명시하고 추가 episode로 세지 않는다. GPU에 기존 compute PID가 있으면 여유 메모리와 관계없이 선택하지 않는다. `report/`의 `action_support.json`, `selector_probe.json`, `historical_forensics.json`, `fixed_update_rollout.json`, `scorer_parity.json`, `sg_boundary_profile.json`에 검증과 수치 정확도·비용을 남긴다. full-sequence scorer는 완전한 인증 전까지 생산 경로에 적용하지 않는다.
