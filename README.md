# CAGE4 + ACD + Skill-Pro

고정 Qwen3-8B와 KEEP 4개가 CAGE4 가상 기업망을 방어하고, 실행 경험으로 외부 Security Skill을 개선하는 연구 코드다. Qwen 가중치는 업데이트하지 않는다.

현재 실행 코드는 루트 `research/` 하나다. 과거 실행 결과·원시 로그·revision 보존본·일회성 분석 도구는 제거했다. 모델·가상환경·원본 소스·runtime은 실행에 필요한 로컬 자원으로 유지하며 Git에는 포함하지 않는다. 과거 성능 수치나 학습 완료 Pool을 제공하는 저장소는 아니다.

[파일별 기능·데이터 구조](docs/code_and_data_guide.md), [전체 파일 목록 CSV](docs/code_and_data_inventory.csv), [논문·원본 출처](references/README.md)를 제공한다.

## 비교 실험군

| 군 | 구성 | 내부 arm |
|---|---|---|
| B0 | Qwen ACD + 공통 관찰/사건 이력 | E0 |
| B1 | B0 + 고정 초기 Security Skill | E1 |
| B2 | Skill-Pro 논문식 SG·batch·RTG·PPO Gate | E2 |
| B3 | B2 + evidence/diagnosis 검증·독립 episode 지원 | E2 |
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

# B0–B4 label을 구분하고 train/test/미학습 대조군을 분리한 비교
./research.sh report research_outputs/new --reasoning
python3 scripts/recompute_public_metrics.py --episodes research_outputs/new/report/episodes.json
```

B2–B4의 학습 후 성능은 완료한 훈련 Pool로 평가한다. 초기 Pool을 명시해 평가한 경우 새 비교 도구는 `untrained_control`로 분리한다. 기존 출력 경로를 덮어쓰지 말고 새 실험 경로를 지정한다. 생성한 결과는 Git에서 제외한다.

4개 GPU의 점유 시간 합계를 제한하는 별도 pilot 관리자가 있다. 이 관리자는 모델 서버의 기동·종료도 담당한다.

```bash
.venv/bin/python scripts/run_campaign.py --output "$PWD/research_outputs/pilot-new" --gpu-hours 8
./research.sh report research_outputs/pilot-new --reasoning
```

full 설정과 작은 pilot 계획을 같은 실험으로 합산하지 않는다. 학습·후보 검증·최종 test seed는 분리돼 있다. 실험 설정은 `research_configs/`, 현재 경로용 생성 설정은 `.local-configs/`에 있다.

## 검사

```bash
.venv/bin/python -m pytest -q
./scripts/run_research.sh B4 evaluate --help
```

정리 후 회귀 검사 175개가 통과했다. 모델 추론/방어 성능 검증을 대신하는 결과는 아니다. 공식 reward·PPO 수식·고정 모델 및 KEEP lock은 유지하며 테스트의 기준 hash는 `tests/fixtures/reference_integrity.json`에 있다.

구조화된 진단 검증은 행동·대상·시점을 검사하지만 자유 서술 전체의 의미나 인과관계를 보증하지 않는다. 공동 reward를 개별 Skill의 직접 기여로 해석하지 않으며, PPO Gate 양수만으로 방어 성능 개선을 주장하지 않는다.
