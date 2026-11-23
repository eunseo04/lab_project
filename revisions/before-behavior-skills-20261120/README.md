# CC4 LLM+RL / KEEP 실험 환경

## ACD + Skill-Pro + CAGE4 연구 확장 (2026-11-20)

연구명: **Closed-Loop Self-Improvement of Autonomous Cyber Defense Agents through
Non-Parametric PPO-Based Security Skill Optimization**

새 연구 구현은 **`/home/jeonges/proposed`** 에 있다. 기존
`/home/jeonges/acd-experiments`와 `acd/` 코드는 보존했다. 새 진입점은
`research.sh`이며 E0(Qwen), E1(Static Skills), E2(Skill-Pro 이식),
E3(E2 + CAGE4 관찰 기반 검증)를 독립 실행한다. Qwen3-8B의 가중치는 고정하고
외부 Skill Pool만 갱신한다. 기존 주석 API 예시와 분리된 실제 로컬 추론 경로다.

Skill 최적화의 현재 권장 설정은 [`paper-cage4-fixed.yaml`](research_configs/paper-cage4-fixed.yaml)이다.
출판 논문 Algorithm 1의 현재 배치 진화, RTG Gate, 점수 기반 용량 관리를 적용하고
CAGE4의 정상 대기와 실행 대상 형식 문제를 수정했다.
[수정 근거·검증·실측 결과](docs/skillpro_paper_fix.md)와
[새 비교 결과](research_outputs/paper-fix/report/performance.md)를 참고한다.
수정본의 테스트 59개는 통과했다. 동일 4개 평가 시드의 FiniteState 평균 보상은
ACD −287.75, 학습 전 E3 −303.50, loop 1 −319.50, loop 2 −320.25로,
이번 표본에서는 성능 개선을 확인하지 못했다. 코드·수식 검증과 보상 개선은 구분한다.
AggressiveFSM 전이도 ACD −261.75, 최종 E3 −338.50이었다. 등록한 평가 24회와
훈련 4회·검증 2회가 모두 완료됐으며 시드별 결과와 부진 원인을 보고서에 기록했다.
아래 `controlled.yaml`, `E2.yaml`, `E3.yaml`은 기존 공개 코드 모드의 비교 설정이다.

```bash
# 8018의 고정 Qwen 서버가 준비된 상태: 500-step, 2 loops, 새 평가 seed 4개
./research.sh train --config research_configs/paper-cage4-fixed.yaml --output research_outputs/paper-next/E3-hybrid-finite/train
./research.sh evaluate --config research_configs/paper-cage4-fixed.yaml --output research_outputs/paper-next/E3-hybrid-finite/eval
```

현재 설치된 CPU 환경과 CUDA 모델 환경은 분리되어 있다. RTX A5000 24GB에서
BF16 Qwen3-8B 실행을 확인했다. 초기 설치에는 Python 3.10, CUDA 호환 NVIDIA
드라이버, 모델 다운로드 네트워크와 충분한 디스크가 필요하다. 새 머신에서는:

```bash
cd /home/jeonges/proposed
./scripts/setup_research.sh --download
```

이미 준비된 현재 환경에서는 설치를 반복할 필요 없이 모델 서버를 시작한다.
서버 포트가 이미 사용 중이면 새 서버를 중복 실행하지 않는다.

```bash
# 터미널 1: API 키 없이 로컬 Qwen. GPU 번호를 바꿀 수 있다.
CUDA_VISIBLE_DEVICES=0 ./scripts/start_qwen.sh --port 8018

# 터미널 2
cd /home/jeonges/proposed
./research.sh doctor
./research.sh test
.venv/bin/python scripts/validate_qwen_scoring.py
```

설정의 `model.endpoint`를 서버 주소에 맞춘다. 모델 ID/revision은 검증하며
장애 시 다른 LLM으로 대체하지 않는다. `research_models.lock.json`에는 실제
다운로드한 가중치·토크나이저 파일 hash가 있다. `.model-venv`에는 원본 실험의
CPU torch/Ray를 설치하지 않는다.

현재 실험 설정은 모두 같은 Qwen3-8B revision을 사용한다. 추후 다른 모델로
새 실험을 만들려면 YAML의 `model.model`, `revision`, `endpoint`를 명시적으로
바꾸고 별도 output을 사용한다. 로컬 HF 서버는 `--model`, `--model-id`,
`--revision`으로 가중치/identity를 지정할 수 있다. 다른 모델의 Skill Pool은
섞어 불러오지 않는다. 일반 백엔드는 `/health`의 model/revision/weights_frozen,
`/generate`의 text/token_ids, `/score`의 정확한 token logprob 합,
`/embed`의 고정 encoder 계약을 구현해야 한다. 기존 주석 SDK 예시는 유지했다.

실제 CLI 사용 예시는 다음과 같다. 기존 출력 디렉터리는 `--resume` 없이
덮어쓰지 않는다. 기본 최종 평가는 100 episodes다.

```bash
# E0/E1: nominal 500 steps, 원논문 규모의 2-episode 점검
./research.sh evaluate --config research_configs/controlled.yaml --arm E0 --episodes 2 --output research_outputs/check-E0
./research.sh evaluate --config research_configs/controlled.yaml --arm E1 --episodes 2 --output research_outputs/check-E1

# E2/E3: 훈련 6 episodes / 3 iterations → 검증 → 고정 Pool 평가
./research.sh train --config research_configs/E2.yaml
./research.sh evaluate --config research_configs/E2.yaml --limit 2
./research.sh evaluate --config research_configs/E2.yaml --limit 2 --resume
./research.sh train --config research_configs/E3.yaml
./research.sh evaluate --config research_configs/E3.yaml --limit 2

# 특정 Pool은 --pool로 지정한다. eval은 학습 경험을 추가하지 않는다.
./research.sh evaluate --config research_configs/E2.yaml --pool research_outputs/full/E2-hybrid-finite/train/checkpoint.json --output research_outputs/another-eval --limit 2

# 기존 Pool에서 후속 훈련을 시작할 때는 새 train_seed/output 설정과 --pool을 쓴다.
# Pool뿐 아니라 baseline과 경험 버퍼도 이어받고 전체 학습 시드를 보존한다.

# E0–E3 × Red 5종. --limit은 계획을 유지한 이번 호출의 새 episode 예산이다.
./research.sh compare --config research_configs/controlled.yaml --train --limit 2
./research.sh compare --config research_configs/controlled.yaml --train --limit 2 --resume

# 전체 완료할 때까지 일괄 실행
./research.sh compare --config research_configs/controlled.yaml --train --resume

# 원본 threshold=6 이후에도 학습 episode가 이어지는 12-episode / 6-iteration 계획
./research.sh compare --config research_configs/closed-loop.yaml --train --limit 2

# All LLM은 별도 실험 root에 저장한다.
./research.sh compare --config research_configs/controlled.yaml --composition all_llm --train --output research_outputs/full-all-llm

# 실제 원본 평가기: 원본 reset stream 및 official 보상 누적 대조
./research.sh official --config research_configs/paper-compatible.yaml --composition no_blue --episodes 2 --output research_outputs/new-official-parity

# paper RTG/online-score 모드는 공개 코드 모드와 별도 비교한다.
./research.sh compare --config research_configs/paper-rtg.yaml --train --output research_outputs/paper-formulas

# 결과 CSV/Markdown/PNG. --reasoning은 실제 이유의 로컬 embedding 분석이다.
./research.sh report research_outputs/real-500 --reasoning
```

`--red`는 `finite/aggressive/stealthy/impact/degrade`, `--composition`은
`hybrid/all_llm/no_blue/all_rl`이다. `--train-episodes`, `--iterations`,
`--episodes`, `--steps`, `--arm`, `--pool`, `--endpoint`, `--output`으로 개별 조건을 정한다.
긴 실행은 `--limit`과 `--resume`을 사용한다. 설정을 바꿀 때는 새 output을 쓴다.
PPO/SG/score ablation은 `research_configs/without_*.yaml`에 있다.

터미널/실행 도구의 시간 제한을 피하려면 아래처럼 독립 프로세스로 실행한다.
PID와 stdout/stderr 로그는 `research_outputs/jobs/`에 남고, 진행 상태는 각
실험의 `summary.json`에서 확인한다. 같은 이름의 살아 있는 작업은 중복 실행하지 않는다.

```bash
python3 scripts/research_background.py --name full-batch -- compare --config research_configs/closed-loop.yaml --train --limit 2
```

결과에는 episode JSON/JSONL, manifest, frozen Pool, checkpoint, iteration별
Skill/gradient/Gate 기록, 검증 Reward, 호출·토큰·시간, 실패 이유가 남는다.
`report/`에는 4개 결과표와 7개 그림의 데이터/상태가 생성된다. Mock은 제외하고
미실행 칸은 N/A다. 논문 수치는 Paper Reference로만 표시한다.
여러 길이·보상·모델 조건이 있으면 학습 곡선도 `report/training-*`별로 분리한다.
구체적인 source와 조건은 `report_manifest.json`에 기록한다.

원본 평가기 특성상 nominal 500은 실제 공동 전이 499개 / 점수 전이 498개다.
기존 legacy 실행기의 terminal-inclusive/fixed horizon과 혼합하지 않는다.
공통 wrapper 수정, 공개 KEEP의 통신 호환 변환, Qwen 모델 변경 때문에
원논문 숫자의 무수정 재현이라고 주장하지 않는다.

- [실제 실행 결과·완료 범위](docs/execution_report.md)
- [원본 파일/함수·커밋 분석](docs/reproduction_audit.md)
- [Skill-MDP·Semantic Gradient·PPO Gate](docs/methodology.md)
- [시드·예산·평가·재개 규칙](docs/experimental_protocol.md)
- [Baseline 비교 가능 범위](docs/baseline_comparison.md)
- [원본 대비 코드 변경](docs/implementation_changes.md)

아래는 보존된 기존 `acd-experiments` 실행 경로의 설명이다.

---

사용자 환경에 설치된 독립 실험 경로: `/home/jeonges/acd-experiments`.
Python 3.10.12와 CPU PyTorch를 사용한다. 이전 `/tmp/cage4-*` 환경을 수정하지 않는다.

다음 원본을 고정해서 사용한다.

| 자료 | 용도 |
|---|---|
| [Large Language Models are Autonomous Cyber Defenders, v2](https://arxiv.org/abs/2505.04843v2) | 5개 방어 에이전트, 4개 팀 구성, 공격자 5종, 2에피소드·500스텝, 통신 프로토콜 |
| [r4wd3r/llms-are-acd](https://github.com/r4wd3r/llms-are-acd) | 원본 프롬프트·자연어 관측 포매터·공격자 정책·KEEP 확장 |
| [cybermonic/cage-4-submission](https://github.com/cybermonic/cage-4-submission) | 실제 GCN/PPO 구현, 그래프 표현, 공개 가중치 5개 |
| [cage-challenge-4](https://github.com/cage-challenge/cage-challenge-4) | CybORG 시뮬레이터 |

커밋과 공개 가중치 SHA256은 [sources.lock.json](sources.lock.json)에 기록했다.
`third_party/`는 원본 체크아웃이고, 조립·최소 패치는 `runtime/`에만 적용한다.
각 원본의 라이선스·저작권 표시는 해당 체크아웃에 보존되어 있다.

## 바로 실행

현재 머신에는 가상환경과 짧은 학습 검증용 체크포인트가 준비되어 있다.

```bash
cd /home/jeonges/acd-experiments
./run.sh doctor

# 논문 통신 방식: RL 4개 + 모의 LLM 1개, API 호출 없이 12스텝
./run.sh evaluate --config configs/smoke.yaml

# 공개 KEEP 가중치 평가: 2개 에피소드, 에피소드당 정확히 500번 환경 진행
./run.sh evaluate --config configs/keep.yaml --episodes 2 --horizon fixed

# 논문의 4개 팀 × 공격자 5종을 짧게 확인. PNG/CSV도 생성한다.
./run.sh matrix --config configs/smoke.yaml --episodes 1 --steps 12
```

`run.sh`는 작업 디렉터리를 자동으로 프로젝트 루트로 옮긴다.
직접 Python을 쓰려면 프로젝트 루트에서 `.venv/bin/python -m acd ...`로 실행한다.
WandB·Weave·모델 SDK는 로컬 실행 경로에서 import하지 않는다.

새로 설치하거나 원본을 다시 조립할 때:

```bash
./setup.sh
# 새 설치에서는 모의 실험용 RL 체크포인트도 생성한다.
./run.sh train --config configs/train_paper_smoke.yaml
.venv/bin/python -m pytest -q
```

`setup.sh`는 Python 3.10 환경과 인터넷이 필요하다. pip의 `--python` 옵션으로
전용 가상환경만 설치한다. `requirements.lock.txt`는 실제 검증한 전체 CPU 의존성,
`requirements.txt`는 핵심 의존성이다. CUDA 패키지는 필요하지 않다.
기존 결과나 체크포인트가 있는 출력 경로에는 덮어쓰지 않는다.

## 무엇이 원본과 같은가

- KEEP의 GCN actor/critic, 액션 분포, PPO clipped objective와 공개 가중치를 사용한다.
- 그래프 관측·호스트 및 서브넷 액션 매핑은 원본을 사용한다. 원본 그래프 텐서와
  원본 actor 확률이 동일한지 테스트로 비교한다.
- 논문 저장소의 `acd2025/base.yml` 프롬프트와 자연어 관측 포매터를 사용한다.
- 매 스텝 system 1개 + 현재 관측 user 1개로 요청을 구성한다. 과거 대화는 누적하지 않는다.
- JSON `action`/`reason`을 CybORG 행동으로 변환한다. 잘못된 응답은 Sleep과 오류 기록으로 처리한다.
- 혼합 팀은 `blue_agent_4`가 LLM 역할을 맡고 0~3은 RL이다.
- green은 EnterpriseGreenAgent, red는 원본 FiniteState/Aggressive/Stealthy/Impact/DegradeService 정책이다.

다음은 실행·학습 오류를 바로잡기 위한 의도적인 차이다. 논문의 숫자가 그대로
나온다고 주장하지 않으며, 모든 결과에도 `paper_scores_reproduced: false`를 기록한다.

| 원본에서 확인한 사항 | 이 환경의 처리 |
|---|---|
| 혼합 `CombinedWrapper`가 동일 환경을 LLM 래퍼와 RL 래퍼로 각각 step/reset | 그래프·LLM 행동을 합쳐 한 번만 진행/초기화. 매 턴 실제 `step_count` 증가량 검사 |
| 그래프 래퍼가 원본 관측의 `message`를 `pop()`으로 제거 | 복사한 관측만 파싱해 LLM에 메시지를 보존 |
| 평가 코드가 종료 검사 뒤에 보상을 더해 마지막 보상을 누락 | 종료 턴 보상을 포함하고 5명에게 중복 전달된 팀 보상을 평균해서 기록 |
| CC4는 `steps-1`에서 done을 반환 | `horizon: native`는 원본 유지. `fixed`는 정확히 설정한 횟수 진행. 기본 KEEP 설정은 native, 논문 설정은 fixed |
| 원본 문자열 분할 파서가 JSON 쉼표·따옴표에 취약 | JSON 파싱 및 자기 담당 호스트 검증. 오류/이유를 로컬 trace에 기록 |
| 통신 코드에서 스캔을 user compromise 코드 2로 분류하고, 알 수 없는 IP는 예외 | 논문 정의대로 scan=01, user=10, admin=11. 알 수 없는 IP/contractor는 다른 에이전트 비트를 세우지 않음 |
| 학습에서 에피소드 버퍼 수가 고정 5개라 N=25 중 일부만 사용; 다중 턴 보상·종료 경계 문제 | 모은 에피소드를 모두 사용, 행동 완료까지 보상을 귀속, 에피소드별 terminal을 봉인 |
| 원본 학습은 25개 프로세스와 높은 스레드 수 사용 | 같은 rollout 개수·PPO 설정으로 순차 수집. 설정한 CPU 스레드만 사용 |

## 공개 KEEP와 논문 통신 모드

두 저장소의 `gnn_ppo-0.pt`~`gnn_ppo-4.pt`는 SHA256까지 동일하다.
논문 설명의 새 통신 프로토콜로 재학습한 별도 가중치는 공개 파일에서 구분되지 않는다.
공개 가중치를 새 프로토콜의 학습 결과라고 취급하지 않는다.

| 설정 | 네트워크 입력 | 사용 가중치 |
|---|---|---|
| `communication: keep` | 원본 192차원, 원본 메시지 특징 3개 포함 | 제공된 KEEP 가중치 즉시 사용 |
| `communication: paper` | 197차원, 메시지 특징을 논문 8비트로 교체 | 이 모드로 별도 학습한 체크포인트 |

`paper` 모드는 0~4번 비트에 의심되는 원격 네트워크, 5~6번에 로컬 침해 수준,
7번에 실행 중 여부를 담는다. HQ가 관리하는 3개 서브넷에는 HQ의 동일 벡터를 넣는다.
관측에 없는 실제 red 상태를 보지 않는다. 8비트 형식 자체에는 수신 확인 비트가 없어
모두 0인 메시지와 수신 불가를 구분하는 추가 정보는 만들지 않는다.

192차원 공개 가중치를 197차원 실험에 지정하면 명확한 오류로 멈춘다.
`checkpoints/paper-smoke/`는 작은 네트워크를 아주 짧게 학습한 **동작 검증용**이다.
방어 성능·수렴·논문 점수 검증에 사용하면 안 된다.

## 팀과 공격자 선택

| `team` | 구성 |
|---|---|
| `none` | 5명 모두 Sleep |
| `all_rl` | 5명 모두 KEEP |
| `all_llm` | 5명 모두 모델 어댑터 |
| `mixed` | KEEP 4명 + 모델 어댑터 1명(HQ) |

`red`: `finite`, `aggressive`, `stealthy`, `impact`, `degrade`.
`--seed`, `--steps`, `--episodes`, `--threads`, `--team`, `--red`, `--weights`,
`--communication`, `--horizon`로 평가 설정을 덮어쓸 수 있다.
각 에피소드의 환경·NumPy·Python·Torch RNG를 `seed + episode`로 초기화한다.
원본 논문이 사용한 모든 실행 seed는 공개 자료만으로 확정할 수 없어 기본값 42를 명시했다.

## 모델 교체: API는 주석 상태

기본 `model.backend: mock`은 JSON Sleep만 반환한다. LLM 추론을 하지 않는다.
이 역할이 포함된 실험은 `offline_mock_pipeline_only`로 표시하고 그래프도 따로 그린다.

[acd/backends.py](acd/backends.py)의 `ApiBackend.generate()`에는 요청 예제가 전부
주석으로 들어 있다. API 키를 읽거나 요청을 보내는 실행 코드는 없다.
단순히 YAML의 `enabled`를 바꿔도 요청이 실행되지 않는다.

나중에 모델을 선택하면:

1. `configs/paper.yaml`을 복사하고 `model`, `base_url`, `api_key_env` 등을 지정한다.
2. 선택한 공급자의 SDK·프로토콜에 맞춰 `ApiBackend.generate()`를 구현한다.
3. `backend: api`, `enabled: true`로 바꾼다. 키는 코드나 YAML에 쓰지 않고 환경변수로 둔다.

SDK를 사용하지 않는 로컬 모델이나 다른 API도 다음 인터페이스로 바꿀 수 있다.

```python
# my_model.py: 선택한 모델의 호출부만 구현한다.
class MyBackend:
    def generate(self, messages, *, agent, episode, step) -> str:
        # messages는 [{"role": "system", ...}, {"role": "user", ...}]
        # 선택한 모델을 호출한 뒤 JSON 텍스트를 반환한다.
        raise NotImplementedError("선택한 모델 연결부 구현")

def create_backend(config):
    return MyBackend()
```

```yaml
model:
  backend: custom
  factory: my_model:create_backend
  model: your-model-name
  temperature: 1.0
```

과거 모델 응답을 다시 재생하려면 `backend: replay`, `replay_path: <trace.jsonl>`을
사용한다. `(episode, step, agent)`로 응답을 찾고 누락되면 실패한다.
`trace.jsonl`에는 system prompt, API 키를 반복 저장하지 않으며 system prompt는
`system_prompts.json`에 별도로 저장한다.

## 실제 학습과 논문 크기 평가

```bash
# 논문 통신 방식: 50,000 에피소드, 25 에피소드마다 업데이트,
# 500스텝, GCN hidden=256 / embedding=128, PPO 4 epochs.
# 장시간 학습 명령이다. 현재는 짧은 동작 검증까지만 수행했다.
./run.sh train --config configs/train_paper.yaml

# 논문 크기의 4개 팀 × 5개 red × 2에피소드 × 500스텝
# 모델 미연결 상태의 LLM 역할은 여전히 mock으로 명확히 표시된다.
./run.sh matrix --config configs/paper.yaml

# 원본 KEEP 표현을 새로 학습
./run.sh train --config configs/train_keep.yaml
```

학습은 업데이트마다 가중치 5개와 `history.json`을 저장한다.
완료 시 `training.json`에 가중치 변경·저장/재로딩 일치·학습량을 기록한다.
현재 학습기는 새 출력 디렉터리에서 시작하며 optimizer resume 기능은 없다.
공개 원본과 동일한 PPO 핵심 수식을 사용하되 위 표의 collector 수정 때문에
원본의 버그까지 포함한 학습 궤적과 동일하지는 않다.

## 로그와 분석

실험마다 `runs/<KST시각>-<이름>/`에 다음 파일을 만든다.

- `summary.json`: 설정, seed, 실제 스텝 수·phase, reward 평균·표본 표준편차,
  가중치 SHA256, 소스 커밋, 모의 모델 여부
- `trace.jsonl`: 매 턴·에이전트별 실제 행동, 요청 행동, 보상, 통신 벡터,
  자연어 관측, 모델 응답, 이유, 응답 오류, 시간
- `system_prompts.json`: LLM 역할의 고정 프롬프트

```bash
./run.sh plot runs/<실험또는매트릭스디렉터리>
# comparison.csv와 모의 응답/실제 평가를 구분한 reward PNG

# 논문처럼 blue_agent_4의 첫 에피소드에서 초기화 턴 제외
./run.sh export-reasons runs/<실험>/trace.jsonl --output reasons.jsonl

# reasons.jsonl 순서 그대로 만든 NxD 임베딩 배열을 제공한다.
# 논문의 text-embedding-3-large는 3072차원. 500스텝에서 모두 유효 응답이면 이유 499개.
./run.sh cluster-reasons reasons.jsonl --embeddings embeddings.npy --output reason-clusters
```

군집 분석은 PCA 3차원 → K-Means K=4, Elbow용 inertia와 silhouette를 저장한다.
임베딩 생성·군집 LLM 요약 API도 주석 상태다. 임베딩 모델을 선택한 뒤
`acd/analysis.py`의 예제를 바꾸거나 로컬 임베딩을 만들어 전달한다.
모의 Sleep 응답의 이유를 논문의 LLM reasoning 결과로 해석하면 안 된다.

## 검증

```bash
.venv/bin/python -m pytest -q --junitxml=verification/pytest.xml
.venv/bin/python -m pip check
```

테스트에서는 소켓 연결을 차단한 채 실행하고, 원본 그래프·정책 분포 동등성,
통신 비트, 행동 파싱, 종료 보상, 같은 seed 재현성, API 비활성 상태를 확인한다.
현재 머신에서 수행한 학습·평가 결과와 한계는
`verification/compatibility_report.json`에 기록한다.
