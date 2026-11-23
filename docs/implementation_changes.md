# 원본 대비 구현 변경

기존 ACD의 20개 파일은 `preexisting_files_sha256.json`과 대조한다.
원본 4개 저장소의 tracked source는 수정하지 않는다. Python이 만든
`__pycache__`는 source 수정으로 세지 않는다. 최상위 README와 .gitignore만
연구 확장을 설명하도록 추가 수정했다. 기존 `acd-experiments`는 보존한다.

| 분류 | 위치 | 재사용/변경 이유 |
|---|---|---|
| 기존 ACD 재사용 | `acd/environment.py`, `acd/llm_agent.py`, `acd/runner.py` | 기존 joint-step 래퍼, 원본 포매터/프롬프트 연결, JSON 파서, KEEP 로더 |
| 원본 ACD 재사용 | `third_party/llms-are-acd/CybORG/Agents` | Red 5종, GNN/PPO, graph/action mapping, 관찰 포매터, 프롬프트 |
| 원본 평가기 | `research/official.py` | 원본 evaluator 호출과 동일 로그의 새 집계기 산술 대조 |
| 새로운 CAGE4 연결 | `research/adapter.py` | Blue별 관찰·대상, 1회 공동 step, native horizon, 공통 KEEP 통신 투영 |
| 원본 Skill-Pro 재사용 | `research/upstream.py` | Skill/Experience, FIFO/Golden, SkillPool, SkillEvolution import; 원본 encoder 의존성을 동일 MiniLM 원격 함수로 연결 |
| Skill-MDP | `research/skills.py` | CAGE4 초기 Skills와 Blue별 controller; 원본 고정 selector 사용 |
| Proposed 추가 | `research/skills.py:CAGEVerifier` | 부분 관찰에서 확인할 수 있는 단계·메시지·행동 상태 필터, E3에서만 사용 |
| 진화 이식 | `research/evolution.py` | 실제 단계별 이력, 원본 Semantic Gradient/aggregation/후보 생성, fail-closed Best-of-N, Pool rollback |
| 확률 계산 보완 | `research/model_server.py`, `research/client.py` | 실제 생성 token ID의 정확한 합 logprob, 동일 policy 모델 강제, truncation/모델 대체 금지 |
| 수식 분리 | `research/gate.py`, `research/paper_algorithm.py` | 코드의 상수 advantage/LCB와 논문의 RTG/online score 차이를 별도 모드로 명시 |
| 실험기 | `research/experiment.py` | 에피소드·시드 분리, pool freeze, 재개, validation, 모델·Skill 버전 기록 |
| 평가/그림 | `research/metrics.py`, `research/reporting.py` | 원본 공동 보상·sample std, 실제 데이터만 비교, 누락 N/A, 4개 표·7개 그림 경로 |

공개 Skill-Pro와 다른 주요 연결은 다음과 같다.

1. 최종 상태 요약 대신 전체 시간 순서 이력을 Semantic Gradient에 전달한다.
2. Blue와 Skill별 유효 행동만 Gate 증거로 쓴다. busy로 정책이 실행되지 않은
   행동, 파싱 실패, 토큰이 없는 행동은 확률 검증에 넣지 않는다.
3. 원본의 HF 평균 logprob 및 vLLM 누락 확률 근사를 사용하지 않는다.
4. no-gate ablation은 확률을 전혀 요청하지 않는다. source의 best-by-PPO와 다르다.
5. FIFO/Golden 경험은 실제 저장한다. 원본 코드에서는 생성만 하고 진화에 연결하지 않은 부분이 있다.
6. 평가 Pool의 통계까지 고정한다. score 업데이트는 훈련에서만 한다.
7. 원본 5-iteration warmup을 그대로 복사하지 않고 실험 설정의 훈련 예산을 따른다.
8. 공동 보상의 Skill별 인과 기여 추정기를 새로 만들지 않는다. Skill 점수는
   관찰된 공동 보상에 대한 원본형 bookkeeping이지 개별 책임의 증명이 아니다.

실제 장기 훈련 점검 중 score 통계 분모를 배치 전체가 아닌 Blue별 에피소드의
호출 횟수로 바로잡았다. 완료 episode와 pending 경험을 보존하고, Evolution
직전에 학습기를 재개하여 수정이 반영되도록 했다. 불완전한 재실행 trace는
`interrupted/`에 남는다. 구현 hash 기록은 재개된 invocation부터 제공된다.

`requirements-model.lock.txt`의 CUDA 환경과 기존 CPU 환경은 분리했다.
모델 파일은 revision과 SHA256으로 고정한다. 외부 유료 LLM API 연동은 추가하지
않았다. 이번 요청에 따라 새 연구 경로는 실제 로컬 Qwen을 연결하며, 기존
`acd/backends.py`의 주석 처리된 API 예시는 그대로 보존한다.
