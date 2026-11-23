# 구현 방법

실행 코드는 `research/`, 보존된 기존 ACD 코드는 `acd/`, 고정된 원본은
`third_party/`에 있다. 본 구현은 고정된 Qwen3-8B의 외부 절차형 메모리를
수정한다. 신경망 PPO 학습, LoRA, GRPO 또는 모델 파라미터 업데이트는 하지 않는다.

## Skill-MDP와 관찰 범위

`research/skills.py:SecuritySkillPool`은 원본 `Skill` 자료형에 CAGE4의 초기
7개 Skill을 넣는다. 각 Skill은 Initiation, Policy, Termination과 버전,
부모 ID, 통계, 행동 목록을 가진다. 행동 목록은 메타데이터이며 새로운 환경
행동을 만들지 않는다. E0는 Skill이 없고, E1은 초기 Pool을 고정한다.

`SkillController`는 Blue별 활성 Skill, 시작 시점, 이력, 종료 이벤트를 독립
관리한다. 고정된 원본 LLM selector와 epsilon 탐색을 사용한다. 현재 Skill을
여러 턴 유지하고, LLM 종료 판단 또는 최대 나이 8에서 교체한다. 잘못된 종료
응답은 오류를 기록하며 현재 Skill을 유지한다. 최대 나이는 계속 적용된다.

입력은 해당 Blue의 공식 관찰, 그 Blue의 이전 행동, 허용 대상 메타데이터다.
실제 침해 상태나 Red 세션을 조회하여 Skill을 고르지 않는다. E3만
`CAGEVerifier`의 관찰 기반 조건 필터를 추가한다. 상태 키 검사도 있지만,
관찰 경계의 근거는 `raw_observations()[agent]`와 Blue별 공식 action metadata를
입력으로 삼는 호출 경로다. 키 이름 검사만으로 정보 유출 방지를 주장하지 않는다.

E3는 단계와 메시지 형식, 관찰된 위협, 진행 중인 행동, 직전 실패, 같은 증거에서의
반복, 실행 가능한 행동을 점검한다. Restore에는 관찰된 관리자 침해 경고,
AllowTraffic에는 관찰된 이전 Block 이력을 요구한다. 행동의 대상 검증은
E0–E3 모두 공통 ACD 파서에서 수행한다. 부분 관찰로 확정할 수 없는 성공을
실제 침해 제거로 간주하지 않는다.

## 행동과 경험

`research/adapter.py:DefensePolicy`는 기존 ACD의 프롬프트·관찰 포매터·행동
파서를 사용한다. 현재 관찰과 활성 Skill을 Qwen에 전달하고, 응답 JSON을
CAGE4 기본 행동으로 변환한다. 자동 Monitor와 LLM이 선택한 행동을 구분한다.
잘못된 JSON/대상은 기존 파서의 Sleep과 오류 기록으로 처리한다. 이는 실제
CAGE4 행동 결과로 저장되며, 유효한 PPO 행동 증거로 사용하지 않는다.

`run_episode`는 모든 Blue의 행동을 모아 환경을 한 번만 진행한다. JSONL에는
관찰 전후, seed/step/Blue, 단계, 메시지, Skill ID/버전, 요청과 실제 행동,
상태, 원본 공동 보상, 종료, 정책 호출 여부, 토큰 ID를 기록한다. 보상 집계는
한 공동 턴에 한 번이다. 여러 Blue의 궤적에 같은 공동 보상이 보존되어도
평가 점수에 다섯 번 더하지 않는다.

경험은 Blue와 Skill별로 나누고 Skill 활성 구간만 귀속한다. 전체 에피소드의
미래 보상으로 RTG를 먼저 계산한 뒤 Skill 구간을 추출한다. FIFO와 Golden
Experience Pool에 실제 궤적을 넣는다. 원본의 우선순위/성공·실패 표본 선택이
진화 입력을 정하며, Golden Pool이 별도 새로운 선택 정책을 만들지는 않는다.

## Semantic Gradient와 후보

`research/evolution.py:Evolution`은 원본 `SkillEvolution`을 상속한다.
각 시간 순서 chunk의 행동 전후·결과·보상을 원본 Skill Doctor에 전달하고,
Initiation/Policy/Termination 개선 신호를 얻는다. 모든 chunk가 처리된다.
원본의 배치 aggregation과 refine/new Skill 생성 함수를 재사용한다.
긴 입력을 최종 관찰 하나로 바꾸거나 조용히 자르지 않는다. 모델 또는 문맥
한도 오류는 해당 진화의 실패로 남기고 기존 Pool을 보존한다.

후보마다 구조를 검사하고 Best-of-N에서 Gate가 가장 높은 후보를 선택한다.
기본 승인 조건은 `score > 0`이며 설정에서 margin을 지정할 수 있다. 승인과
실제 평가 성능 개선은 서로 다른 사건이다. Gate 승인만으로 Reward 개선을
주장하지 않는다. 부모 Skill은 유지하며 원본 방식으로 새 버전을 Pool에 추가한다.

## PPO Gate의 두 계산 모드

원본 공개 코드와 논문이 달라 모드를 분리했다.

| 설정 | Advantage와 평균 | Pool 관리 |
|---|---|---|
| `controlled.yaml` / `official_code` | 경험별 `(episode_return − EMA episode baseline) / 유효 행동 수`, 전체 행동 평균 | 원본 LCB·성숙도·semantic dedup |
| `paper-rtg.yaml` / `paper_rtg` | 전체 공동 궤적의 RTG − running baseline, 궤적별 surrogate 평균의 평균 | 활성 구간의 단계 보상 gain / 호출 횟수, nonpositive score pruning |

두 모드 모두 `ratio = exp(clamp(new_logprob − old_logprob, −10, 10))`,
`min(ratio*A, clip(ratio, 1−epsilon, 1+epsilon)*A)`를 사용한다. 로그 비율의
[-10,10] 수치 안정화는 공개 코드에서 유지한 선택이며 논문 수식에 명시된
추가 클리핑은 아니다. 기본 epsilon=.2, gamma=.99, EMA alpha=.1이다.

논문의 running baseline 세부 추정기는 충분히 지정되지 않아 `paper_rtg`에서는
Blue/Skill로 중복하지 않은 전체 공동 시간축의 평균 RTG를 EMA로 갱신한다.
점수용 단계 보상 baseline은 별도로 관리한다. 새 후보의 아직 관찰되지 않은
점수는 0으로 기록하고 임시 보존한다. 관찰 후 nonpositive score면 제거한다.
이 cold-start 규칙은 본 구현의 명시적 보완이며 논문의 무수정 구현이 아니다.

`official_code`의 통계는 원본 `Skill.update_stats`를 사용한다. 호출 횟수 분모는
각 Blue의 각 에피소드 안에서 계산한다. 서로 다른 Blue/에피소드 호출 수를
하나의 궤적처럼 합치지 않는다. 완전한 원본 실행 순서의 복제라고 표시하지 않으며,
ALFWorld/TextArena의 episode 예산·반복 스케줄은 CAGE4 실험 설정으로 대체한다. 고정한 원본 `run.py:44`는 `warm_up_flag=False`이므로, 코드에 있는 5회 warmup 분기는 기본 실행에서 활성화되지 않는다.

## 정확한 로그확률

`research/model_server.py`는 동일한 BF16 Qwen 가중치·토크나이저·chat template로
생성과 점수를 계산한다. thinking=False, full-support temperature sampling,
top_p=1, top_k=0이다. 실제 생성된 completion token ID와 EOS를 보관한다.
점수 계산은 생성과 동일한 cache 경로에서 원래 logits의 확률을 기록한 뒤
과거 토큰을 강제한다. 강제로 만든 확률 1을 점수로 사용하지 않는다.

점수는 이유를 포함한 응답 시퀀스의 조건부 로그확률 합이다. 모든 가능한 이유를
주변화한 환경 행동 확률이라고 해석하지 않는다. 과거/후보 Skill만 바꿔 동일한
관찰·완성 토큰을 재계산한다. 근사 top-logprob, 누락 시 0, 다른 모델 대체,
문맥 truncation은 허용하지 않는다. 오류면 Pool 변경을 되돌린다.

`scripts/validate_qwen_scoring.py`가 실제 생성 점수와 teacher-forcing 점수를
비교한다. 검증 파일은 `research_outputs/scoring-validation.json`이다.
Gate 로그에는 점수, 승인/거부, 이유, 후보 내용, Semantic Gradient가 남는다.
최신 실행에서는 단계별 old/new logprob, ratio, advantage, surrogate도 남긴다.

## 학습·평가 분리와 ablation

훈련에서만 통계·후보·Pool을 갱신한다. 각 iteration 뒤 별도 검증 시드로
현재 Pool의 Reward를 측정하되 검증 경험을 학습 버퍼에 넣지 않는다.
최종 평가에서는 Pool 스냅샷을 고정하고 내용 hash와 동일성을 검사한다.
업데이트가 끝나지 않은 pending batch는 최종 평가 입력으로 거절한다.

No Semantic Gradient는 원본 요약 기반 생성 경로, No PPO Gate는 로그확률 호출
자체를 생략하는 경로, No Score Maintenance는 FIFO 관리를 사용한다.
세 기능의 off-switch는 각각 자동 테스트한다.

원본 비교 근거: [Skill-Pro 공개 코드](https://github.com/Miracle1207/Skill-Pro),
[논문 v3](https://arxiv.org/abs/2602.01869v3). 파일별 차이는
[reproduction_audit.md](reproduction_audit.md)에 기록했다.
