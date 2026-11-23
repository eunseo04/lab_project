# 실험 프로토콜

## 고정 조건

기본은 Hybrid: Blue 0–3 공개 KEEP GNN-PPO, Blue 4 Qwen3-8B다.
All LLM에서는 5명 모두 같은 E arm과 독립 Skill 상태를 사용한다.
No Blue/All RL도 실행할 수 있다. 구성별 결과를 다른 그룹으로 저장한다.
Green은 EnterpriseGreenAgent, Red는 ACD 원본 FiniteState/AggressiveFSM/
StealthyFSM/ImpactFSM/DegradeServiceFSM이다. 네트워크, 작전 단계, 환경 보상은
변경하지 않는다. 각 정책은 별도 seed 대응 결과를 갖는다.

| 조건 | 기본값 |
|---|---|
| 최종 평가 | 100 episodes, nominal 500 steps |
| 훈련 | 6 episodes, 3 iterations |
| 훈련 시드 | 100–105 |
| iteration 검증 시드 | 5000, 각 iteration 동일 |
| 최종 평가 시드 | 10000–10099 |
| 정책 | Qwen/Qwen3-8B, BF16, frozen |
| revision | b968826d9c46dd6066d109eabc6255188de91218 |
| 추론 | Transformers 4.51.3 / torch 2.6.0+cu124, SDPA |
| sampling | temperature=1, top_p=1, top_k=0, thinking=False |
| 정책 출력 제한 | 192 tokens |
| 메타 판단 / 진화 출력 제한 | 48 / 768 tokens |
| selector | 원본 고정 LLM selector, epsilon=.1, 최대 age=8 |
| Pool | 최대 10, 초기 7 Skills |
| Gate | official_code, epsilon=.2, 기본 best-of-3 |

실제 다운로드 모델 파일의 SHA256과 encoder revision은
`research_models.lock.json`, 소스/KEEP 가중치는 `research_sources.lock.json`에 있다.
서버는 127.0.0.1에만 바인딩하며, localhost 요청만 사용한다.
서버가 없거나 identity가 다르면 종료하며 다른 모델로 바꾸지 않는다.
후속 다른 모델 실험은 model ID/revision을 명시하고 새 출력 경로를 사용한다.
그 모델로 만든 Pool만 불러올 수 있으며 비교 그룹도 model ID/revision별로 나눈다.

## 보상, 종료, 초기화

공식 평가 코드와 같은 `mean(blue rewards)`를 턴마다 한 번 더한다. Blue에게
반복 전달된 보상이 서로 다르면 중단한다. 원본은 terminal 검사 후 점수를
더하므로 terminal reward는 제외된다. native 종료 조건상 nominal 500은
실제 공동 전이 499개, 점수 전이 498개다. 원본 terminal reward도 trace에 남는다.
`scoring: terminal_inclusive`는 별도 공통 변경 설정이며 official과 섞지 않는다.

에피소드 총 보상의 평균과 `statistics.stdev` 표본 표준편차(ddof=1)를 사용한다.
N=1의 표준편차는 N/A다. 시간은 에피소드별 wall time의 합이다.
일반 실행기의 시간 구간은 seed/환경 생성/KEEP 로딩부터 행동·메타 판단·trace
저장 종료까지이며, 모델 서버 가중치 로딩과 평가 전 Pool 로딩은 제외된다.
오프라인 진화 시간은 훈련 history의 `seconds`에 따로 저장한다.

`research.sh official`은 실제 원본 `evaluation.run_evaluation`을 호출한다.
환경을 한 번 생성하고 episode reset 시 다시 seed를 주지 않는 원본 stream을
유지한다. `evaluate/compare`는 E0–E3 seed 대응을 위해 각 episode를 명시적으로
초기화한다. 같은 보상 누적 규칙을 사용하지만 초기화 방식은 다르다. 원논문의
정확한 seed와 공개되지 않은 retrained KEEP 가중치가 없어 무수정 수치 재현을
주장하지 않는다. 원본 evaluator의 시간은 환경/정책 사전 구성 이후 시작하므로
논문 시간과 새 실행기 시간의 범위도 완전히 같다고 표시하지 않는다.

## 공통 변경

`paper-compatible`은 원본 ACD 프롬프트의 Sleep 표기 제한을 유지하고 busy 때도
정책을 호출한다. `controlled-comparison`은 E0–E3 모두 Sleep 허용 문구를 추가하고
진행 중인 행동에는 공통으로 정책 호출을 생략한다. Monitor는 환경이 자동 실행한다.

공개 KEEP는 192차원 legacy 통신 특징용이다. `paper_projected_keep`는 wire상
ACD 8비트를 보내고 KEEP 입력에서 severity를 compromised/scanned 특징으로
공통 변환한다. 수신 여부는 any(bit)로 정한다. all-zero 메시지의 수신/미수신은
구분할 수 없다. 이 호환 변환은 E0–E3 모두 동일하다. 원본에서 제공하지 않은
8비트용 재훈련 가중치를 만들었다고 표시하지 않는다. `keep_native`는 별도 그룹이다.

## 이번 실제 실행 예산과 기본 최종 평가의 구분

`executed-E1/E2/E3.yaml`과 E0 실행 manifest는 finite / Hybrid / nominal 500 /
평가 2 episodes를 사용한다. E2/E3는 훈련 2 episodes / 1 iteration,
검증 1 episode, threshold=2, best-of-2, 최대 진화 1개로 예산을 제한했다.
이는 기본 6-episode/3-iteration 학습이나 100-episode 최종 평가 완료를 뜻하지 않는다.
짧은 12-step 실제 LLM 통합 실험도 별도 디렉터리에 저장하며 500-step 표에 넣지 않는다.

기본 threshold=6은 6개 경험이 쌓인 뒤 진화하므로, 이후 학습까지 계속하려면
`closed-loop.yaml`의 12 episodes / 6 iterations를 사용한다. 짧은
`executed-closed-loop.yaml`은 threshold=1 / 4 episodes / 2 iterations로
연결만 검증하며, 보상 변동이 없으면 후보를 승인하지 않는 것이 정상이다.

## 대응 비교와 결과 생성

결과 그룹은 Blue 구성, prompt profile, 보상 profile, 통신, 길이, 모델 revision,
temperature/출력 한도, Gate/maintenance 모드, busy 처리, selector 조건으로 나눈다.
Red별 E0–E3의 공통 완료 seed만 대응 비교에 쓴다. 부족한 arm은 N/A다.
`completed_cases.csv`에는 개별 완료 사례도 보이되 대응 비교와 구분한다.

Table 2는 실행된 arm들의 공통 seed로 표시하며 미실행 arm은 N/A다.
전체 4개 arm이 모두 대응되는 seed 집합은 별도로 기록한다. `pairwise_available.csv`는
각 두 arm의 실제 공통 seed에 대한 보상 차이를 제공한다. Table 1/3과 Figure 2의
전체 arm 비교는 네 arm의 공통 데이터만 사용한다.

Table 1 전체 집계는 5종 Red 모두 대응 결과가 있을 때만 채운다. 별도
`paired_available_policies.csv`와 Figure 2는 완료된 정책 범위를 제목에 명시한다.
전체 집계는 대응 episode return을 합친 표본 통계다. 원논문의 Red 간 집계 및
표준편차 규약의 불확실성을 별도로 표시하고 paper_reference와 동일시하지 않는다.

Figure 4/Table 3은 LLM Blue의 실제 실행 행동을 세고, 자동 Monitor를 별도 표에
둔다. KEEP 4명의 행동을 LLM 행동처럼 넣지 않는다. 원본 reason 분석과 달리
로컬 MiniLM 384차원 → PCA(3) → KMeans(4)를 선택적으로 사용하며, 변경된
embedding과 분석 episode를 기록한다. 군집은 방어 성공률이 아니다.

## 배치·중단·재개

출력 디렉터리는 잠금과 config 검사를 사용한다. `--limit N`은 이번 호출에서
새로 수행할 episode 수이며 계획 수를 바꾸지 않는다. `--resume`은 완료 episode를
건너뛰고 pending 경험까지 이어간다. 미완료 trace와 이전 실패 기록은
`interrupted/`에 보존한다. 최종 평가 데이터는 훈련 버퍼에 들어가지 않는다.
설정이나 seed를 바꾸면 새 출력 디렉터리를 사용한다.

`train --pool checkpoint.json`은 새 계획의 후속 훈련을 시작한다. 이전 Pool,
경험 버퍼와 baseline을 이어받고 누적 훈련 시드/episode 예산을 보존한다.
이미 사용한 평가 seed가 유입된 Pool은 거절한다. 한 계획의 중단 복구에는
`--pool` 변경 대신 `--resume`을 사용한다.

진화/검증 도중 중단되어도 완료 episode의 JSON/전체 trace는 재사용한다.
최신 결과에는 Python RNG와 메타 모델 counter를 저장해 복구한다. 이 상태를
저장하지 않은 초기 결과를 복구하면 Evolution RNG를 명시적으로 reseed하고
`interrupted/recovered-episode-*.json`에 기록한다. 원래 행동 표본을 재생성하거나
완료 reward를 덮어쓰지 않는다. 미완료 episode는 seed를 유지해 처음부터 재실행한다.

초기 episode의 `invalid_json` 보조 카운터는 JSON/행동/대상 검증 오류를 합친
필드다. JSON 문법 오류율로 해석하지 않는다. 정확한 분류에는 원본 `response`와
`parse_error`를 사용한다. 기본 성능 집계에는 이 카운터를 사용하지 않는다.

이번 500-step E0–E3는 같은 RTX A5000의 GPU별 독립 모델 서버에서 병렬로
실행했다(8018–8021). CPU 환경과 디스크를 공유했고 baseline control도 병행했다.
시간은 이 조건에서 측정한 wall time이며 엄격한 단독 실행 속도 비교는 아니다.
기본 CLI 일괄 비교는 하나의 endpoint로 순차 실행하므로 단독 속도 재측정에 쓸 수 있다.

추가 전이 검증은 `executed-transfer-E0/E3.yaml`로 실행했다. FiniteState에서
학습한 E3 Pool을 고정하고 AggressiveFSM의 같은 평가 시드 10000/10001에서
E0와 비교한다. AggressiveFSM 평가를 Skill 학습에 사용하지 않는다. E1/E2의
AggressiveFSM 대응 결과는 미실행이므로 추가 기여의 전 정책 일반화를 주장하지 않는다.

Manifest는 소스 revision과 config를 기록한다. 최신 호출의 `invocations/`에는
실행 시점의 구현 파일 hash도 남긴다. 초기 장기 실행의 manifest에 구현 hash가
없을 수 있으며, 재개 시 추가 기록을 남긴다. E2/E3 Pool은 학습 checkpoint의
경로/hash/학습 시드/알고리즘과 함께 frozen snapshot으로 보존한다.
