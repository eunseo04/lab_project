# Skill-Pro 계산·근거·공동 보상 귀속 감사

출판 [Skill-Pro](https://proceedings.mlr.press/v306/mi26d.html)의 Skill-MDP, Semantic Gradient/집계, 고정 LLM, RTG 기반 clipped PPO Gate, Best-of-N J>0, Algorithm 1 용량 관리를 B2에 유지한다. 공개 코드의 episode-level advantage 모드와 논문의 RTG 모드는 기존처럼 구분한다.

기존 실제 후보 24개의 저장 old/new token logprob·확률비·Advantage·clipped objective를 모두 재계산해 저장 점수와 일치시켰다. 같은 조건에서 수치가 맞는다는 사실은 새 에피소드 성능 향상을 보장하지 않는다.

| Loop | Parent | Candidate | J | Parent r=1 | 유효 결정 | 승인 |
|---|---|---|---:|---:|---:|---|
| 1 | RapidProcessResponse | RapidProcessResponse_v2 | -28.9824 | -30.8175 | 31 | False |
| 1 | RapidProcessResponse | RapidProcessResponse_v2 | -26.5296 | -30.8175 | 31 | False |
| 1 | EscalationDiagnosis | EscalationDiagnosis_v2 | -26.3323 | -22.5970 | 43 | False |
| 1 | EscalationDiagnosis | EscalationDiagnosis_v2 | -25.9767 | -22.5970 | 43 | False |
| 1 | PrivilegedRecovery | PrivilegedRecovery_v2 | -18.7649 | -23.4561 | 2 | False |
| 1 | PrivilegedRecovery | PrivilegedRecovery_v2 | -18.7649 | -23.4561 | 2 | False |
| 1 | SelectiveDeception | SelectiveDeception_v2 | 2.7111 | 7.9638 | 277 | True |
| 1 | SelectiveDeception | SelectiveDeception_v2 | 1.7382 | 7.9638 | 277 | False |
| 1 | NetworkContainment | NetworkContainment_v2 | -14.7156 | -8.1542 | 66 | False |
| 1 | NetworkContainment | NetworkContainment_v2 | -18.3413 | -8.1542 | 66 | False |
| 1 | TrafficRecovery | TrafficRecovery_v2 | 7.4050 | 12.1528 | 2 | False |
| 1 | TrafficRecovery | TrafficRecovery_v2 | 7.4051 | 12.1528 | 2 | True |
| 2 | RapidProcessResponse | RapidProcessResponse_v2 | -36.0247 | 18.6061 | 40 | False |
| 2 | RapidProcessResponse | RapidProcessResponse_v2 | 1.4592 | 18.6061 | 40 | True |
| 2 | EscalationDiagnosis | EscalationDiagnosis_v2 | 2.3129 | 18.8302 | 130 | False |
| 2 | EscalationDiagnosis | EscalationDiagnosis_v2 | 2.4870 | 18.8302 | 130 | True |
| 2 | PrivilegedRecovery | PrivilegedRecovery_v2 | 24.5878 | 40.2299 | 2 | True |
| 2 | PrivilegedRecovery | PrivilegedRecovery_v2 | 0.0000 | 40.2299 | 2 | False |
| 2 | SelectiveDeception | SelectiveDeception_v2_c0 | 8.4322 | 18.6027 | 30 | False |
| 2 | SelectiveDeception | SelectiveDeception_v2_c1 | 9.1444 | 18.6027 | 30 | True |
| 2 | NetworkContainment | NetworkContainment_v2 | 16.9748 | 27.7217 | 9 | False |
| 2 | NetworkContainment | NetworkContainment_v2 | 19.4607 | 27.7217 | 9 | True |
| 2 | SelectiveDeception_v2 | SelectiveDeception_v3 | 15.8000 | 20.7037 | 410 | True |
| 2 | SelectiveDeception_v2 | SelectiveDeception_v3 | 15.6412 | 20.7037 | 410 | False |

첫 갱신의 승인 후보가 포함된 Pool은 정적 Pool보다 FiniteState 평가 평균 −88.75였다. 두 번째 Pool은 같은 이력 ACD보다 −56.25였다. 원래 실험에 후보별 환경 검증은 없었으므로 모든 승인 후보를 나열하되 어느 하나가 손실을 일으켰다고 단정하지 않는다.

수치 오류와 일반화 실패를 구분한다. 기존 8개 실제 행동의 독립 재점수는 최대 차이 4.77×10⁻⁷이었다. 저장 Gate 계산에서 확률·Advantage 구현 오류를 확인하지 못했다. 반면 공동 RTG는 다른 KEEP 4명과 Red·Green의 영향까지 포함하고, 조용한 구간과 복구 구간의 조건부 분포가 다르다. 후보의 조건부 행동 확률 점수는 selector가 그 Skill을 선택할 빈도나 실제 환경 보상을 직접 검증하지 않는다.

확률은 실제 생성한 `action`과 `reason`을 포함한 JSON 전체 토큰의 합이다. 같은 primitive를 택하더라도 reason 표현의 확률 변화가 Gate에 반영될 수 있다. 원본 출력 확률 경로를 유지했으며 primitive-only 점수로 바꾸지 않았다. 이것이 실제 손실을 얼마나 설명하는지는 별도 분리 실험을 하지 않았으므로 가설로 남긴다. Gate에서 평가한 특정 Skill의 조건부 likelihood와 새 Pool에서 selector가 해당 Skill을 선택하는 과정도 구분해야 한다.

첫 배치에서 PrivilegedRecovery와 TrafficRecovery는 각각 유효 결정 2개였고 SelectiveDeception은 277개였다. v5는 부모별 유효 결정·궤적·phase·Red와 승인 여부를 기록하며, Proposed의 제한된 환경 검증은 가장 많은 근거를 가진 부모부터 수행한다. 미검증 희귀 Skill은 원래 상태를 유지한다. 원본 B2에는 이 순서·보류 규칙을 넣지 않는다.

Proposed의 Semantic Gradient는 각 변경 성분에 실제 step·Blue ID·원시 필드의 정확한 인용을 요구한다. 근거가 없으면 NEW fallback도 차단하고, 후보가 근거 없는 다른 성분을 바꾸거나 훈련 host/IP를 외우면 보류한다. 인용의 정확성은 의미 해석의 정답을 보장하지 않으므로 환경 검증을 추가한다.

최소 경험 수는 임의의 고정값으로 활성화하지 않는다. 0·4·8·16·32의 검증 민감도를 기록하며, 한 후보에서 이 값들의 효과가 같으면 최적 cutoff가 식별됐다고 주장하지 않는다. 예산상 검증되지 않은 다른 부모는 유지한다.

[후보별 수치·Skill 통계·phase/Red별 사용·공동 보상 노출](../research_outputs/rollout-v5/skillpro-audit.json). 이 노출 합계는 개별 Skill의 인과적 기여도가 아니다.


추가 원본 소스 감사에서는 BlueRewardMachine, scenario generator, GreenLocalWork/AccessService, Restore/DegradeServices, SimulationController와 ControlTraffic가 고정한 CAGE4 원본과 모두 byte 단위로 같았다. [환경 감사](../research_outputs/rollout-v5/environment_source_audit.json). 공식 출판 PDF도 확보해 RTG·clipped gate·Algorithm 1을 재확인했다. [출판 원문 hash](../research_outputs/rollout-v5/references/source.json).

새 실제 Qwen 행동 8건의 생성 확률과 독립 uncached teacher forcing 확률의 최대 차이는 4.76837158203125e−7이었다. [새 확률 대조](../research_outputs/rollout-v5/generation_scoring_parity.json). 원본 24개 Gate 재계산과 이 새 입력 대조를 별개로 보존한다.


새 B4의 실제 훈련 두 궤적에서 부모의 r=1 surrogate와 가능한 Gate의 수학적 상한도 계산했다. 각 Advantage A에 대해 clipped 항의 상한은 A≥0일 때 (1+ε)A, A<0일 때 (1−ε)A다. 이 값을 원래 Gate처럼 궤적별 평균한 결과 RiskBasedInvestigation의 상한은 −1.7160, SelectiveDeception은 −9.5705였다. 이 배치에서는 어떤 후보의 행동 확률을 사용해도 두 Skill의 J가 양수가 될 수 없다. 반면 EscalationDiagnosis의 부모 r=1 값은 +4.9479로, 부모 점수가 0으로 정규화돼 있지도 않다. 이는 공동 RTG의 조건부 표본·baseline 문제를 보여주는 진단이며 원래 Gate를 바꾸거나 후보를 생략하는 데 적용하지 않았다. [전체 값과 유도](../research_outputs/rollout-v5/gate_feasibility_diagnostics.json).

135개 회귀 테스트 뒤에 용량 초과 시 pruning을 적용한 후보가 거부되면 live 부모가 유지되는지, 승인되면 실제 검증한 pruning snapshot과 동일한지 확인하는 별도 두 테스트도 통과했다. [용량·rollback 추가 검사](../revisions/rollout-v5/research_outputs/v5-capacity-branch-tests.xml). 현재 한 loop의 Pool은 최대 14개로 capacity24에 닿지 않으므로 실측 성능 차이를 pruning의 효과로 설명하지 않는다.


활성 Skill 프롬프트를 포함한 실제 행동도 별도로 검증했다. 새 B4 훈련의 6개 Skill family에서 유효 행동 8건을 골라 실제 생성 토큰을 재구성한 Skill 프롬프트로 독립 uncached teacher forcing했다. 최대 로그확률 차이는 9.5367431640625e−7이었다. 앞의 초기 무활성 행동 8건 점검과 구분하며, 새 성능 에피소드를 추가하지 않았다. [활성 Skill 확률 대조](../research_outputs/rollout-v5/active_skill_scoring_parity.json).

저장한 훈련 experience에서 SG 청크를 모델 호출 없이 다시 구성해 hash와 인용을 대조했다. B4의 실제 48개 청크와 124개 유효 인용은 모두 원본 step·agent·필드에 연결됐고, 전체 성분 출처 검증을 통과한 청크는 27개였다. 다만 124개 중 84개는 요청 행동을 인용했고 실제 `behavior_outcome` 인용은 1개였다. 첫 청크의 SG는 ‘성공한 Remove 뒤 종료하지 않았다’고 썼지만 그 청크의 완료 행동은 Analyse 두 번이며 cmd.sh를 관측했다. 정확한 출처 ID만으로 이 진단의 의미가 입증되지는 않는다. 이 한계는 고친 것으로 주장하지 않고 실제 rollout의 독립 승인 조건과 함께 보고한다. [청크 재구성·출처·원시 SG 및 완료 행동](../research_outputs/rollout-v5/semantic_catalog_audit.json).


보상 원시 시계열에서 RTG와 EMA baseline을 독립 재구성하고 Gate transition의 step·seed·Blue·Skill 정렬도 확인했다. 기존 v2 두 loop의 Advantage 2,084건과 새 B4 첫 loop의 400건이 일치했고, 저장 trace RTG의 최대 차이는 0이었다. 원본 공동 보상을 한 번만 센 수식은 맞지만 이 값은 개인 Skill의 인과적 기여도가 아니다. 후속 완료 arm도 같은 감사에 추가한다. [원시 보상부터의 RTG/Advantage 감사](../research_outputs/rollout-v5/rtg_alignment_audit.json).
