# Rollout 검증을 추가한 B0–B4 실제 비교

상태: running. 실제 최종 평가 0회. 각 500-step, KEEP 4 + 고정 Qwen3-8B 1. 새 보상·모델 가중치 학습 없음.

별도 실제 훈련 6회, 후보 검증 16회가 완료됐다. 훈련·검증·알려진 시드 회귀·40-step 점검은 최종 평가 N에 더하지 않는다.

검증 Pool 메타데이터의 episodes_completed 누락으로 최초 worker들이 평가 시작 전에 실패했다. 원본 소스·실패·후보·훈련 결과를 보존하고 metadata-fix revision에서 필수 count 두 필드만 추가했다. 같은 후보 snapshot hash, 검증 시드, 기준과 최초 8시간 종료 시각을 유지한 재검증이다. 모델·정책·보상은 바꾸지 않았다.

최종 테스트 전 B1의 지정 Pool 로드 누락도 수정했다. B1은 기본 6개가 아닌 등록한 공통 수동 7개 Skill을 실제로 읽는다. 별도 static-pool-fix revision의 변경은 이 E1 로드 조건 한 줄이며 E0/E2/E3 동작·진행 중 검증은 같다. 139개 회귀 검사를 통과했고 수정 시점에 최종 테스트는 없었다.

## Table 1. FiniteState

| 지표 | B0 ACD+History | B1 Static | B2 Original Skill-Pro | B3 Rollout | B4 Proposed |
|---|---:|---:|---:|---:|---:|
| 실제 N | 0 | 0 | 0 | 0 | 0 |
| Mean Joint Reward | 미실행/자료 없음 | 미실행/자료 없음 | 미실행/자료 없음 | 미실행/자료 없음 | 미실행/자료 없음 |
| Reward sample SD | 미실행/자료 없음 | 미실행/자료 없음 | 미실행/자료 없음 | 미실행/자료 없음 | 미실행/자료 없음 |
| Execution time 합계(s) | 미실행/자료 없음 | 미실행/자료 없음 | 미실행/자료 없음 | 미실행/자료 없음 | 미실행/자료 없음 |

실행 시간은 병렬 서버 대기까지 포함한 실제 평가 시간이다. 모델 자체 추론 속도의 인과 비교가 아니다.

| 비교: Bx−B0 | 대응 N | 평균 차이 | 개선 비율 | 중앙값 | 대응 차이 95% CI |
|---|---:|---:|---:|---:|---|
| B1 | 0 | 미실행/자료 없음 | 미실행/자료 없음 | 미실행/자료 없음 | None |
| B2 | 0 | 미실행/자료 없음 | 미실행/자료 없음 | 미실행/자료 없음 | None |
| B3 | 0 | 미실행/자료 없음 | 미실행/자료 없음 | 미실행/자료 없음 | None |
| B4 | 0 | 미실행/자료 없음 | 미실행/자료 없음 | 미실행/자료 없음 | None |

CI는 고정 난수의 대응 시드 percentile bootstrap 10,000회다. N<2에서는 정의하지 않으며 적은 시드의 CI를 확정적 통계 우위로 해석하지 않는다.

## Table 2. Red Strategy

| Red | B0 N / 평균 | B2 N / 평균 | B4 N / 평균 | B4−B0 대응 N / 차이 |
|---|---|---|---|---|
| finite | 0 / 미실행/자료 없음 | 0 / 미실행/자료 없음 | 0 / 미실행/자료 없음 | 0 / 미실행/자료 없음 |
| aggressive | 0 / 미실행/자료 없음 | 0 / 미실행/자료 없음 | 0 / 미실행/자료 없음 | 0 / 미실행/자료 없음 |
| stealthy | 0 / 미실행/자료 없음 | 0 / 미실행/자료 없음 | 0 / 미실행/자료 없음 | 0 / 미실행/자료 없음 |
| impact | 0 / 미실행/자료 없음 | 0 / 미실행/자료 없음 | 0 / 미실행/자료 없음 | 0 / 미실행/자료 없음 |
| degrade | 0 / 미실행/자료 없음 | 0 / 미실행/자료 없음 | 0 / 미실행/자료 없음 | 0 / 미실행/자료 없음 |

## Table 3. FiniteState Reward Breakdown

| 원본 항목 | B0 | B4 | 차이 |
|---|---:|---:|---:|
| action_cost | 미실행/자료 없음 | 미실행/자료 없음 | 미실행/자료 없음 |
| RIA | 미실행/자료 없음 | 미실행/자료 없음 | 미실행/자료 없음 |
| LWF | 미실행/자료 없음 | 미실행/자료 없음 | 미실행/자료 없음 |
| ASF | 미실행/자료 없음 | 미실행/자료 없음 | 미실행/자료 없음 |
| mean | 미실행/자료 없음 | 미실행/자료 없음 | 미실행/자료 없음 |

구성요소는 중복 없는 원본 BlueRewardMachine 사건과 action_cost다. N이 다르면 이 표의 단순 평균 차이는 대응 효과가 아니며 시드별 CSV를 확인한다.

## Table 4. Skill Evolution

| Version | PPO Gate | 실제 검증 | 자동 채택 수 | Test Reward |
|---|---|---|---:|---|
| Initial | 미적용 / 수동 7개 | 이번 후보 검증의 previous 역할로 별도 기록 | 0 | 중간 Pool 테스트로 선택하지 않음 |
| B2 Loop 1 | 12 후보 점수 계산 | 완료된 대응 검증 없음 | 4 | 최종 선택 결과만 아래 기록 |
| B3 Loop 1 | 0 후보 점수 계산 | 완료된 대응 검증 없음 | 0 | 최종 선택 결과만 아래 기록 |
| B4 Loop 1 | 2 후보 점수 계산 | 완료된 대응 검증 없음 | 0 | 최종 선택 결과만 아래 기록 |
| Loop 2 | 이번 예산에서 미실행 | 미실행 | — | 미실행 |
| B2 Final | 테스트 전 동결 | 미동결개 학습 Skill version | — | N=0, 평균 미실행/자료 없음 |
| B3 Final | 테스트 전 동결 | 미동결개 학습 Skill version | — | N=0, 평균 미실행/자료 없음 |
| B4 Final | 테스트 전 동결 | 미동결개 학습 Skill version | — | N=0, 평균 미실행/자료 없음 |

후보별 승인·거부와 보류는 다음과 같다.

| Arm | Loop | Parent / Candidate | PPO J | 검증 N / 평균 차이 | 채택 | 거부·보류 |
|---|---:|---|---:|---|---|---|
| B2 | 1 | RapidProcessResponse / RapidProcessResponse_v2 | -1.53 | 0 / 미실행/자료 없음 | False | None |
| B2 | 1 | RapidProcessResponse / RapidProcessResponse_v2 | -0.81 | 0 / 미실행/자료 없음 | False | None |
| B2 | 1 | EscalationDiagnosis / EscalationDiagnosis_v2 | 1.65 | 0 / 미실행/자료 없음 | True | None |
| B2 | 1 | EscalationDiagnosis / EscalationDiagnosis_v2 | 0.18 | 0 / 미실행/자료 없음 | False | None |
| B2 | 1 | PrivilegedRecovery / PrivilegedRecovery_v2 | 0.27 | 0 / 미실행/자료 없음 | False | None |
| B2 | 1 | PrivilegedRecovery / PrivilegedRecovery_v2 | 0.71 | 0 / 미실행/자료 없음 | True | None |
| B2 | 1 | SelectiveDeception / SelectiveDeception_v2 | 7.44 | 0 / 미실행/자료 없음 | True | None |
| B2 | 1 | SelectiveDeception / SelectiveDeception_v2 | 4.43 | 0 / 미실행/자료 없음 | False | None |
| B2 | 1 | NetworkContainment / NetworkContainment_v2 | 3.63 | 0 / 미실행/자료 없음 | False | None |
| B2 | 1 | NetworkContainment / NetworkContainment_v2 | 3.87 | 0 / 미실행/자료 없음 | True | None |
| B2 | 1 | RiskBasedInvestigation / RiskBasedInvestigation_v2 | -7.95 | 0 / 미실행/자료 없음 | False | None |
| B2 | 1 | RiskBasedInvestigation / RiskBasedInvestigation_v2 | -6.09 | 0 / 미실행/자료 없음 | False | None |
| B4 | 1 | EscalationDiagnosis / EscalationDiagnosis_v2 | 1.91 | 0 / 미실행/자료 없음 | False | validation_error |
| B4 | 1 | EscalationDiagnosis / EscalationDiagnosis_v2 | -0.41 | 0 / 미실행/자료 없음 | False | None |
| B4 | 1 | RiskBasedInvestigation / None | 미실행/자료 없음 | 0 / 미실행/자료 없음 | False | registered_rollout_budget_exhausted; keep previous skill including rare skills |
| B4 | 1 | RapidProcessResponse / None | 미실행/자료 없음 | 0 / 미실행/자료 없음 | False | registered_rollout_budget_exhausted; keep previous skill including rare skills |
| B4 | 1 | SelectiveDeception / None | 미실행/자료 없음 | 0 / 미실행/자료 없음 | False | registered_rollout_budget_exhausted; keep previous skill including rare skills |
| B4 | 1 | NetworkContainment / None | 미실행/자료 없음 | 0 / 미실행/자료 없음 | False | registered_rollout_budget_exhausted; keep previous skill including rare skills |
| B4 | 1 | PrivilegedRecovery / None | 미실행/자료 없음 | 0 / 미실행/자료 없음 | False | registered_rollout_budget_exhausted; keep previous skill including rare skills |

이번 8시간 계획은 자동 갱신 loop 1회/학습 2 episodes를 등록했다. 추가 loop 2와 100시드 전체를 실행했다고 주장하지 않는다. Test Reward는 selection.json에 먼저 고정한 최종 Pool만 측정하며 학습 중 후보 선택에 사용하지 않는다.

## 행동·Reasoning 진단

최초로 다른 primitive 요청을 고른 step, 그때의 두 reason·Skill·직전 자기 관측 hash를 decision_comparisons.csv에 기록했다. 후속 Sleep→Restore·Remove→Analyse 등의 패턴은 상태가 갈라진 뒤의 기술 통계이며 통제된 인과 실험으로 해석하지 않는다.

skill_usage.csv는 Skill별 실제 결정·phase·Red 정책·완료/실패 상태·공동 RTG 노출을 기록한다. TRUE 완료를 제거 성공으로 바꾸거나 공동 보상을 개인 보상으로 재정의하지 않는다. 유효 모델 선택, 파싱 fallback, 자동 busy Sleep·Monitor는 results.json에서 구분한다.

## 공정성·해석

B0–B4는 같은 v4 관측/사건 이력·Sleep·primitive 메뉴를 공유한다. B1/B2/B3의 selector는 원본 경로이며 B4만 관측 기반 domain initiation 검증을 추가한다. B3/B4는 SG 근거 검증과 실제 rollout admission을 추가한 Proposed 확장이다. 기존 B2의 PPO 수식과 용량 관리는 바꾸지 않았다.

근거 없는 수정·미완료 환경 검증은 이전 Pool을 유지한다. 희귀 Skill 최소 경험 수의 최적값은 한 후보 비교만으로 식별되지 않으므로 임의로 고정하지 않는다. 검증하지 못한 부모는 별도 보류로 기록한다.

원시 상태: {}. 중단·실패는 0점이나 성공 표본으로 바꾸지 않는다. 수동 초기 Skill, B4의 domain 규칙, 자동으로 채택한 새 Skill의 기여를 구분해야 한다.

[사건별 성능 저하 원인](../../../reports/reward_root_cause.md), [Skill-Pro 계산 감사](../../../reports/skillpro_algorithm_audit.md).
