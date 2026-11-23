# 2026-11-24 연구 snapshot

이 문서는 이번 반출 시 실제 저장된 완료 결과에서 상태를 정리했다. 이전 문서의 진행 예정·running 표시는 당시 캡처 상태로 남겨 두었다. 새 모델 실험을 실행하지 않았다.

## 완료 및 미완료 범위

| 구분 | 실제 상태 |
|---|---|
| v2 | 500-step 44회 실험 완료. FiniteState 최종−동일 이력 ACD −56.25, AggressiveFSM +50.75. |
| v4 | 수동 Skill 대조 8회 완료. 평균 +35.50이나 1/4 개선으로 안정성 기준 미달. |
| v5 B2 훈련 | 2 episodes, 1 loop. 자동 채택 4개. |
| v5 B3 훈련 | 2 episodes, 1 loop. 자동 채택 0개. |
| v5 B4 훈련 | 2 episodes, 1 loop. 자동 채택 0개. |
| v5 B4 실환경 검증 | 20개 대응 시드, 이전/후보 총 40회 완료. 후보 거부. |
| v5 최종 테스트 | 0회. B0–B4 100시드 비교와 새 Red 정책 일반화는 미완료. |

## B4의 실제 Rollout 결과

| 지표 | 값 |
|---|---:|
| 이전 Pool 평균 | -190.55 |
| 후보 Pool 평균 | -189.00 |
| 평균 대응 차이 | +1.55 |
| 대응 중앙값 | +0.50 |
| 승률 | 50% |
| 최악 시드 차이 | -92 |
| LWF 평균 차이 | +1.00 |
| ASF 평균 차이 | -3.35 |

평균만 +1.55였으나 승률 60% 기준, 최악 손실 −50 기준, Green AccessService 손실 증가 없음 조건을 충족하지 못했다. 후보를 거부하고 이전 Pool을 유지했다. 이는 이 검증 후보의 결과이며 최종 테스트의 개선이나 통계적 우위를 뜻하지 않는다.

근거: [실제 결정 JSON](../research_outputs/rollout-v5/recovery/B4/train/rollout-verification/iteration-0001-001-EscalationDiagnosis_v2/decision.json). 완료된 훈련·검증 요약, 프로토콜, source hash와 이전 실패 기록을 함께 포함했다. 대용량 원시 로그가 필요한 전체 재감사는 로컬 원본 archive가 필요하다.

## 재검사

반출된 재사용판을 기존 CPU dependency 환경으로 검사한다. API 연결은 fixture에서 금지한다. 새 모델 inference나 GPU 학습은 수행하지 않는다. 결과는 [반출 검증](publication_validation.json)에 기록한다.
