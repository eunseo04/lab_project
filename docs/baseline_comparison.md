# ACD와 비교할 수 있는 범위

| 항목 | 원논문 / 원본 코드 | 이 연구 경로 |
|---|---|---|
| 환경·토폴로지·기본 행동·공동 보상 | CAGE Challenge 4 | 같은 고정 source |
| Red / Green | FSM 5종 / EnterpriseGreenAgent | 같은 정책 |
| LLM | GPT-4o-mini | 고정 Qwen3-8B BF16 |
| KEEP | 공개 GNN-PPO 및 논문 설정 | 공개 5개 weight hash 고정 |
| Hybrid wrapper | 혼합 시 동일 simulator 중복 호출 문제 | 공동 tick을 한 번 진행하는 공통 수정 |
| 통신 | 논문 8비트와 공개 KEEP 특징 불일치 | 공통 8비트→KEEP 특징 투영, 또는 분리된 keep_native |
| 종료·보상 | terminal 이전 누적 | official profile에서 같은 규칙 |
| 초기화 | 한 env의 reset stream | official 명령은 보존, 대응 실험은 episode seed 명시 |
| 행동 prompt | Sleep 제한, 자동 Monitor | paper-compatible 또는 공통 controlled 설정 |
| 학습 | Skill 없음 | E1 정적, E2 진화, E3 관찰 검증 추가 |

`research_configs/paper_reference.json`의 -493.0/95.9/45.2와
-2547.2/498.8/4704.6은 논문 KEEP/GPT-4o-mini 참고 수치다. Qwen 실측값이 아니고
단일 FiniteState episode 기준도 아니다. Red 간 가중 및 표준편차 집계 단위가
완전히 확인되지 않아 새 통계와 직접 동등한 수치 재현으로 표시하지 않는다.
원논문: [Large Language Models are Autonomous Cyber Defenders](https://arxiv.org/abs/2505.04843v2).

E0–E3의 같은 구성·시드·Red·모델·관찰·행동·보상·실행기를 사용한 대응 비교가
주된 검증 대상이다. Hybrid와 All LLM, 짧은 통합 실험과 500-step 평가,
2-episode 점검과 100-episode 평가는 각각 별도 그룹이다.

RQ1은 E1−E0, RQ2는 E2−E1, RQ3은 E3−E2로 확인한다. 음수 보상이므로
차이가 양수일수록 개선이다. 작은 N의 평균 차이는 일반화의 증거가 아니다.
RQ4에는 다른 Red 정책과 충분한 미사용 평가 seed가 필요하다. 누락 정책은 N/A로
남기고, Gate 승인이나 reasoning 군집을 Reward 개선으로 대체하지 않는다.

실제 완료 수, 측정값, Gate 승인/거부 및 아직 답할 수 없는 RQ는 자동 생성한
[execution_report.md](execution_report.md)에 기록한다. 원본 보상 집계 대조는
`research_outputs/official-evaluator-parity/arithmetic_parity.json`, 실제 Qwen
확률 검증은 `research_outputs/scoring-validation.json`에 있다.
