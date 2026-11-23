# 행동 분석에 따른 Security Skill 보완

2026-11-21, Asia/Seoul. 기존 [출판 논문 수정 실험](skillpro_paper_fix.md)의 결과는 보존하고, 다른 훈련·평가 시드를 사용했다. 실측 구성은 **KEEP 4명 + Qwen3-8B Blue 4 한 명의 Hybrid**다. 단기 개발 검증, 고정한 closed-loop 평가, 관측 오류를 수정한 수동 Skill 평가를 구분한다.

**평가를 완료했다.** v2의 최종 학습 Skill은 같은 이력의 ACD보다 FiniteState에서 평균 −56.25, 처음 보는 AggressiveFSM에서 +50.75였다. 관측 처리를 고친 v4의 수동 Skill은 새 FiniteState 시드에서 평균 +35.50이지만 개선은 1/4여서 사전 안정성 기준을 통과하지 못했다. 코드의 오류 수정과 조건별 개선은 확인했으나 일관된 성능 우위는 확인하지 못했다. v4로 새 closed-loop 훈련을 수행한 결과는 아니며, 아래에서 버전과 실험을 구분한다.

## 행동 빈도를 해석하는 조건

[ACD 원논문 §IV-A](https://arxiv.org/html/2505.04843v1)는 Figure 7의 대표 에피소드 행동 비교와 함께 Sleep을 프롬프트에서 제외했다는 점, Monitor가 자동 수행된다는 점을 설명한다. 이 대표 행동 분포를 모든 시나리오 평균이나 모델 능력만의 차이로 해석하지 않는다. 또한 여기서 사용하는 공개 KEEP는 저자들이 통신 기능을 추가해 별도 학습했다고 설명한 정책과 같은 가중치라고 주장할 수 없다.

현재 환경의 실제 메뉴는 다음과 같다.

| 항목 | Qwen 대조군 / Proposed | 현재 공개 KEEP |
|---|---|---|
| Analyse, Remove, Restore, DeployDecoy | 선택 가능 | 선택 가능 |
| BlockTrafficZone, AllowTrafficZone | 선택 가능 | 선택 가능 |
| 명시적 Sleep | 선택 가능 | **GraphWrapper의 출력 메뉴에 없음** |
| 명시적 Monitor | 선택하지 않음 | 선택 가능 |
| 자동 Monitor | 매 tick | 매 tick |
| busy 중 요청 | 실제 개입과 구분 | 실제 개입과 구분 |

KEEP의 실제 `NODE_ACTIONS`, `EDGE_ACTIONS`, `GLOBAL_ACTIONS`와 action translator를 확인했다. [메뉴·source hash 감사](../research_outputs/behavior-v2/report/action_menu_audit.json). 분포는 전체 요청 수와 **행동을 실제로 선택할 수 있었던 free tick**을 따로 센다. 자동 Monitor도 별도로 센다. Sleep/명시적 Monitor는 보조적인 ‘수동적 선택’ 범주로 묶을 수 있으나, 두 정책의 MDP와 반환 관측까지 같다는 뜻은 아니다.

현재 All RL의 동일 네 시드 Blue 4에서는 free tick Sleep 요청이 0회였다. 이는 Sleep을 잘 판단해 안 쓴다는 증거가 아니라 메뉴에 해당 행동이 없는 조건의 결과다. 논문의 Sleep 261회를 수치 목표로 복사하지 않는다.

## 보완한 상태와 Skill

기존 Qwen은 현재 한 tick의 관측만 actor에게 전달했다. 방금 전의 경보가 현재 관측에서 사라지면, 대응이나 확인이 끝나지 않아도 조용한 상황으로 처리할 수 있었다. 새 `ObservableCaseLedger`는 **그 Blue가 실제로 본 관측과 자기 행동 결과**만 유지한다.

- busy 중 잠시 나타난 의심 PID·파일 지표도 남긴다. 다른 Blue의 내부 상태나 실제 Red 세션 수를 읽지 않는다.
- 처음 요청한 행동을 완료까지 추적한다. 중간 자동 Sleep을 새 행동으로 잘못 연결하지 않는다.
- Remove의 TRUE는 완료 사실이다. 제거가 실제로 성공했거나 host가 깨끗하다는 표시는 아니다.
- Remove 후에도 `cmd.sh` 같은 파일 흔적은 남을 수 있다. 오래된 동일 파일만으로 새 공격이라고 단정하지 않는다.
- 관측된 관리자 지표와 아직 검토하지 않은 후속 작업을 유지한다. 현재 경보 부재가 과거 사건을 지우지는 않는다.
- peer ID·위협 기원·심각도·busy bit를 분리해 보여준다. peer의 busy bit만으로 공격을 추정하지 않는다.

내부 기록은 구조화하지만 actor와 selector에는 읽기 쉬운 사건 문장으로 제시한다. 현재 진행 중인 작업이 없는 경우 **FREE**임을 명시해 이미 끝난 작업을 계속 기다리는 혼동을 줄인다. 최대 12개 우선 사건과 최근 완료 4건을 입력에 제시하며, 내부 이력과 원시 궤적은 유지한다.

사건 이력은 공정한 대조를 위해 E0에도 동일하게 제공한다. 이것을 Skill 학습 효과로 모두 귀속하지 않는다.

`baseline_context − baseline_raw`는 이력 저장과 읽기 쉬운 표현·실행 상태 해석 계약을 함께 바꾼 효과다. 이 둘의 기여를 별도 ablation으로 분리한 수치라고 해석하지 않는다.

새 초기 Skill은 다음 여섯 개다. 원래 seed set은 `legacy`로 유지한다.

| Skill | 시작 조건과 절차 | 종료 |
|---|---|---|
| RapidProcessResponse | 새 local PID나 미대응 user 지표 → 신속한 Remove → 결과 검토·필요 시 진단 | 대응과 후속 확인 완료 또는 관리자 복구로 전환 |
| EscalationDiagnosis | 아직 검토하지 않은 사건·Remove 후 확인 → Analyse 결과에서 user/admin 지표 확인 | 결과 소비 또는 다음 대응 필요 |
| PrivilegedRecovery | 남아 있는 관리자 지표 → Restore, 가용성 비용과 복구 필요를 함께 고려 | 완료 상태 확인 또는 실패 재평가 |
| SelectiveDeception | 긴급 사건이 없을 때 필요한 host에 제한적 배치 | 배치 결과 확인; 같은 성공 배치를 반복하지 않음 |
| NetworkContainment | 실제 관측된 교차 zone 위협에 한정해 선택적 차단 | 결과 확인과 필요 재평가 |
| TrafficRecovery | 자신의 성공한 차단 기록에 근거해 복구 | 복구 결과 확인 |

자동 busy 대기를 별도의 선택 가능한 Wait Skill로 두지 않는다. **Sleep primitive는 E0와 Proposed 모두 계속 선택할 수 있다.** actor의 행동을 스크립트로 대신하거나 RL 행동 비율에 맞춰 강제하지 않는다. E3의 Skill 시작 검증은 같은 관측 이력에 근거한다. 이는 CAGE4용 검증 확장이며 무수정 Skill-Pro 환경 재현과 구분한다.

## 경험이 다음 Skill에 전달되는 경로

현재 훈련 배치의 실제 Skill 구간을 Semantic Gradient에 전달한다. 새로 기록하는 정보는 다음과 같다.

- 행동 전 사건 이력과 행동 후 완료 상태
- 자동 busy 대기와 모델이 직접 고른 Sleep의 구분
- 열린 사건이 있을 때의 Sleep
- 새 지표 없이 반복한 Analyse
- 같은 host의 이미 성공한 decoy를 다시 요청한 경우
- 미대응 PID가 있을 때 먼저 Analyse를 선택한 경우

이 정보는 **진단 자료**이며 새 보상이 아니다. 공동 보상, RTG, 정확한 old/new token 확률, clipped surrogate, Best-of-N의 `J > 0`, 용량 초과 시 score maintenance는 앞서 구현한 논문 모드를 유지한다. LLM 가중치와 selector를 학습하지 않는다.

`duplicate_successful_decoy`는 **같은 host에 과거 TRUE 배치 후 다시 요청했다**는 뜻이다. 현재 DeployDecoy는 Apache·Tomcat·Haraka·Vsftpd 중 호환 서비스를 선택하며, 같은 host의 다른 서비스 추가나 Restore 이후 재배치가 가능하다. 원시 완료 관측의 `service_name`과 자기 Restore 이력으로 상세 진단을 추가했다. 첫 loop의 같은 host 재배치 42회 중 새 관측 서비스가 기록된 경우는 23회였다. 같은 서비스가 다시 관측돼도 외부 공격에 의해 기존 decoy가 사라졌는지는 이 기록만으로 확정할 수 없다. 따라서 42회 전체를 낭비라고 표시하지 않는다. 초기 Skill의 host별 배치 제한은 보수적인 수동 사전 지식이며 최적 배치 수를 증명한 규칙이 아니다. 상세 진단은 소급해 actor·훈련 입력을 변경하지 않는다.

‘동일 지표 Analyse’에는 새 가시 지표가 없는 정당한 재조사도 포함될 수 있다. 관리자 지표 없이 요청한 Restore 역시 그 사실만으로 모두 오답이라고 판정하지 않는다. 사건 처리·가용성·실제 공동 보상과 함께 해석한다. free tick에 적용된 요청에는 파싱 실패의 fallback Sleep이 포함되므로, 유효 모델 선택과 fallback도 CSV·JSON에서 별도로 제공한다.

Pool 용량은 24, 후보는 Best-of-2, 최대 Skill 나이는 16 tick으로 설정했다. 비동기 Remove(3)·Analyse(2)·Restore(5)의 절차를 고려한 실행 hyperparameter이며 새 neural PPO나 상대 점수 Gate는 아니다. 설정은 최종 평가 전에 고정했다. 아직 경험이 적은 초기 Skill을 임의로 보호하는 별도 점수 보정은 넣지 않았다.

### 피드백이 올바른 경험 활용을 뜻하는지는 별도 확인한다

첫 loop는 6개 Parent의 전체 활성 구간을 60개 시간 조각으로 나누어 Semantic Gradient에 전달했다. 실제 정책 표본은 421개였지만 PrivilegedRecovery는 2개, TrafficRecovery도 2개뿐이었다. SelectiveDeception은 277개였다. 피드백 호출이 정상 실행됐다는 사실과 희귀 복구 절차를 충분히 배웠다는 주장은 구분한다.

일부 실제 피드백은 제공한 환경 설명을 잘못 해석했다. RapidProcessResponse의 첫 피드백은 경보 PID에 신속히 Remove할 수 있다는 입력 설명과 달리 Analyse로 먼저 확인하도록 권했다. SelectiveDeception의 예시 피드백은 조용한 구간에서 배치를 수행하지 않은 것을 절차 실패로 다뤘다. 이들은 선택한 정성적 사례이며 전체 Semantic Gradient의 오류율이나 전체 손실의 인과 비율을 추정한 것은 아니다. [원문 예시와 출처](../research_outputs/behavior-v2/semantic_gradient_examples.json).

승인된 SelectiveDeception_v2의 PPO 점수는 `2.7111`이지만 같은 저장 행동·Advantage에서 부모를 `r=1`로 둔 값은 `7.9638`이었다. TrafficRecovery의 부모 `12.1528`과 승인 후보 `7.4051`도 같은 관계였다. **논문의 `J>0`은 부모 대비 개선 검사가 아니다.** 이를 이유로 승인 수식·원래 보상을 바꾸지 않았고, 실제 동결 Pool 평가를 성능 판단 기준으로 삼았다. 복구·제거 구간의 음수 Advantage와 조용한 구간의 양수 Advantage는 공동 RTG의 조건부 표본 차이에도 영향을 받는다. 이를 해당 행동의 인과적 유용성으로 동일시하지 않는다.

두 번째 loop의 PrivilegedRecovery_v2는 관리자 지표가 있으면서 이미 분석되지 않은 host라는 시작 조건을 만들었다. 관리자 지표를 Analyse로 확인하는 절차와 충돌할 수 있다. Deception 후보의 위협 요구 조건과 E3의 긴급 사건 우선 검증도 함께 확인해야 한다. 이는 문구에 대한 정성적 진단이며 해당 문장만의 인과 효과를 분리한 실험은 아니다. Gate는 같은 저장 상태에서 그 Skill이 주어졌을 때 행동 확률을 계산한다. 이후 메타 선택이 그 Skill을 실제로 선택하는 확률이나 전체 새 에피소드를 직접 검증한 점수는 아니다.

## 개발 검증과 재현 확인

JSON 이력을 사용한 첫 시도에서는 이력에 확인할 사건이 남아 있는데도 “현재 경보 없음”, “이전 작업 완료 대기”를 이유로 Sleep을 반복했다. 해당 결과는 `behavior-v1-smoke`에 보존했다. 이어 진행하던 두 500-step 개발 작업은 표시 방식을 바꾸면서 중단했고, 부분 보상은 최종 표본에서 제외했다.

읽기 쉬운 이력의 `behavior-v2-smoke`는 실제 40-step 훈련 2회와 별도 검증 1회를 완료했다. `Remove → Analyse → Restore`의 실제 전환을 확인했다. 짧은 결과를 장기 성능 개선으로 주장하지 않는다.

[회귀 테스트 75개](../research_outputs/behavior-v2-final-tests.xml)가 통과했다. 실제 행동 8건을 캐시 없이 teacher forcing으로 다시 계산한 최대 sequence logprob 차이는 `4.76837158203125e-7`이었다. selector XML 48건, termination XML 24건과 actor 파싱도 확인했다. [실제 smoke 감사](../research_outputs/behavior-v2-smoke/integrity.json).

500-step 개발 시드 23000의 초기 검증에서는 사건 이력만 받은 ACD −132, 새 정적 E3 −225였다. 정적 E3의 열린 사건 Sleep은 줄었으나 Restore는 4→9회로 늘었다. 행동 지표 개선과 보상 개선이 다를 수 있어, 이 결과는 과잉 개입을 확인할 개발 진단으로 기록한다. 이 시드는 최종 일반화 평가에 포함하지 않는다.

### 개발 검증의 원본 보상 분해

기록된 실제 LLM 응답과 원래 `run_episode` 경로로 두 검증 에피소드를 재생했다. **모든 Blue의 행동·전후 관측·공동 보상이 원본과 일치한 뒤에만** 원본 `BlueRewardMachine`의 항목을 분해했다. 새로운 모델 성능 표본이 아니며, 이 분해값이나 시뮬레이터 내부 상태를 actor·Semantic Gradient에 제공하지 않았다.

| 원본 보상 항목 | 이력만 받은 ACD | 정적 E3 | E3 − ACD |
|---|---:|---:|---:|
| 팀 전체 행동 비용 | −43 | −45 | −2 |
| Red Impact | −61 | −42 | +19 |
| Green 로컬 작업 실패 | −28 | −126 | −98 |
| Green 서비스 접근 실패 | 0 | −12 | −12 |
| 합계 | −132 | −225 | −93 |

따라서 손실 전체를 ‘Restore가 5회 늘어서’로 설명할 수 없다. 현재 자료에서는 사용자 작업·서비스 실패가 증가한 것이 더 큰 손실 항목이다. [완전 일치 재생·분해](../research_outputs/behavior-v2/reward-decomposition/validation_diagnosis.json).

가장 큰 로컬 작업 손실은 검증 시드의 공개 구역 한 host에 집중됐다. ACD는 그 host를 Analyse 3회·Restore 1회 처리했지만 정적 E3는 처리하지 않았다. 정적 E3의 해당 host 관측은 연결 INFO였고 PID 경보는 없었다. 일부 INFO는 busy 중에만 보였다. 이 관측은 **반복 조사 억제가 필요한 선제 정보 수집까지 줄일 수 있다**는 가설을 뒷받침하지만, 단일 시드의 행동 차이에 대한 인과 효과를 분리한 실험은 아니다. 특정 host 이름을 정책에 하드코딩하지 않는다.

이 가설에 따라 연결 근거가 있는 한정적 선제 조사 Skill을 하나 추가한 **별도 개발 후보**도 시드 23000에서 검증했다. [초기 Skill 자료](../research_outputs/behavior-risk-development/initial_pool.json). 보상은 **−173**으로 여섯 Skill보다 +52였지만, 같은 이력의 ACD −132보다 −41 낮았다. 결과를 보기 전에 정한 `후보 > −132` 기준을 통과하지 못해 채택하지 않았다. [사전 기준과 판정](../research_outputs/behavior-risk-development/decision.json). 이 수동 초기화 후보를 경험으로 학습한 Skill이라고 표시하지 않는다.

이 후보도 원본 경로에서 모든 전이를 일치시켜 보상을 분해했다. 행동 비용 −47, RIA −68, LWF −56, ASF −2였다. 정적 여섯 Skill의 사용자 작업 손실은 줄었지만 Red Impact 손실은 커졌다. 첫 학습 loop의 같은 개발 시드는 −175(행동 비용 −57, RIA −20, LWF −98)로, 공격 피해 감소와 사용자 작업 보존의 차이가 계속 남았다. [검증 loop 1](../research_outputs/behavior-v2/reward-decomposition/validation_loop1-seed-23000.json), [추가 초기 Skill](../research_outputs/behavior-v2/reward-decomposition/static.json).

### 격리한 후속 개발 검증

등록 실험을 재실행하거나 중간 코드를 바꾸지 않고, [별도 source fork](../revisions/behavior-v3-dev-20261121/fork_provenance.json)에서 두 누락을 보완했다.

- 기존 반복 검사는 최근 두 world tick만 보므로, Analyse 사이에 자동 busy Sleep이 끼면 실제 연속 결정을 놓쳤다. 선택 가능한 모드 `accepted_decisions`는 최근의 유효한 실제 결정을 검사한다. 자발적 free Sleep은 유지해 반복의 연속성을 끊고, 새 관측이 있으면 반복 억제를 적용하지 않는다. 기존 `legacy_ticks`가 기본값이다.
- 선택 가능한 `connection_leads`는 자기 관측에 나온 연결 INFO를 잠시 유지한다. 연결은 침해 확정 지표로 승격하지 않는다. ephemeral remote port는 중복 근거로 세지 않고, 성공한 Analyse·Restore 검토 후 16-step cooldown과 마지막 관측 후 32-step 만료를 적용한다. 높은 우선순위의 기존 사건이 없을 때만 조사 후보가 되며, 조사할 근거가 남았으면 재량 decoy보다 먼저 검토한다.

후속 source의 테스트 **82개가 통과**했다. 최초 fork에서 누락한 두 참조 자료 때문에 발생한 테스트 오류는 자료 복사 후 해결했다. [테스트 결과](../revisions/behavior-v3-dev-20261121/research_outputs/behavior-v3-tests.xml). 이 상태를 E0와 E3 모두에 제공하는 **동일 개발 시드의 500-step 대조 2회**를 [별도로 등록](../research_outputs/behavior-v3-development/protocol.json)했다. 이전 E0 −132를 새 입력의 직접 대조군으로 재사용하지 않는다. 이 검증은 최종 평가나 closed-loop 학습 효과의 표본에 포함하지 않는다.

개발 결과는 E0 **−130**, 수동 7개 Skill의 E3 **−125**였다. [동일 입력·원시 이력 감사](../research_outputs/behavior-v3-development/results.json). 모든 전이를 일치시킨 원본 보상 분해는 E0 `(행동 비용 −40, RIA −38, LWF −52)`, E3 `(행동 비용 −43, RIA −69, LWF −13)`이었다. +5는 사용자 작업 보존과 공격 피해의 차이가 합산된 작은 개발 결과이며, 학습 Skill의 일반화 성능이 아니다.

이후 **코드와 초기 Pool을 추가 수정하지 않고** 새 FiniteState 시드 24000–24003의 E0/E3 8회를 완료했다. E0는 `[-264, -208, -237, -194]`, E3는 `[-214, -498, -263, -221]`이었다. 평균은 **−225.75 대 −299.00**, 대응 차이는 `[+50, −290, −26, −27]`, 평균 **−73.25**, 개선은 **1/4**였다. 개발 시드의 +5는 새 시드에서 유지되지 않았다. 사전 기준을 통과하지 못해 이 수동 후보의 AggressiveFSM 전이는 미실시했다. 기존 `behavior-v2`의 학습 Pool 전이 평가는 이 조건과 관계없이 전부 수행한다. [전향적 고정 프로토콜](../research_outputs/behavior-v3-prospective/protocol.json), [전체 결과·감사](../research_outputs/behavior-v3-prospective/report/performance.md).

v3 source와 `research_v3.sh`는 이 결과의 재현용으로 보존한다. 아래 관측 파서 문제가 발견돼 새 실험의 권장 버전으로 표시하지 않는다. [당시 설정](../revisions/behavior-v3-dev-20261121/research_configs/behavior-v3.yaml).

### 유효한 부분 관측과 Restore 이후 재침해 처리

실패 시드 24001을 분석하니 연결 관측에 `remote_port`는 있지만 `local_port`는 없거나, `local_address`만 있는 경우가 있었다. v3 파서는 `remote_address`와 `local_port`를 함께 요구해 이런 유효한 관측을 버렸다. 이는 모델 능력과 별개의 구현 오류다. 새 v4는 제공된 필드만 불확실한 연결 단서로 보존하며, 빠진 endpoint를 추정해 채우지 않는다. 기존 완전한 endpoint의 표시와 remote ephemeral port 중복 처리는 유지했다.

실제 궤적 7개를 이용한 입력 감사에서 의미 있는 연결 관측은 654개 host/tick이었고, 그중 부분 필드만 있는 관측은 468개, busy 중 부분 관측은 101개였다. 수정한 파서는 654개 모두 이력에 보존했다. 이는 입력 처리 감사이며 추가 모델 성능 표본이 아니다. [범위·원시 궤적 hash](../research_outputs/behavior-v4-development/input_coverage_audit.json).

실패한 Remove를 대응 완료로 세던 경로, Remove 뒤 새 PID가 나타났는데 과거 Remove로 사건을 닫던 경로, 대소문자를 다르게 쓴 유효 행동의 추적도 수정했다. 성공한 Restore는 과거 지표의 중복 제거 구간을 끝내고, 이후 같은 파일명·PID가 다시 관측되면 새 사건으로 처리한다. Restore 완료와 함께 반환된 지표는 보존하고 실패한 Restore는 이전 성공 경계를 바꾸지 않는다. 오래된 Remove로 Restore 이후의 새 침해를 처리 완료로 간주하지 않는다.

새 회귀 테스트에서 기존 파서·상태 처리의 실패를 먼저 재현한 뒤 전체 **99개 테스트가 통과**했다. [현재 결과](../revisions/behavior-v4-observation-20261121/research_outputs/behavior-v4-tests.xml). PPO Gate·score maintenance·selector·환경 보상·모델 가중치는 바꾸지 않았다. Semantic Gradient에는 연결 단서와 같은 host의 decoy 재배치에 관한 사실 설명만 보완했다.

중간 v4 개발 실행 2회는 Restore 경계 오류를 재현한 뒤 중단했고 부분 로그와 당시 source를 보존했다. 최종 성능 표본에 넣지 않는다. [중단 기록](../research_outputs/behavior-v4-development/retired.json). 수정본의 **개발 시드 24001** 대조는 E0 −272, 수동 Skill E3 −220으로 +52였다. 이 시드는 이미 오류 분석에 사용했으므로 heldout이 아니다. [개발 고정 조건](../research_outputs/behavior-v4-lifecycle-development/protocol.json), [원시 관측·동일 입력·방법 유지 감사](../research_outputs/behavior-v4-lifecycle-development/results.json).

두 궤적을 완전히 일치 재생한 보상 항목은 E0 `(행동 비용 −60, RIA −137, LWF −45, ASF −30)`, E3 `(행동 비용 −66, RIA −154, LWF 0, ASF 0)`이었다. 이전 v3의 같은 시드 E0 −208, E3 −498과도 함께 보존한다. 관측 표현과 처리 상태가 바뀌었으므로 새 후보의 직접 대조는 같은 수정 입력을 사용하는 E0 −272다. 새 입력이 E0 자체의 보상을 높였다고 주장하지 않는다.

개발 대조의 사전 조건을 통과해 코드·초기 Pool을 동결하고 **새 시드 29000–29003의 E0/E3 8회를 완료**했다. [수정본의 전향적 고정 조건](../research_outputs/behavior-v4-prospective/protocol.json), [시드별 결과·원시 입력 감사](../research_outputs/behavior-v4-prospective/report/performance.md).

| 수정본 v4, 수동 초기 Skill 평가 | 29000 | 29001 | 29002 | 29003 | 평균 |
|---|---:|---:|---:|---:|---:|
| 동일 v4 이력의 ACD | −155 | −168 | −79 | −376 | −194.50 |
| 수동 7개 Skill E3 | −175 | −176 | −108 | −177 | −159.00 |
| 대응 차이 | −20 | −8 | −29 | +199 | **+35.50** |

평균은 개선됐지만 **개선 시드는 1/4**였다. 큰 손실을 줄인 시드 29003의 +199가 평균 상승을 이끌었다. 미리 고정한 ‘평균 차이 양수·4개 중 3개 이상 개선’ 기준은 실패했다. 따라서 조건부 AggressiveFSM 30000–30003의 8개 작업은 사전 규칙으로 미실시했고 0점으로 채우지 않았다. 결과를 본 뒤 코드나 Pool을 다시 조정하지 않는다. 이는 수동 초기 Skill·관측 기반 검증의 결과로, 경험 기반 Skill-Pro 학습의 개선 수치와 합산하지 않는다.

| v4 원본 보상 분해, N=4 | 팀 행동 비용 | Red Impact | 로컬 작업 실패 | 서비스 접근 실패 | 합계 |
|---|---:|---:|---:|---:|---:|
| 동일 v4 이력의 ACD | −61.00 | −72.50 | −26.50 | −34.50 | −194.50 |
| 수동 7개 Skill E3 | −68.00 | −78.00 | −13.00 | 0 | −159.00 |
| E3 − ACD | −7.00 | −5.50 | +13.50 | +34.50 | +35.50 |

서비스 접근·사용자 작업 손실은 줄었지만 행동 비용과 Red Impact는 커졌다. 8개 궤적 모두 원본 관측·행동·보상을 완전히 일치 재생했고 summary·trace hash를 확인했다. 이 분해는 추가 모델 표본이나 행동별 인과 효과가 아니다. [원시 분해·출처 검증](../research_outputs/behavior-v4-prospective/report/reward_components.json). 최신 수정본에서도 ‘반복 행동 감소’만으로 성공을 판정할 수 없으며, 초기 상태에 따른 방어·가용성의 차이가 남았다.

현재 구현 진입점은 `research_v4.sh`다. 새 훈련을 시작할 수 있는 설정 예시는 다음과 같다. **아래 미래 훈련의 성능은 현재 실측에 포함하지 않는다.**

```bash
./research_v4.sh train --config research_configs/behavior-v4.yaml
./research_v4.sh evaluate --config research_configs/behavior-v4.yaml \
  --pool /home/jeonges/proposed/research_outputs/behavior-v4-next/E3-hybrid-finite/train/pool.json
```

[v4 설정](../revisions/behavior-v4-observation-20261121/research_configs/behavior-v4.yaml)의 기본 `pool`은 수동 초기화 7개 Skill이다. 학습 후 평가에서는 생성된 `train/pool.json`을 명시한다.

## 고정한 장기 비교

[실험 사전 등록](../research_outputs/behavior-v2/protocol.json)에 코드 hash와 비교·시드를 기록했다.

| 구분 | 조건 |
|---|---|
| 모델 | 동일 BF16 Qwen3-8B revision, 가중치 고정 |
| 훈련 | seeds 21000–21003, 500-step 4 episodes, 2 loops |
| 개발·검증 | seed 23000, 학습 전 E0/E3 및 각 loop 동결 Pool을 별도로 평가 |
| 최종 평가 | seeds 22000–22003, 각 방식 4 episodes |
| FiniteState 비교 | 이력 없는 ACD, 같은 이력을 받은 ACD, 정적 E3, loop 1, loop 2 |
| 공격 정책 전이 | FiniteState에서 학습한 최종 Pool을 동결해 AggressiveFSM의 이력 대조군과 비교 |
| KEEP 대조 | All RL × 두 Red 정책 × 같은 4개 시드, Blue 4 행동도 별도 집계 |
| 규모 | 훈련 4 + 외부 검증 4 + 최종 평가 28 + All RL 대조 8 = 44개 500-step episodes |

학습 도중의 검증은 별도 worker에서 **동결한 snapshot**으로 수행한다. 검증이 훈련 baseline·Pool 통계를 갱신하지 않는다. 최종 평가 보상으로 Pool이나 iteration을 선택하지 않고 모든 시드·loop 결과를 보고한다. 새 시드의 평가는 동일 CAGE4 시나리오 생성기 안에서의 초기 상태 변화이며 다른 토폴로지 전체에 대한 보증은 아니다.

다음 차이를 분리해서 판단한다.

1. `baseline_context − baseline_raw`: 사건 이력 추가의 효과
2. `static − baseline_context`: 새 초기 Skill과 CAGE4 Skill 검증의 효과
3. `loop1/loop2 − static`: 경험 기반 Skill 갱신의 추가 효과
4. `transfer_loop2 − transfer_baseline`: 학습하지 않은 Red 정책의 전이 효과

## 결과와 실행

등록된 **44개 500-step 훈련·검증·평가·KEEP 대조가 모두 완료**됐다. [CSV·그래프·상태](../research_outputs/behavior-v2/report/performance.md), [시드별 결과와 감사](../research_outputs/behavior-v2/report/results.json). 아래 표는 고정한 v2의 학습 비교이며, 관측 처리 버그를 수정한 v4의 수동 Skill 평가와 합산하지 않는다.

| 방식 | 22000 | 22001 | 22002 | 22003 | 평균 공동 보상 |
|---|---:|---:|---:|---:|---:|
| 이력 없는 ACD, 동일 제어 행동 계약 | −294 | −265 | −876 | −230 | −416.25 |
| 동일 사건 이력의 ACD | −193 | −201 | −218 | −123 | −183.75 |
| 초기 6개 정적 Skill | −231 | −224 | −207 | −236 | −224.50 |
| Closed-loop 1 | −240 | −406 | −309 | −298 | −313.25 |
| Closed-loop 2 | −161 | −365 | −243 | −191 | −240.00 |

두 번째 갱신은 loop 1보다 모든 시드에서 높았고 평균 +73.25였다. 그러나 최종 loop 2는 초기 정적 Skill보다 평균 −15.50, 같은 사건 이력의 ACD보다 −56.25였다. 따라서 두 번째 갱신의 회복과 전체 학습의 추가 성능 개선을 구분한다.

| 단계 | 훈련 시드와 갱신 전 보상 | 학습 후 별도 검증 23000 | 승인 | Pool |
|---|---|---:|---:|---:|
| 초기 | 훈련 없음 | −225 | 없음 | 6 |
| Loop 1 | 21000: −323, 21001: −304, 평균 −313.50 | −175 | 2 | 8 |
| Loop 2 | 21002: −198, 21003: −207, 평균 −202.50 | −80 | 6 | 14 |

훈련 평균은 서로 다른 시드이며 각 갱신 전 Pool의 결과다. 개발 시드에서는 −225→−175→−80으로 좋아졌지만 최종 평가에서는 같은 이력의 ACD를 넘지 못했다. 전체 입력·확률·보상 수식이 정상 실행된다는 사실이 초기 상태 변화에 대한 보상 개선을 보장하지 않는다.

### 학습하지 않은 Red 정책

FiniteState에서 학습한 최종 Pool을 동결하고 AggressiveFSM에서 평가했다.

| 방식 | 22000 | 22001 | 22002 | 22003 | 평균 |
|---|---:|---:|---:|---:|---:|
| 동일 이력의 ACD | −236 | −262 | −279 | −273 | −262.50 |
| 최종 loop 2 | −207 | −252 | −174 | −214 | −211.75 |
| 대응 차이 | +29 | +10 | +105 | +59 | **+50.75** |

네 시드 모두 개선됐다. FiniteState에서는 −56.25, AggressiveFSM에서는 +50.75여서 공격 정책에 따라 효과가 달랐다. 두 정책을 같은 비중으로 합친 보조 평균 차이는 −2.75다. 이 동일 가중 합산은 실제 공격 분포의 성능을 추정한 값은 아니며, 모든 조건의 일관된 개선을 주장하지 않는다. 다른 topology 전체나 Red 정책 전체에 대한 보증도 아니다.

### 행동·경험 활용과 실제 보상의 관계

학습 Skill은 실제로 재사용됐다. 최종 FiniteState에서 1603개 free 결정 중 1430개(89.2%), AggressiveFSM에서 1622개 중 1517개(93.5%)가 새 버전에 조건화됐다. 최종 FiniteState의 EscalationDiagnosis_v2 사용은 819회, AggressiveFSM에서는 992회였다. 이 이름이 붙은 모든 결정이 Analyse라는 뜻은 아니다. 원시 행동은 별도로 집계한다. PrivilegedRecovery_v2 사용은 각각 3회·1회로, 희귀 전용 복구 Skill의 경험과 재사용은 여전히 적었다.

| FiniteState의 Blue 4 요청 합계, N=4 | Analyse | Decoy | Remove | Restore | 직접 Sleep |
|---|---:|---:|---:|---:|---:|
| 이력 없는 ACD | 285 | 6 | 34 | 16 | 1188 |
| 동일 이력 ACD | 200 | 2 | 25 | 19 | 1426 |
| 정적 Skill | 193 | 39 | 44 | 33 | 1229 |
| Loop 1 | 135 | 88 | 39 | 24 | 1313 |
| Loop 2 | 195 | 47 | 24 | 26 | 1311 |

현재 Qwen은 논문의 o3-mini 대표 에피소드와 달리 초기 Decoy 선택이 적었다. Skill 추가·갱신 후 Decoy가 늘어난 점도 함께 보고한다. 모든 행은 같은 Qwen의 free 요청이며, busy Sleep·자동 Monitor·KEEP의 서로 다른 메뉴를 이 표의 모델 선택에 더하지 않는다.

원본 궤적의 모든 Blue 행동·관측·보상을 일치 재생한 보상 분해는 다음과 같다. 추가 성능 표본이나 단일 행동의 인과적 ablation이 아니다.

| 조건 | 팀 행동 비용 | Red Impact | 로컬 작업 실패 | 서비스 접근 실패 | 합계 |
|---|---:|---:|---:|---:|---:|
| FiniteState 이력 ACD | −66.25 | −90.00 | −25.75 | −1.75 | −183.75 |
| FiniteState 정적 Skill | −70.00 | −112.50 | −33.50 | −8.50 | −224.50 |
| FiniteState loop 1 | −63.50 | −125.25 | −120.25 | −4.25 | −313.25 |
| FiniteState loop 2 | −67.75 | −96.50 | −75.75 | 0 | −240.00 |
| AggressiveFSM 이력 ACD | −66.75 | −102.50 | −88.75 | −4.50 | −262.50 |
| AggressiveFSM loop 2 | −64.00 | −100.75 | −40.25 | −6.75 | −211.75 |

FiniteState 최종 손실 차이 −56.25 중 −50은 로컬 작업 실패 증가였다. AggressiveFSM 개선 +50.75 중 +48.5는 로컬 작업 실패 감소였다. 따라서 단순히 Restore·Analyse·Decoy 횟수를 일정 비율로 만드는 대신 임무 가용성을 함께 평가해야 한다. 팀 비용과 실패에는 KEEP 4명, Red·Green 상호작용도 기여한다. [분해 CSV·원시 일치 감사](../research_outputs/behavior-v2/report/reward_components.json).

최종 감사는 평가 누락 0, 테스트 75개, 실제 Gate 후보 점수 24개 재계산, 평가 관측 이력 28회와 훈련 이력 4회의 일치 확인을 포함한다. 모델 가중치는 고정됐고 원래 ACD 파일 20개와 이전 원시 결과 171개의 hash가 유지됐다. wall-clock 실행 시간에는 동시 추론 대기도 포함되므로 추론 효율 비교에는 사용하지 않는다.

네 평가 시드에서 공통 이력·읽기 쉬운 실행 상태 해석의 차이는 `[+101, +64, +658, +107]`, 평균 +232.50이었다. 네 시드 모두 높지만 효과 크기는 한 시드의 큰 개선에도 영향을 받는다. 이 공통 입력 효과와 별도로 학습 Skill의 추가 기여를 위 표에서 측정했다.

```bash
cd /home/jeonges/proposed
# 현재 등록 실험의 집계·원시 로그·수식 감사
.venv/bin/python scripts/report_behavior_campaign.py --require-complete
.venv/bin/python scripts/report_behavior_rewards.py --require-complete
.venv/bin/python scripts/report_behavior_v4_prospective.py --require-complete

# 새 디렉터리에서 단일 GPU로 재실행하는 예시
./research.sh train --config research_configs/behavior-v2.yaml --output research_outputs/behavior-next/E3-hybrid-finite/train
./research.sh evaluate --config research_configs/behavior-v2.yaml --output research_outputs/behavior-next/E3-hybrid-finite/eval
```

이미 실행한 캠페인은 `--launch`로 중복 시작하지 않는다. 같은 디렉터리의 source와 설정이 바뀌면 기존 실행을 재개하지 않는다. 이전 코드·문서 52개와 JSON prototype 코드를 `revisions/`에 보존했고, 등록 시점의 실행 코드는 `revisions/behavior-v2-registered-20261120/`에 고정했다.

최종 문서·분석 스크립트와 버전별 프로토콜·결과의 hash는 [완료 시점 manifest](../revisions/behavior-final-20261121/manifest.json)에 기록했다. 실험에 사용한 복제 서버 8119·8120·8121은 모두 종료했으며 기존 8018 Qwen 서버의 PID·실행 명령·모델 상태가 유지됨을 확인했다. [정리 기록](../research_outputs/behavior-v4-prospective/server_cleanup.json).
