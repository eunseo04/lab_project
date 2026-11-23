# FiniteState 성능 저하의 사건별 원인 추적

기존 v2의 동일 이력 ACD·정적 Skill·최종 학습 Pool 각 4회, 총 12개 실제 궤적을 재생했다. 모든 Blue의 전후 관측·행동·공동 보상이 원본과 일치했다. 새로운 모델 성능 표본이 아니다.

| Seed | ACD | 최종 Skill | 행동 비용 차이 | RIA 차이 | LWF 차이 | ASF 차이 | 총 차이 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 22000 | -193.0 | -161.0 | 3 | 37 | -8 | 0 | 32.0 |
| 22001 | -201.0 | -365.0 | 5 | -20 | -156 | 7 | -164.0 |
| 22002 | -218.0 | -243.0 | -3 | -21 | -1 | 0 | -25.0 |
| 22003 | -123.0 | -191.0 | -11 | -22 | -35 | 0 | -68.0 |

평균 차이: {'action_cost': -1.5, 'RIA': -6.5, 'LWF': -50, 'ASF': 1.75, 'total_difference': -56.25}. 공식 공동 보상은 Blue 수만큼 중복 합산하지 않았다. 사건별 보상과 행동 비용의 합계도 각 원본 총점과 일치한다.

## 실행 분기로 확인한 Green 실패

GreenLocalWork의 실제 선택 서비스·신뢰도·활성 상태와 마지막 서비스 변경 행동을 추적했다. Red DegradeServices 직후 서비스 신뢰도가 낮아졌고, 실제 실패가 그 신뢰도 검사 분기에서 반환된 경우만 `red_degraded_service_reliability`로 분류한다. GreenAccessService는 실제 연결 차단 분기를 확인한다. 0점 가중치 구역의 실패 사건도 보존하되 점수 손실과 구분한다.

| 실행 | 시드 | 실행 분기별 원본 보상 |
|---|---:|---|
| baseline_context | 22000 | {'successful_red_impact': -90, 'red_degraded_service_reliability': -33} |
| static | 22000 | {'successful_red_impact': -103, 'red_degraded_service_reliability': -58, 'observed_zone_firewall_block': -2} |
| loop2 | 22000 | {'successful_red_impact': -53, 'red_degraded_service_reliability': -41} |
| baseline_context | 22001 | {'red_degraded_service_reliability': -2, 'successful_red_impact': -119, 'observed_zone_firewall_block': -7} |
| static | 22001 | {'red_degraded_service_reliability': -1, 'successful_red_impact': -129, 'observed_zone_firewall_block': -32} |
| loop2 | 22001 | {'successful_red_impact': -139, 'red_degraded_service_reliability': -158} |
| baseline_context | 22002 | {'red_degraded_service_reliability': -64, 'successful_red_impact': -93} |
| static | 22002 | {'red_degraded_service_reliability': -11, 'successful_red_impact': -121} |
| loop2 | 22002 | {'red_degraded_service_reliability': -65, 'successful_red_impact': -114} |
| baseline_context | 22003 | {'successful_red_impact': -58, 'red_degraded_service_reliability': -4} |
| static | 22003 | {'successful_red_impact': -97, 'red_degraded_service_reliability': -64} |
| loop2 | 22003 | {'successful_red_impact': -80, 'red_degraded_service_reliability': -39} |

시드 22001의 최종 Skill은 public_access_zone_subnet_user_host_1에서 LWF −154를 기록했다. Red가 step 15·19·29·33·108에 서비스 신뢰도를 80·60·40·20·0으로 낮췄다. 첫 관련 실패는 step 16이었다. 해당 host의 Blue Restore나 Remove가 실행된 기록은 없었다. 이 host의 실패를 Restore 중단으로 설명할 근거가 없다.

Restore는 원본 구현에서 5-step 지연 후 backup 상태로 복원하며 직접 행동 비용 −1을 가진다. 대기 중이라는 이유로 서비스를 끄는 구현은 없다. Restore의 임무 가용성 영향은 이 직접 비용과 실제 서비스 상태·다른 행동을 못 하는 지연을 구분해야 한다.

현재 GreenAccessService.execute는 `available_dest_service`를 호출하지 않고 bound method의 truth value를 확인한다. 따라서 이 환경에서 ASF를 일반적인 원격 서비스 신뢰도 실패로 해석할 수 없다. 환경은 사용자 조건대로 수정하지 않았다. 실제 ASF는 관측된 zone block 분기로 분류했다.

## 처음 다른 결정을 한 시점

| Seed | 비교 | Step | ACD 요청 | Skill 요청 | 직전 자기 관측 동일 |
|---|---|---:|---|---|---|
| 22000 | static | 44 | Sleep | Analyse admin_network_subnet_user_host_8 | True |
| 22000 | loop2 | 0 | Sleep | DeployDecoy public_access_zone_subnet_user_host_0 | True |
| 22001 | static | 34 | Sleep | DeployDecoy public_access_zone_subnet_server_host_0 | True |
| 22001 | loop2 | 0 | Sleep | DeployDecoy public_access_zone_subnet_server_host_0 | True |
| 22002 | static | 9 | Analyse office_network_subnet_server_host_1 | DeployDecoy office_network_subnet_server_host_1 | True |
| 22002 | loop2 | 12 | Sleep | Analyse office_network_subnet_user_host_3 | True |
| 22003 | static | 34 | Analyse public_access_zone_subnet_user_host_3 | DeployDecoy public_access_zone_subnet_server_host_3 | True |
| 22003 | loop2 | 22 | Sleep | Analyse admin_network_subnet_server_host_3 | True |

후속 Sleep/Restore, Remove/Analyse, Analyse/Decoy 차이는 원시 JSON의 `later_request_patterns`에 보존했다. 첫 분기 이후에는 Red·Green 난수 소비와 상태가 달라질 수 있어, 동일 시드의 후속 차이를 한 행동의 인과 효과로 단정하지 않는다.

## 확정된 오류와 남은 가설

- 이전 v3의 부분 연결 누락과 Remove/Restore evidence epoch 오류는 v4에서 수정돼 99개 회귀 테스트로 보존됐다. v5의 공통 관측·이력은 v4와 동일하게 유지한다.
- 현재 원시 로그는 서비스 신뢰도 손상을 누가 일으켰는지 보여준다. Qwen이 더 일찍 조사·복구했다면 전체 손실이 얼마나 줄었는지는 별도 개입 검증 없이 확정할 수 없다.
- 과도한 조사 억제, Skill 시작 조건의 충돌, 드문 복구 Skill의 적은 경험, 공동 RTG와 Skill 선택의 관계는 검증할 가설이다. 행동 빈도 자체를 새 보상으로 만들지 않는다.

[사건별 실행·시드·phase·Skill·pending·보상과 CSV](../research_outputs/rollout-v5/root-cause/results.json). 각 에피소드 JSON은 Blue 5명의 당시 요청·완료 결과·행동 전 사건 이력, 실제 Red/Green 실행 순서와 서비스 상태를 포함한다.

## 상태·난수 복원 검증

단순 deepcopy는 mappingproxy 직렬화 오류로 실패했다. cloudpickle로 전체 wrapper·controller·state·Red/Green agent와 외부 Python/NumPy/Torch RNG를 보존하는 방식은 한 상태에서 같은 5-step branch의 관측·보상·전체 State hash·내부 RNG가 일치했다. 검증 범위 밖의 장기 인과 효과를 주장하지 않는다. [검증 기록](../research_outputs/rollout-v5/root-cause/cloudpickle-snapshot-audit.json).

전체 실제 replay에서의 첫 분기 개입도 점검했다. 시드 22000·22001의 step 0에서는 동일 snapshot/RNG의 반복 실행과 원본의 모든 Blue 관측·보상이 일치했고, 첫 행동만 바꾼 5-step 보상 차이는 모두 0이었다. 이후 요청·메시지를 고정한 짧은 결과이며 전체 에피소드의 효과가 아니다. 시드 22002·22003은 사전 관측 동등성 검사에서 통과하지 못해 개입 보상을 보고하지 않았다. 이를 일반적인 상태 복원 성공으로 확대하지 않는다. [검증·실패 기록](../research_outputs/rollout-v5/counterfactual/summary.json).

작전 단계별 RIA/LWF/ASF는 [phase CSV](../research_outputs/rollout-v5/root-cause/phase_reward_breakdown.csv), 자기 관측의 첫 PID/admin 지표와 최초 대응 요청은 [시각 CSV](../research_outputs/rollout-v5/root-cause/response_timing.csv)에 따로 기록했다. 요청 지연은 사건의 전체 인과효과가 아니며, 현재 증거 없이 숨겨진 Red 공격 시각부터 계산한 값을 방어자의 실수라고 부르지 않는다. 최초 요청이 없으면 지연을 임의의 숫자로 채우지 않는다.

## 방어 차단으로 인한 ASF의 직접 실행 주체

BlockTrafficZone/AllowTrafficZone의 실제 from/to subnet과 실행 순서를 추가로 재생했다. 원본은 `blocks[to].append(from)`으로 저장하므로 이 방향을 그대로 사용했고, Green의 실패 시점에 남아 있는 차단 edge의 마지막 성공 실행 주체를 연결했다. 여러 차단 edge가 동시에 원인이 되는 경우에도 보상은 한 번만 센다.

| 궤적 | ASF | 실패 시점 차단 edge를 만든 Blue |
|---|---:|---|
| ACD seed 22001 | −7 | KEEP blue_agent_2 |
| Static seed 22000 | −2 | Qwen blue_agent_4 |
| Static seed 22001 | −32 | KEEP blue_agent_3 |

따라서 초기 Static의 ASF 총 −34 중 −32는 다른 KEEP가 만든 활성 차단 edge에 직접 연결됐다. 이 −32 전체를 Qwen의 NetworkContainment Skill이 직접 막아서 생긴 실패로 해석하면 잘못이다. 반대로 seed 22000의 −2는 실제 Qwen 차단에 연결됐다. 팀 상태·메시지·행동 순서가 달라진 장기적인 간접 효과까지 분리한 결과는 아니다. [실행 순서·edge·주체·원시 hash](../research_outputs/rollout-v5/block-root-cause/attribution.json).

가장 큰 LWF host의 ACD 궤적에서는 최초 PID 관측 step 58, 최초 Remove 요청 step 60이었다. 최종 Skill 궤적에는 그 host의 PID/admin 관측이 없었다. 두 궤적의 잠재 공격 상태가 같다고 놓고 ‘늦은 Remove’의 효과를 계산할 수 없다. 추가 조사와 관측 단서 보존의 필요성은 검증 가설이며 숨겨진 공격 host를 정책에 하드코딩하지 않는다.

## 초기 정적 Skill의 손실 분해

| 항목 | 동일 이력 ACD | Static | Static−ACD |
|---|---:|---:|---:|
| action_cost | -66.25 | -70.00 | -3.75 |
| RIA | -90.00 | -112.50 | -22.50 |
| LWF | -25.75 | -33.50 | -7.75 |
| ASF | -1.75 | -8.50 | -6.75 |

합계 -40.75는 −224.50−(−183.75)와 일치한다. 행동 비용만으로 손실을 설명할 수 없으며 Red 영향과 Green 실패가 주요 차이다. 차단의 실제 실행 주체는 위 ASF 추적처럼 다른 KEEP일 수 있다. 같은 시드에서 첫 분기 뒤 상태가 달라지므로 이 표의 각 차이를 Qwen의 특정 Skill 하나의 직접 인과효과로 바꾸지 않는다.
