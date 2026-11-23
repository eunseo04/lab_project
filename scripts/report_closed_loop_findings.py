"""Write evidence-based RQ answers from audited completed runs, with actual N."""
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_outputs/rollout-v5'
def read(path):return json.loads(Path(path).read_text())
def fmt(v):return '미완료/자료 없음' if v is None else f'{v:.2f}'

def main():
    result=read(OUT/'report/results.json');terminal=result['status']=='budgeted_campaign_complete'
    finite={m['label']:m for m in result['metrics'] if m['red']=='finite'}
    comparisons={p['label']:p for p in result['paired'] if p['red']=='finite'}
    main_pair=comparisons['B4'];evolution=result['evolution']
    selected=(result.get('selection') or {}).get('selection',{})
    learned=selected.get('B4',{}).get('learned_skill_versions')
    lines=['# ACD + Skill-Pro 원인 검증과 실제 Closed-loop 결과','',
        ('8시간 한도 캠페인을 종료하고 완료된 에피소드만 감사했다.' if terminal else '캠페인이 진행 중이다. 아래 수치는 현재 완료·감사된 표본이며 최종 결론이 아니다.'),
        f"500-step 최종 평가 {len(result['episodes'])}회, 별도 훈련 {sum(len(v) for v in result['actual_training_episodes'].values())}회, 후보 검증 {len(result['actual_validation_episodes'])}회가 실제 완료됐다.",
        '',f"주 비교 B4−B0는 대응 N={main_pair['n']}, 평균 차이 {fmt(main_pair['mean'])}, 개선 {main_pair['wins']}/{main_pair['n']}, 중앙값 {fmt(main_pair['median'])}, 대응 bootstrap95% CI={main_pair['difference_95ci']}다.",
        f"최종 B4의 초기 수동 Pool에 없던 학습 Skill version 수는 {learned if learned is not None else '아직 미동결'}개다. 수동 Skill과 domain 규칙의 효과를 자동 학습 성과로 합치지 않는다.",
        '', '## 동일 조건의 실제 최종 비교','',
        '| Arm | 실제 N | Mean Joint Reward | Sample SD | 대응 N | 평균 차이 vs B0 | 개선/악화/동점 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for label in ['B0','B1','B2','B3','B4']:
        m=finite[label];p=comparisons.get(label)
        row=f"| {label} | {m['n']} | {fmt(m['mean'])} | {fmt(m['std'])} | "
        row+=f"{p['n']} | {fmt(p['mean'])} | {p['wins']}/{p['n']-p['wins']-p['ties']}/{p['ties']} |" if p else '— | 기준 | — |'
        lines.append(row)
    lines+=['','최종 Pool은 테스트 전에 동결했고, 시드·기준·iteration 선택에 테스트 보상을 사용하지 않았다. 양쪽 완료 시드의 대응 차이를 우선 해석한다. 작은 N의 bootstrap 구간으로 통계적 우위를 확정하지 않는다.',
        '', '## RQ1. 왜 정적 Skill이 동일 이력 ACD보다 낮았는가?','',
        '기존 v2의 Static−ACD −40.75는 action_cost −3.75, RIA −22.50, Green LocalWork −7.75, Green AccessService −6.75로 정확히 분해된다. 행동 비용보다 실제 공격 피해와 Green 실패 차이가 크다. 다만 Static의 ASF 총 −34 중 −32는 KEEP agent3가 만든 활성 block edge에 직접 연결돼 Qwen의 NetworkContainment 하나의 직접 피해라고 볼 수 없다.',
        '', '최종 v2의 −56.25는 cost −1.50, RIA −6.50, LWF −50.00, ASF +1.75다. seed22001의 한 host에서 LWF −154가 발생했고 실제 Red DegradeServices의 서비스 신뢰도 저하 분기를 확인했다. 그 host의 Restore/Remove 실행은 없었다. Restore를 곧바로 원인으로 지목한 설명은 지지되지 않는다. Baseline과 Skill의 후속 공격 상태가 달라 단일 Blue 행동의 전체 인과효과는 분리하지 못했다.',
        '', '공통 v4 이력의 사건 epoch·새 PID·실패한 Remove·Restore 후 재침해 처리는 유지했다. B4에는 현재 admin/새 PID 대응이 불필요한 재진단보다 먼저 후보가 되도록 initiation 충돌을 수정했다. primitive는 여전히 Qwen이 고르며 Restore와 Sleep도 가능하다. [사건·phase·실행 주체·최초 분기 근거](reward_root_cause.md).',
        '', '## RQ2. PPO 통과와 실제 보상이 왜 일치하지 않는가?','',
        '이전 24개 후보의 확률비·clipped 항·J를 모두 재계산했고, 별도 원시 보상에서 RTG·EMA baseline·2,084개 Gate Advantage도 재구성해 일치시켰다. 새 활성 Skill 행동 8건/6종의 생성 확률과 uncached teacher forcing의 최대 차이는 9.54×10⁻⁷이다. 이 검사들에서 확률·Advantage의 수치 오류는 확인하지 못했다.',
        '', '원본 조건은 Best-of-N 후보의 J>0이며 Jcandidate>Jparent가 아니다. Skill 조건부 공동 RTG는 다른 KEEP·Red·Green·시간/위치의 영향을 포함한다. Gate는 그 궤적에서 나온 행동 토큰의 확률을 평가하므로, 다음 에피소드에서 selector가 Skill을 고르는 빈도나 실제 보상의 개선을 보장하지 않는다. 새 배치의 RiskBasedInvestigation과 SelectiveDeception은 가능한 clipped J의 상한부터 음수였고, 이를 임의로 다른 Gate로 바꾸지 않았다.',
        '', 'SG의 출처와 의미도 구분해야 한다. B4의 최초 48개 청크 중 grounding이 통과한 것은 27개였으며, 저장한 실제 필드 인용 124개는 모두 원본 훈련 trace에 연결됐다. 하지만 Analyse만 완료된 청크에서 Remove 성공 후 종료 실패라는 해석이 나왔다. 정확한 인용도 잘못된 원인 해석을 막지 못하는 사례다. [원본 수식·공동 보상·출처 및 한계](skillpro_algorithm_audit.md).',
        '', '## RQ3. 실제 Rollout 검증은 잘못된 갱신을 줄였는가?','']
    checks=[r for r in evolution if r.get('ppo_selected') and r['label'] in ['B3','B4']]
    if not checks:lines+=['현재 완료된 자동 갱신에서 실제 환경 검증까지 기록된 PPO 선택 후보가 없다. 기능 테스트 통과만으로 실험적 효과를 주장하지 않는다.']
    for r in checks:
        lines.append(f"- {r['label']} {r['candidate']}: PPO J={fmt(r['score'])}, 완전 대응 검증 N={r['validation_n']}, 실제 평균 차이={fmt(r['validation_mean_difference'])}, 채택={r['accepted']}, 사유={r['rollout_reason']}, 사전 기준={r.get('validation_checks')}.")
    lines+=['','전체20개 시드의 양쪽500-step 실행과 사전 보상·승률·최악 손실·Green·SD 조건이 충족돼야 새 snapshot을 채택한다. 미완료 검증은 성능 실패와 구분해 기존 Pool을 유지한다. 실제 비교가 끝나지 않은 후보를 나쁜 후보로 분류하지 않는다. 검증 Pool 메타데이터 누락으로 생긴 최초 worker 실패는 점수 표본에서 제외하고, 별도 수정본에서 동일 후보 hash로 다시 검증했다.',
        '', '채택·거부는 이 표본과 설정에서의 결정이다. B3에는 SG 출처 검증·검증 순서 제한도 들어가므로 B3−B2를 rollout 하나만의 순수 인과효과로 해석하지 않는다. 검증 통과 뒤의 테스트 안정성은 RQ4에서 별도로 확인한다.',
        '', '## RQ4. 동일 이력 ACD보다 여러 시드에서 일관되게 좋은가?','']
    if main_pair['n']==0:lines.append('미사용 최종 테스트의 대응 표본이 아직 없다.')
    else:
        lines.append(f"실제 FiniteState 대응 차이는 {main_pair['differences']}다. 개선 {main_pair['wins']}, 악화 {main_pair['n']-main_pair['wins']-main_pair['ties']}, 동점 {main_pair['ties']}회다.")
        if main_pair['wins']<main_pair['n']:lines.append('모든 평가 시드에서의 일관된 개선은 확인하지 못했다.')
        else:lines.append('현재 완료된 시드에서는 모두 개선했지만, 이 표본만으로 일반적인 일관성이나 통계적 우위를 확정하지 않는다.')
    if learned==0:lines.append('B4에서 실제 채택된 학습 Skill이 없으므로, 최종 B4 점수가 좋아도 자동 Closed-loop Self-Improvement가 성공했다고 표현하지 않는다.')
    lines+=['','## RQ5. 미학습 Red 정책에도 개선이 유지되는가?','',
        '| Red 정책 | 실제 대응 N | B4−B0 평균 차이 | 개선/악화/동점 | 95% CI |',
        '|---|---:|---:|---|---|']
    for p in result['paired']:
        if p['label']!='B4' or p['red']=='finite':continue
        lines.append(f"| {p['red']} | {p['n']} | {fmt(p['mean'])} | {p['wins']}/{p['n']-p['wins']-p['ties']}/{p['ties']} | {p['difference_95ci']} |")
    lines+=['','각 정책을 섞어 하나의 큰 표본으로 부풀리지 않는다. N<2에서는 분산·CI를 정의하지 않는다. 평가하지 못한 정책이나 시드에는 점수를 채우지 않는다. 기존 v2 AggressiveFSM의 +50.75/4승은 별도 과거 Pool·시드 결과이며 현재 Proposed의 전이 성과로 가져오지 않는다.',
        '', '## 변경·검증·재현 및 한계','',
        '- 원본 Skill-MDP, SG/집계 경로, RTG clipped PPO, Best-of-N J>0, Algorithm1 용량 관리는 B2에 유지했다. 모델 가중치와 공식 보상은 바꾸지 않았다.',
        '- Proposed는 실제 step/actor/field의 SG 출처 검사, unsupported/host-instance 수정 거부, transactional real rollout 및 rollback, B4의 관측 기반 initiation 검증을 추가했다.',
        '- 기존99개를 포함한 실측판139개 회귀 검사가 통과했다. 실제 snapshot 메타데이터와 정적 B1의 공통 Pool 로드 누락을 수정 전 재현·수정 후 정상 로더 통과로 확인했다. 여러 loop/warm start의 provenance를 보강한 재사용판은141개가 통과했으며 별도 성능 실험으로 합산하지 않았다.',
        '- 희귀 Skill의 임의 최소 경험 수를 활성화하지 않았다. 0/4/8/16/32 민감도를 기록하되 한 후보만으로 최적 cutoff를 식별했다고 주장하지 않는다. 제한된 검증 예산의 다른 부모는 별도 deferred다.',
        '- 두 상태에서 같은 snapshot/RNG 개입을 검증했지만 첫 행동만 바꾼 고정5-step 보상 차이는 모두0이었다. 다른 두 시드는 사전 관측 동등성에 실패해 개입 효과를 보고하지 않았다. 개별 보상 보정은 추가하지 않았다.',
        '- 이번 등록은 arm당 훈련2 episodes/자동 loop1회다. loop2 및 100시드 전체를 수행했다고 주장하지 않는다. 추가 loop는 재현 설정으로 지원한다.',
        '', '[4개 결과표·8개 그래프·개별 시드·CSV·Gate 진단](../research_outputs/rollout-v5/report/performance.md), [방법과 실제 재현 명령](rollout_methodology.md), [동결 및 선택 기록](../research_outputs/rollout-v5/selection.json).']
    debug=OUT/'debugging/audit.json'
    if debug.exists():
        d=read(debug)['paired']
        lines+=['','알려진 기존4시드의 별도 Debugging은 수동 초기 Pool+B4 검증 규칙의 회귀 확인이다. B0 평균−217.25, B4 평균−210.00, 대응 차이+7.25, 3/4 개선, CI[−37.00,36.25]였다. 학습 후보 검증이나 미사용 최종 테스트 N에 합치지 않았고 이 결과로 코드·기준을 변경하지 않았다.']
    (ROOT/'reports/closed_loop_findings.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'terminal':terminal,'finite_paired_n':main_pair['n'],'final_episodes':len(result['episodes'])}))

if __name__=='__main__':main()
