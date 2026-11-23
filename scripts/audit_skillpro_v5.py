"""Audit all historical gates and report shared-reward exposure without causal credit."""
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.gate import clipped_terms


def main():
    out=ROOT/'research_outputs/rollout-v5';train=json.loads((ROOT/'research_outputs/behavior-v2/E3-hybrid-finite/train/summary.json').read_text())
    gates=[];skills={};uses=defaultdict(Counter)
    for h in train['history']:
        for entry in h['evolution']:
            for index,c in enumerate(entry['candidates']):
                if c.get('score') is None:continue
                means=[];parents=[];samples=0
                for d in c['diagnostics']:
                    ratios,terms=clipped_terms(d['old_logprobs'],d['new_logprobs'],d['advantages'],.2,log_ratio_bound=None)
                    assert np.allclose(ratios,d['importance_ratios'],rtol=0,atol=1e-12)
                    assert np.allclose(terms,d['clipped_surrogates'],rtol=0,atol=1e-12)
                    means.append(float(terms.mean()));parents.append(float(np.mean(d['advantages'])));samples+=d['n']
                value=float(np.mean(means));assert abs(value-c['score'])<1e-10
                gates.append({'loop':h['iteration'],'parent':entry['parent'],'candidate':c['skill']['name'],
                    'score':value,'parent_r1_surrogate':float(np.mean(parents)),'valid_decisions':samples,
                    'accepted':c['accepted'],'positive_gate':value>0,
                    'semantic_chunks':len(entry.get('semantic_gradients',[])),
                    'individual_candidate_online_reward':'not_measured_in_original_campaign'})
        for skill in h['pool_snapshot']['skills']:skills[skill['name']]={'loop':h['iteration'],**skill}
    queue=json.loads((ROOT/'research_outputs/behavior-v2/queue.json').read_text())
    for task in queue:
        if task['stage'] not in ['static','loop1','loop2','transfer_loop2']:continue
        summary=json.loads(Path(task['summary']).read_text())
        for line in Path(summary['episodes'][0]['trace']).read_text().splitlines():
            r=json.loads(line)
            if r['agent']!='blue_agent_4' or not r.get('skill_id'):continue
            counts=uses[r['skill_id']];counts['ticks']+=1;counts['phase_'+str(r['mission_phase'])]+=1
            counts['red_'+task['red']]+=1
            counts['valid_decisions']+=bool(r.get('policy_invoked') and r.get('request_applied') and not r.get('parse_error'))
            counts['busy_ticks']+=not r.get('request_applied',False)
            counts['true_status_ticks']+='TRUE'==r['action_status'];counts['false_status_ticks']+='FALSE'==r['action_status']
            # Exposure, not causal contribution; score is shared by all Blue agents.
            if r['score_included']:counts['joint_reward_during_skill']+=r['joint_reward']
    data={'passed':True,'all_historical_candidate_scores_recomputed':len(gates),'gates':gates,
        'skill_online_statistics':skills,'heldout_usage_and_shared_reward_exposure':dict(uses),
        'reward_not_reweighted':True,'rare_support_cutoff_not_retrospectively_invented':True,
        'limitations':['Gate J>0 is not candidate J > parent J.',
            'Original candidate-level rollout rewards were not measured; a bad final pool does not identify one child as its cause.',
            'Status tick counts include asynchronous continuations, not unique successful cleanup counts.',
            'Joint reward exposure is not causal credit to the acting Skill.']}
    (out/'skillpro-audit.json').write_text(json.dumps(data,indent=2)+'\n')
    with (out/'gate-audit.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(gates[0]));writer.writeheader();writer.writerows(gates)
    report=ROOT/'reports/skillpro_algorithm_audit.md'
    lines=['# Skill-Pro 계산·근거·공동 보상 귀속 감사','',
        '출판 [Skill-Pro](https://proceedings.mlr.press/v306/mi26d.html)의 Skill-MDP, Semantic Gradient/집계, 고정 LLM, RTG 기반 clipped PPO Gate, Best-of-N J>0, Algorithm 1 용량 관리를 B2에 유지한다. 공개 코드의 episode-level advantage 모드와 논문의 RTG 모드는 기존처럼 구분한다.',
        '',f'기존 실제 후보 {len(gates)}개의 저장 old/new token logprob·확률비·Advantage·clipped objective를 모두 재계산해 저장 점수와 일치시켰다. 같은 조건에서 수치가 맞는다는 사실은 새 에피소드 성능 향상을 보장하지 않는다.',
        '', '| Loop | Parent | Candidate | J | Parent r=1 | 유효 결정 | 승인 |','|---|---|---|---:|---:|---:|---|']
    for row in gates:lines.append(f"| {row['loop']} | {row['parent']} | {row['candidate']} | {row['score']:.4f} | {row['parent_r1_surrogate']:.4f} | {row['valid_decisions']} | {row['accepted']} |")
    lines+=['', '첫 갱신의 승인 후보가 포함된 Pool은 정적 Pool보다 FiniteState 평가 평균 −88.75였다. 두 번째 Pool은 같은 이력 ACD보다 −56.25였다. 원래 실험에 후보별 환경 검증은 없었으므로 모든 승인 후보를 나열하되 어느 하나가 손실을 일으켰다고 단정하지 않는다.',
        '', '수치 오류와 일반화 실패를 구분한다. 기존 8개 실제 행동의 독립 재점수는 최대 차이 4.77×10⁻⁷이었다. 저장 Gate 계산에서 확률·Advantage 구현 오류를 확인하지 못했다. 반면 공동 RTG는 다른 KEEP 4명과 Red·Green의 영향까지 포함하고, 조용한 구간과 복구 구간의 조건부 분포가 다르다. 후보의 조건부 행동 확률 점수는 selector가 그 Skill을 선택할 빈도나 실제 환경 보상을 직접 검증하지 않는다.',
        '', '첫 배치에서 PrivilegedRecovery와 TrafficRecovery는 각각 유효 결정 2개였고 SelectiveDeception은 277개였다. v5는 부모별 유효 결정·궤적·phase·Red와 승인 여부를 기록하며, Proposed의 제한된 환경 검증은 가장 많은 근거를 가진 부모부터 수행한다. 미검증 희귀 Skill은 원래 상태를 유지한다. 원본 B2에는 이 순서·보류 규칙을 넣지 않는다.',
        '', 'Proposed의 Semantic Gradient는 각 변경 성분에 실제 step·Blue ID·원시 필드의 정확한 인용을 요구한다. 근거가 없으면 NEW fallback도 차단하고, 후보가 근거 없는 다른 성분을 바꾸거나 훈련 host/IP를 외우면 보류한다. 인용의 정확성은 의미 해석의 정답을 보장하지 않으므로 환경 검증을 추가한다.',
        '', '최소 경험 수는 임의의 고정값으로 활성화하지 않는다. 0·4·8·16·32의 검증 민감도를 기록하며, 한 후보에서 이 값들의 효과가 같으면 최적 cutoff가 식별됐다고 주장하지 않는다. 예산상 검증되지 않은 다른 부모는 유지한다.',
        '', '[후보별 수치·Skill 통계·phase/Red별 사용·공동 보상 노출](../research_outputs/rollout-v5/skillpro-audit.json). 이 노출 합계는 개별 Skill의 인과적 기여도가 아니다.']
    report.write_text('\n'.join(lines)+'\n');print(json.dumps({'passed':True,'gates':len(gates),'accepted':sum(g['accepted'] for g in gates),'report':str(report)},indent=2))


if __name__=='__main__':main()
