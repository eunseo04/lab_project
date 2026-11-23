"""Report exact event branches, paired divergences and limits without causal guessing."""
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research_outputs/rollout-v5/root-cause'


def main():
    rows=[];host_deltas=[];divergences=[];summaries={}
    queue=json.loads((ROOT/'research_outputs/behavior-v2/queue.json').read_text())
    for seed in range(22000,22004):
        cases={}
        traces={}
        for stage in ['baseline_context','static','loop2']:
            case=json.loads((OUT/f'{stage}-seed-{seed}.json').read_text());assert case['passed']
            assert sum(case['components'].values())==case['joint_return']
            cases[stage]=case
            task=next(t for t in queue if t['stage']==stage and t['seed']==seed)
            episode=json.loads(Path(task['summary']).read_text())['episodes'][0]
            trace=[json.loads(s) for s in Path(episode['trace']).read_text().splitlines()]
            traces[stage]={r['step']:r for r in trace if r['agent']=='blue_agent_4'}
            summaries[f'{stage}-{seed}']={k:v for k,v in case.items() if k not in ['events','executions']}
        base=cases['baseline_context'];proposed=cases['loop2']
        row={'seed':seed,'ACD':base['joint_return'],'Proposed':proposed['joint_return']}
        row.update({k:proposed['components'].get(k,0)-base['components'].get(k,0) for k in ['action_cost','RIA','LWF','ASF']})
        row['total_difference']=row['Proposed']-row['ACD'];assert sum(row[k] for k in ['action_cost','RIA','LWF','ASF'])==row['total_difference'];rows.append(row)
        counts={}
        for stage,case in cases.items():
            hosts=Counter()
            for event in case['events']:
                if event['kind']=='LWF':hosts[event['host']]+=event['reward']
            counts[stage]=hosts
        for host in sorted(counts['baseline_context'].keys()|counts['loop2'].keys()):
            if counts['baseline_context'][host]!=counts['loop2'][host]:host_deltas.append({'seed':seed,'host':host,'baseline_lwf':counts['baseline_context'][host],'proposed_lwf':counts['loop2'][host],'difference':counts['loop2'][host]-counts['baseline_context'][host]})
        for stage in ['static','loop2']:
            a,b=traces['baseline_context'],traces[stage]
            different=[s for s in a if a[s]['requested_action']!=b[s]['requested_action']]
            s=different[0]
            divergence={'seed':seed,'comparison':stage,'first_different_Qwen_request_step':s,
                'baseline':a[s]['requested_action'],'candidate':b[s]['requested_action'],
                'observation_before_equal':a[s]['observation_before']==b[s]['observation_before'],
                'later_state_divergence_is_not_a_controlled_action_ablation':True}
            patterns=Counter()
            for t in different:
                x,y=a[t]['requested_action'] or '',b[t]['requested_action'] or ''
                for label,first,second in [('Sleep_Restore','Sleep','Restore'),('Remove_Analyse','Remove','Analyse'),('Analyse_Decoy','Analyse','DeployDecoy')]:
                    if x.startswith(first) and y.startswith(second):patterns[label]+=1
            divergence['later_request_patterns']=dict(patterns);divergences.append(divergence)
    means={k:statistics.mean(r[k] for r in rows) for k in ['action_cost','RIA','LWF','ASF','total_difference']}
    assert means['LWF']==-50 and means['total_difference']==-56.25
    result={'passed':True,'episodes_audited':12,'paired_rows':rows,'mean_differences':means,
        'host_lwf_differences':host_deltas,'first_decision_divergences':divergences,'episode_diagnostics':summaries,
        'scope':'Historical v2 replays, not v5 performance or a policy-level causal estimate.'}
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,data in [('paired_reward_breakdown.csv',rows),('host_lwf_differences.csv',host_deltas)]:
        with (OUT/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    report=ROOT/'reports';report.mkdir(exist_ok=True)
    text=['# FiniteState 성능 저하의 사건별 원인 추적','',
        '기존 v2의 동일 이력 ACD·정적 Skill·최종 학습 Pool 각 4회, 총 12개 실제 궤적을 재생했다. 모든 Blue의 전후 관측·행동·공동 보상이 원본과 일치했다. 새로운 모델 성능 표본이 아니다.',
        '', '| Seed | ACD | 최종 Skill | 행동 비용 차이 | RIA 차이 | LWF 차이 | ASF 차이 | 총 차이 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:text.append('| '+' | '.join(str(r[k]) for k in ['seed','ACD','Proposed','action_cost','RIA','LWF','ASF','total_difference'])+' |')
    text+=['',f'평균 차이: {means}. 공식 공동 보상은 Blue 수만큼 중복 합산하지 않았다. 사건별 보상과 행동 비용의 합계도 각 원본 총점과 일치한다.',
        '', '## 실행 분기로 확인한 Green 실패','',
        'GreenLocalWork의 실제 선택 서비스·신뢰도·활성 상태와 마지막 서비스 변경 행동을 추적했다. Red DegradeServices 직후 서비스 신뢰도가 낮아졌고, 실제 실패가 그 신뢰도 검사 분기에서 반환된 경우만 `red_degraded_service_reliability`로 분류한다. GreenAccessService는 실제 연결 차단 분기를 확인한다. 0점 가중치 구역의 실패 사건도 보존하되 점수 손실과 구분한다.',
        '', '| 실행 | 시드 | 실행 분기별 원본 보상 |','|---|---:|---|']
    for key,d in summaries.items():text.append(f"| {key.rsplit('-',1)[0]} | {d['seed']} | {d['classified_reward']} |")
    text+=['','시드 22001의 최종 Skill은 public_access_zone_subnet_user_host_1에서 LWF −154를 기록했다. Red가 step 15·19·29·33·108에 서비스 신뢰도를 80·60·40·20·0으로 낮췄다. 첫 관련 실패는 step 16이었다. 해당 host의 Blue Restore나 Remove가 실행된 기록은 없었다. 이 host의 실패를 Restore 중단으로 설명할 근거가 없다.',
        '', 'Restore는 원본 구현에서 5-step 지연 후 backup 상태로 복원하며 직접 행동 비용 −1을 가진다. 대기 중이라는 이유로 서비스를 끄는 구현은 없다. Restore의 임무 가용성 영향은 이 직접 비용과 실제 서비스 상태·다른 행동을 못 하는 지연을 구분해야 한다.',
        '', '현재 GreenAccessService.execute는 `available_dest_service`를 호출하지 않고 bound method의 truth value를 확인한다. 따라서 이 환경에서 ASF를 일반적인 원격 서비스 신뢰도 실패로 해석할 수 없다. 환경은 사용자 조건대로 수정하지 않았다. 실제 ASF는 관측된 zone block 분기로 분류했다.',
        '', '## 처음 다른 결정을 한 시점','',
        '| Seed | 비교 | Step | ACD 요청 | Skill 요청 | 직전 자기 관측 동일 |','|---|---|---:|---|---|---|']
    for d in divergences:text.append(f"| {d['seed']} | {d['comparison']} | {d['first_different_Qwen_request_step']} | {d['baseline']} | {d['candidate']} | {d['observation_before_equal']} |")
    text+=['','후속 Sleep/Restore, Remove/Analyse, Analyse/Decoy 차이는 원시 JSON의 `later_request_patterns`에 보존했다. 첫 분기 이후에는 Red·Green 난수 소비와 상태가 달라질 수 있어, 동일 시드의 후속 차이를 한 행동의 인과 효과로 단정하지 않는다.',
        '', '## 확정된 오류와 남은 가설','',
        '- 이전 v3의 부분 연결 누락과 Remove/Restore evidence epoch 오류는 v4에서 수정돼 99개 회귀 테스트로 보존됐다. v5의 공통 관측·이력은 v4와 동일하게 유지한다.',
        '- 현재 원시 로그는 서비스 신뢰도 손상을 누가 일으켰는지 보여준다. Qwen이 더 일찍 조사·복구했다면 전체 손실이 얼마나 줄었는지는 별도 개입 검증 없이 확정할 수 없다.',
        '- 과도한 조사 억제, Skill 시작 조건의 충돌, 드문 복구 Skill의 적은 경험, 공동 RTG와 Skill 선택의 관계는 검증할 가설이다. 행동 빈도 자체를 새 보상으로 만들지 않는다.',
        '', '[사건별 실행·시드·phase·Skill·pending·보상과 CSV](../research_outputs/rollout-v5/root-cause/results.json). 각 에피소드 JSON은 Blue 5명의 당시 요청·완료 결과·행동 전 사건 이력, 실제 Red/Green 실행 순서와 서비스 상태를 포함한다.',
        '', '## 상태·난수 복원 검증','',
        '단순 deepcopy는 mappingproxy 직렬화 오류로 실패했다. cloudpickle로 전체 wrapper·controller·state·Red/Green agent와 외부 Python/NumPy/Torch RNG를 보존하는 방식은 한 상태에서 같은 5-step branch의 관측·보상·전체 State hash·내부 RNG가 일치했다. 검증 범위 밖의 장기 인과 효과를 주장하지 않는다. [검증 기록](../research_outputs/rollout-v5/root-cause/cloudpickle-snapshot-audit.json).']
    (report/'reward_root_cause.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({'passed':True,'episodes':12,'means':means,'report':str(report/'reward_root_cause.md')},indent=2))


if __name__=='__main__':main()
