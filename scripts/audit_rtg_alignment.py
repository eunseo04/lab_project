"""Independently reconstruct shared RTG, EMA baseline and every gate advantage."""
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_outputs/rollout-v5'

def read(path):return json.loads(Path(path).read_text())
def rtg(rewards,gamma):
    values=[];total=0.
    for reward in reversed(rewards):total=float(reward)+gamma*total;values.append(total)
    return list(reversed(values))

def audit(directory):
    summary=read(directory/'summary.json');cfg=summary['config'];histories=summary.get('history',[])
    if not histories:return {'status':'no_committed_iteration'}
    episodes={int(p.stem[-4:]):read(p) for p in directory.glob('episode-????.json')}
    returns={};rows={};maximum_rtg_difference=0.
    for episode in episodes.values():
        assert episode['joint_transitions']==499 and episode['scored_transitions']==498
        values=episode['scored_step_rewards'];assert len(values)==499 and values[-1]==0
        assert sum(values)==episode['joint_return']
        independently=rtg(values,cfg['evolution']['gamma'])
        returns[episode['seed']]=independently
        for line in Path(episode['trace']).read_text().splitlines():
            row=json.loads(line);key=(row['seed'],row['agent'],row['step']);rows[key]=row
            if row['policy']=='llm':
                difference=abs(independently[row['step']]-row['return_to_go'])
                maximum_rtg_difference=max(maximum_rtg_difference,difference)
                assert difference<1e-10
    baseline=None;previous=0;checked=0;hist=[]
    for h in histories:
        batch=[episodes[i] for i in range(previous,h['episodes'])];previous=h['episodes']
        rtgs=[value for e in batch for value in rtg(e['scored_step_rewards'][:498],cfg['evolution']['gamma'])]
        measured=statistics.mean(rtgs);alpha=cfg['evolution']['ema_alpha']
        baseline=measured if baseline is None else (1-alpha)*baseline+alpha*measured
        assert abs(baseline-h['baseline'])<1e-10
        samples=0
        for parent in h['evolution']:
            for candidate in parent['candidates']:
                for d in candidate.get('diagnostics',[]):
                    for t,advantage in zip(d['transitions'],d['advantages']):
                        key=(t['seed'],t['agent'],t['step']);row=rows[key]
                        assert row['skill_id']==t['skill_id']==parent['parent']
                        assert row['policy_invoked'] and row['request_applied'] and not row['parse_error']
                        expected=returns[t['seed']][t['step']]-baseline
                        assert abs(expected-advantage)<1e-10
                        samples+=1;checked+=1
        hist.append({'iteration':h['iteration'],'baseline':baseline,'gate_advantages_checked':samples})
    return {'passed':True,'source':str(directory),'loops':hist,'gate_advantages_checked':checked,
        'max_trace_rtg_difference':maximum_rtg_difference,
        'official_joint_rewards_counted_once_per_tick':True,'causal_individual_credit_claimed':False}

def main():
    directories={'historical_v2':ROOT/'research_outputs/behavior-v2/E3-hybrid-finite/train'}
    directories.update({label:OUT/label/'train' for label in ['B2','B3','B4']})
    result={label:audit(path) for label,path in directories.items() if (path/'summary.json').exists()}
    (OUT/'rtg_alignment_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
