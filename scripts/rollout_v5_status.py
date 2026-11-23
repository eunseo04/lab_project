"""Small read-only progress snapshot for the budgeted local experiment."""
from collections import Counter
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_outputs/rollout-v5'


def main():
    budget=json.loads((OUT/'budget.json').read_text());now=time.time()
    result={'elapsed_minutes':round((now-budget['started_unix'])/60,2),
        'model_minutes_remaining':round(max(0,budget['model_deadline_unix']-now)/60,2),
        'validation_minutes_remaining':round(max(0,budget['validation_deadline_unix']-now)/60,2),
        'training':{},'debug_completed':len(list((OUT/'debugging').glob('*/summary.json')))}
    for label in ['B2','B3','B4']:
        train=OUT/label/'train';summary=train/'summary.json';value=json.loads(summary.read_text()) if summary.exists() else {}
        log=OUT/f'{label}-train.log';lines=log.read_text().splitlines() if log.exists() else []
        validations=[]
        for case in (train/'rollout-verification').glob('*'):
            if not case.is_dir():continue
            counts=Counter(path.parent.name.split('-seed-')[0] for path in case.glob('*/summary.json')
                if json.loads(path.read_text()).get('status')=='completed')
            decision=case/'decision.json'
            validations.append({'case':case.name,'completed':dict(counts),
                'decision':{k:v for k,v in json.loads(decision.read_text()).items() if k in
                    ['accepted','reason','mean_difference','win_rate','previous_n','candidate_n']} if decision.exists() else None})
        for case in (OUT/'recovery'/label/'train/rollout-verification').glob('*'):
            if not case.is_dir():continue
            counts=Counter(path.parent.name.split('-seed-')[0] for path in case.glob('*/summary.json')
                if json.loads(path.read_text()).get('status')=='completed')
            decision=case/'decision.json'
            validations.append({'metadata_repair':True,'case':case.name,'completed':dict(counts),
                'decision':{k:v for k,v in json.loads(decision.read_text()).items() if k in
                    ['accepted','reason','mean_difference','win_rate','previous_n','candidate_n']} if decision.exists() else None})
        result['training'][label]={'summary_status':value.get('status','not_started'),
            'completed_500_step_training_episodes':len(list(train.glob('episode-????.json'))),
            'committed_iterations':len(value.get('history',[])),
            'last_progress_line':lines[-1] if lines else None,'validations':validations}
    queue=OUT/'queue.json'
    if queue.exists():result['final_queue']=dict(Counter(t['status'] for t in json.loads(queue.read_text())))
    result['campaign_terminal']=(OUT/'campaign_complete.json').exists()
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
