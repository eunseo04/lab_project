"""Read-only campaign monitor; periodically refresh audits, never change a policy."""
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_outputs/rollout-v5'

def read(path):return json.loads(path.read_text())

def snapshot():
    budget=read(OUT/'budget.json');training={};validation={};running=[]
    for label in ('B2','B3','B4'):
        path=OUT/label/'train/summary.json'
        summary=read(path) if path.exists() else {}
        training[label]={'episodes':len(list((OUT/label/'train').glob('episode-????.json'))),
            'committed_loops':len(summary.get('history',[])),'status':summary.get('status')}
        for base in (OUT/label/'train/rollout-verification',OUT/'recovery'/label/'train/rollout-verification'):
            for case in base.glob('*'):
                if not case.is_dir():continue
                counts=Counter()
                for d in case.glob('*-seed-*'):
                    path=d/'summary.json'
                    if path.exists() and read(path).get('status')=='completed':counts[d.name.split('-seed-')[0]]+=1
                    elif (d/'episode-0000.partial.jsonl').exists():
                        lines=(d/'episode-0000.partial.jsonl').read_text().splitlines()
                        if lines:
                            row=json.loads(lines[-1]);running.append(f'{label}/{d.name}:{row["step"]}')
                if counts or base.parts[-4:-3]==('recovery',):validation[str(case.relative_to(OUT))]=dict(counts)
    queue=read(OUT/'queue.json') if (OUT/'queue.json').exists() else []
    final=Counter(f"{q['red']}:{q['label']}" for q in queue if q['status']=='completed')
    return {'elapsed_min':round((time.time()-budget['started_unix'])/60,2),
        'remaining_min':round(max(0,budget['model_deadline_unix']-time.time())/60,2),
        'training':training,'validation':validation,'running_validation_steps':running,
        'final_completed':dict(final),'queue':dict(Counter(q['status'] for q in queue)),
        'terminal':(OUT/'campaign_complete.json').exists()}

def refresh():
    results=[]
    for script in ('audit_rtg_alignment.py','audit_semantic_catalog.py','report_rollout_v5.py','report_closed_loop_findings.py'):
        process=subprocess.run([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,
            capture_output=True,text=True,timeout=180)
        results.append({'script':script,'exit_code':process.returncode,
            'error':process.stderr[-1800:] if process.returncode else None})
    path=OUT/'monitor-audit-refresh.json';path.write_text(json.dumps({'updated_unix':time.time(),'results':results},indent=2)+'\n')
    if any(r['exit_code'] for r in results):print(json.dumps({'audit_refresh_failed':results}),flush=True)

def main():
    refreshed=0
    while True:
        status=snapshot();print(json.dumps(status,separators=(',',':')),flush=True)
        if status['terminal'] or time.time()-refreshed>=600:
            refresh();refreshed=time.time()
        if status['terminal']:break
        time.sleep(50)

if __name__=='__main__':main()
