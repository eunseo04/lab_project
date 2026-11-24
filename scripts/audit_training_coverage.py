"""Recover Skill coverage from completed training traces even before an optimizer commits."""
from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.attribution import support_statistics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',type=Path,default=ROOT/'research_outputs/pilot')
    args=parser.parse_args();pilot=args.pilot.resolve();records=[];episode_runs=0
    for label in ['B2','B3','B4']:
        groups=defaultdict(list);sources=[]
        cfg=json.loads((pilot/f'{label}-config.json').read_text())
        initial=json.loads(Path(cfg['pool']).read_text())
        for skill in initial.get('pool',initial).get('skills',[]):groups[skill['name']]=[]
        for path in sorted((pilot/f"runs/{label}-{cfg['red']}-train").glob('episode-????.json')):
            episode=json.loads(path.read_text())
            if episode.get('status')!='completed' or episode.get('kind')!='real_cage4':continue
            episode_runs+=1
            trace=Path(episode['trace']);content=trace.read_bytes()
            sources.append({'path':str(trace.relative_to(ROOT)),'sha256':hashlib.sha256(content).hexdigest()})
            local=defaultdict(list)
            for line in content.splitlines():
                row=json.loads(line)
                if row.get('policy')=='llm' and row.get('skill_id'):local[row['skill_id']].append(row)
            for skill,rows in local.items():groups[skill].append(SimpleNamespace(transitions=rows,env_name=f"{cfg['composition']}:{cfg['red']}"))
        for skill,experiences in groups.items():
            records.append({'label':label,'skill':skill,**support_statistics(experiences),
                'raw_training_sources':sources,'source':'completed_training_traces',
                'semantic_gradient_count':None,'ppo_gate_positive_candidates':None,
                'rollout_approved':None,'rollout_rejected':None,'automatic_updates':None,
                'deferred_reason':None,'error':None,
                'optimizer_metric_note':'Unknown until a committed optimizer record or individual rollout decision is present.'})
    target=pilot/'report'/'training_skill_coverage.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps({'pilot':str(pilot),'records':records,
        'independence_unit':'episode seed within each arm; arms share the same paired training seeds'},indent=2)+'\n')
    print(json.dumps({'skill_arm_records':len(records),'total_trace_episode_runs':episode_runs}))


if __name__=='__main__':main()
