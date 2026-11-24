"""Count planned SG chunks from real training records, without any model calls."""
from collections import defaultdict
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',type=Path,default=ROOT/'research_outputs/pilot')
    args=parser.parse_args();pilot=args.pilot.resolve()
    fields=('step','agent','observation_before','requested_action','executed_actions','action_status',
        'observation_after','joint_reward','skill_id','skill_version','skill_terminated','response',
        'parse_error','policy_invoked','request_applied','behavior_context','behavior_outcome')
    output=[]
    for label in ['B2','B3','B4']:
        cfg=json.loads((pilot/f'{label}-config.json').read_text())
        experiences=defaultdict(list)
        for path in sorted((pilot/f"runs/{label}-{cfg['red']}-train").glob('episode-????.jsonl')):
            grouped=defaultdict(list)
            with path.open() as stream:
                for row in map(json.loads,stream):
                    if row.get('policy')=='llm' and row.get('skill_id'):grouped[row['skill_id']].append(row)
            for name,rows in grouped.items():experiences[name].append(rows)
        for name,trajectories in experiences.items():
            chunks=chars=decisions=0
            for rows in trajectories:
                pending=used=0
                for row in rows:
                    record={k:row.get(k) for k in fields if k in row}
                    size=len(json.dumps(record,ensure_ascii=False));chars+=size
                    decisions+=bool(row.get('policy_invoked') and row.get('request_applied') and not row.get('parse_error') and row.get('token_ids'))
                    if pending and (pending>=cfg['evolution']['semantic_chunk_steps'] or used+size>cfg['evolution']['semantic_chunk_chars']):
                        chunks+=1;pending=used=0
                    pending+=1;used+=size
                chunks+=bool(pending)
            output.append({'label':label,'skill':name,'independent_training_episodes':len(trajectories),
                'valid_policy_decisions':decisions,'record_characters':chars,'planned_semantic_gradient_chunks':chunks,
                'minimum_episode_guard_eligible':len(trajectories)>=cfg['evolution']['minimum_independent_episodes']})
    report={'pilot':str(pilot),'records':output,'actual_model_call_count':None,
        'interpretation':'Planned per-experience chunk counts using the recorded width/character budget. Interrupted model calls are not fabricated. Cache reuse across Best-of-N avoids repeating the same per-experience SG; aggregation/candidate/scoring calls cost extra.',
        'new_performance_samples':0}
    target=pilot/'report'/"training_cost.json"
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({label:{'planned_eligible_sg_chunks':sum(r['planned_semantic_gradient_chunks'] for r in output if r['label']==label and r['minimum_episode_guard_eligible']),
        'characters':sum(r['record_characters'] for r in output if r['label']==label)} for label in ['B2','B3','B4']}))


if __name__=='__main__':main()
