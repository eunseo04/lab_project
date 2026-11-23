"""Resolve firewall owner from recorded execution order and native edge orientation."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_outputs/rollout-v5/block-root-cause'


def main():
    rows=[]
    for path in sorted(OUT.glob('*-seed-*.json')):
        data=json.loads(path.read_text());active={};owners_by_execution={}
        for r in data['executions']:
            if r['name'] in ['BlockTrafficZone','AllowTrafficZone'] and r['success']=='TRUE':
                edge=(r['to_subnet'],r['from_subnet'])
                if r['name']=='BlockTrafficZone':active[edge]={k:r[k] for k in ['step','order','agent','from_subnet','to_subnet']}
                else:active.pop(edge,None)
            if r['name']=='GreenAccessService' and r.get('connection_failure_flag'):
                owners_by_execution[r['order']]=[{**edge,'owner':active.get((edge['from_subnet'],edge['to_subnet']))}
                    for edge in r['blocking_edges']]
        totals=Counter();events=[]
        for e in data['events']:
            if e['kind']!='ASF':continue
            edges=owners_by_execution[e['green_execution']['order']]
            agents=sorted({x['owner']['agent'] if x['owner'] else 'UNKNOWN' for x in edges})
            totals['+'.join(agents) or 'UNKNOWN']+=e['reward']
            events.append({'step':e['step'],'phase':e['phase'],'host':e['host'],'reward':e['reward'],'edges':edges})
        assert sum(totals.values())==data['components'].get('ASF',0)
        rows.append({'case':path.stem,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'ASF':sum(totals.values()),'exclusive_actor_group_rewards':dict(totals),'events':events})
    source=ROOT/'runtime/CybORG/Simulator/Actions/ConcreteActions/ControlTraffic.py'
    result={'passed':True,'cases':rows,'native_storage_rule':'state.blocks[to_subnet].append(from_subnet)',
        'native_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'multi_edge_rewards_counted_once':True,'scope':'Executed firewall failure branch; not a full-policy causal effect.'}
    previous=OUT/'attribution.json'
    if previous.exists() and not (OUT/'attribution-before-edge-orientation-fix.json').exists():previous.rename(OUT/'attribution-before-edge-orientation-fix.json')
    previous.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k!='events'} for r in rows],indent=2))


if __name__=='__main__':main()
