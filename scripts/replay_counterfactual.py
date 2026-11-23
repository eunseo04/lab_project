"""Short open-loop interventions at the first real policy divergence, not new scores."""
from copy import deepcopy
import cloudpickle
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
import replay_behavior_pipeline as replay
from research.adapter import make_environment,plain
from research.experiment import seed_everything
from CybORG.Simulator.Actions import Sleep,Analyse

ROOT=replay.ROOT;OUT=ROOT/'research_outputs/rollout-v5/counterfactual'
factory=make_environment


def run(seed):
    summary=ROOT/f'research_outputs/behavior-v2/paired/loop2-seed-{seed}/summary.json'
    data=json.loads(summary.read_text());cfg=data['config']
    original=[json.loads(s) for s in Path(data['episodes'][0]['trace']).read_text().splitlines()]
    baseline=json.loads((ROOT/f'research_outputs/behavior-v2/paired/baseline_context-seed-{seed}/summary.json').read_text())
    base={r['step']:r for r in (json.loads(s) for s in Path(baseline['episodes'][0]['trace']).read_text().splitlines()) if r['agent']=='blue_agent_4'}
    own={r['step']:r for r in original if r['agent']=='blue_agent_4'}
    step=next(s for s in own if own[s]['requested_action']!=base[s]['requested_action'])
    batches=[];snapshots={}
    def capture(config,case_seed):
        env=factory(config,case_seed);native=env.step
        def wrapped(actions,messages=None):
            if len(batches)==step:
                # Snapshot the ACTUAL replay state, including wrapper caches and
                # RNG consumption from prior controller decisions. Remove only
                # instrumentation closures, then restore them before execution.
                overrides={key:env.__dict__.pop(key) for key in ('reset','step') if key in env.__dict__}
                calc=env.env.environment_controller.team_reward_calculators['Blue']['BlueRewardMachine']
                observer=calc.__dict__.pop('calculate_reward',None)
                try:
                    snapshots['env']=cloudpickle.dumps(env)
                    snapshots['rng']=(random.getstate(),np.random.get_state(),torch.get_rng_state())
                finally:
                    env.__dict__.update(overrides)
                    if observer is not None:calc.calculate_reward=observer
            batches.append((deepcopy(actions),deepcopy(messages)))
            return native(actions,messages=messages)
        env.step=wrapped;return env
    replay.make_environment=capture;replay.ARTIFACT_ROOT=OUT
    replay.replay(summary)
    snapshot=snapshots['env'];rng=snapshots['rng'];env=cloudpickle.loads(snapshot)
    observed=plain(env.raw_observations()['blue_agent_4']);expected=own[step]['observation_before']
    if observed!=expected:
        mismatch={k:{'snapshot':observed.get(k),'recorded':expected.get(k)} for k in observed.keys()|expected.keys() if observed.get(k)!=expected.get(k)}
        (OUT/f'failed-snapshot-{seed}.json').write_text(json.dumps({'passed':False,'seed':seed,'step':step,
            'differences':mismatch,'no_counterfactual_reward_reported':True},indent=2)+'\n')
        raise AssertionError('Snapshot observation differs; see failed snapshot audit')
    alternative=base[step]['requested_action']
    if alternative=='Sleep':replacement=Sleep()
    elif alternative.startswith('Analyse '):replacement=Analyse(agent='blue_agent_4',session=0,hostname=alternative.split(' ',1)[1])
    else:raise ValueError('Unsupported first-divergence alternative; never substitute another action')
    def branch(replace):
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])
        fork=cloudpickle.loads(snapshot);values=[];observations=[]
        for index in range(step,min(step+5,len(batches))):
            actions,messages=deepcopy(batches[index])
            if index==step and replace:actions['blue_agent_4']=deepcopy(replacement)
            _,rewards,*_=fork.step(actions,messages=messages)
            joint=list(rewards.values())[0];assert len(set(rewards.values()))==1
            values.append(joint);observations.append(plain(fork.raw_observations()))
        return {'rewards':values,'observations':observations,'state_sha256':hashlib.sha256(cloudpickle.dumps(fork.env.environment_controller.state)).hexdigest()}
    actual=branch(False);repeat=branch(False);assert actual==repeat
    for offset,raw in enumerate(actual['observations']):
        for row in original:
            if row['step']==step+offset:assert raw[row['agent']]==row['observation_after']
        assert actual['rewards'][offset]==own[step+offset]['joint_reward']
    counter=branch(True)
    result={'passed':True,'seed':seed,'step':step,'actual_action':own[step]['requested_action'],
        'intervention':alternative,'horizon':len(actual['rewards']),'actual_rewards':actual['rewards'],
        'intervention_rewards':counter['rewards'],'joint_reward_difference':sum(counter['rewards'])-sum(actual['rewards']),
        'repeated_same_action_same_full_state':True,'actual_branch_matches_recorded_all_blue_observations_and_rewards':True,
        'snapshot_sha256':hashlib.sha256(snapshot).hexdigest(),'new_model_performance_sample':False,
        'causal_scope':'First action changed in identical simulator and RNG state; subsequent Blue requests and messages frozen to recorded sequence for five steps. Does not estimate adaptive future-policy or episode-level effect.'}
    (OUT/f'seed-{seed}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('seed',type=int)
    OUT.mkdir(parents=True,exist_ok=True);run(parser.parse_args().seed)
