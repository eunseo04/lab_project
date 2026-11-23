"""Verify whole-environment snapshots before making any intervention claims."""
import cloudpickle
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from research.adapter import make_environment,plain,Sleep
from research.config import load
from research.experiment import seed_everything,BLUE_AGENTS


def main():
    c=load('research_configs/behavior-v2.yaml');seed_everything(22001,1)
    env=make_environment(c,22001);env.reset(seed=22001)
    actions={a:Sleep() for a in BLUE_AGENTS}
    for _ in range(8):env.step(actions,messages=env.paper_messages())
    snapshot=cloudpickle.dumps(env)
    gs=(random.getstate(),np.random.get_state(),torch.get_rng_state())
    def branch():
        random.setstate(gs[0]);np.random.set_state(gs[1]);torch.set_rng_state(gs[2])
        e=cloudpickle.loads(snapshot);rewards=[]
        for _ in range(5):
            _,r,*_=e.step(actions,messages=e.paper_messages());rewards.append(r)
        ec=e.env.environment_controller
        return {'raw':plain(e.raw_observations()),'rewards':rewards,
            'state_sha256':hashlib.sha256(cloudpickle.dumps(ec.state)).hexdigest(),
            'rng':plain(ec.state.np_random.bit_generator.state)}
    a,b=branch(),branch();assert a==b
    result={'passed':True,'snapshot':'cloudpickle full environment/controller/state/Red-Green agents',
        'identical_five_step_branches':True,'tested_seed':22001,'snapshot_after_steps':8,
        'size_bytes':len(snapshot),'external_rng_restored':['random','numpy','torch CPU'],
        'compared':['raw Blue observations','joint rewards','full serialized simulator State','internal RNG state'],
        'new_model_performance_sample':False,
        'limitation':'One tested state and five steps; future policy effects still require matched intervention replay.'}
    target=Path('research_outputs/rollout-v5/root-cause/cloudpickle-snapshot-audit.json')
    target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
