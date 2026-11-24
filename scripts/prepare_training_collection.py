"""Create a versioned training-only configuration for additional independent seeds."""
import argparse
from pathlib import Path
import re
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.config import validate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-id',required=True)
    parser.add_argument('--arm',choices=['B2','B3','B4'],default='B3')
    parser.add_argument('--train-seed',type=int,required=True)
    parser.add_argument('--train-episodes',type=int,default=12)
    parser.add_argument('--iterations',type=int,default=3)
    parser.add_argument('--red',choices=['finite','aggressive','stealthy','impact','degrade'],default='finite')
    parser.add_argument('--pool',type=Path)
    parser.add_argument('--support-ablation',action='store_true',help='Disable only the Proposed minimum-episode guard in a separately named study')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',args.experiment_id):parser.error('Use a simple unique experiment ID')
    proposed=set(range(args.train_seed,args.train_seed+args.train_episodes))
    reserved=set(range(97000,97020))|{99000,102000}|set(range(101000,101100))
    if proposed & reserved:parser.error('Additional training seeds overlap registered validation/test/generalization seeds')
    original=ROOT/f'research_configs/rollout-{args.arm}.yaml'
    cfg=yaml.safe_load(original.read_text())
    cfg.update(experiment_version='cage4-collection:'+args.experiment_id,train_seed=args.train_seed,
        train_episodes=args.train_episodes,iterations=args.iterations,red=args.red,
        output=str(ROOT/'research_outputs'/args.experiment_id))
    if args.pool:cfg['pool']=str(args.pool.resolve())
    if args.support_ablation:
        cfg['evolution']['minimum_independent_episodes']=0
        cfg['ablation']='minimum_independent_episode_guard_disabled'
    cfg=validate(cfg)
    directory=ROOT/'.local-configs/collections';directory.mkdir(exist_ok=True,parents=True)
    destination=directory/f'{args.experiment_id}-{args.arm}.yaml'
    with destination.open('x') as stream:yaml.safe_dump(cfg,stream,sort_keys=False)
    print(destination)
    print(f'./research.sh train --config {destination}')
    print('Training only; no model or simulation was started. Original test/validation splits remain reserved.')


if __name__=='__main__':main()
