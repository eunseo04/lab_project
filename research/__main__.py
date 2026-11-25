import argparse
from pathlib import Path
import json
import sys
from . import ROOT
from .config import load, validate, ARMS, REDS, VERIFICATION_VARIANTS, CONTROLLER_VARIANTS, variant_config


def resource_args(config,args):
    for key in ('endpoint','scoring_endpoints','generation_endpoints','sg_workers'):
        value=getattr(args,key,None)
        if value is not None:config['model'][key]=value
    if getattr(args,'rollout_ports',None) is not None and config.get('evolution',{}).get('rollout_verification'):
        config['rollout_validation']['ports']=args.rollout_ports
    return config


def main():
    import signal
    def terminate(signum,frame):raise KeyboardInterrupt('Worker interrupted; save state and clean owned children')
    signal.signal(signal.SIGTERM,terminate)
    p = argparse.ArgumentParser(description="Frozen-Qwen ACD / Skill-Pro / CAGE4 research pipeline")
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("evaluate", "train", "compare", "official"):
        s = sub.add_parser(command)
        s.add_argument("--config", default=str(ROOT/"research_configs/rollout-B0.yaml"))
        s.add_argument('--variant',choices=[*VERIFICATION_VARIANTS,*CONTROLLER_VARIANTS],help='Named verification ablation built from the existing configs')
        s.add_argument("--arm", choices=ARMS)
        s.add_argument("--red", choices=REDS)
        s.add_argument("--composition", choices=("hybrid", "all_llm", "no_blue", "all_rl"))
        s.add_argument("--episodes", type=int)
        s.add_argument("--train-episodes", type=int)
        s.add_argument("--iterations", type=int)
        s.add_argument("--steps", type=int)
        s.add_argument("--pool")
        s.add_argument("--endpoint", help="Explicit local policy server; model identity must still match")
        s.add_argument('--scoring-endpoints',nargs='+')
        s.add_argument('--generation-endpoints',nargs='+')
        s.add_argument('--sg-workers',type=int)
        s.add_argument('--rollout-ports',type=int,nargs='+')
        s.add_argument("--output")
        s.add_argument("--resume", action="store_true")
        s.add_argument("--limit", type=int, help="Maximum new episodes in this invocation; plan and resume remain intact")
        if command == "compare":
            s.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS))
            s.add_argument('--variants',choices=['B0','B1','B2','B3-no-support','B3','B4',*VERIFICATION_VARIANTS,*CONTROLLER_VARIANTS],nargs='+',
                help='Use the registered public configs and distinct ablation IDs, rather than changing internal E arms')
            s.add_argument("--reds", choices=REDS, nargs="+", default=list(REDS))
            s.add_argument("--train", action="store_true")
    sub.add_parser("doctor")
    s = sub.add_parser("report"); s.add_argument("directory"); s.add_argument("--reasoning", action="store_true")
    sub.add_parser("test")
    args = p.parse_args()
    if args.command in ("train", "evaluate", "compare", "official"):
        c = variant_config(args.variant) if args.variant else load(args.config)
        for key in ("arm", "red", "composition", "episodes", "train_episodes", "iterations", "steps", "pool"):
            if getattr(args, key, None) is not None: c[key] = getattr(args, key)
        resource_args(c,args)
        c = validate(c)
        if args.command=='compare' and args.output:
            c['output']=str(Path(args.output).resolve())
        if args.command == "official":
            from .official import run_official
            print(json.dumps(run_official(c, args.output or "research_outputs/official-reference")))
            return
        from .experiment import run
        if args.command == "compare":
            if args.variants:
                base=Path(args.output).resolve() if args.output else ROOT/'research_outputs/variant-comparison'
                for red in args.reds:
                    for variant in args.variants:
                        case=variant_config(variant)
                        for key in ('composition','episodes','train_episodes','iterations','steps','pool'):
                            if getattr(args,key,None) is not None:case[key]=getattr(args,key)
                        resource_args(case,args)
                        case.update(red=red,output=str(base));case=validate(case)
                        directory=base/variant/red
                        if args.train and case['arm'] in ('E2','E3'):
                            training=run(case,'train',directory/'train',resume=args.resume,limit=args.limit)
                            if training['status']!='completed':continue
                            case['pool']=str(directory/'train/checkpoint.json')
                        run(case,'eval',directory/'eval',resume=args.resume,limit=args.limit)
                from scripts.report_experiment import report_experiment
                report_experiment(base)
                return
            for red in args.reds:
                for arm in args.arms:
                    case = {**c, "arm": arm, "red": red, "comparison_id": arm}
                    if args.train and arm in ("E2", "E3"):
                        training = run(case, "train", resume=args.resume, limit=args.limit)
                        if training["status"] != "completed":
                            print(f"{arm}/{red}: training batch saved; frozen evaluation will follow after training completes", flush=True)
                            continue
                    run(case, "eval", resume=args.resume, limit=args.limit)
            from .reporting import report
            report(c["output"])
        else:
            run(c, "train" if args.command == "train" else "eval", args.output, args.resume, args.limit)
    elif args.command == "doctor":
        from .client import ModelClient
        c = load(ROOT/"research_configs/rollout-B0.yaml")
        import importlib.metadata as m
        import shutil
        output = {"workspace": str(ROOT), "cpu_environment": sys.prefix,
            "versions": {k: m.version(k) for k in ("torch", "ray", "numpy", "torch_geometric")},
            "free_disk_gib": round(shutil.disk_usage(ROOT).free/1024**3, 2),
            "sources": json.loads((ROOT/"research_sources.lock.json").read_text())}
        try: output["local_qwen"] = ModelClient(c["model"]).identity
        except Exception as exc: output["local_qwen_error"] = str(exc)
        print(json.dumps(output, indent=2))
    elif args.command == "report":
        from scripts.report_experiment import report_experiment
        print(json.dumps(report_experiment(args.directory, include_trace=args.reasoning)))
    elif args.command == "test":
        import subprocess
        sys.exit(subprocess.call([sys.executable, "-m", "pytest", "tests", "research_tests", "-q", "--junitxml=research_outputs/tests.xml"], cwd=ROOT))


if __name__ == "__main__":
    try: main()
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
