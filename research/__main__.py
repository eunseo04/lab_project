import argparse
from pathlib import Path
import json
import sys
from . import ROOT
from .config import load, validate, ARMS, REDS


def main():
    p = argparse.ArgumentParser(description="Frozen-Qwen ACD / Skill-Pro / CAGE4 research pipeline")
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("evaluate", "train", "compare", "official"):
        s = sub.add_parser(command)
        s.add_argument("--config", default=str(ROOT/"research_configs/rollout-B0.yaml"))
        s.add_argument("--arm", choices=ARMS)
        s.add_argument("--red", choices=REDS)
        s.add_argument("--composition", choices=("hybrid", "all_llm", "no_blue", "all_rl"))
        s.add_argument("--episodes", type=int)
        s.add_argument("--train-episodes", type=int)
        s.add_argument("--iterations", type=int)
        s.add_argument("--steps", type=int)
        s.add_argument("--pool")
        s.add_argument("--endpoint", help="Explicit local policy server; model identity must still match")
        s.add_argument("--output")
        s.add_argument("--resume", action="store_true")
        s.add_argument("--limit", type=int, help="Maximum new episodes in this invocation; plan and resume remain intact")
        if command == "compare":
            s.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS))
            s.add_argument("--reds", choices=REDS, nargs="+", default=list(REDS))
            s.add_argument("--train", action="store_true")
    sub.add_parser("doctor")
    s = sub.add_parser("report"); s.add_argument("directory"); s.add_argument("--reasoning", action="store_true")
    sub.add_parser("test")
    args = p.parse_args()
    if args.command in ("train", "evaluate", "compare", "official"):
        c = load(args.config)
        for key in ("arm", "red", "composition", "episodes", "train_episodes", "iterations", "steps", "pool"):
            if getattr(args, key, None) is not None: c[key] = getattr(args, key)
        if args.endpoint: c["model"]["endpoint"] = args.endpoint
        c = validate(c)
        if args.command=='compare' and args.output:
            from pathlib import Path
            c['output']=str(Path(args.output).resolve())
        if args.command == "official":
            from .official import run_official
            print(json.dumps(run_official(c, args.output or "research_outputs/official-reference")))
            return
        from .experiment import run
        if args.command == "compare":
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
