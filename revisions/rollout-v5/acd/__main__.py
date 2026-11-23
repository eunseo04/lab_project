import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

from . import ROOT
from .config import load_config, validate, TEAMS, REDS


def overrides(config, args):
    for name in ("episodes", "steps", "seed", "team", "red", "threads", "weights", "communication", "horizon"):
        if getattr(args, name, None) is not None:
            config[name] = getattr(args, name)
    return validate(config)


def main():
    parser = argparse.ArgumentParser(description="CC4 paper/KEEP local experiments (API disabled)")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("evaluate", "matrix", "train"):
        p = commands.add_parser(command)
        default_config = "configs/train_paper_smoke.yaml" if command == "train" else "configs/smoke.yaml"
        p.add_argument("--config", default=str(ROOT / default_config))
        p.add_argument("--output")
        p.add_argument("--steps", type=int)
        p.add_argument("--seed", type=int)
        p.add_argument("--threads", type=int)
        p.add_argument("--weights")
        p.add_argument("--communication", choices=("keep", "paper"))
        p.add_argument("--horizon", choices=("native", "fixed"))
        p.add_argument("--red", choices=REDS)
        if command != "train":
            p.add_argument("--episodes", type=int)
            p.add_argument("--team", choices=TEAMS)
        if command == "matrix":
            p.add_argument("--teams", choices=TEAMS, nargs="+", default=list(TEAMS))
            p.add_argument("--reds", choices=REDS, nargs="+", default=list(REDS))
    commands.add_parser("doctor")
    p = commands.add_parser("plot")
    p.add_argument("directory")
    p = commands.add_parser("export-reasons")
    p.add_argument("trace")
    p.add_argument("--output", required=True)
    p.add_argument("--agent", default="blue_agent_4")
    p.add_argument("--episode", type=int, default=0)
    p = commands.add_parser("cluster-reasons")
    p.add_argument("input")
    p.add_argument("--embeddings", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--clusters", type=int, default=4)
    args = parser.parse_args()
    if args.command in ("evaluate", "matrix", "train"):
        config = overrides(load_config(args.config), args)
        from .runner import evaluate, write_json
        if args.command == "train":
            from .training import train
            train(config, args.output)
        elif args.command == "evaluate":
            evaluate(config, args.output)
        else:
            stamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S-%f")
            output = Path(args.output or Path(config["output"]) / f"{stamp}-matrix")
            output.mkdir(parents=True, exist_ok=False)
            rows = []
            for red in args.reds:
                for team in args.teams:
                    case = {**deepcopy(config), "red": red, "team": team, "name": f"{team}-{red}"}
                    folder, summary = evaluate(case, output / case["name"])
                    rows.append({"team": team, "red": red, "summary": str(folder / "summary.json"),
                                 "result_kind": summary["result_kind"], "reward": summary["reward"]})
                    write_json(output / "matrix.json", rows)
            from .analysis import plot_results
            plot_results(output)
            print("Matrix complete:", output)
    elif args.command == "doctor":
        import importlib.metadata as metadata
        import platform
        import shutil
        import torch
        from .environment import make_env
        versions = {name: metadata.version(name) for name in
                    ("torch", "torch-geometric", "ray", "numpy", "pyarrow", "joblib", "tqdm")}
        features = {}
        for protocol in ("keep", "paper"):
            env = make_env(seed=42, steps=6, communication=protocol)
            obs, _ = env.reset(seed=42)
            features[protocol] = {"expected": env.feature_dimension,
                                 "actual": obs["blue_agent_4"][0][0].shape[1]}
        print(json.dumps({"python": platform.python_version(), "environment": sys.prefix,
            "dependencies": versions, "features": features,
            "disk_free_gib": round(shutil.disk_usage(ROOT).free / 1024 ** 3, 1),
            "model_api_code": "commented out", "wandb": "not imported by local runner",
            "sources": json.loads((ROOT / "sources.lock.json").read_text())}, indent=2))
    elif args.command == "plot":
        from .analysis import plot_results
        print("Plotted", len(plot_results(args.directory)), "results")
    elif args.command == "export-reasons":
        from .analysis import export_reasons
        print("Exported", export_reasons(args.trace, args.output, args.agent, args.episode), "reasons")
    elif args.command == "cluster-reasons":
        from .analysis import cluster_reasons
        result = cluster_reasons(args.input, args.embeddings, args.output, args.clusters)
        print("Clustered", result["samples"], "reasons")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
