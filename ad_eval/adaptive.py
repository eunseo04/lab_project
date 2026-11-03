"""Commit trajectory GRPO updates and new vLLM role policies before the next match."""
from __future__ import annotations

from dataclasses import asdict
import json
import hashlib
import math
import os
from pathlib import Path
import subprocess

from .docker_lab import ROOT
from .learning import empty_planner_state, trajectory_reward, update_planner
from .models import ModelSpec, checkpoint_info, load_adapter, select_model, verify_model
from .llm import served_model
from .trajectory import export_trajectories
from .grpo_rewards import config_from_args


def save(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def enabled(args, condition="proposed") -> bool:
    return condition == "proposed" and getattr(args, "adaptive_proposed", True)


def initial_checkpoints(args) -> dict[str, dict]:
    shared = args.grpo_checkpoint
    result = {}
    for role in ("red", "blue"):
        path = getattr(args, role + "_grpo_checkpoint", None) or shared
        if not path:
            path = verify_model(select_model(args, "proposed"))["server_model"].get("root")
        if not path:
            raise ValueError("adaptive GRPO requires a locally accessible initial LoRA checkpoint")
        result[role] = checkpoint_info(path)
    return result


def training_environment(args) -> dict:
    env = dict(os.environ)
    if args.training_cuda_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.training_cuda_devices
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    return env


def validate_learning_args(args):
    if not enabled(args) or "proposed" not in args.conditions:
        return
    if args.grpo_num_generations < 2 or args.temperature <= 0:
        raise ValueError("adaptive GRPO requires num-generations>=2 and temperature>0")
    if (args.grpo_epochs <= 0 or not math.isfinite(args.grpo_epochs) or
            not math.isfinite(args.grpo_learning_rate) or args.grpo_learning_rate <= 0 or
            not math.isfinite(args.grpo_beta) or args.grpo_beta < 0 or
            args.grpo_max_steps == 0 or args.grpo_max_steps < -1):
        raise ValueError("invalid GRPO epochs, learning rate, beta, or max steps")
    if args.frequency_penalty or args.presence_penalty:
        raise ValueError("the adaptive Transformers sampler currently requires zero frequency/presence penalties")
    config_from_args(args, required=not args.dry_run)


def learning_preflight(args) -> dict:
    initial = initial_checkpoints(args)
    code = ("import json,importlib.metadata as m,torch; from dataclasses import fields; "
            "from trl import GRPOConfig; "
            "print(json.dumps({'cuda_available':torch.cuda.is_available(),"
            "'versions':{p:m.version(p) for p in ['torch','transformers','trl','peft']},"
            "'grpo_fields':[f.name for f in fields(GRPOConfig)]}))")
    process = subprocess.run([args.training_python, "-c", code], cwd=ROOT,
                             env=training_environment(args), text=True, capture_output=True, timeout=60)
    if process.returncode:
        raise RuntimeError("GRPO training environment check failed: " + process.stderr[-1200:])
    info = json.loads(process.stdout.strip().splitlines()[-1])
    if not info["cuda_available"]:
        raise RuntimeError("allocate a CUDA training device before starting adaptive matches")
    required = {"chat_template_kwargs", "loss_type", "scale_rewards", "num_generations", "min_p"}
    if required - set(info.pop("grpo_fields")):
        raise RuntimeError("training TRL lacks required on-policy GRPO settings")
    config = config_from_args(args)
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(prefix="ad-eval-reward-preflight-") as temporary:
        path = Path(temporary) / "red_reward.json"
        path.write_text(json.dumps(config.to_dict()))
        checked = subprocess.run([args.training_python, "-m", "ad_eval.grpo_train", "--check-red-reward", str(path)],
                                 cwd=ROOT, env=training_environment(args), capture_output=True, text=True, timeout=120)
        if checked.returncode:
            raise RuntimeError("Red GRPO reward backend check failed: " + checked.stderr[-1200:])
    return {"initial_role_checkpoints": initial, "training_runtime": info, "red_reward": config.to_dict(),
            "training_python": args.training_python, "updates_per_match": 4,
            "planned_updates": 4 * args.repeats, "policy_updates": "after_each_complete_team_role_trajectory"}


class AdaptiveExperiment:
    def __init__(self, args):
        self.args = args
        self.directory = Path(args.out_dir).resolve() / "learning"
        self.prefix = served_model(args.trained_model) + "-online-" + hashlib.sha256(str(self.directory).encode()).hexdigest()[:8]
        self.state_path = self.directory / "state.json"
        self.initial = initial_checkpoints(args)
        self.red_reward = config_from_args(args)
        self.state = {"schema_version": 1, "planners": empty_planner_state(), "policies": {}, "updates": []}
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))

    def initialize_policies(self):
        if not self.state["policies"]:
            for role in ("red", "blue"):
                spec = ModelSpec(self.prefix + f"-{role}-initial",
                                 select_model(self.args, "proposed").endpoint,
                                 "grpo", self.initial[role]["path"])
                load_adapter(spec)
                self.state["policies"][role] = asdict(spec)
            save(self.state_path, self.state)

    def prepare(self, rep: int) -> tuple[dict[str, ModelSpec], Path]:
        self.initialize_policies()
        snapshot = self.directory / f"rep{rep:03d}_planner_input.json"
        if snapshot.exists():
            raise FileExistsError("planner input snapshot already exists")
        save(snapshot, self.state["planners"])
        return {role: ModelSpec(**data) for role, data in self.state["policies"].items()}, snapshot

    def update(self, match: dict, directory: Path, events: dict) -> list[dict]:
        trajectories = export_trajectories(directory, match, events)
        for team_score in match["teams"]:
            for role in ("red", "blue"):
                trajectory_reward(team_score, role)
        updates = []
        # Four distinct trajectories: team1 Red, team1 Blue, team2 Red, team2 Blue.
        # Each role continues its own previous checkpoint; no reward/weights are shared with CAI.
        for path in trajectories:
            trajectory = json.loads(path.read_text(encoding="utf-8"))
            if any(item["trajectory_id"] == trajectory["trajectory_id"] for item in self.state["updates"]):
                raise RuntimeError("trajectory was already committed")
            if not trajectory["steps"]:
                raise RuntimeError(f"{trajectory['trajectory_id']}: no completed policy decisions; cannot invent GRPO data")
            role, team = trajectory["role"], trajectory["team"]
            reward = trajectory_reward(trajectory["team_score"], role)
            planner = update_planner(self.state["planners"], trajectory, reward)
            current = ModelSpec(**self.state["policies"][role])
            slug = f"rep{match['rep']:03d}_team{team}_{role}"
            output = self.directory / "checkpoints" / slug
            job = {"trajectory": str(path), "input_checkpoint": current.checkpoint,
                   "base_model": self.args.training_base_model or self.initial[role]["base_model"] or served_model(self.args.base_model),
                   "output_dir": str(output), "replay_directory": str(self.directory / "replay_range"),
                   "num_generations": self.args.grpo_num_generations, "epochs": self.args.grpo_epochs,
                   "max_steps": self.args.grpo_max_steps, "learning_rate": self.args.grpo_learning_rate,
                   "beta": self.args.grpo_beta}
            if role == "red":
                job["red_reward"] = self.red_reward.to_dict()
                job["red_success_memory"] = str(self.directory / "red_successes.json")
            job_path = self.directory / "jobs" / (slug + ".json")
            save(job_path, job)
            log_path = job_path.with_suffix(".log")
            print(f"[ad_eval] GRPO {trajectory['trajectory_id']} from {current.model}", flush=True)
            with log_path.open("w", encoding="utf-8") as handle:
                log_path.chmod(0o600)
                process = subprocess.run([self.args.training_python, "-m", "ad_eval.grpo_train", "--job", str(job_path)],
                                         cwd=ROOT, env=training_environment(self.args), stdout=handle,
                                         stderr=subprocess.STDOUT, check=False)
            if process.returncode:
                raise RuntimeError(f"GRPO job failed rc={process.returncode}; see {log_path}")
            report = json.loads((output / "training_result.json").read_text(encoding="utf-8"))
            if report.get("trajectory_id") != trajectory["trajectory_id"] or report.get("optimizer_steps", 0) < 1:
                raise RuntimeError("trainer did not report an optimization step for this trajectory")
            if report.get("red_reward") != job.get("red_reward"):
                raise RuntimeError("trainer did not apply the declared Red research reward configuration")
            before = checkpoint_info(current.checkpoint)
            after = checkpoint_info(report["checkpoint"])
            before_weights = {key: value for key, value in before["sha256"].items() if key.startswith("adapter_model.")}
            after_weights = {key: value for key, value in after["sha256"].items() if key.startswith("adapter_model.")}
            changed = before_weights != after_weights
            if changed != report["parameters_changed"]:
                raise RuntimeError("trainer's parameter-change claim does not match checkpoint weight hashes")
            spec = ModelSpec(self.prefix + f"-{role}-r{match['rep']:03d}-t{team}",
                             current.endpoint, "grpo", report["checkpoint"])
            deployed = load_adapter(spec)  # register unique version and verify root/hash before committing
            update = {"trajectory_id": trajectory["trajectory_id"], "terminal_reward": reward,
                      "input_policy": asdict(current), "output_policy": asdict(spec),
                      "training_result": str(output / "training_result.json"),
                      "parameters_changed": report["parameters_changed"],
                      "optimizer_steps": report["optimizer_steps"],
                      "reward_groups_with_variation": report["reward_groups_with_variation"],
                      "red_reward": self.red_reward.to_dict() if role == "red" else None,
                      "deployed_checkpoint": deployed.get("checkpoint_artifact")}
            self.state["planners"] = planner
            self.state["policies"][role] = asdict(spec)
            self.state["updates"].append(update)
            save(self.state_path, self.state)
            updates.append(update)
        save(directory / "learning_updates.json", {"updates": updates, "planner_after": self.state["planners"]})
        return updates
