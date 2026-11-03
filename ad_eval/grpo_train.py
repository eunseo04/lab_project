"""One genuine on-policy GRPO training job per completed Red/Blue CTF trajectory.

GPU libraries are imported only in main(). New completions receive fresh private
range rewards; recorded terminal returns are used by MCTS, not copied to GRPO samples.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

from .inference import InferenceSettings
from .models import checkpoint_info
from .replay_lab import ReplayLab
from .trajectory import sanitize
from .grpo_rewards import RedRewardConfig, RedTrainingReward, build_red_reward


def completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        return str(completion[-1].get("content", ""))
    raise ValueError("unsupported GRPO completion format")


def grpo_parameters(job: dict, policy: InferenceSettings) -> dict:
    group = int(job["num_generations"])
    if group < 2 or policy.temperature <= 0:
        raise ValueError("GRPO needs at least two sampled completions and temperature>0")
    if policy.frequency_penalty or policy.presence_penalty:
        raise ValueError("the Transformers GRPO backend requires zero frequency/presence penalties")
    return {"output_dir": job["output_dir"], "per_device_train_batch_size": group,
            "gradient_accumulation_steps": 1, "num_generations": group,
            "num_train_epochs": job["epochs"], "max_steps": job["max_steps"],
            "learning_rate": job["learning_rate"], "beta": job["beta"],
            "loss_type": "grpo", "scale_rewards": "group", "num_iterations": 1,
            "temperature": policy.temperature, "top_p": policy.top_p,
            "top_k": max(0, policy.top_k), "min_p": policy.min_p,
            "repetition_penalty": policy.repetition_penalty,
            "max_completion_length": policy.max_tokens,
            "chat_template_kwargs": {"enable_thinking": policy.enable_thinking},
            "seed": policy.seed, "data_seed": policy.seed,
            "use_vllm": False, "remove_unused_columns": False,
            "save_strategy": "no", "report_to": "none", "logging_steps": 1,
            "gradient_checkpointing": True, "gradient_checkpointing_kwargs": {"use_reentrant": False}}


class RolloutRewards:
    def __init__(self, trajectory: dict, lab: ReplayLab, log: Path, num_generations: int,
                 *, red_reward: RedTrainingReward | None = None):
        self.rows = {row["row_id"]: row for row in trajectory["steps"]}
        self.team, self.role = trajectory["team"], trajectory["role"]
        self.lab, self.log, self.group = lab, log, num_generations
        self.trajectory_id = trajectory.get("trajectory_id", "unknown")
        if (self.role == "red") != (red_reward is not None):
            raise ValueError("Red GRPO requires an explicit training reward; Blue must use only its environment reward")
        self.red_reward = red_reward
        self.samples, self.nonzero_spread_groups = 0, 0

    def evaluate(self, completions, row_ids: list[str]) -> list[float]:
        if len(completions) != len(row_ids):
            raise ValueError("GRPO samples and trajectory row IDs must align")
        rewards = []
        pending = []
        history = self.red_reward.snapshot() if self.red_reward else []
        for completion, row_id in zip(completions, row_ids):
            text = completion_text(completion)
            reward, evidence = self.lab.reward(self.rows[row_id], text, self.team, self.role)
            components = None
            if self.red_reward:
                reward, components = self.red_reward.score(self.rows[row_id], text, reward, evidence, history)
            rewards.append(float(reward))
            pending.append({"trajectory_id": self.trajectory_id, "row_id": row_id, "completion": sanitize(text),
                            "reward": reward, "evidence": evidence, "components": components})
        # A single frozen history per reward batch prevents arbitrary candidate
        # order from changing group-relative advantages.
        if self.red_reward:
            self.red_reward.commit_successes(pending)
        self.samples += len(pending)
        with self.log.open("a", encoding="utf-8") as handle:
            self.log.chmod(0o600)
            for sample in pending:
                handle.write(json.dumps(sanitize(sample), ensure_ascii=False) + "\n")
        for offset in range(0, len(rewards), self.group):
            group = rewards[offset:offset + self.group]
            if len(group) == self.group and max(group) > min(group):
                self.nonzero_spread_groups += 1
        return rewards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--job", type=Path)
    mode.add_argument("--check-red-reward", type=Path)
    args = parser.parse_args()
    if args.check_red_reward:
        config = RedRewardConfig(**json.loads(args.check_red_reward.read_text(encoding="utf-8")))
        # Read-only component preparation, without Docker, student weights or history writes.
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temporary:
            reward = build_red_reward(config, Path(temporary) / "successes.json")
            reward.prepare()
        print(json.dumps({"red_reward": config.to_dict(), "backends_prepared": True}))
        return
    job = json.loads(args.job.read_text(encoding="utf-8"))
    trajectory = json.loads(Path(job["trajectory"]).read_text(encoding="utf-8"))
    if not trajectory["steps"]:
        raise ValueError("empty trajectory cannot produce a GRPO training update")
    if len({row["row_id"] for row in trajectory["steps"]}) != len(trajectory["steps"]):
        raise ValueError("trajectory row IDs must be unique")
    previous = checkpoint_info(job["input_checkpoint"])
    policy = InferenceSettings(**trajectory["inference"]).for_role(trajectory["team"], trajectory["role"])
    parameters = grpo_parameters(job, policy)
    red_reward = None
    if trajectory["role"] == "red":
        if "red_reward" not in job:
            raise ValueError("Red GRPO job must declare the research reward configuration")
        red_reward = build_red_reward(RedRewardConfig(**job["red_reward"]),
                                      Path(job["red_success_memory"]) if job.get("red_success_memory") else None)

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    required = set(parameters) - {"output_dir"}
    missing = required - {field.name for field in fields(GRPOConfig)}
    if missing:
        raise RuntimeError(f"installed TRL cannot enforce required GRPO settings: {sorted(missing)}")
    if not torch.cuda.is_available():
        raise RuntimeError("the Qwen GRPO training job requires an allocated CUDA device")
    if red_reward:
        red_reward.prepare()  # fail before loading the large student policy
    output = Path(job["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite a prior trajectory training result")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(job["base_model"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(job["base_model"], torch_dtype="auto")
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, previous["path"], is_trainable=True)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    parameters["bf16"] = torch.cuda.is_bf16_supported()
    parameters["fp16"] = not parameters["bf16"]
    config = GRPOConfig(**parameters)
    dataset = Dataset.from_list([{"prompt": row["prompt"], "row_id": row["row_id"]}
                                 for row in trajectory["steps"]])
    with ReplayLab(Path(job["replay_directory"])) as lab:
        rewards = RolloutRewards(trajectory, lab, output / "rollout_rewards.jsonl", job["num_generations"],
                                 red_reward=red_reward)

        def environment_reward(prompts, completions, row_id, **_kwargs):
            return rewards.evaluate(completions, row_id)

        trainer = GRPOTrainer(model=model, args=config, train_dataset=dataset,
                              processing_class=tokenizer, reward_funcs=environment_reward)
        if trainer.accelerator.num_processes != 1:
            raise RuntimeError("private range rewards currently require one training process")
        trainer.train()
        if trainer.state.global_step < 1 or rewards.samples < 2:
            raise RuntimeError("no GRPO optimization step or sampled reward group was executed")
        adapter = output / "adapter"
        trainer.save_model(str(adapter))
        tokenizer.save_pretrained(str(adapter))
        trainer.save_state()
        artifact = checkpoint_info(adapter)
        old_weights = {k: v for k, v in previous["sha256"].items() if k.startswith("adapter_model.")}
        new_weights = {k: v for k, v in artifact["sha256"].items() if k.startswith("adapter_model.")}
        result = {"trajectory_id": trajectory["trajectory_id"], "role": trajectory["role"],
                  "team": trajectory["team"], "input_checkpoint": previous,
                  "checkpoint": artifact["path"], "artifact": artifact,
                  "optimizer_steps": trainer.state.global_step, "sampled_completions": rewards.samples,
                  "reward_groups_with_variation": rewards.nonzero_spread_groups,
                  "parameters_changed": old_weights != new_weights, "training_parameters": parameters,
                  "red_reward": red_reward.config.to_dict() if red_reward else None,
                  "red_success_memory": str(red_reward.memory.path) if red_reward and red_reward.memory else None,
                  "algorithm": "on-policy GRPO, newly sampled commands evaluated in private reset/replay twins"}
        (output / "training_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
