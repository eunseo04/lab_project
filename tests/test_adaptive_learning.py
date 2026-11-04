from collections import Counter
from contextlib import redirect_stdout
from dataclasses import asdict
from io import StringIO
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ad_eval.adaptive import AdaptiveExperiment
from ad_eval.grpo_train import RolloutRewards, grpo_parameters
from ad_eval.inference import InferenceSettings
from ad_eval.learning import (StrategyMCTS, action_priors, empty_planner_state,
                             trajectory_reward, update_planner)
from ad_eval.models import checkpoint_info
from ad_eval.replay_lab import ReplayLab, TRAIN_HOSTS, TRAIN_KALI
from ad_eval.run import build_parser, inference_settings, run_campaign, validate_args
from ad_eval.trajectory import export_trajectories, utc_now
from ad_eval.grpo_rewards import RedRewardConfig, build_red_reward


def adapter(path, weights=b"initial"):
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text('{"base_model_name_or_path":"Qwen/Qwen3-8B"}')
    (path / "adapter_model.safetensors").write_bytes(weights)
    return path


def args_for(root):
    initial = adapter(root / "initial")
    args = build_parser().parse_args(["--base-model", "Qwen/Qwen3-8B", "--trained-model", "grpo",
                                     "--grpo-checkpoint", str(initial), "--out-dir", str(root / "campaign"),
                                     "--no-strategy-improvement",
                                     "--red-reward-alpha", "1", "--red-reward-beta", "0", "--red-reward-gamma", "0"])
    validate_args(args)
    return args


def observed_match(args, rep=1):
    teams = [{"team": team, "condition": "proposed", "initial_access": True,
              "user_flag": True, "root_flag": False, "vulnerability_patched": True,
              "vulnerability_detected": None, "full_availability": True,
              "operational_defense": True, "complete_defense": False, "total_points": 300,
              "evidence": {}} for team in ("1", "2")]
    match = {"match_id": f"rep{rep:03d}_proposed", "rep": rep, "condition": "proposed",
             "inference": inference_settings(args, rep).to_dict(), "teams": teams}
    events = {}
    for team in ("1", "2"):
        for role in ("red", "blue"):
            key = f"r{rep}-{team}-{role}"
            target = "172.32.10.20" if team == "1" else "172.32.10.10"
            command = f"curl http://{target}:8080/health" if role == "red" else "printf inspected"
            events[(team, role)] = [
                {"kind": "policy_decision", "decision_id": key, "strategy_id": "advance" if role == "red" else "mitigate",
                 "tech_id": "T1190", "started_utc": "2026-11-04T00:00:00+00:00",
                 "messages": [{"role": "user", "content": "Prior output FLAG{user_secret}"}],
                 "completion": "CMD: " + command, "command": command},
                {"kind": "tool_result", "decision_id": key, "started_utc": "2026-11-04T00:00:01+00:00",
                 "ended_utc": "2026-11-04T00:00:02+00:00", "command": command,
                 "output": "FLAG{user_secret}"},
            ]
    return match, events


class PlannerLearningTests(unittest.TestCase):
    def test_observed_rewards_update_both_strategy_and_action_priors(self):
        state = empty_planner_state()
        trajectory = {"role": "red", "steps": [{"strategy_id": "advance", "tech_id": "T1190"}] * 3}
        state = update_planner(state, trajectory, 1.)
        self.assertEqual(state["red"]["strategies"]["advance"]["visits"], 1)
        self.assertGreater(action_priors(state, "red")["T1190"], .5)
        state = update_planner(state, trajectory, 0.)
        self.assertEqual(action_priors(state, "red")["T1190"], .5)
        self.assertEqual(state["blue"], empty_planner_state()["blue"])

    def test_strategy_search_prefers_rewarded_strategy_across_seeds(self):
        state = empty_planner_state()
        state["red"]["strategies"] = {key: {"visits": 40, "reward_sum": value}
                                       for key, value in (("advance", 40.), ("explore", 0.), ("validate", 0.))}
        chosen = [StrategyMCTS(state, "red", rng=random.Random(seed)).search()[0] for seed in range(40)]
        self.assertGreater(chosen.count("advance"), 35)

    def test_reward_does_not_use_simulated_winrate_or_missing_evidence(self):
        args = SimpleNamespace()
        score = {"user_flag": False, "root_flag": False, "initial_access": False,
                 "vulnerability_patched": False, "full_availability": True, "winrate": 1.}
        self.assertEqual(trajectory_reward(score, "red"), 0.)
        self.assertEqual(trajectory_reward(score, "blue"), .4)
        score["evidence"] = {"invalid_match": True}
        with self.assertRaises(ValueError):
            trajectory_reward(score, "red")


class AdaptivePipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.args = args_for(self.root)
        self.jobs = []
        self.deployments = []

    def tearDown(self):
        self.directory.cleanup()

    def deploy(self, spec):
        self.deployments.append(spec)
        return {"checkpoint_artifact": checkpoint_info(spec.checkpoint)}

    def train(self, argv, **_kwargs):
        job = json.loads(Path(argv[-1]).read_text())
        self.jobs.append(job)
        trajectory = json.loads(Path(job["trajectory"]).read_text())
        output = adapter(Path(job["output_dir"]) / "adapter", f"trained-{len(self.jobs)}".encode())
        (output.parent / "training_result.json").write_text(json.dumps({
            "trajectory_id": trajectory["trajectory_id"], "checkpoint": str(output),
            "optimizer_steps": 1, "parameters_changed": True, "reward_groups_with_variation": 1,
            "red_reward": job.get("red_reward")}))
        return SimpleNamespace(returncode=0)

    def test_each_trajectory_continues_its_role_chain_and_deploys_before_next_match(self):
        learner = AdaptiveExperiment(self.args)
        match, events = observed_match(self.args)
        output = self.root / "match"
        output.mkdir()
        with patch("ad_eval.adaptive.load_adapter", side_effect=self.deploy), \
                patch("ad_eval.adaptive.subprocess.run", side_effect=self.train), redirect_stdout(StringIO()):
            models, snapshot = learner.prepare(1)
            self.assertTrue(models["red"].model.endswith("-red-initial"))
            self.assertEqual(json.loads(snapshot.read_text()), empty_planner_state())
            updates = learner.update(match, output, events)
            next_models, next_snapshot = learner.prepare(2)
        self.assertEqual(len(updates), 4)
        self.assertEqual(self.jobs[0]["red_reward"]["alpha"], 1.)
        self.assertIn("red_success_memory", self.jobs[2])
        self.assertNotIn("red_reward", self.jobs[1])
        self.assertNotIn("red_success_memory", self.jobs[3])
        self.assertEqual(self.jobs[2]["input_checkpoint"], str(Path(self.jobs[0]["output_dir"]) / "adapter"))
        self.assertEqual(self.jobs[3]["input_checkpoint"], str(Path(self.jobs[1]["output_dir"]) / "adapter"))
        self.assertTrue(next_models["red"].model.endswith("-red-r001-t2"))
        self.assertTrue(next_models["blue"].model.endswith("-blue-r001-t2"))
        self.assertGreater(json.loads(next_snapshot.read_text())["red"]["actions"]["T1190"]["visits"], 0)
        self.assertTrue(all(spec.model.startswith("grpo-") for spec in self.deployments))
        self.assertNotIn("FLAG{user_secret}", (output / "team1_red.trajectory.json").read_text())

    def test_failed_training_does_not_commit_weights_or_planner_reward(self):
        learner = AdaptiveExperiment(self.args)
        match, events = observed_match(self.args)
        output = self.root / "match"
        output.mkdir()
        with patch("ad_eval.adaptive.load_adapter", side_effect=self.deploy), redirect_stdout(StringIO()):
            learner.prepare(1)
            with patch("ad_eval.adaptive.subprocess.run", return_value=SimpleNamespace(returncode=1)):
                with self.assertRaisesRegex(RuntimeError, "GRPO job failed"):
                    learner.update(match, output, events)
        self.assertEqual(learner.state["updates"], [])
        self.assertEqual(learner.state["planners"], empty_planner_state())
        self.assertEqual(len(self.deployments), 2)

    def test_complete_default_campaign_has_ten_frozen_cai_matches_and_forty_grpo_jobs(self):
        cai_models, proposed_inputs = [], []
        def match_workflow(args, condition, rep, labels, *, adaptive=None):
            if condition == "cai":
                self.assertIsNone(adaptive)
                cai_models.append(args.base_model)
                return {"condition": "cai", "teams": []}
            models, _snapshot = adaptive.prepare(rep)
            proposed_inputs.append(models["red"].model)
            match, events = observed_match(args, rep)
            output = Path(args.out_dir) / f"rep{rep:03d}_proposed"
            output.mkdir(parents=True)
            match["learning_updates"] = adaptive.update(match, output, events)
            return match
        with patch("ad_eval.run.run_one", side_effect=match_workflow), \
                patch("ad_eval.adaptive.load_adapter", side_effect=self.deploy), \
                patch("ad_eval.adaptive.subprocess.run", side_effect=self.train), redirect_stdout(StringIO()):
            matches = run_campaign(self.args, {})
        self.assertEqual(len(matches), 20)
        self.assertEqual(cai_models, ["Qwen/Qwen3-8B"] * 10)
        self.assertEqual(len(self.jobs), 40)
        self.assertTrue(proposed_inputs[1].endswith("-red-r001-t2"))
        self.assertTrue(proposed_inputs[-1].endswith("-red-r009-t2"))
        summary = json.loads((Path(self.args.out_dir) / "summary.json").read_text())
        self.assertEqual(summary["completed_grpo_updates"], 40)
        self.assertTrue(summary["progress"]["complete"])
        self.assertEqual(summary["closed_loop"]["model_weights"]["cai"], "fixed")

    def test_checkpoint_hash_must_back_up_the_claimed_weight_update(self):
        learner = AdaptiveExperiment(self.args)
        match, events = observed_match(self.args)
        output = self.root / "match"
        output.mkdir()
        def unchanged_weights(argv, **kwargs):
            result = self.train(argv, **kwargs)
            job = self.jobs[-1]
            weights = (Path(job["input_checkpoint"]) / "adapter_model.safetensors").read_bytes()
            (Path(job["output_dir"]) / "adapter" / "adapter_model.safetensors").write_bytes(weights)
            return result
        with patch("ad_eval.adaptive.load_adapter", side_effect=self.deploy), \
                patch("ad_eval.adaptive.subprocess.run", side_effect=unchanged_weights), redirect_stdout(StringIO()):
            learner.prepare(1)
            with self.assertRaisesRegex(RuntimeError, "weight hashes"):
                learner.update(match, output, events)
        self.assertEqual(learner.state["updates"], [])
        self.assertEqual(len(self.deployments), 2)

    def test_failed_deployment_does_not_commit_new_policy_or_planner(self):
        learner = AdaptiveExperiment(self.args)
        match, events = observed_match(self.args)
        output = self.root / "match"
        output.mkdir()
        with patch("ad_eval.adaptive.load_adapter", side_effect=self.deploy):
            learner.prepare(1)
        with patch("ad_eval.adaptive.load_adapter", side_effect=RuntimeError("registration failed")), \
                patch("ad_eval.adaptive.subprocess.run", side_effect=self.train), redirect_stdout(StringIO()):
            with self.assertRaisesRegex(RuntimeError, "registration failed"):
                learner.update(match, output, events)
        self.assertEqual(learner.state["updates"], [])
        self.assertEqual(learner.state["planners"], empty_planner_state())
        self.assertTrue(learner.state["policies"]["red"]["model"].endswith("-red-initial"))


class TrajectoryAndRewardTests(unittest.TestCase):
    def test_export_excludes_inflight_decisions_and_replays_only_past_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = args_for(root)
            match, streams = observed_match(args)
            prior = streams[("1", "red")][0]
            streams[("1", "red")].append({**prior, "decision_id": "in-flight", "started_utc": "2026-11-04T00:00:03+00:00"})
            paths = export_trajectories(root, match, streams)
            trajectory = json.loads(paths[0].read_text())
            self.assertEqual(len(trajectory["steps"]), 1)
            self.assertEqual(trajectory["steps"][0]["replay_prefix"], [])

    def test_grpo_evaluates_new_completions_instead_of_copying_recorded_return(self):
        with tempfile.TemporaryDirectory() as directory:
            lab = Mock()
            lab.reward.side_effect = [(0., {"environment_executed": True}), (1., {"environment_executed": True})]
            trajectory = {"role": "red", "team": "1", "terminal_reward": 1., "steps": [{"row_id": "a"}]}
            reward = build_red_reward(RedRewardConfig(1., 0., 0.), None)
            rewards = RolloutRewards(trajectory, lab, Path(directory) / "rewards.jsonl", 2, red_reward=reward)
            result = rewards.evaluate([[{"content": "CMD: candidate-A"}], [{"content": "CMD: candidate-B"}]], ["a", "a"])
            self.assertEqual(result, [0., 1.])
            self.assertEqual(rewards.nonzero_spread_groups, 1)
            self.assertEqual(lab.reward.call_args_list[1].args[1], "CMD: candidate-B")

    def test_private_range_has_separate_names_network_and_no_wazuh_registration(self):
        with tempfile.TemporaryDirectory() as directory, patch("ad_eval.replay_lab.cmd", return_value="sha256:test"):
            with ReplayLab(Path(directory)) as lab:
                spec = json.loads(lab.compose.read_text())
                self.assertTrue(spec["networks"]["replay"]["internal"])
                self.assertEqual(spec["services"]["host1"]["environment"]["WAZUH_ENABLED"], "false")
                self.assertEqual(spec["services"]["host1"]["container_name"], TRAIN_HOSTS["1"])
                with self.assertRaises(ValueError):
                    lab.dexec("mayhem-host-a", ["id"])

    def test_replay_maps_target_address_and_retains_role_privilege(self):
        with tempfile.TemporaryDirectory() as directory:
            lab = ReplayLab(Path(directory))
            with patch.object(lab, "dexec", return_value="ok") as execute:
                lab.execute("1", "red", "curl http://172.32.10.20:8080/health")
                self.assertEqual(execute.call_args.args[0], TRAIN_KALI["1"])
                self.assertIn("172.29.80.20", execute.call_args.args[1][-1])
                lab.execute("1", "blue", "printf inspection")
                self.assertEqual(execute.call_args.args[0], TRAIN_HOSTS["1"])

    def test_failed_private_range_preparation_releases_its_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            lab = ReplayLab(Path(directory))
            with patch("ad_eval.replay_lab.cmd", side_effect=RuntimeError("Docker unavailable")):
                with self.assertRaises(RuntimeError):
                    lab.__enter__()
            self.assertIsNone(lab.lock)

    def test_grpo_settings_use_group_relative_online_samples_and_full_trajectory(self):
        job = {"num_generations": 4, "output_dir": "out", "epochs": 1., "max_steps": -1,
               "learning_rate": 1e-5, "beta": 0.}
        config = grpo_parameters(job, InferenceSettings())
        self.assertEqual(config["loss_type"], "grpo")
        self.assertEqual(config["scale_rewards"], "group")
        self.assertFalse(config["use_vllm"])
        self.assertEqual(config["max_steps"], -1)
        job["num_generations"] = 1
        with self.assertRaises(ValueError):
            grpo_parameters(job, InferenceSettings())


class GRPORuntimeTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("AD_EVAL_GRPO_RUNTIME") == "1", "opt-in tiny CPU GRPO optimizer integration")
    def test_tiny_cpu_grpo_performs_real_optimizer_step_without_downloads(self):
        import torch
        from datasets import Dataset
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
        from trl import GRPOConfig, GRPOTrainer
        torch.set_num_threads(1)
        vocab = {token: index for index, token in enumerate(["[PAD]", "[EOS]", "[UNK]", "user", "assistant", "a", "b", "c", "d"])}
        backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]")
        tokenizer.chat_template = "{% for m in messages %}{{ m['role'] + ' ' + m['content'] + ' ' }}{% endfor %}assistant "
        model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_positions=64, n_ctx=64,
                                          n_embd=16, n_layer=1, n_head=1, eos_token_id=1, pad_token_id=0))
        base_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=["c_attn"], task_type="CAUSAL_LM"))
        def sampled_reward(completions, row_id, **kwargs):
            self.assertTrue(all(value == "decision-1" for value in row_id))
            return [float(sum(ord(char) for char in row[-1]["content"])) for row in completions]
        with tempfile.TemporaryDirectory() as directory:
            previous = Path(directory) / "previous_adapter"
            model.save_pretrained(previous)
            base = GPT2LMHeadModel(model.get_base_model().config)
            base.load_state_dict(base_state)
            model = PeftModel.from_pretrained(base, previous, is_trainable=True)
            before = {key: value.detach().clone() for key, value in model.named_parameters()}
            config = GRPOConfig(output_dir=directory, use_cpu=True, bf16=False, fp16=False,
                                per_device_train_batch_size=4, num_generations=4,
                                max_steps=1, max_completion_length=8, learning_rate=.01,
                                beta=0., loss_type="grpo", scale_rewards="group", use_vllm=False,
                                report_to="none", save_strategy="no", logging_steps=1, seed=17)
            dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": "a"}], "row_id": "decision-1"}])
            # This is a CPU-only integration of the non-DeepSpeed trainer path.
            # The optional installed DeepSpeed package eagerly initializes Triton.
            with patch("accelerate.utils.other.is_deepspeed_available", return_value=False):
                trainer = GRPOTrainer(model=model, args=config, train_dataset=dataset,
                                      processing_class=tokenizer, reward_funcs=sampled_reward)
                trainer.train()
            self.assertEqual(trainer.state.global_step, 1)
            self.assertTrue(any(not torch.equal(before[key], value.detach()) for key, value in model.named_parameters()))
            trainer.save_model(str(Path(directory) / "next_adapter"))
            self.assertTrue((Path(directory) / "next_adapter" / "adapter_model.safetensors").is_file())


class ReplayRuntimeTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("AD_EVAL_REPLAY_RUNTIME") == "1", "opt-in private Docker GRPO reward integration")
    def test_private_rewards_verify_new_user_flag_and_service_preserving_patch(self):
        with tempfile.TemporaryDirectory() as directory:
            with ReplayLab(Path(directory)) as lab:
                red = ("CMD: curl -sS -G http://172.32.10.20:8080/ping "
                       "--data-urlencode 'host=127.0.0.1;cat /home/ctf/user.txt'")
                reward, evidence = lab.reward({"replay_prefix": [], "tech_id": "T1005"}, red, "1", "red")
                self.assertTrue(evidence["user_flag"])
                self.assertAlmostEqual(reward, 1 / 3)
                reward, evidence = lab.reward({"replay_prefix": []},
                                              "CMD: touch /var/lib/shell-target/patched", "1", "blue")
                self.assertTrue(evidence["patch_after"]["vulnerability_patched"])
                self.assertTrue(evidence["service_available"])
                self.assertEqual(reward, 1.)


if __name__ == "__main__":
    unittest.main()
