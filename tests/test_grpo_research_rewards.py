import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from ad_eval.diversity_reward import DiversityPenalty, EmbeddingEncoder, SuccessMemory, cosine, self_bleu
from ad_eval.grpo_rewards import RedRewardConfig, RedTrainingReward, build_red_reward
from ad_eval.grpo_train import RolloutRewards
from ad_eval.replay_lab import ReplayLab
from ad_eval.run import build_parser, closed_loop_configuration, validate_args
from ad_eval.semantic_reward import SemanticJudge

ROOT = Path(__file__).resolve().parents[1]
COMMAND = "curl http://172.32.10.20:8080/health"


class FixedEncoder:
    model_name, revision = "frozen-test-encoder", "test-revision"
    def encode(self, texts):
        return [[1., 0.] if text.startswith(("curl", "wget")) else [0., 1.] for text in texts]


def config(**overrides):
    values = dict(alpha=1., beta=.5, gamma=.25, semantic_model="frozen-judge",
                  semantic_endpoint="http://localhost:9001", embedding_model="test-encoder")
    return RedRewardConfig(**(values | overrides))


def row():
    return {"row_id": "one", "prompt": [{"role": "user", "content":
            "Goal: retrieve the user flag. Observed state: verified foothold. Strategy: advance."}],
            "strategy_id": "advance", "tech_id": "T1190", "observation": "FUTURE_RESULT_MUST_NOT_BE_USED"}


class SemanticTests(unittest.TestCase):
    def test_judge_uses_goal_observed_context_strategy_and_excludes_future_result(self):
        response = Mock()
        response.json.return_value = {"choices": [{"message": {"content": json.dumps({
            "goal_alignment": 1., "state_alignment": .5, "strategy_alignment": .75, "rationale": "aligned"})}}]}
        judge = SemanticJudge("http://localhost:9001/v1", "frozen-judge")
        with patch("ad_eval.semantic_reward.requests.post", return_value=response) as post:
            value, evidence = judge.score(row(), COMMAND)
        self.assertEqual(value, .75)
        payload = post.call_args.kwargs["json"]
        self.assertNotIn("FUTURE_RESULT", json.dumps(payload))
        self.assertIn("verified foothold", json.dumps(payload))
        self.assertIn("advance", json.dumps(payload))
        self.assertEqual(payload["temperature"], 0.)
        self.assertEqual(payload["model"], "frozen-judge")
        self.assertEqual(evidence["dimensions"]["goal_alignment"], 1.)

    def test_invalid_judge_results_fail_without_a_fabricated_score(self):
        judge = SemanticJudge("http://localhost:9001", "frozen-judge")
        for labels in ({"goal_alignment": 2}, {"goal_alignment": True}, {"goal_alignment": float("nan")}):
            response = Mock()
            response.json.return_value = {"choices": [{"message": {"content": json.dumps(labels)}}]}
            with self.subTest(labels=labels), patch("ad_eval.semantic_reward.requests.post", return_value=response):
                with self.assertRaises(ValueError):
                    judge.score(row(), COMMAND)


class DiversityTests(unittest.TestCase):
    def test_duplicates_including_opposite_team_address_have_maximum_penalty(self):
        reference = {"command": COMMAND.replace("172.32.10.20", "TARGET_IP")}
        penalty, detail = DiversityPenalty(FixedEncoder()).score(COMMAND.replace(".20:", ".10:"), [reference])
        self.assertEqual(penalty, 1.)
        self.assertEqual(detail["self_bleu"], 1.)
        self.assertEqual(detail["cosine_similarity"], 1.)

    def test_lexically_different_similar_embedding_still_contributes_penalty(self):
        penalty, detail = DiversityPenalty(FixedEncoder(), .5).score("wget http://TARGET_IP:8080/health", [{"command": COMMAND}])
        self.assertEqual(detail["cosine_similarity"], 1.)
        self.assertAlmostEqual(penalty, .5 * detail["self_bleu"] + .5)
        self.assertLess(detail["self_bleu"], 1.)

    def test_empty_history_and_unrelated_commands_have_no_duplicate_penalty(self):
        calculator = DiversityPenalty(FixedEncoder())
        self.assertEqual(calculator.score(COMMAND, [])[0], 0.)
        self.assertEqual(calculator.score("printf inspected", [{"command": COMMAND}])[0], 0.)
        self.assertEqual(self_bleu("id", ["id"]), 1.)
        self.assertEqual(cosine([1., 0.], [-1., 0.]), 0.)
        with self.assertRaises(ValueError):
            cosine([0., 0.], [1., 0.])

    def test_embedding_endpoint_indices_are_sorted_and_cache_is_reused(self):
        ready = Mock()
        ready.json.return_value = {"data": [{"id": "emb"}]}
        response = Mock()
        response.json.return_value = {"data": [{"index": 1, "embedding": [0., 1.]}, {"index": 0, "embedding": [1., 0.]}]}
        encoder = EmbeddingEncoder("emb", endpoint="http://localhost:9002")
        with patch("ad_eval.diversity_reward.requests.get", return_value=ready), \
                patch("ad_eval.diversity_reward.requests.post", return_value=response) as post:
            self.assertEqual(encoder.encode(["a", "b"]), [[1., 0.], [0., 1.]])
            encoder.encode(["a", "b"])
        self.assertEqual(post.call_count, 1)

    def test_history_rejects_unverified_entries_and_persists_valid_unique_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.json"
            memory = SuccessMemory(path)
            with self.assertRaises(ValueError):
                memory.add([{"command": COMMAND, "verified_success": False}])
            memory.add([{"command": COMMAND, "verified_success": True}])
            memory.add([{"command": COMMAND.replace(".20:", ".10:"), "verified_success": True}])
            self.assertEqual(len(SuccessMemory(path).snapshot()), 1)


class CompositeRewardTests(unittest.TestCase):
    def make_reward(self, path, **weights):
        judge = Mock()
        judge.score.return_value = (.8, {"model": "frozen-judge"})
        reward = RedTrainingReward(config(**weights), SuccessMemory(path), semantic=judge,
                                   diversity=DiversityPenalty(FixedEncoder()))
        return reward, judge

    def test_signed_weighted_formula_is_logged_without_clipping(self):
        with tempfile.TemporaryDirectory() as directory:
            reward, _ = self.make_reward(Path(directory) / "history.json", alpha=2., beta=.5, gamma=.25)
            total, detail = reward.score(row(), "CMD: " + COMMAND, .5, {"environment_executed": True},
                                         [{"command": COMMAND}])
            self.assertAlmostEqual(total, 2 * .5 + .5 * .8 - .25)
            self.assertEqual(detail["R_Attack"], .5)
            self.assertEqual(detail["R_Semantic"], .8)
            self.assertEqual(detail["P_Diversity"], 1.)

    def test_invalid_candidate_gets_only_negative_attack_component(self):
        with tempfile.TemporaryDirectory() as directory:
            reward, judge = self.make_reward(Path(directory) / "history.json", alpha=2.)
            value, detail = reward.score(row(), "not a command", -1., {"environment_executed": False}, [])
            self.assertEqual(value, -2.)
            self.assertFalse(detail["semantic"]["computed"])
            judge.score.assert_not_called()

    def test_group_uses_frozen_history_then_next_batch_penalizes_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reward, _ = self.make_reward(root / "history.json")
            lab = Mock()
            lab.reward.return_value = (1., {"environment_executed": True, "initial_access": True,
                                          "user_flag": True, "root_flag": True})
            trajectory = {"trajectory_id": "trajectory-red", "role": "red", "team": "1", "steps": [row()]}
            evaluator = RolloutRewards(trajectory, lab, root / "reward.jsonl", 2, red_reward=reward)
            completions = [[{"content": "CMD: " + COMMAND}]] * 2
            first = evaluator.evaluate(completions, ["one", "one"])
            self.assertEqual(first, [1.4, 1.4])
            self.assertEqual(len(reward.memory.snapshot()), 1)
            second = evaluator.evaluate(completions, ["one", "one"])
            self.assertEqual(second, [1.15, 1.15])
            logs = [json.loads(line) for line in (root / "reward.jsonl").read_text().splitlines()]
            self.assertEqual(logs[0]["components"]["diversity"]["reference_count"], 0)
            self.assertEqual(logs[2]["components"]["diversity"]["reference_count"], 1)
            # Order alone did not create a fake nonzero GRPO advantage.
            self.assertEqual(evaluator.nonzero_spread_groups, 0)

    def test_positive_semantics_without_verified_attack_never_enters_success_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reward, _ = self.make_reward(root / "history.json")
            lab = Mock()
            lab.reward.return_value = (0., {"environment_executed": True, "initial_access": False,
                                          "user_flag": False, "root_flag": False})
            trajectory = {"role": "red", "team": "1", "steps": [row()]}
            evaluator = RolloutRewards(trajectory, lab, root / "log.jsonl", 2, red_reward=reward)
            values = evaluator.evaluate([[{"content": "CMD: " + COMMAND}]], ["one"])
            self.assertGreater(values[0], 0)
            self.assertEqual(reward.memory.snapshot(), [])

    def test_backend_error_does_not_admit_partial_batch_successes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reward, judge = self.make_reward(root / "history.json")
            judge.score.side_effect = [(.8, {}), RuntimeError("judge offline")]
            lab = Mock()
            lab.reward.return_value = (1., {"environment_executed": True, "root_flag": True})
            evaluator = RolloutRewards({"role": "red", "team": "1", "steps": [row()]}, lab, root / "log", 2,
                                        red_reward=reward)
            with self.assertRaisesRegex(RuntimeError, "judge offline"):
                evaluator.evaluate([[{"content": "CMD: " + COMMAND}]] * 2, ["one"] * 2)
            self.assertEqual(reward.memory.snapshot(), [])
            self.assertEqual(evaluator.samples, 0)

    def test_blue_environment_reward_and_explicit_red_ablation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            lab = Mock()
            lab.reward.return_value = (.4, {"environment_executed": True})
            trajectory = {"role": "blue", "team": "1", "steps": [row()]}
            evaluator = RolloutRewards(trajectory, lab, path / "log", 2)
            self.assertEqual(evaluator.evaluate(["CMD: printf inspected"], ["one"]), [.4])
            ablation = build_red_reward(RedRewardConfig(1., 0., 0.), None)
            with self.assertRaises(ValueError):
                RolloutRewards(trajectory, lab, path / "log", 2, red_reward=ablation)
            with self.assertRaises(ValueError):
                RolloutRewards({**trajectory, "role": "red"}, lab, path / "log", 2)


class IntegrationBoundaryTests(unittest.TestCase):
    def test_adaptive_weights_are_explicit_and_never_required_for_cai(self):
        parser = build_parser()
        adaptive = parser.parse_args(["--base-model", "Qwen/Qwen3-8B", "--trained-model", "grpo"])
        with self.assertRaisesRegex(ValueError, "explicitly"):
            validate_args(adaptive)
        baseline = parser.parse_args(["--base-model", "Qwen/Qwen3-8B", "--conditions", "cai"])
        validate_args(baseline)
        self.assertIsNone(closed_loop_configuration(baseline)["red_grpo_reward"])

    def test_unknown_weights_or_trainable_judge_are_rejected(self):
        for changes in ({"alpha": -1}, {"gamma": float("nan")}, {"cosine_weight": 2}, {"history_limit": -1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                config(**changes)
        args = build_parser().parse_args(["--base-model", "base", "--trained-model", "grpo",
            "--red-reward-alpha", "1", "--red-reward-beta", ".5", "--red-reward-gamma", "0",
            "--red-semantic-model", "grpo"])
        with self.assertRaisesRegex(ValueError, "frozen"):
            validate_args(args)

    def test_recon_text_alone_cannot_fabricate_verified_attack_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            lab = ReplayLab(Path(directory))
            with patch.object(lab, "reset"), patch.object(lab, "execute", return_value="8080/tcp open"), \
                    patch.object(lab, "flags", return_value={"user": "FLAG{user_real}", "root": "FLAG{root_real}"}), \
                    patch.object(lab, "dexec", return_value=""), \
                    patch.object(lab, "observed_http_recon", return_value=False) as observed:
                value, evidence = lab.reward({"replay_prefix": [], "tech_id": "T1595"}, "CMD: " + COMMAND, "1", "red")
                self.assertEqual(value, 0.)
                self.assertFalse(evidence["recon_http_verified"])
                observed.return_value = True
                self.assertEqual(lab.reward({"replay_prefix": [], "tech_id": "T1595"}, "CMD: " + COMMAND, "1", "red")[0], .05)


if __name__ == "__main__":
    unittest.main()
