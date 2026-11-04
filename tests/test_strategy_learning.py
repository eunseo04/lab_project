from copy import deepcopy
from contextlib import redirect_stdout
from dataclasses import asdict
from io import StringIO
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ad_eval.idea_queue import IdeaQueue
from ad_eval.learning import (StrategyMCTS, add_strategy, commit_comparison, empty_planner_state,
                             normalize_planner, strategy_node, update_planner)
from ad_eval.mcts import CyberState, MCTS, Objective
from ad_eval.models import ModelSpec, checkpoint_info
from ad_eval.reflection import reflect
from ad_eval.self_play import PairedSelfPlay, compare_returns, incumbent_for, pinned_state
from ad_eval.strategy_improve import improve_strategy
from ad_eval.run import inference_settings, run_campaign
from test_adaptive_learning import args_for, observed_match, adapter
from ad_eval.adaptive import AdaptiveExperiment, validate_learning_args
from ad_eval.trajectory import export_trajectories


def candidate(key="improved", parent="advance"):
    return {"id": key, "parent_id": parent, "instruction": "Inspect actual observations and verify prerequisites before advancing.",
            "preferred_tactics": ["initial-access"], "preferred_techniques": ["T1190"],
            "max_depth": 3, "discount": .9, "status": "candidate", "children": []}


def pairs(a=.2, b=.6):
    return [{"team": team, "valid": True, "baseline_return": a, "candidate_return": b} for team in ("1", "2")]


def report(key="improved", parent="advance", status="accepted"):
    return {"parent_id": parent, "candidate_id": key, "status": status, "valid": True,
            "pairs": pairs(), "minimum_gain": 0.}


def reflection_response(prompt):
    context = json.loads(prompt)
    return json.dumps({"findings": [{"kind": "feedback_alignment",
        "hypothesis": "Observed actions may leave the current objective insufficiently verified.",
        "proposal": "Verify the latest feedback before advancing the next strategy action.",
        "step_ids": [context["recent_steps"][0]["row_id"]], "terminal_fields": [], "priority": 4}]})


def fresh_conditions():
    return {"reset": "compose_down_volumes_then_recreate",
            "patches": {side: {"vulnerability_patched": False} for side in ("1", "2")},
            "services": {side: {"up": True} for side in ("1", "2")}}


class StrategyTreeTests(unittest.TestCase):
    def test_legacy_statistics_migrate_and_dynamic_lineage_is_selectable_after_comparison(self):
        legacy = {role: {"strategies": {}, "actions": {}} for role in ("red", "blue")}
        legacy["red"]["strategies"]["advance"] = {"visits": 4, "reward_sum": 3.}
        tree = add_strategy(legacy, "red", candidate())
        self.assertEqual(tree["red"]["strategies"], legacy["red"]["strategies"])
        self.assertIn("improved", tree["red"]["tree"]["nodes"]["advance"]["children"])
        for seed in range(10):
            self.assertNotEqual(StrategyMCTS(tree, "red", rng=random.Random(seed)).search()[0], "improved")
        pinned = pinned_state(tree, "red", "1", "improved", {"red": "advance", "blue": "mitigate"})
        self.assertEqual(StrategyMCTS(pinned, "red", team="1", rng=random.Random(1)).search()[0], "improved")
        self.assertEqual(StrategyMCTS(pinned, "red", team="2", rng=random.Random(1)).search()[0], "advance")
        tree = commit_comparison(tree, "red", "improved", report())
        self.assertEqual(StrategyMCTS(tree, "red", rng=random.Random(1)).search()[0], "improved")
        tree = add_strategy(tree, "red", candidate("grandchild", "improved"))
        tree = commit_comparison(tree, "red", "grandchild", report("grandchild", "improved"))
        self.assertEqual(StrategyMCTS(tree, "red", rng=random.Random(1)).search()[0], "grandchild")
        self.assertNotIn("tree", legacy["red"])

    def test_observed_reward_is_backed_up_once_per_ancestor_not_per_command(self):
        state = add_strategy(empty_planner_state(), "red", candidate())
        state = add_strategy(state, "red", candidate("grandchild", "improved"))
        updated = update_planner(state, {"role": "red", "steps": [
            {"strategy_id": "grandchild", "tech_id": "T1190"},
            {"strategy_id": "grandchild", "tech_id": "T1190"},
            {"strategy_id": "improved", "tech_id": "T1190"}]}, .75)
        for key in ("advance", "improved", "grandchild"):
            self.assertEqual(updated["red"]["strategies"][key], {"visits": 1, "reward_sum": .75})
        self.assertEqual(updated["red"]["actions"]["T1190"]["visits"], 1)
        self.assertEqual(updated["blue"], state["blue"])

    def test_rejected_inconclusive_and_unsubstantiated_candidates_never_promote(self):
        state = add_strategy(empty_planner_state(), "red", candidate())
        for status in ("rejected", "inconclusive"):
            result = commit_comparison(state, "red", "improved", report(status=status))
            self.assertIsNone(result["red"]["tree"]["incumbent"])
        bad = report()
        bad["pairs"][0]["candidate_return"] = 0.
        with self.assertRaisesRegex(ValueError, "do not support"):
            commit_comparison(state, "red", "improved", bad)
        state["red"]["tree"]["nodes"]["advance"]["parent_id"] = "improved"
        with self.assertRaisesRegex(ValueError, "cycle"):
            normalize_planner(state)


class ReflectionAndIdeasTests(unittest.TestCase):
    def trajectory(self):
        return {"trajectory_id": "match:1:red", "role": "red",
                "team_score": {"initial_access": False, "user_flag": False, "root_flag": False},
                "steps": [{"row_id": "a", "strategy_id": "advance", "tech_id": "T1190", "command": "probe",
                           "observation": "Permission denied FLAG{secret}", "rejected": False},
                          {"row_id": "b", "strategy_id": "advance", "tech_id": "T1190", "command": "probe",
                           "observation": "policy rejection", "rejected": True}]}

    def test_reflection_links_failures_to_steps_and_does_not_claim_causality(self):
        reflected = reflect(self.trajectory())
        findings = {finding["kind"]: finding for finding in reflected["findings"]}
        self.assertEqual(findings["policy_rejection"]["step_ids"], ["b"])
        self.assertEqual(findings["execution_error"]["step_ids"], ["a"])
        self.assertEqual(findings["repeated_action"]["step_ids"], ["a", "b"])
        self.assertIn("incomplete_objective", findings)
        self.assertNotIn("FLAG{secret}", json.dumps(reflected))
        self.assertTrue(all("hypothesis" in finding["causal_status"] for finding in findings.values()))

    def test_queue_deduplicates_persists_priority_and_retries_inconclusive_with_new_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "ideas.json"
            queue = IdeaQueue(path)
            reflected = reflect(self.trajectory())
            queue.generate(reflected, "advance")
            count = len(queue.items)
            queue.generate(reflected, "advance")
            self.assertEqual(len(queue.items), count)
            idea = queue.next("red", "advance")
            self.assertEqual(idea["kind"], "policy_rejection")
            queue.finish(idea["id"], {"status": "inconclusive"})
            restored = IdeaQueue(path)
            self.assertEqual(restored.items, queue.items)
            new = self.trajectory()
            new["trajectory_id"] = "next:1:red"
            restored.generate(reflect(new), "advance")
            self.assertEqual(restored.next("red", "advance")["id"], idea["id"])
            self.assertEqual(len(restored.next("red", "advance")["sources"]), 2)

    def test_generation_uses_parent_idea_evidence_and_validates_frozen_techniques(self):
        parent = strategy_node(empty_planner_state(), "red", "advance")
        idea = {"id": "idea", "proposal": "Verify prerequisites"}
        data = {key: value for key, value in candidate().items() if key in
                {"instruction", "preferred_tactics", "preferred_techniques", "max_depth", "discount"}}
        llm = Mock()
        llm.complete.return_value = json.dumps(data)
        result = improve_strategy(parent, idea, [reflect(self.trajectory())], llm, {"T1190": {}})
        self.assertEqual(result["parent_id"], "advance")
        self.assertEqual(result["idea_id"], "idea")
        prompt = json.loads(llm.complete.call_args.args[0])
        self.assertEqual(prompt["parent"]["id"], "advance")
        self.assertNotIn("FLAG{secret}", json.dumps(prompt))
        data["preferred_techniques"] = ["T9999"]
        llm.complete.return_value = json.dumps(data)
        with self.assertRaisesRegex(ValueError, "frozen memory"):
            improve_strategy(parent, idea, [], llm, {"T1190": {}})

    def test_llm_reflection_produces_new_ideas_with_grounded_evidence_and_rejects_invented_refs(self):
        llm = Mock()
        llm.complete.side_effect = lambda prompt, **kwargs: reflection_response(prompt)
        result = reflect(self.trajectory(), llm)
        self.assertEqual(result["analysis_source"], "frozen_role_model")
        self.assertEqual(result["findings"][0]["kind"], "feedback_alignment")
        self.assertIn("observed_signals", result)
        with tempfile.TemporaryDirectory() as directory:
            queue = IdeaQueue(Path(directory) / "queue.json")
            queue.generate(result, "advance")
            self.assertEqual(queue.next("red", "advance")["proposal"], result["findings"][0]["proposal"])
        invalid = deepcopy(result["findings"])
        invalid[0]["step_ids"] = ["fabricated-step"]
        llm.complete.side_effect = None
        llm.complete.return_value = json.dumps({"findings": invalid})
        with self.assertRaisesRegex(ValueError, "invented evidence"):
            reflect(self.trajectory(), llm)


class MultiStepSearchTests(unittest.TestCase):
    memory = {"T1595": {"name": "scan", "tactics": ["reconnaissance"]},
              "T1583": {"name": "prepare", "tactics": ["resource-development"]},
              "T1190": {"name": "access", "tactics": ["initial-access"]},
              "T1059": {"name": "execute", "tactics": ["execution"]}}

    def test_transitions_advance_capabilities_on_success_without_mutating_real_state(self):
        planner = MCTS(self.memory, rng=Mock(random=Mock(return_value=0.)))
        state = CyberState()
        action = {"tech_id": "T1595", "tactics": ["reconnaissance"]}
        next_state, reward = planner.transition(state, action, "red")
        self.assertEqual(next_state.foothold, 1)
        self.assertIn("recon", next_state.capabilities)
        self.assertEqual(state, CyberState())
        planner.rng.random.return_value = .99
        failed, _ = planner.transition(state, action, "red")
        self.assertEqual(failed.foothold, 0)
        self.assertEqual(failed.capabilities, ())

    def test_multiple_depths_have_state_dependent_candidates_and_backed_up_visits(self):
        planner = MCTS(self.memory, n_sims=24, max_depth=4,
                       rng=Mock(random=Mock(return_value=0.)))
        action, value = planner.search(CyberState())
        self.assertEqual(action["tech_id"], "T1595")
        self.assertEqual(planner.last_search["max_depth_reached"], 4)
        root = planner.last_tree
        self.assertEqual(root.visits, 24)
        edge = root.edges[0]
        self.assertEqual(edge.visits, 24)
        child = next(iter(edge.outcomes.values()))
        self.assertEqual(child.state.foothold, 1)
        self.assertEqual(child.edges[0].action["tech_id"], "T1583")
        self.assertEqual(child.edges[0].visits, 24)
        self.assertGreater(edge.value_sum, 0.)
        self.assertTrue(0 <= value <= 1)

    def test_strategy_affects_prior_depth_and_llm_candidate_context(self):
        memory = {"T1595": self.memory["T1595"], "T1592": {"name": "other", "tactics": ["reconnaissance"]}}
        planner = MCTS(memory, n_sims=8, rng=random.Random(5))
        strategy = {"instruction": "Verify the specific service before changing phases.", "max_depth": 2,
                    "preferred_techniques": ["T1592"], "discount": .8}
        planner.search(CyberState(), strategy=strategy)
        priors = {edge.action["tech_id"]: edge.prior for edge in planner.last_tree.edges}
        self.assertGreater(priors["T1592"], priors["T1595"])
        self.assertEqual(planner.last_search["horizon"], 2)
        llm = Mock(available=True)
        llm.complete.return_value = "T1595,T1592"
        MCTS(memory, llm=llm).search(CyberState(), strategy=strategy)
        self.assertIn(strategy["instruction"], llm.complete.call_args.args[0])

    def test_blue_transition_changes_defended_state_for_future_candidates(self):
        planner = MCTS(self.memory, rng=Mock(random=Mock(return_value=0.)))
        state = CyberState(detected=("T1190", "T1059"))
        action, _ = planner.search(state, role="blue", focus_tid="T1190")
        after, _ = planner.transition(state, action, "blue", "T1190")
        self.assertIn(action["tech_id"], after.defended)
        self.assertNotIn(action["tech_id"], state.defended)


class ControlledSelfPlayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.args = args_for(self.root)
        self.args.strategy_improvement = True
        self.models = {role: ModelSpec(role + "-frozen", "http://local/v1", "grpo") for role in ("red", "blue")}
        self.state = add_strategy(empty_planner_state(), "red", candidate())
        self.incumbents = {"red": "advance", "blue": "mitigate"}
        self.calls = []
        self.evaluations = 0

    def tearDown(self):
        self.temporary.cleanup()

    def run_game(self, args, condition, rep, labels, *, role_models, planner_state):
        state = json.loads(planner_state.read_text())
        self.calls.append((args, state, role_models))
        selected = state["red"]["selection_overrides"]
        changed = next((team for team in ("1", "2") if selected[team] == "improved"), None)
        teams = [{"team": team, "initial_access": True, "user_flag": bool(changed == team),
                  "root_flag": False, "full_availability": True, "vulnerability_patched": True,
                  "evidence": {}} for team in ("1", "2")]
        self.assertEqual(labels, {})
        self.assertFalse(args.adaptive_proposed)
        return {"teams": teams, "role_models": {key: asdict(spec) for key, spec in role_models.items()},
                "inference": inference_settings(args, rep).to_dict(), "duration_s": args.match_seconds,
                "initial_conditions": fresh_conditions(),
                "strategy_selection_audit": {str(i): {"valid": True} for i in range(4)}}

    def evaluate(self, **kwargs):
        evaluator = PairedSelfPlay(self.args, runner=kwargs.pop("runner", self.run_game),
                                   environment=kwargs.pop("environment", lambda: {"images": "pinned"}),
                                   policies=kwargs.pop("policies", lambda models: {"weights": "sha-pinned"}))
        suffix = str(self.evaluations) if self.evaluations else ""
        self.evaluations += 1
        return evaluator.evaluate(state=self.state, role="red", parent_id="advance", candidate_id="improved",
                                  models=self.models, rep=1, directory=self.root / ("evaluation" + suffix),
                                  incumbents=self.incumbents)

    def test_real_workflow_callback_gets_single_role_intervention_equal_seed_models_and_both_positions(self):
        result = self.evaluate()
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(len(self.calls), 4)
        self.assertEqual({pair["team"] for pair in result["pairs"]}, {"1", "2"})
        self.assertNotEqual(result["pairs"][0]["order"], result["pairs"][1]["order"])
        for left, right in (self.calls[:2], self.calls[2:]):
            self.assertEqual(left[0].seed, right[0].seed)
            self.assertEqual(left[0].match_seconds, right[0].match_seconds)
            self.assertEqual(left[2], right[2])
            self.assertEqual(left[1]["blue"], right[1]["blue"])
            a, b = deepcopy(left[1]), deepcopy(right[1])
            a["red"].pop("selection_overrides")
            b["red"].pop("selection_overrides")
            self.assertEqual(a, b)
        self.assertTrue((self.root / "evaluation" / "comparison.json").exists())

    def test_environment_or_weight_drift_is_inconclusive(self):
        identities = iter([{"images": value} for value in ("a", "b", "a", "a")])
        self.assertEqual(self.evaluate(environment=lambda: next(identities))["status"], "inconclusive")
        checks = iter(["original", "original", "changed", "original", "original", "original", "original", "original", "original"])
        result = self.evaluate(policies=lambda models: {"weights": next(checks)})
        self.assertEqual(result["status"], "inconclusive")

    def test_missing_strategy_audit_worker_error_or_unknown_sla_cannot_promote(self):
        for problem in ("strategy_selection_audit", "worker_error", "availability", "initial_state"):
            def broken(*args, **kwargs):
                match = self.run_game(*args, **kwargs)
                if problem == "strategy_selection_audit":
                    match.pop(problem)
                elif problem == "worker_error":
                    match["teams"][1]["evidence"]["invalid_match"] = True
                elif problem == "initial_state":
                    match["initial_conditions"]["patches"]["1"]["vulnerability_patched"] = True
                else:
                    match["teams"][0]["full_availability"] = None
                return match
            self.assertEqual(self.evaluate(runner=broken)["status"], "inconclusive")

    def test_tie_regression_margin_and_missing_position_cannot_promote(self):
        self.assertEqual(compare_returns(pairs(.4, .4), 0.)[0], "rejected")
        self.assertEqual(compare_returns(pairs(.4, .5), .2)[0], "rejected")
        mixed = pairs(.4, .8)
        mixed[1]["candidate_return"] = .3
        self.assertEqual(compare_returns(mixed, 0.)[0], "rejected")
        self.assertEqual(compare_returns(pairs()[:1], 0.)[0], "inconclusive")
        mixed[1]["candidate_return"] = float("nan")
        self.assertEqual(compare_returns(mixed, 0.)[0], "inconclusive")


class StrategistIntegrationTests(unittest.TestCase):
    def test_invalid_comparison_budgets_fail_before_starting_a_campaign(self):
        with tempfile.TemporaryDirectory() as root:
            args = args_for(Path(root))
            args.strategy_improvement = True
            for field, value in (("strategy_self_play_pairs", 1), ("strategy_self_play_pairs", 3),
                                 ("strategy_self_play_seconds", 0), ("strategy_min_gain", float("nan"))):
                invalid = deepcopy(args)
                setattr(invalid, field, value)
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    validate_learning_args(invalid)

    def test_reflection_queue_generation_comparison_commit_and_next_worker_snapshot(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            args = args_for(root)
            args.strategy_improvement = True
            learner = AdaptiveExperiment(args)
            learner.state["policies"] = {role: asdict(ModelSpec(role + "-fixed", "http://local/v1", "grpo",
                                                             str(root / "initial"))) for role in ("red", "blue")}
            match, events = observed_match(args)
            paths = export_trajectories(root, match, events)
            llm = Mock()
            llm.complete.side_effect = lambda prompt, **kwargs: reflection_response(prompt)
            from ad_eval.inference import InferenceSettings
            llm.inference = InferenceSettings()
            llm.usage, llm.usage_complete = {"input_tokens": 20, "output_tokens": 30}, True
            def generate(parent, idea, reflections, client, memory):
                return candidate("new-" + idea["role"], parent["id"])
            def evaluate(**kwargs):
                return {**report(kwargs["candidate_id"], kwargs["parent_id"]), "mean_paired_gain": .4}
            with patch("ad_eval.llm.LLMClient", return_value=llm), \
                    patch("ad_eval.strategy_improve.improve_strategy", side_effect=generate) as generator, \
                    patch("ad_eval.self_play.PairedSelfPlay.evaluate", side_effect=evaluate) as compare:
                updates = learner.improve_strategies(paths, match)
            self.assertEqual(len(updates), 2)
            self.assertEqual(generator.call_count, 2)
            self.assertEqual(compare.call_count, 2)
            self.assertEqual(compare.call_args_list[0].kwargs["models"], compare.call_args_list[1].kwargs["models"])
            self.assertEqual(learner.state["planners"]["red"]["tree"]["incumbent"], "new-red")
            self.assertEqual(learner.state["planners"]["blue"]["tree"]["incumbent"], "new-blue")
            _, snapshot = learner.prepare(2)
            next_state = json.loads(snapshot.read_text())
            self.assertEqual(StrategyMCTS(next_state, "red", rng=random.Random(1)).search()[0], "new-red")
            self.assertEqual(len(learner.state["updates"]), 0)  # no GRPO jobs from development bouts
            self.assertEqual(len(IdeaQueue(learner.directory / "idea_queue.json").items), 2)

    def test_default_ten_round_campaign_keeps_eighty_strategy_bouts_out_of_forty_grpo_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = args_for(root)
            args.strategy_improvement = True
            training_jobs, bouts, cai_models = [], [], []

            def train(argv, **kwargs):
                job = json.loads(Path(argv[-1]).read_text())
                training_jobs.append(job)
                trajectory = json.loads(Path(job["trajectory"]).read_text())
                checkpoint = adapter(Path(job["output_dir"]) / "adapter", str(len(training_jobs)).encode())
                (checkpoint.parent / "training_result.json").write_text(json.dumps({
                    "trajectory_id": trajectory["trajectory_id"], "checkpoint": str(checkpoint),
                    "optimizer_steps": 1, "parameters_changed": True, "reward_groups_with_variation": 1,
                    "red_reward": job.get("red_reward")}))
                return SimpleNamespace(returncode=0)

            def workflow(local_args, condition, rep, labels, *, adaptive=None, role_models=None, planner_state=None):
                if condition == "cai":
                    cai_models.append(local_args.base_model)
                    self.assertIsNone(adaptive)
                    return {"condition": condition, "teams": []}
                if adaptive:
                    _, snapshot = adaptive.prepare(rep)
                    state = json.loads(snapshot.read_text())
                    match, events = observed_match(local_args, rep)
                    for (team, role), stream in events.items():
                        key = incumbent_for(state, role)
                        for event in stream:
                            if "strategy_id" in event:
                                event["strategy_id"] = key
                    output = Path(local_args.out_dir) / f"rep{rep:03d}_proposed"
                    output.mkdir(parents=True)
                    match["learning_updates"] = adaptive.update(match, output, events)
                    return match
                bouts.append((rep, local_args, role_models))
                state = json.loads(planner_state.read_text())
                intervention = next(((role, team) for role in ("red", "blue") for team in ("1", "2")
                                     if state[role]["tree"]["nodes"][state[role]["selection_overrides"][team]]["status"] == "candidate"), None)
                teams = [{"team": team, "initial_access": False, "user_flag": bool(intervention == ("red", team)),
                          "root_flag": False, "vulnerability_patched": bool(intervention == ("blue", team)),
                          "full_availability": True, "evidence": {}} for team in ("1", "2")]
                return {"condition": "proposed", "teams": teams, "duration_s": local_args.match_seconds,
                        "inference": inference_settings(local_args, rep).to_dict(),
                        "initial_conditions": fresh_conditions(),
                        "role_models": {role: asdict(spec) for role, spec in role_models.items()},
                        "strategy_selection_audit": {str(i): {"valid": True} for i in range(4)}}

            def completion(client, prompt, **kwargs):
                context = json.loads(prompt)
                if context.get("task") == "reflection":
                    return reflection_response(prompt)
                parent = context["parent"]
                return json.dumps({"instruction": parent["instruction"] + " Verify the latest observation before continuing.",
                                   "preferred_tactics": [], "preferred_techniques": [], "max_depth": 4, "discount": .95})

            with patch("ad_eval.run.run_one", side_effect=workflow), \
                    patch("ad_eval.adaptive.load_adapter", side_effect=lambda spec: {"checkpoint_artifact": checkpoint_info(spec.checkpoint)}), \
                    patch("ad_eval.adaptive.subprocess.run", side_effect=train), \
                    patch("ad_eval.llm.LLMClient.complete", autospec=True, side_effect=completion), \
                    patch("ad_eval.self_play.policy_identity", side_effect=lambda models: {
                        role: checkpoint_info(spec.checkpoint) for role, spec in models.items()}), \
                    patch("ad_eval.self_play.range_identity", return_value={"images": "fixed"}), \
                    redirect_stdout(StringIO()):
                matches = run_campaign(args, {})
            self.assertEqual(len(matches), 20)
            self.assertEqual(cai_models, ["Qwen/Qwen3-8B"] * 10)
            self.assertEqual(len(training_jobs), 40)
            self.assertEqual(len(bouts), 80)
            summary = json.loads((Path(args.out_dir) / "summary.json").read_text())
            self.assertEqual(summary["completed_grpo_updates"], 40)
            self.assertEqual(summary["progress"]["total_completed"], 20)
            state = json.loads((Path(args.out_dir) / "learning" / "state.json").read_text())
            self.assertEqual(len(state["strategy_updates"]), 20)
            self.assertTrue(all(update["status"] == "accepted" for update in state["strategy_updates"]))
            for role in ("red", "blue"):
                current, depth = state["planners"][role]["tree"]["incumbent"], 0
                while state["planners"][role]["tree"]["nodes"][current]["parent_id"]:
                    depth += 1
                    current = state["planners"][role]["tree"]["nodes"][current]["parent_id"]
                self.assertEqual(depth, 10)


if __name__ == "__main__":
    unittest.main()
