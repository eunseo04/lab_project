import asyncio
from collections import Counter
from contextvars import ContextVar
from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from ad_eval.agents import make_workers
from ad_eval.cai_parallel import install_inference_controls
from ad_eval.inference import InferenceSettings
from ad_eval.llm import LLMClient, LLMConfig
from ad_eval.run import (build_parser, campaign_plan, collect_records, inference_audit,
                         inference_settings, run_campaign, validate_args)


def arguments(*extra):
    args = build_parser().parse_args(["--base-model", "Qwen/Qwen3-8B", "--trained-model", "grpo",
                                     "--no-adaptive-proposed", *extra])
    validate_args(args)
    return args


class CampaignTests(unittest.TestCase):
    def test_ten_complete_workflows_each_with_paired_settings_and_balanced_order(self):
        args = arguments()
        schedule = campaign_plan(args)
        self.assertEqual(len(schedule), 20)
        self.assertEqual(Counter(item["condition"] for item in schedule), {"cai": 10, "proposed": 10})
        self.assertEqual([x["condition"] for x in schedule[:4]], ["cai", "proposed", "proposed", "cai"])
        for rep in range(1, 11):
            pair = [x for x in schedule if x["rep"] == rep]
            self.assertEqual(pair[0]["inference"], pair[1]["inference"])
        with tempfile.TemporaryDirectory() as directory:
            args.out_dir = directory
            def completed(_args, condition, rep, _labels):
                return {"condition": condition, "rep": rep, "teams": []}
            with patch("ad_eval.run.run_one", side_effect=completed) as workflow, redirect_stdout(StringIO()):
                run_campaign(args, {})
            self.assertEqual(workflow.call_count, 20)
            summary = json.loads((Path(directory) / "summary.json").read_text())
            self.assertTrue(summary["progress"]["complete"])
            self.assertEqual(summary["progress"]["completed_by_condition"], {"cai": 10, "proposed": 10})
            self.assertEqual(summary["closed_loop"]["model_weights"], {"cai": "fixed", "proposed": "fixed"})

    def test_worker_configuration_overrides_conflicting_ambient_settings(self):
        policy = InferenceSettings(temperature=.25, top_p=.8, max_tokens=256, seed=100)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                "QWEN_TEMPERATURE": "1.8", "QWEN_TOP_P": ".1", "CAI_TEMPERATURE": "1.7"}):
            creds = {"1": "a", "2": "b"}
            base = make_workers("cai", Path(directory), "Qwen/Qwen3-8B", "http://localhost:8000",
                                blue_credentials=creds, inference=policy)
            proposed = make_workers("proposed", Path(directory), "grpo", "http://localhost:8000",
                                    blue_credentials=creds, inference=policy)
            for worker in base:
                self.assertEqual(json.loads(worker.env["AD_EVAL_INFERENCE"]), policy.to_dict())
            for worker in proposed:
                expected = policy.for_role(worker.team, worker.role)
                self.assertEqual(json.loads(worker.env["AD_EVAL_INFERENCE"]), expected.to_dict())
                with patch.dict(os.environ, worker.env):
                    self.assertEqual(LLMConfig.from_env().inference_settings(), expected)

    def test_invalid_generation_parameters_are_rejected_before_running(self):
        for params in ({"temperature": float("nan")}, {"top_p": 0}, {"top_k": 0},
                       {"max_tokens": 0}, {"seed": -1}, {"min_p": 2}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                InferenceSettings(**params)

    def test_inference_audit_requires_actual_request_logs_for_completed_calls(self):
        policy = InferenceSettings()
        declaration = {"kind": "inference_settings", "inference": policy.request_fields()}
        usage = {"kind": "llm_usage", "requests": 1}
        self.assertFalse(inference_audit([declaration, usage], policy)["valid"])
        request = {"kind": "inference_request", "inference": policy.request_fields()}
        self.assertTrue(inference_audit([declaration, usage, request], policy)["valid"])
        request["inference"]["temperature"] = 1.5
        self.assertFalse(inference_audit([declaration, usage, request], policy)["valid"])

    def test_mismatched_inference_invalidates_both_teams(self):
        args = arguments()
        policy = inference_settings(args)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            for team in ("1", "2"):
                for role in ("red", "blue"):
                    fields = policy.for_role(team, role).request_fields()
                    if (team, role) == ("1", "red"):
                        fields["top_p"] = .1
                    events = [{"kind": "agent_started"}, {"kind": "inference_settings", "inference": fields}]
                    (out / f"team{team}_{role}.events.jsonl").write_text(
                        "\n".join(json.dumps(e) for e in events))
            flags = {t: {"user": {"FLAG{user_real}"}, "root": {"FLAG{root_real}"}} for t in ("1", "2")}
            with patch("ad_eval.run.patch_evidence", return_value={"vulnerability_patched": False}):
                records = collect_records(args, "cai", 1, out, {"1": [], "2": []}, flags, {},
                                          {"1": "", "2": ""}, {"1": [], "2": []})
            self.assertTrue(all("invalid_match" in row["evidence"] for row in records))
            self.assertTrue(all(row["initial_access"] is None for row in records))


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_cai_and_proposed_send_identical_generation_fields(self):
        policy = InferenceSettings(temperature=.25, top_p=.8, max_tokens=256, seed=73)
        role = ContextVar("test_role", default="red")
        completions = SimpleNamespace(acompletion=AsyncMock(return_value={}), completion=Mock(return_value={}))
        original = completions.acompletion
        install_inference_controls(completions, policy, "2", role, Mock())
        await completions.acompletion(model="openai/Qwen/Qwen3-8B", temperature=1.9, top_p=.1,
                                      max_tokens=9999, max_completion_tokens=9999,
                                      extra_body={"temperature": 2., "chat_template_kwargs": {"enable_thinking": True}})
        cai_call = original.call_args.kwargs
        flattened = {**cai_call, **cai_call["extra_body"]}
        self.assertNotIn("max_completion_tokens", cai_call)
        with patch.dict(os.environ, {**policy.for_role("2", "red").to_env(),
                                     "QWEN_ENDPOINT": "http://localhost:8000", "QWEN_ENDPOINT_MODEL": "grpo"}):
            client = LLMClient()
        response = Mock()
        response.json.return_value = {"usage": {"prompt_tokens": 1, "completion_tokens": 1},
                                      "choices": [{"message": {"content": "CMD: true"}}]}
        with patch("ad_eval.llm.requests.post", return_value=response) as post:
            client.complete("test")
        proposed_call = post.call_args.kwargs["json"]
        for key, expected in policy.for_role("2", "red").request_fields().items():
            self.assertEqual(flattened[key], expected, key)
            self.assertEqual(proposed_call[key], expected, key)
        with self.assertRaises(ValueError):
            client.complete("test", temperature=1.9)

    async def test_concurrent_red_blue_calls_keep_their_own_paired_seeds(self):
        policy = InferenceSettings(seed=100)
        active = ContextVar("concurrent_role", default=None)
        received = []
        async def capture(**kwargs):
            received.append((kwargs["model"], kwargs["seed"]))
        completions = SimpleNamespace(acompletion=capture, completion=Mock())
        install_inference_controls(completions, policy, "1", active, Mock())
        async def call(role):
            token = active.set(role)
            try:
                await asyncio.sleep(0)
                await completions.acompletion(model=role)
            finally:
                active.reset(token)
        await asyncio.gather(call("red"), call("blue"))
        self.assertCountEqual(received, [("red", 100), ("blue", 101)])

    @unittest.skipUnless(os.environ.get("AD_EVAL_CAI_RUNTIME") == "1", "opt-in installed CAI adapter integration")
    async def test_installed_cai_adapter_retains_controlled_fields_after_its_filter(self):
        import litellm
        from cai.sdk.agents import ModelSettings
        from cai.sdk.agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
        policy = InferenceSettings()
        with patch.object(litellm, "acompletion", new=AsyncMock(return_value={})) as dispatch:
            restore = install_inference_controls(litellm, policy, "1", ContextVar("runtime_role", default="red"), Mock())
            try:
                model = OpenAIChatCompletionsModel.__new__(OpenAIChatCompletionsModel)
                model.model = "openai/Qwen/Qwen3-8B"
                with patch.dict(os.environ, {"OLLAMA_API_BASE": "http://localhost:8000/v1"}):
                    await model._fetch_response_litellm_ollama(
                        {"model": model.model, "messages": [], "stream": False}, ModelSettings(), None, False, False)
                actual = dispatch.call_args.kwargs
                flattened = {**actual, **actual["extra_body"]}
                for key, expected in policy.request_fields().items():
                    self.assertEqual(flattened[key], expected, key)
            finally:
                restore()


class ClosedLoopTests(unittest.TestCase):
    def test_proposed_red_next_decision_uses_actual_previous_tool_result(self):
        from ad_eval import proposed_worker as worker
        prompts = []
        def complete(prompt, **kwargs):
            prompts.append(prompt)
            return "CMD: curl http://172.32.10.20:8080/health"
        llm = SimpleNamespace(check=Mock(), inference=InferenceSettings(), complete=complete,
                              usage_complete=True, usage={"input_tokens": 0, "output_tokens": 0, "requests": 0})
        planner = Mock()
        planner.last_search = {}
        planner.search.return_value = ({"tech_id": "T1595", "name": "Scan"}, .5)
        with patch.dict(os.environ, {}, clear=True), patch.object(worker, "get_llm", return_value=llm), \
                patch.object(worker.P, "load_shared_memory", return_value={"T1595": {"name": "Scan"}}), \
                patch.object(worker.P, "MCTS", return_value=planner), \
                patch.object(worker.executor, "run_command", side_effect=["8080/tcp open", "uid=1000(ctf)", KeyboardInterrupt]), \
                redirect_stdout(StringIO()), self.assertRaises(KeyboardInterrupt):
            worker.run("1", "red")
        self.assertIn("Current observation: none", prompts[0])
        self.assertIn("8080/tcp open", prompts[1])
        self.assertIn("initial-access", prompts[1])
        self.assertIn("uid=1000(ctf)", prompts[2])
        self.assertIn("privilege-escalation", prompts[2])

    def test_proposed_blue_next_decision_uses_ssh_result_and_new_siem_observation(self):
        from ad_eval import proposed_worker as worker
        prompts = []
        def complete(prompt, **kwargs):
            prompts.append(prompt)
            return "CMD: printf checked"
        llm = SimpleNamespace(check=Mock(), inference=InferenceSettings(), complete=complete,
                              usage_complete=True, usage={"input_tokens": 0, "output_tokens": 0, "requests": 0})
        planner = Mock()
        planner.last_search = {}
        planner.search.return_value = ({"tech_id": "T1190", "name": "Contain"}, .5)
        wazuh = Mock(available=True, last_error=None)
        wazuh.query_alerts.return_value = [{"rule": {"description": "Observed injection", "mitre": {"id": ["T1190"]}}}]
        wazuh.technique_ids.return_value = {"T1190"}
        with patch.dict(os.environ, {"AD_EVAL_BLUE_PASSWORD": "test"}, clear=True), \
                patch.object(worker, "get_llm", return_value=llm), \
                patch.object(worker.P, "load_shared_memory", return_value={"T1190": {"name": "Exploit"}}), \
                patch.object(worker.P, "MCTS", return_value=planner), \
                patch("ad_eval.wazuh.get_wazuh_client", return_value=wazuh), \
                patch.object(worker, "blue_ssh_exec", side_effect=["inspection-result", "patch-result", KeyboardInterrupt]), \
                patch.object(worker.time, "sleep"), redirect_stdout(StringIO()), self.assertRaises(KeyboardInterrupt):
            worker.run("1", "blue")
        self.assertIn("inspection-result", prompts[1])
        self.assertIn("patch-result", prompts[2])
        self.assertTrue(all("Observed injection" in prompt for prompt in prompts))


if __name__ == "__main__":
    unittest.main()
