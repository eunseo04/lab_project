import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ad_eval.agents import make_workers
from ad_eval.cai_parallel import LocalTraceSink
from ad_eval.evidence import flags_and_access
from ad_eval.executor import execute
from ad_eval.llm import LLMClient, LLMConfig, endpoint_base
from ad_eval.mcts import CyberState, MCTS, Objective, load_shared_memory
from ad_eval.models import ModelSpec, checkpoint_info, select_model, verify_model
from ad_eval.run import read_events, usage_totals
from ad_eval.scoring import availability_score
from ad_eval.wazuh import WazuhClient, WazuhConfig, verify_lab_ingestion

ROOT = Path(__file__).resolve().parents[1]


class RefactorTests(unittest.TestCase):
    def test_runtime_has_no_legacy_imports_or_gpu_load(self):
        result = subprocess.run([sys.executable, "-c",
            "import sys; import ad_eval.run, ad_eval.proposed_worker; "
            "assert not {'pipeline','llm','redtools','wazuh_client','torch','transformers'} & sys.modules.keys()"],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_dependency_audit(self):
        from scripts.audit_dependencies import audit
        report = audit()
        self.assertEqual(report["evaluation_legacy_imports"], [])
        self.assertEqual(report["missing_local_modules"], [])

    def test_endpoint_normalization_and_hf_name(self):
        self.assertEqual(endpoint_base("http://127.0.0.1:8000/v1/"), "http://127.0.0.1:8000/v1")
        with self.assertRaises(ValueError):
            endpoint_base("file:///tmp/model")
        args = SimpleNamespace(base_model="openai/Qwen/Qwen3-8B", endpoint="http://localhost:8000",
                               trained_model="qwen3-grpo", trained_endpoint=None, grpo_checkpoint=None)
        self.assertEqual(select_model(args, "cai").model, "Qwen/Qwen3-8B")
        self.assertEqual(select_model(args, "proposed").model, "qwen3-grpo")
        args.trained_model = "Qwen/Qwen3-8B"
        with self.assertRaises(ValueError):
            select_model(args, "proposed")

    def test_native_cai_two_sessions_and_proposed_four_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            creds = {"1": "BLUE_SECRET_1", "2": "BLUE_SECRET_2"}
            cai = make_workers("cai", Path(directory), "Qwen/Qwen3-8B", "http://localhost:8000/v1",
                               blue_credentials=creds)
            self.assertEqual(len(cai), 2)
            for worker in cai:
                self.assertIn("ad_eval.cai_parallel", worker.command)
                self.assertNotIn("/v1/v1", worker.env["OPENAI_BASE_URL"])
                config = json.loads(Path(worker.command[-1]).read_text())
                self.assertEqual([x["name"] for x in config["parallel_agents"]],
                                 ["redteam_agent", "blueteam_agent"])
                self.assertTrue(all(not x["unified_context"] for x in config["parallel_agents"]))
                self.assertTrue(all(secret not in config["parallel_agents"][0]["prompt"]
                                    for secret in creds.values()))
            proposed = make_workers("proposed", Path(directory), "qwen3-grpo", "http://localhost:8000",
                                    blue_credentials=creds)
            self.assertEqual(len(proposed), 4)
            self.assertTrue(all(w.env["QWEN_ENDPOINT_MODEL"] == "qwen3-grpo" for w in proposed))
            self.assertTrue(all("AD_EVAL_BLUE_PASSWORD" not in w.env for w in proposed if w.role == "red"))

    def test_endpoint_failure_raises_instead_of_fallback(self):
        client = LLMClient(LLMConfig(endpoint="http://localhost:8000", model="qwen"))
        with patch("ad_eval.llm.requests.post", side_effect=RuntimeError("offline")):
            with self.assertRaisesRegex(RuntimeError, "offline"):
                client.complete("test")

    def test_token_usage_and_explicit_price_reach_team_totals(self):
        client = LLMClient(LLMConfig(endpoint="http://localhost:8000", model="qwen"))
        events = []
        client.event_callback = lambda **event: events.append(event)
        response = Mock()
        response.json.return_value = {"usage": {"prompt_tokens": 12, "completion_tokens": 3},
                                      "choices": [{"message": {"content": "CMD: true"}}]}
        with patch("ad_eval.llm.requests.post", return_value=response):
            client.complete("test")
        totals = usage_totals({"red": events, "blue": events}, 1., 2.)
        self.assertEqual(totals["input_tokens"], 24)
        self.assertEqual(totals["output_tokens"], 6)
        self.assertEqual(totals["cost_usd"], 0.000036)
        self.assertIsNone(usage_totals({"red": events, "blue": events}, None, None)["cost_usd"])
        incomplete = [*events, {"kind": "llm_request_started"}]
        self.assertIsNone(usage_totals({"red": incomplete}, 1., 2.)["input_tokens"])

    def test_unknown_usage_is_not_fabricated_zero(self):
        result = usage_totals({"red": [], "blue": []}, 0., 0.)
        self.assertIsNone(result["input_tokens"])
        self.assertIsNone(result["cost_usd"])

    def test_executor_cannot_fall_back_to_host(self):
        with self.assertRaises(ValueError):
            execute("id", container="")

    def test_frozen_dataset_does_not_expose_canonical_exploits(self):
        memory = load_shared_memory()
        self.assertTrue(memory)
        self.assertTrue(all("command" not in info for info in memory.values()))

    def test_multistep_search_keeps_stage_prerequisites_and_is_reproducible(self):
        memory = {
            "T1595": {"name": "Active Scanning", "tactics": ["reconnaissance"]},
            "T1190": {"name": "Exploit Public-Facing Application", "tactics": ["initial-access"]},
            "T1548": {"name": "Abuse Elevation Control Mechanism", "tactics": ["privilege-escalation"]},
        }
        for stage, caps, tid in [(0, (), "T1595"),
                  (2, ("recon", "infra"), "T1190"),
                  (5, ("recon", "infra", "access", "exec", "persistence"), "T1548")]:
            def search():
                return MCTS(memory, n_sims=32, rng=random.Random(7),
                            objective=Objective(capture_tactic="privilege-escalation")).search(
                                CyberState(foothold=stage, capabilities=caps))
            action, value = search()
            self.assertEqual(action["tech_id"], tid)
            self.assertTrue(0 <= value <= 1)
            self.assertEqual((action, value), search())


class EvidenceCorrelationTests(unittest.TestCase):
    def setUp(self):
        self.tool = {"role": "red", "target_ip": "172.32.10.20", "command": "curl target",
                     "tool_call_id": "tool-1", "output": "uid=1000(ctf)\nAD_EXECUTION_ID=abc\n",
                     "started_utc": "2026-11-03T00:00:00+00:00",
                     "ended_utc": "2026-11-03T00:00:03+00:00"}
        self.host = {"execution_id": "abc", "injection": True, "uid_observed": True,
                     "source_ip": "172.32.10.101", "command": "ping ; id",
                     "started_utc": "2026-11-03T00:00:01+00:00",
                     "ended_utc": "2026-11-03T00:00:02+00:00"}

    def result(self):
        return flags_and_access(self.tool["output"], "EXEC inj=1", {"FLAG{user_real}"},
                                {"FLAG{root_real}"}, tool_events=[self.tool], host_events=[self.host],
                                target_ip="172.32.10.20", source_ip="172.32.10.101")

    def test_same_execution_target_and_time_establish_access(self):
        self.assertTrue(self.result()["initial_access"])
        self.assertEqual(len(self.result()["evidence"]["linked_executions"]), 1)

    def test_unrelated_uid_marker_or_time_requires_review(self):
        for field, value in [("execution_id", "different"), ("source_ip", "172.32.10.102"),
                             ("started_utc", "2026-11-03T01:00:00+00:00")]:
            before = self.host[field]
            self.host[field] = value
            self.assertIsNone(self.result()["initial_access"])
            self.host[field] = before

    def test_exact_flag_match_not_substring(self):
        result = flags_and_access("FLAG{user_real}suffix", "", {"FLAG{user_real}"}, set())
        self.assertTrue(result["user_flag"])
        result = flags_and_access("FLAG{user_realextra}", "", {"FLAG{user_real}"}, set())
        self.assertFalse(result["user_flag"])

    def test_invalid_monitor_samples_never_award_availability(self):
        for samples in ([{"elapsed_s": float("nan"), "up": True}],
                        [{"elapsed_s": -1, "up": True}],
                        [{"elapsed_s": 0, "up": True}, {"elapsed_s": 0, "up": True}]):
            self.assertIsNone(availability_score(samples, 60, 10)["availability_points"])


class NativeTracingTests(unittest.TestCase):
    def test_native_agent_trace_separates_roles_and_collects_actual_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            sink = LocalTraceSink(Path(directory), "1")
            agent = SimpleNamespace(span_id="agent", parent_id=None,
                                    span_data=SimpleNamespace(type="agent", name="Red Team Agent #1"))
            sink.on_span_start(agent)
            generation = SimpleNamespace(span_id="gen", parent_id="agent", error=None,
                span_data=SimpleNamespace(type="generation", usage={"input_tokens": 10, "output_tokens": 5}, output=[]))
            sink.on_span_start(generation)
            sink.on_span_end(generation)
            tool = SimpleNamespace(span_id="tool", parent_id="agent", error=None,
                started_at="2026-11-03T00:00:00Z", ended_at="2026-11-03T00:00:02Z",
                span_data=SimpleNamespace(type="function", input='{"command":"curl target"}', output="uid=1000(ctf)"))
            sink.on_span_start(tool)
            sink.on_span_end(tool)
            events = read_events(Path(directory) / "team1_red.events.jsonl")
            self.assertEqual(usage_totals({"red": events}, 0., 0.)["input_tokens"], 10)
            self.assertEqual([e for e in events if e["kind"] == "tool_result"][0]["target_ip"], "172.32.10.20")
            self.assertFalse((Path(directory) / "team1_blue.events.jsonl").exists())


class ModelRegistrationTests(unittest.TestCase):
    def test_checkpoint_weights_and_server_root_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "adapter_config.json").write_text('{"base_model_name_or_path":"Qwen/Qwen3-8B"}')
            with self.assertRaises(ValueError):
                checkpoint_info(path)
            (path / "adapter_model.safetensors").write_bytes(b"test-fixture")
            self.assertIn("adapter_model.safetensors", checkpoint_info(path)["sha256"])
            spec = ModelSpec("grpo", "http://localhost:8000/v1", "grpo", directory)
            response = Mock()
            response.json.return_value = {"data": [{"id": "grpo", "root": directory}]}
            with patch("ad_eval.models.requests.get", return_value=response):
                self.assertEqual(verify_model(spec)["kind"], "grpo")
                response.json.return_value = {"data": [{"id": "grpo", "root": "/wrong-checkpoint"}]}
                with self.assertRaises(RuntimeError):
                    verify_model(spec)


class WazuhIntegrationTests(unittest.TestCase):
    def test_host_scoped_query_keeps_millisecond_precision(self):
        from datetime import datetime, timezone
        config = WazuhConfig()
        config.password = "test"
        client = WazuhClient(config)
        response = Mock()
        response.json.return_value = {"hits": {"hits": []}}
        with patch("ad_eval.wazuh.requests.post", return_value=response) as post:
            client.query_alerts(datetime(2026, 11, 3, microsecond=123000, tzinfo=timezone.utc), "mayhem-host-a")
        body = post.call_args.kwargs["json"]
        self.assertIn({"term": {"agent.name": "mayhem-host-a"}}, body["query"]["bool"]["must"])
        self.assertIn(".123Z", str(body))

    def test_registered_hosts_must_emit_fresh_mitre_tagged_canaries(self):
        client = Mock()
        client.ping.return_value = True
        client.last_error = None
        client.technique_ids.side_effect = WazuhClient.technique_ids
        tokens = {}
        def execute(container, argv, **kwargs):
            ip = argv[argv.index("-G") + 1]
            team = "1" if ".10:8080" in ip else "2"
            tokens[team] = argv[-1].split("printf ")[1]
            return "ok"
        def query(**kwargs):
            name = kwargs["agent_name"]
            team = "1" if name.endswith("-a") else "2"
            return [{"agent": {"name": name, "id": team}, "full_log": tokens[team],
                     "rule": {"id": "100714", "mitre": {"id": ["T1190"]}}}]
        client.query_alerts.side_effect = query
        with patch("ad_eval.docker_lab.dexec", side_effect=execute):
            self.assertEqual(set(verify_lab_ingestion(client, timeout=2)), {"1", "2"})
        client.query_alerts.return_value = []
        client.query_alerts.side_effect = lambda **kwargs: [{"agent": {"name": kwargs["agent_name"]},
                                                           "full_log": "missing-mitre"}]
        with patch("ad_eval.docker_lab.dexec", side_effect=execute):
            with self.assertRaisesRegex(RuntimeError, "MITRE"):
                verify_lab_ingestion(client, timeout=.01)

    @unittest.skipUnless(os.environ.get("AD_EVAL_LIVE_WAZUH") == "1", "opt-in live Docker/Wazuh integration")
    def test_live_both_agents_registered_and_mitre_events_ingested(self):
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        result = verify_lab_ingestion(WazuhClient(), timeout=60)
        self.assertTrue(all("T1190" in row["technique_ids"] for row in result.values()))


if __name__ == "__main__":
    unittest.main()
