import json
from pathlib import Path
import socket
import sys

import numpy as np
import pytest
import torch

from acd import ROOT
from acd.backends import ApiBackend, MockBackend, ReplayBackend, create_backend
from acd.communication import create_message
from acd.config import validate
from acd.environment import BLUE_AGENTS, make_env
from acd.llm_agent import parse_response
from acd.runner import evaluate, load_agents
from CybORG.Shared.Enums import TernaryEnum
from CybORG.Simulator.Actions import Sleep


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline experiments must not open a network connection")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    torch.set_num_threads(1)


def test_paper_wire_severity_busy_unknown_and_remote_owners():
    data = {"success": TernaryEnum.IN_PROGRESS,
        "own_host": {"Files": [{"File Name": "cmd.sh"}, {"File Name": "escalate.exe"}],
        "Processes": [{"Connections": [{"remote_address": "10.1.2.3"}]},
                      {"Connections": [{"remote_address": "203.0.113.99"}]}]}}
    wire = create_message(data, {"operational_zone_b_subnet_user_host_0": "10.1.2.3"})
    assert wire.tolist() == [False, False, False, True, False, True, True, True]
    assert wire.dtype == np.bool_


def test_scan_is_01_and_user_compromise_is_10():
    scan = {"host": {"Processes": [{"Connections": [{"remote_address": "1.2.3.4"}]}] * 2}}
    assert create_message(scan, {})[5:7].tolist() == [False, True]
    assert create_message({"host": {"Files": [{"File Name": "cmd.sh"}]}}, {})[5:7].tolist() == [True, False]


@pytest.mark.parametrize("protocol,dimension", [("keep", 192), ("paper", 197)])
def test_all_five_agents_single_tick_reset_and_termination(protocol, dimension):
    env = make_env(seed=42, steps=6, communication=protocol)
    obs, _ = env.reset(seed=42)
    assert {o[0][0].shape[1] for o in obs.values()} == {dimension}
    for i in range(6):
        obs, reward, term, trunc, _ = env.step({a: Sleep() for a in BLUE_AGENTS})
        assert env.env.environment_controller.step_count == i + 1
        assert set(reward) == set(BLUE_AGENTS)
        assert "message" in env.env.get_observation("blue_agent_4")
    assert trunc["__all__"]


def test_eight_bits_are_not_interpreted_as_native_keep_checkbits():
    env = make_env(seed=42, steps=6, communication="paper")
    wire = np.arange(32).reshape(4, 8) % 2
    decoded = env._message_features(0, wire)
    assert np.array_equal(decoded[:3], wire[:3])
    assert np.array_equal(decoded[3:], np.repeat(wire[-1:], 3, axis=0))
    assert np.array_equal(env._message_features(4, wire), wire)


@pytest.mark.parametrize("response", ["nonsense", '{"action":"Sleep"}',
    '{"action":"Remove host:other_host","reason":"x"}',
    '{"action":"Restore host:allowed Sleep","reason":"x"}'])
def test_invalid_response_is_logged_sleep(response):
    action, _, error = parse_response(response, "blue_agent_0", ["allowed"], [])
    assert isinstance(action, Sleep) and error


def test_parser_preserves_json_reason_and_builds_real_action():
    action, reason, error = parse_response(json.dumps({"action": "Analyse host:allowed",
        "reason": 'Comma, quote: "ok"'}), "blue_agent_0", ["allowed"], [])
    assert type(action).__name__ == "Analyse"
    assert action.hostname == "allowed" and error is None
    assert reason == 'Comma, quote: "ok"'


def test_api_requires_manual_implementation_and_never_imports_sdk():
    with pytest.raises(ValueError):
        create_backend({"backend": "api", "enabled": False})
    with pytest.raises(RuntimeError, match="disabled"):
        ApiBackend({}).generate([])
    assert not {"openai", "wandb", "weave", "transformers"}.intersection(sys.modules)


def test_replay_uses_context_and_rejects_missing_rows(tmp_path):
    p = tmp_path / "responses.jsonl"
    p.write_text(json.dumps({"episode": 0, "step": 1, "agent": "blue_agent_4", "response": "test"}) + "\n")
    backend = ReplayBackend(p)
    assert backend.generate([], episode=0, step=1, agent="blue_agent_4") == "test"
    with pytest.raises(ValueError):
        backend.generate([], episode=0, step=2, agent="blue_agent_4")


def test_published_weights_require_native_message_features():
    config = validate({"team": "all_rl", "communication": "keep"})
    agents, metadata = load_agents(config, 192)
    assert len(agents) == len(metadata) == 5
    with pytest.raises(ValueError, match="192 features"):
        load_agents(config, 197)


def test_mock_pipeline_counts_steps_rewards_and_is_reproducible(tmp_path):
    config = validate({"name": "test", "team": "all_llm", "communication": "paper",
                       "episodes": 1, "steps": 9, "seed": 42})
    folder, result = evaluate(config, tmp_path / "first")
    _, repeated = evaluate(config, tmp_path / "second")
    assert result["episodes"] == repeated["episodes"]
    assert result["result_kind"] == "offline_mock_pipeline_only"
    records = [json.loads(line) for line in (folder / "trace.jsonl").read_text().splitlines()]
    assert len(records) == 9 * 5
    assert result["episodes"][0]["simulator_steps"] == 9
    assert result["episodes"][0]["mission_phases"] == [0, 1, 2]
    assert sum(r["reward"] for r in records) / 5 == result["reward"]["mean"]
    assert not result["invalid_responses"]
    assert {r["simulator_step"] for r in records if r["step"] == 8} == {9}


def test_native_graph_matches_unmodified_keep_graph():
    # Strong parity check: compare full graph tensors, rewards and the original
    # published actor distributions on the same observations and action stream.
    sys.path.insert(0, str(ROOT / "third_party/cage-4-submission"))
    try:
        from wrapper.graph_wrapper import GraphWrapper as OriginalGraph
        from models.cage4 import load as original_load
    finally:
        sys.path.pop(0)
    left = make_env(seed=7, steps=9, communication="keep", horizon="native")
    right_container = make_env(seed=7, steps=9, communication="keep", horizon="native")
    right = OriginalGraph(right_container.env)
    # Both constructors consumed the same RNG sequence; the original reset()
    # does not accept a seed, so reset both without reseeding for this parity run.
    lobs, _ = left.reset()
    robs, _ = right.reset()
    config = validate({"team": "all_rl", "communication": "keep"})
    agents, _ = load_agents(config, 192)
    original_agents = {a: original_load(str(ROOT / f"third_party/cage-4-submission/weights/gnn_ppo-{a[-1]}.pt")) for a in BLUE_AGENTS}
    for _ in range(8):
        for name in BLUE_AGENTS:
            for a, b in zip(lobs[name][0], robs[name][0]):
                assert torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b
            with torch.no_grad():
                assert torch.equal(agents[name].actor(*lobs[name][0]).probs,
                                   original_agents[name].actor(*robs[name][0]).probs)
        actions = {a: None for a in BLUE_AGENTS}
        lobs, lr, _, _, _ = left.step(actions)
        robs, rr, _, _, _ = right.step(actions)
        assert lr == rr
