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


@pytest.mark.parametrize('case,bits',[
    ('pid',(1,0,0)),('duplicate_first_remote',(1,0,0)),('duplicate_second_remote',(0,0,0)),
    ('user_file',(1,0,0)),('admin_file',(1,1,0)),('sessions',(1,0,0)),('metadata_only',(0,0,1)),
    ('two_keys',(0,0,0)),
])
def test_original_encoder_matches_pinned_observation_on_actual_runtime(case,bits):
    from ipaddress import IPv4Address
    from CybORG.Agents.LLMAgents.comm_vector import CommVectorGenerator
    from CybORG.Shared.Observation import Observation
    from acd.config import ORIGINAL_ENCODER
    one,two,three=map(IPv4Address,('10.0.0.1','10.0.0.2','10.0.0.3'))
    cases={
        'pid':{'Processes':[{'PID':7}]},
        'duplicate_first_remote':{'Processes':[{'Connections':[{'remote_address':one}]}]*2},
        'duplicate_second_remote':{'Processes':[{'Connections':[{'remote_address':ip},{'remote_address':three}]} for ip in (one,two)]},
        'user_file':{'Files':[{'File Name':'cmd.sh'}]},'admin_file':{'Files':[{'File Name':'escalate.exe'}]},
        'sessions':{'Sessions':[{},{}]},'metadata_only':{},'two_keys':{'Processes':[{'PID':7}]},
    }
    raw={'success':TernaryEnum.IN_PROGRESS if case=='metadata_only' else TernaryEnum.UNKNOWN,'host':cases[case]}
    if case!='two_keys':raw.update(message=[],phase=1,action=None)
    mapping={'operational_zone_b_subnet_user_host_0':one,'office_network_subnet_user_host_0':two}
    observation=Observation();observation.data=raw
    direct=CommVectorGenerator.create_comm_message(observation,None,mapping)
    for supplied in (observation,raw):
        actual=create_message(supplied,mapping,ORIGINAL_ENCODER)
        assert actual.dtype==np.bool_ and actual.shape==(8,)
        assert actual.tolist()==direct.astype(bool).tolist()
        assert tuple(actual[5:])==bits
        if case=='duplicate_first_remote':assert actual[3]
    assert observation.data is raw


@pytest.mark.parametrize('raw,error',[
    ({},KeyError),
    ({'success':TernaryEnum.UNKNOWN,'phase':1,'host':{'Processes':[{'Connections':[{'remote_address':'unmapped'}]}]*2}},StopIteration),
    ({'success':TernaryEnum.UNKNOWN,'phase':1,'host':{'Processes':[{'Connections':[]}]}},IndexError),
])
def test_original_encoder_preserves_source_exceptions(raw,error):
    from acd.config import ORIGINAL_ENCODER
    with pytest.raises(error):create_message(raw,{},ORIGINAL_ENCODER)


def test_original_wire_reaches_all_five_outgoing_and_self_graph_inputs(monkeypatch):
    from copy import deepcopy
    from CybORG.Agents.LLMAgents.comm_vector import CommVectorGenerator
    from research.adapter import make_environment
    from research.config import variant_config
    from acd.config import ORIGINAL_ENCODER
    env=make_environment({**variant_config('B0'),'steps':6},42)
    assert env.communication_encoder==ORIGINAL_ENCODER
    env.reset(seed=42)
    ec=env.env.environment_controller
    # The source receives each agent's real Observation and original map.
    actual=CommVectorGenerator.create_comm_message
    calls=[]
    def record(obs,last,mapping):
        calls.append((obs,mapping));return actual(obs,last,mapping)
    monkeypatch.setattr(CommVectorGenerator,'create_comm_message',record)
    expected={agent:actual(ec.get_last_observation(agent),None,ec.hostname_ip_map).astype(bool) for agent in BLUE_AGENTS}
    original_step=env.env.parallel_step
    delivered=[]
    def step(*args,**kwargs):
        delivered.append(deepcopy(kwargs['messages']));return original_step(*args,**kwargs)
    monkeypatch.setattr(env.env,'parallel_step',step)
    graphs,*_=env.step({agent:Sleep() for agent in BLUE_AGENTS})
    assert len(calls)==10 and all(mapping is ec.hostname_ip_map for _,mapping in calls)
    assert set(delivered[0])==set(BLUE_AGENTS)
    assert all(np.array_equal(delivered[0][agent],expected[agent]) for agent in BLUE_AGENTS)
    assert all(np.array_equal(env.msg[agent],actual(ec.get_last_observation(agent),None,ec.hostname_ip_map)) for agent in BLUE_AGENTS)
    assert {entry[0][0].shape[1] for entry in graphs.values()}=={192}
    assert ec.step_count==1


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


@pytest.mark.parametrize('command',['BlockTrafficZone','AllowTrafficZone'])
@pytest.mark.parametrize('agent',BLUE_AGENTS)
def test_traffic_parser_matches_original_command_parameters(command,agent):
    import ast
    from types import SimpleNamespace
    from CybORG.Simulator.Actions.ConcreteActions.ControlTraffic import AllowTrafficZone, BlockTrafficZone
    from CybORG.Agents.LLMAgents.llm_adapter.utils.constants import CAGE4_SUBNETS

    pinned = ROOT/'third_party/llms-are-acd/CybORG/Agents/LLMAgents/llm_policy.py'
    policy_class = next(node for node in ast.parse(pinned.read_text()).body
                        if isinstance(node, ast.ClassDef) and node.name=='LLMDefenderPolicy')
    initializer, extractor = [next(node for node in policy_class.body if node.name==name)
                              for name in ('__init__','extract_action')]
    fixed_source = next(node for node in initializer.body if isinstance(node, ast.Assign)
                        and any(isinstance(t, ast.Attribute) and t.attr=='subnet' for t in node.targets))
    scope = {'self':SimpleNamespace(name=agent), 'CAGE4_SUBNETS':CAGE4_SUBNETS,
             'BlockTrafficZone':BlockTrafficZone, 'AllowTrafficZone':AllowTrafficZone,
             'Logger':SimpleNamespace(debug=lambda *_:None),
             'wandb':SimpleNamespace(log=lambda *_, **__:None, run=SimpleNamespace(step=0))}
    exec(compile(ast.Module(body=[fixed_source,extractor],type_ignores=[]),str(pinned),'exec'),scope)
    dest='admin_network_subnet' if command=='BlockTrafficZone' else 'office_network_subnet'
    response=json.dumps({'action':f'{command} subnet:{dest}','reason':'fixture'})
    expected=scope['extract_action'](scope['self'],response)
    action,_,error=parse_response(response,agent,[],[dest])
    assert error is None
    assert type(action) is type(expected)
    assert (type(action).__name__,action.agent,action.session,action.from_subnet,action.to_subnet)==(
        command,agent,0,CAGE4_SUBNETS[int(agent[-1])],dest)
    assert (action.agent,action.session,action.from_subnet,action.to_subnet)==(
        expected.agent,expected.session,expected.from_subnet,expected.to_subnet)


@pytest.mark.parametrize('dest',['admin_network_subnet','office_network_subnet'])
def test_hq_traffic_changes_only_one_directed_pair_in_actual_cage4(dest):
    from copy import deepcopy
    from research.adapter import traffic_zone_pair

    env=make_env(seed=42,steps=6)
    env.reset(seed=42)
    state=env.env.environment_controller.state
    agent='blue_agent_4'
    source='public_access_zone_subnet'
    before=deepcopy(state.blocks)
    for command in ('BlockTrafficZone','AllowTrafficZone'):
        response=json.dumps({'action':f'{command} subnet:{dest}','reason':'fixture'})
        action,_,error=parse_response(response,agent,[],[dest])
        assert error is None and traffic_zone_pair(action)=={'from_subnet':source,'to_subnet':dest}
        assert action.execute(state).data['success']==TernaryEnum.TRUE
        assert (source in state.blocks[dest])==(command=='BlockTrafficZone')
        assert {key:value for key,value in state.blocks.items() if key!=dest}=={
            key:value for key,value in before.items() if key!=dest}
    assert state.blocks[dest]==before.get(dest,[])


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
