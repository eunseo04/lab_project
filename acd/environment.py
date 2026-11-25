"""One simulator tick per joint action; reuse the actual published graph code."""
from copy import deepcopy
import numpy as np
import torch

from CybORG import CybORG
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent
from CybORG.Simulator.Actions import Action
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers.CybermonicWrappers.graph_wrapper import GraphWrapper
from CybORG.Agents.Wrappers.CybermonicWrappers.observation_graph import ObservationGraph

from .communication import create_message
from .config import LEGACY_ENCODER, validate_communication_encoder

BLUE_AGENTS = tuple(f"blue_agent_{i}" for i in range(5))
RED_CLASSES = {
    "finite": ("FiniteStateRedAgent", "FiniteStateRedAgent"),
    "aggressive": ("AggressiveFSMAgent", "AggressiveFSMAgent"),
    "stealthy": ("StealthyFSMAgent", "StealthyFSMAgent"),
    "impact": ("ImpactFSMAgent", "ImpactFSMAgent"),
    "degrade": ("DegradeServiceFSMAgent", "DegradeServiceFSMAgent"),
}


class FixedHorizonScenario(EnterpriseScenarioGenerator):
    """Keep mission timing, but execute exactly the configured action count.

    Upstream determine_done() returns true at steps-1. Native mode below leaves
    that behavior intact; fixed mode matches the paper's stated 500 actions.
    """
    def determine_done(self, env_controller):
        return env_controller.step_count >= self.steps


def red_class(name):
    import importlib
    module, cls = RED_CLASSES[name]
    return getattr(importlib.import_module("CybORG.Agents.SimpleAgents." + module), cls)


class ExperimentEnv(GraphWrapper):
    def __init__(self, env, communication="paper", communication_encoder=LEGACY_ENCODER):
        if communication not in ("paper", "keep"):
            raise ValueError("communication must be paper or keep")
        self.communication = communication
        self.communication_encoder = validate_communication_encoder(communication, communication_encoder)
        super().__init__(env)

    @property
    def feature_dimension(self):
        # KEEP has 2 tabular + 3 message features. Paper has 2 tabular + 8 bits.
        return ObservationGraph.DIM + (10 if self.communication == "paper" else 5)

    def action_translator(self, agent_name, action):
        if isinstance(action, Action):
            return action
        return super().action_translator(agent_name, action)

    def raw_observations(self):
        observations = {}
        for agent in BLUE_AGENTS:
            observations[agent] = deepcopy(self.env.get_observation(agent))
            observations[agent]["phase"] = self.env.environment_controller.state.mission_phase
            observations[agent].setdefault("message", [np.zeros(8, dtype=bool) for _ in range(4)])
        return observations

    def _paper_message(self, agent):
        ec = self.env.environment_controller
        return create_message(ec.get_last_observation(agent), ec.hostname_ip_map, self.communication_encoder)

    def paper_messages(self):
        return {agent: self._paper_message(agent) for agent in BLUE_AGENTS}

    def step(self, action, messages=None):
        if self.communication == "paper" and messages is None:
            messages = self.paper_messages()
        before = self.env.environment_controller.step_count
        result = super().step(action, messages=messages)
        after = self.env.environment_controller.step_count
        if after != before + 1:
            raise RuntimeError(f"Expected one simulator step, got {after - before}")
        return result

    def _message_features(self, i, msg):
        if self.communication == "keep":
            return super()._message_features(i, msg)
        # Incoming messages are ordered by the other four agent IDs. HQ's one
        # vector applies to all three HQ subnets; keep every paper bit intact.
        msg = msg.astype(float)
        if i != 4:
            msg = np.concatenate([msg[:-1], np.repeat(msg[-1:], 3, axis=0)], axis=0)
        return msg

    def _parse_tabular(self, x, graph):
        tab, phase, legacy = super()._parse_tabular(x, graph)
        if self.communication == "keep":
            return tab, phase, legacy
        owner = next(name for name, g in self.graphs.items() if g is graph)
        return tab, phase, self._paper_message(owner)

    def _to_obs(self, x, ei, masks, phase, other_msg, my_msg):
        if self.communication == "keep":
            return super()._to_obs(x, ei, masks, phase, other_msg, my_msg)
        # Preserve the upstream batching, action masks, topology and subnet
        # indexing. Replace only its three legacy message features with 8 bits.
        own_routers = {int(mask[2][0][0]) for mask in masks}
        peer_routers = [int(r) for r in masks[0][3] if int(r) not in own_routers]
        packed = super()._to_obs(x, ei, masks, phase,
                                np.zeros((len(peer_routers), 3)), np.zeros(8))
        features = torch.zeros((x.size(0), 8))
        if other_msg.shape[-1] != 8:  # upstream reset supplies legacy zero placeholders
            other_msg = np.zeros((len(peer_routers), 8))
        features[peer_routers] = torch.as_tensor(other_msg, dtype=torch.float32)
        for mask in masks:
            features[mask[2][0][0]] = torch.as_tensor(my_msg, dtype=torch.float32)
        return (torch.cat([packed[0][:, :-3], features], dim=1), *packed[1:])


def make_env(*, seed, steps, red="finite", communication="paper", horizon="fixed",
             communication_encoder=LEGACY_ENCODER):
    scenario_class = EnterpriseScenarioGenerator if horizon == "native" else FixedHorizonScenario
    scenario = scenario_class(blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent, red_agent_class=red_class(red), steps=steps)
    return ExperimentEnv(CybORG(scenario, "sim", seed=seed), communication, communication_encoder)
