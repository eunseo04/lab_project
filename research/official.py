"""Invoke the actual unmodified CAGE4 evaluator and compare its arithmetic."""
from copy import deepcopy
import json
from pathlib import Path
from statistics import mean, stdev
import time

from CybORG.Agents import SleepAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Evaluation import evaluation as official_evaluator
from acd.runner import seed_everything, load_agents
from acd.config import LEGACY_ENCODER
from .adapter import ResearchEnv, DefensePolicy, make_state, red_class, BLUE_AGENTS
from .client import ModelClient, TextModel, stable_seed
from .experiment import atomic_json, policy_roles
from .metrics import JointRewardLedger
from .config import execution_config


def run_official(config, output):
    config=execution_config(config,'eval')
    if config["arm"] != "E0" or config["steps"] != 500 or config["episodes"] < 2:
        raise ValueError("Official entrypoint is E0, 500 steps and >=2 episodes")
    directory = Path(output); directory.mkdir(parents=True, exist_ok=False)
    seed_everything(config["eval_seed"], config["threads"])
    roles = policy_roles(config["composition"])
    client = ModelClient(config["model"]) if "llm" in roles.values() else None
    text_model = TextModel(client)
    ledgers, wrapper_ref = [], []

    class Wrapper(ResearchEnv):
        def reset(self, *args, **kwargs):
            result = super().reset(*args, **kwargs)
            ledgers.append(JointRewardLedger())
            self.policies = {a: DefensePolicy(a, client, text_model, config) for a, r in roles.items() if r == "llm"}
            return result

        def step(self, actions, messages=None):
            result = super().step(actions, messages)
            ledgers[-1].add(result[1], self.env.environment_controller.determine_done())
            return result

    class ModelProxy:
        def __init__(self, name): self.name = name
        def get_action(self, observation, action_space):
            wrapper = wrapper_ref[0]
            raw = wrapper.raw_observations()[self.name]
            policy = wrapper.policies[self.name]
            state = make_state(wrapper, self.name, raw, policy.base)
            action, details = policy.decide(state, episode=len(ledgers)-1,
                step=wrapper.ts, seed=stable_seed(config["eval_seed"], len(ledgers), wrapper.ts, self.name))
            return action

    agents = {a: SleepAgent(a) for a in BLUE_AGENTS}
    if "rl" in roles.values():
        c = {**config, "team": "mixed" if config["composition"] == "hybrid" else "all_rl",
             "communication": "keep", "communication_encoder": LEGACY_ENCODER, "deterministic_rl": False}
        agents.update(load_agents(c, 192)[0])
    for a, role in roles.items():
        if role == "llm": agents[a] = ModelProxy(a)

    class Submission:
        NAME, TEAM, TECHNIQUE = "Official evaluator bridge", "Local ACD research", "Frozen Qwen / fixed KEEP"
        AGENTS = agents
        @staticmethod
        def wrap(env):
            wrapper = Wrapper(env, config["communication"], config['communication_encoder'])
            wrapper_ref.append(wrapper)
            return wrapper

    original_scenario = official_evaluator.EnterpriseScenarioGenerator
    def scenario_factory(**kwargs):
        kwargs["red_agent_class"] = red_class(config["red"])
        return EnterpriseScenarioGenerator(**kwargs)
    # Configure the requested existing red policy without editing evaluator source.
    official_evaluator.EnterpriseScenarioGenerator = scenario_factory
    try:
        official_evaluator.run_evaluation(Submission, str(directory), max_eps=config["episodes"],
            write_to_file=True, seed=config["eval_seed"])
    finally:
        official_evaluator.EnterpriseScenarioGenerator = original_scenario
    original = json.loads((directory/"summary.json").read_text())
    totals = [ledger.total for ledger in ledgers]
    assert mean(totals) == original["reward"]["mean"]
    assert stdev(totals) == original["reward"]["stdev"]
    parity = {"official_evaluator_invoked": True, "reward_aggregation_matches": True,
        "episode_returns": totals, "official_reward": original["reward"],
        "executed_transitions": [len(l.raw_rewards) for l in ledgers],
        "scored_transitions": [len(l.rewards) for l in ledgers],
        "config": config, "model_identity": client.identity if client else None,
        "scoring_implementation_changed": False}
    atomic_json(directory/"arithmetic_parity.json", parity)
    return parity
