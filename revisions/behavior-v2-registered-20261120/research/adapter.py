"""CAGE4 bridge: a single joint step, original reward and native episode termination."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import numpy as np

from acd.environment import ExperimentEnv, BLUE_AGENTS, red_class
from acd.runner import json_default, load_agents
from acd.communication import create_message
from acd.llm_agent import LLMDefender, parse_response
from CybORG import CybORG
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent
from CybORG.Simulator.Actions import Sleep
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from .skills import SkillController
from .behavior import ObservableCaseLedger,readable_context

ACTIONS = ("Analyse", "DeployDecoy", "Remove", "Restore", "Monitor", "Sleep",
           "BlockTrafficZone", "AllowTrafficZone")


def plain(value):
    return json.loads(json.dumps(value, default=json_default))


def validate_visible_observation(value):
    forbidden = {"true_state", "global_state", "red_internal_state", "hidden_red_sessions",
                 "environment_controller", "agent_interfaces", "hostname_ip_map", "full_state"}
    if isinstance(value, dict):
        if forbidden.intersection(str(k) for k in value):
            raise ValueError("Hidden simulator/Red state must not enter a Blue model input")
        for item in value.values(): validate_visible_observation(item)
    elif isinstance(value, (list, tuple)):
        for item in value: validate_visible_observation(item)


class ResearchEnv(ExperimentEnv):
    def __init__(self, env, communication="paper_projected_keep"):
        self.wire_protocol = communication
        super().__init__(env, communication="keep")

    def step(self, actions, messages=None):
        if messages is None and self.wire_protocol == "paper_projected_keep":
            messages = self.paper_messages()
        return super().step(actions, messages=messages)

    def _message_features(self, i, msg):
        if self.wire_protocol == "keep_native":
            return super()._message_features(i, msg)
        # Explicit common hybrid compatibility projection. The public KEEP
        # checkpoint cannot ingest 197 paper features without retraining.
        severity = msg[:, 5].astype(int)*2 + msg[:, 6].astype(int)
        projected = np.stack([severity >= 2, severity >= 1, msg.any(axis=1)], axis=1).astype(float)
        if i != 4:
            projected = np.concatenate([projected[:-1], np.repeat(projected[-1:], 3, axis=0)])
        return projected


def make_environment(config, seed):
    scenario = EnterpriseScenarioGenerator(blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent, red_agent_class=red_class(config["red"]), steps=config["steps"])
    return ResearchEnv(CybORG(scenario, "sim", seed=seed), config["communication"])


def make_state(env, agent, raw, base_policy):
    validate_visible_observation(raw)
    from CybORG.Agents.LLMAgents.llm_adapter import obs_formatter
    # Target names come only from the official, agent-specific action metadata.
    host_set = {a.hostname for a, valid in zip(env.actions(agent), env.action_mask(agent))
                if valid and getattr(a, "hostname", None)}
    from CybORG.Simulator.Scenarios.EnterpriseScenarioGenerator import SUBNET
    subnets = [s.value for s in SUBNET]
    text = obs_formatter.format_observation(deepcopy(raw), base_policy.last_action, agent)
    return {"agent": agent, "observation": plain(raw), "user_prompt": text,
            "hosts": sorted(host_set), "subnets": subnets, "allowed_actions": list(ACTIONS)}


def messages_for(state, policy, skill=None):
    system = deepcopy(policy.system_prompt)
    if skill is not None:
        system["content"] += "\n# ACTIVE SECURITY SKILL\n" + skill.format_for_llm()
    return [system, {"role": "user", "content": state["user_prompt"]}]


def explicit_action_contract(state):
    """An observation-only interface shared by E0--E3, not a new action policy."""
    return ("\n\n# EXECUTABLE ACTION CONTRACT\n"
        "Return one JSON object with string fields action and reason. Choose one primitive action, "
        "not a Skill name. A Skill is internal procedural guidance, never an environment action.\n"
        "Primitive forms: Sleep; Analyse host:<hostname>; Remove host:<hostname>; "
        "Restore host:<hostname>; DeployDecoy host:<hostname>; "
        "BlockTrafficZone subnet:<subnet_id>; AllowTrafficZone subnet:<subnet_id>.\n"
        "Monitor is automatic. Use exact IDs from the lists below, not IP addresses, CIDRs or zone aliases.\n"
        "Assigned host IDs: " + json.dumps(state["hosts"]) + "\n"
        "Subnet IDs: " + json.dumps(state["subnets"]) + "\n"
        "Read the observed result of the previous action before advancing a Skill. "
        "IN_PROGRESS means the previous action is still running; Sleep waits for its completion.\n")


class DefensePolicy:
    def __init__(self, agent, client, text_model, config, pool=None):
        self.agent, self.client, self.config = agent, client, config
        self.base = LLMDefender(agent, None)
        if config["profile"] == "controlled-comparison":
            self.base.system_prompt["content"] += (
                "\n## COMMON CONTROLLED ACTION RULES\nSleep is permitted while an action is in progress or to wait for its observed result. "
                "Monitor runs automatically at every turn; do not select it explicitly. Return JSON with action and reason.\n")
        self.controller = SkillController(pool, text_model, config["selector"], proposed=config["arm"] == "E3") if pool else None
        self.ledger=ObservableCaseLedger(agent) if config.get("behavior_context")=="evidence_v1" else None

    def prepare_state(self,state,step):
        if self.ledger is None or "behavior_context" in state:return state
        context=self.ledger.context(state,step)
        rendered=readable_context(context) if self.config.get("case_presentation")=="readable" else "# OBSERVABLE EPISODE CASE LEDGER\n"+json.dumps(context,ensure_ascii=False)
        return {**state,"behavior_context":context,"behavior_seed_set":self.config.get("skill_seed_set","legacy"),
            "user_prompt":state["user_prompt"]+"\n\n"+rendered}

    def observe_transition(self,row):
        if self.ledger is not None:self.ledger.transition(row)

    def decide(self, state, *, episode, step, seed):
        state=self.prepare_state(state,step)
        busy = "IN_PROGRESS" in str(state["observation"].get("success"))
        if busy and self.config.get("skip_busy", False):
            return Sleep(), {"response": None, "token_ids": None, "reason": "Common busy-action handling; no model invocation",
                "parse_error": None, "skill_id": self.controller.active.name if self.controller and self.controller.active else None,
                "skill_version": self.controller.active.version if self.controller and self.controller.active else None,
                "policy_invoked": False, "state": json.dumps(messages_for(state, self.base))}
        skill = self.controller.choose(state, step) if self.controller else None
        # Action JSON belongs ONLY to the primitive actor. Meta-selection and
        # termination consume observations and keep their original XML format.
        if self.config.get("action_contract") == "explicit":
            state = {**state, "user_prompt": state["user_prompt"] + explicit_action_contract(state)}
        base_messages = messages_for(state, self.base)
        messages = messages_for(state, self.base, skill)
        result = self.client.generate(messages, seed=seed, purpose="policy", return_logprobs=True)
        action, reason, error = parse_response(result["text"], self.agent, state["hosts"], state["subnets"])
        self.base.last_action = str(action)
        return action, {"response": result["text"], "token_ids": result["token_ids"],
            "generation_logprob": result.get("generation_logprob"), "prompt_sha256": result.get("prompt_sha256"),
            "reason": reason, "parse_error": error, "skill_id": skill.name if skill else None,
            "skill_version": skill.version if skill else None, "policy_invoked": True,
            "state": json.dumps(base_messages), "user_prompt": state["user_prompt"],
            "skill_controller": self.controller.last_event if self.controller else None,
            "model_identity": result.get("identity", self.client.identity),
            "prompt_tokens": result.get("prompt_tokens"), "output_tokens": result.get("output_tokens")}


def executed_actions(env, agent):
    value = env.env.get_last_action(agent)
    items = value if isinstance(value, (tuple, list)) else [value]
    return [{"name": type(a).__name__, "text": str(a), "target": getattr(a, "hostname", None)
             or getattr(a, "to_subnet", None)} for a in items if a is not None]
