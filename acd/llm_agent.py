"""Paper prompt/observation flow with a replaceable, offline-by-default backend."""
from copy import deepcopy
import json
import re
import yaml

from CybORG.Simulator.Actions import Sleep, Remove, Restore, DeployDecoy, Analyse
from CybORG.Simulator.Actions.ConcreteActions.ControlTraffic import BlockTrafficZone, AllowTrafficZone
from CybORG.Agents.LLMAgents.llm_adapter import obs_formatter
from CybORG.Agents.LLMAgents.llm_adapter.utils.constants import CAGE4_SUBNETS
from CybORG.Agents.LLMAgents.llm_adapter.utils.logger import Logger

from . import RUNTIME

Logger.set_debug_mode(False)

HOST_ACTIONS = {"remove": Remove, "restore": Restore,
                "deploydecoy": DeployDecoy, "analyse": Analyse}
ZONE_ACTIONS = {"blocktrafficzone": BlockTrafficZone, "allowtrafficzone": AllowTrafficZone}


def parse_response(response, agent, allowed_hosts, allowed_subnets):
    """Malformed/invalid outputs become Sleep as in the paper; no eval()."""
    try:
        text = response.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
        value = json.loads(text)
        if not isinstance(value, dict) or not isinstance(value.get("reason"), str):
            raise ValueError("Response must contain a string reason")
        requested = value.get("action")
        if not isinstance(requested, str):
            raise ValueError("Response must contain a string action")
        parts = requested.split()
        if len(parts) == 1 and parts[0].lower() == "sleep":
            return Sleep(), value["reason"], None
        if len(parts) != 2:
            raise ValueError("Expected one action and one host:/subnet: parameter")
        name, parameter = parts
        name = name.lower()
        if name in HOST_ACTIONS and parameter.startswith("host:"):
            host = parameter[5:]
            if host not in allowed_hosts:
                raise ValueError(f"Host is outside the agent's assigned hosts: {host}")
            return HOST_ACTIONS[name](session=0, agent=agent, hostname=host), value["reason"], None
        if name in ZONE_ACTIONS and parameter.startswith("subnet:"):
            subnet = parameter[7:]
            if subnet not in allowed_subnets:
                raise ValueError(f"Unknown subnet: {subnet}")
            # Match the published policy: HQ traffic actions originate at public access.
            origin = CAGE4_SUBNETS[int(agent[-1])]
            return ZONE_ACTIONS[name](session=0, agent=agent, from_subnet=origin,
                                     to_subnet=subnet), value["reason"], None
        raise ValueError("Unsupported action or parameter")
    except (ValueError, TypeError, AttributeError, IndexError) as exc:
        return Sleep(), "Invalid response; Sleep fallback.", str(exc)


class LLMDefender:
    def __init__(self, agent, backend, prompt_path=None):
        self.name = agent
        self.backend = backend
        path = prompt_path or RUNTIME / "CybORG/Agents/LLMAgents/config/prompts/acd2025/base.yml"
        from pathlib import Path
        prompts = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["prompts"]
        if len(prompts) != 1:
            raise ValueError("Paper mode expects exactly one system prompt")
        prefix = (f"AGENT_NAME: {agent}\n"
            "Prioritize this `# DESCRIPTION` section to generate your answer. Remember your `AGENT_NAME` is the BLUE AGENT you are. \n"
            "IMPORTANT: Return ONLY ONE action from `## AVAILABLE ACTIONS`\n# INSTRUCTIONS\n")
        self.system_prompt = {"role": "system", "content": prefix + prompts[0]["content"]}
        self.last_action = None

    def decide(self, observation, *, hosts, subnets, episode, step):
        formatted = obs_formatter.format_observation(deepcopy(observation), self.last_action, self.name)
        messages = [self.system_prompt, {"role": "user", "content": formatted}]
        response = self.backend.generate(messages, agent=self.name, episode=episode, step=step)
        action, reason, error = parse_response(response, self.name, hosts, subnets)
        self.last_action = str(action)
        return action, {"response": response, "reason": reason, "parse_error": error,
                        "user_prompt": formatted}
