"""Provider-independent text interface. All model API code is commented out."""
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import Protocol


class TextBackend(Protocol):
    def generate(self, messages: list[dict[str, str]], *, agent: str,
                 episode: int, step: int) -> str:
        """Return JSON text containing action and reason."""


@dataclass
class MockBackend:
    """A plumbing test, NOT an LLM, policy baseline, or paper result."""
    action: str = "Sleep"

    def generate(self, messages, **context):
        return json.dumps({"action": self.action,
                           "reason": "Offline mock response; no language model was called."})


class ReplayBackend:
    """Replay recorded outputs by (episode, step, agent); missing records fail."""
    def __init__(self, path):
        self.records = {}
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("response") is not None:
                key = (row["episode"], row["step"], row["agent"])
                if key in self.records:
                    raise ValueError(f"Duplicate replay key: {key}")
                self.records[key] = row["response"]

    def generate(self, messages, *, agent, episode, step):
        key = (episode, step, agent)
        if key not in self.records:
            raise ValueError(f"No replay response for {key}")
        return self.records[key]


class ApiBackend:
    def __init__(self, config):
        self.config = config

    def generate(self, messages, **context):
        # MODEL API INTEGRATION IS INTENTIONALLY COMMENTED OUT.
        # Replace this example with the SDK/protocol of your chosen model.
        # Do not put keys in YAML or in experiment output.
        #
        # import os
        # from openai import OpenAI
        # client = OpenAI(
        #     base_url=self.config["base_url"],
        #     api_key=os.environ[self.config["api_key_env"]],
        #     timeout=self.config.get("timeout_seconds", 60),
        #     max_retries=0,
        # )
        # result = client.chat.completions.create(
        #     model=self.config["model"], messages=messages,
        #     temperature=self.config.get("temperature", 1.0),
        # )
        # return result.choices[0].message.content
        raise RuntimeError("Model API is disabled. Implement/uncomment acd/backends.py after choosing a provider.")


def create_backend(config):
    kind = config.get("backend", "mock")
    if kind == "mock":
        return MockBackend(config.get("mock_action", "Sleep"))
    if kind == "replay":
        return ReplayBackend(config["replay_path"])
    if kind == "api":
        if not config.get("enabled", False):
            raise ValueError("model.enabled must be true before selecting an API backend; API code is still commented out.")
        return ApiBackend(config)
    if kind == "custom":
        # e.g. backend factory 'my_model:create_backend', receiving this config.
        module, name = config["factory"].split(":", 1)
        backend = getattr(importlib.import_module(module), name)(config)
        if not callable(getattr(backend, "generate", None)):
            raise TypeError("Custom backend must implement generate(messages, *, agent, episode, step)")
        return backend
    raise ValueError(f"Unknown model backend: {kind}")
