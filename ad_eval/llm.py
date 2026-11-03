"""Endpoint-only inference. A failed request is an experiment error, never a policy fallback."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import urlsplit, urlunsplit

import requests

from .inference import DEFAULT_INFERENCE, InferenceSettings


def endpoint_base(endpoint: str) -> str:
    value = endpoint.strip().rstrip("/")
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.netloc or parts.query or parts.fragment:
        raise ValueError("endpoint must be an HTTP(S) server URL, optionally ending in /v1")
    path = parts.path.rstrip("/")
    if not path.endswith("/v1"):
        path += "/v1"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def served_model(model: str) -> str:
    """Strip only the explicit LiteLLM provider prefix; keep HF organization names."""
    return model.removeprefix("openai/").strip()


@dataclass
class LLMConfig:
    endpoint: str = ""
    model: str = ""
    timeout: float = 120.
    max_new_tokens: int = DEFAULT_INFERENCE.max_tokens
    temperature: float = DEFAULT_INFERENCE.temperature
    top_p: float = DEFAULT_INFERENCE.top_p
    enable_thinking: bool = DEFAULT_INFERENCE.enable_thinking
    top_k: int = DEFAULT_INFERENCE.top_k
    min_p: float = DEFAULT_INFERENCE.min_p
    repetition_penalty: float = DEFAULT_INFERENCE.repetition_penalty
    frequency_penalty: float = DEFAULT_INFERENCE.frequency_penalty
    presence_penalty: float = DEFAULT_INFERENCE.presence_penalty
    seed: int = DEFAULT_INFERENCE.seed

    @classmethod
    def from_env(cls) -> LLMConfig:
        settings = InferenceSettings.from_env().to_dict()
        settings["max_new_tokens"] = settings.pop("max_tokens")
        return cls(
            endpoint=os.environ.get("QWEN_ENDPOINT", ""),
            model=served_model(os.environ.get("QWEN_ENDPOINT_MODEL") or
                               os.environ.get("QWEN_MODEL", "Qwen/Qwen3-8B")),
            timeout=float(os.environ.get("QWEN_TIMEOUT", "120")),
            **settings,
        )

    def inference_settings(self) -> InferenceSettings:
        fields = {name: getattr(self, name) for name in InferenceSettings.__dataclass_fields__
                  if name != "max_tokens"}
        return InferenceSettings(max_tokens=self.max_new_tokens, **fields)


class LLMClient:
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig.from_env()
        self.inference = self.config.inference_settings()
        self.base = endpoint_base(self.config.endpoint)
        if not self.config.model:
            raise ValueError("a served model must be specified")
        key = os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.headers = {"Authorization": "Bearer " + key}
        self.usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0}
        self.usage_complete = True
        self.event_callback = None

    def check(self) -> list[str]:
        response = requests.get(self.base + "/models", headers=self.headers,
                                timeout=min(self.config.timeout, 15))
        response.raise_for_status()
        models = [item["id"] for item in response.json()["data"]]
        if self.config.model not in models:
            raise RuntimeError(f"requested model {self.config.model!r} is not served: {models!r}")
        return models

    @property
    def available(self) -> bool:
        # Construction requires an endpoint; preflight/startup checks connectivity.
        return True

    def chat(self, messages: list[dict], max_new_tokens: int | None = None,
             temperature: float | None = None) -> str:
        if ((max_new_tokens is not None and max_new_tokens != self.inference.max_tokens) or
                (temperature is not None and temperature != self.inference.temperature)):
            raise ValueError("per-call generation overrides would invalidate the shared inference policy")
        if self.event_callback:
            self.event_callback(kind="llm_request_started", inference=self.inference.request_fields())
        response = requests.post(
            self.base + "/chat/completions", headers=self.headers,
            json={"model": self.config.model, "messages": messages, **self.inference.request_fields()},
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        data = response.json()
        self.usage["requests"] += 1
        usage = data.get("usage") or {}
        if "prompt_tokens" in usage and "completion_tokens" in usage:
            self.usage["input_tokens"] += usage["prompt_tokens"]
            self.usage["output_tokens"] += usage["completion_tokens"]
        else:
            self.usage_complete = False
        if self.event_callback:
            self.event_callback(kind="llm_usage", complete=self.usage_complete, **self.usage)
        answer = data["choices"][0]["message"].get("content")
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("LLM returned no visible content")
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.S)
        return answer.rsplit("</think>", 1)[-1].replace("<think>", "").strip()

    def complete(self, prompt: str, system: str | None = None, **kwargs) -> str:
        messages = ([{"role": "system", "content": system}] if system else [])
        return self.chat(messages + [{"role": "user", "content": prompt}], **kwargs)


@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    return LLMClient()
