"""One explicit generation policy shared by CAI and the proposed workers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
import os


@dataclass(frozen=True)
class InferenceSettings:
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    enable_thinking: bool = False
    seed: int = 42

    def __post_init__(self):
        bounds = (("temperature", 0., 2.), ("top_p", 0., 1.), ("min_p", 0., 1.),
                  ("frequency_penalty", -2., 2.), ("presence_penalty", -2., 2.))
        for name, low, high in bounds:
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be finite and between {low} and {high}")
        if self.top_p == 0:
            raise ValueError("top_p must be greater than zero")
        if not math.isfinite(self.repetition_penalty) or self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be finite and positive")
        for name in ("max_tokens", "top_k", "seed"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.max_tokens < 1 or self.top_k == 0 or self.top_k < -1 or not 0 <= self.seed < 2**31:
            raise ValueError("max_tokens>0, top_k=-1 or >0, and 0<=seed<2^31 required")
        if type(self.enable_thinking) is not bool:
            raise ValueError("enable_thinking must be boolean")

    @classmethod
    def from_env(cls) -> InferenceSettings:
        canonical = os.environ.get("AD_EVAL_INFERENCE")
        if canonical:
            return cls(**json.loads(canonical))
        return cls(temperature=float(os.environ.get("QWEN_TEMPERATURE", "0.7")),
                   top_p=float(os.environ.get("QWEN_TOP_P", "0.9")),
                   max_tokens=int(os.environ.get("QWEN_MAX_NEW_TOKENS", "512")),
                   enable_thinking=os.environ.get("QWEN_ENABLE_THINKING", "false").lower() in {"1", "true"})

    def for_role(self, team: str, role: str) -> InferenceSettings:
        if team not in {"1", "2"} or role not in {"red", "blue"}:
            raise ValueError("expected a known team and red/blue role")
        # Corresponding team/role in each condition receives the same seed.
        return replace(self, seed=self.seed + 2 * (int(team) - 1) + int(role == "blue"))

    def to_dict(self) -> dict:
        return asdict(self)

    def to_env(self) -> dict[str, str]:
        return {"AD_EVAL_INFERENCE": json.dumps(self.to_dict(), sort_keys=True),
                "QWEN_TEMPERATURE": str(self.temperature), "QWEN_TOP_P": str(self.top_p),
                "QWEN_MAX_NEW_TOKENS": str(self.max_tokens),
                "QWEN_ENABLE_THINKING": str(self.enable_thinking).lower(),
                "CAI_TEMPERATURE": str(self.temperature), "CAI_TOP_P": str(self.top_p)}

    def request_fields(self) -> dict:
        return {"temperature": self.temperature, "top_p": self.top_p,
                "max_tokens": self.max_tokens, "top_k": self.top_k, "min_p": self.min_p,
                "repetition_penalty": self.repetition_penalty,
                "frequency_penalty": self.frequency_penalty, "presence_penalty": self.presence_penalty,
                "seed": self.seed, "n": 1, "stream": False,
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking}}

    def litellm_kwargs(self, kwargs: dict) -> dict:
        """Apply at the API boundary, after CAI's Ollama adapter drops extra settings."""
        result = dict(kwargs)
        fields = self.request_fields()
        extra_names = {"top_k", "min_p", "repetition_penalty", "chat_template_kwargs"}
        extra = dict(result.get("extra_body") or {})
        for name in fields:
            extra.pop(name, None)
            result.pop(name, None)
        result.pop("max_completion_tokens", None)
        result.update({key: value for key, value in fields.items() if key not in extra_names})
        extra.update({key: fields[key] for key in extra_names})
        result["extra_body"] = extra
        return result


DEFAULT_INFERENCE = InferenceSettings()
