"""
Central LLM client for the closed-loop workflow.

Two backends, selected automatically:
  1. ENDPOINT (preferred for Qwen3-8B): an OpenAI-compatible vLLM server. Set
     QWEN_ENDPOINT (e.g. http://localhost:8000) and the client talks to
     `/v1/chat/completions`. This is how an 8B model is actually served -- it does
     NOT fit a local CPU load, so this is the real connection for Qwen3-8B.
  2. LOCAL transformers: a small model loaded in-process via transformers. Used
     only when no endpoint is set (and torch is installed).

If neither backend is usable the client reports failures as empty strings and
callers fall back to their rule-based path, so the pipeline still runs.

Set LLM_DISABLE=1 to force the rule-based path. The same
`.chat()/.complete()/.available` interface is used by pipeline.py.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover
    pass

# Default policy model for the whole closed loop (red/blue strategy + judging).
DEFAULT_MODEL = "Qwen/Qwen3-8B"


class LLMConfig:
    def __init__(self):
        self.model = os.environ.get("QWEN_MODEL", DEFAULT_MODEL).strip()
        # Endpoint backend: an OpenAI-compatible vLLM server. Accept several common
        # env names so the pipeline LLM can share the reward/eval server.
        self.endpoint = (os.environ.get("QWEN_ENDPOINT")
                         or os.environ.get("LLM_ENDPOINT")
                         or os.environ.get("REWARD_MODEL_URL")
                         or "").strip()
        # model name as registered on the server (vLLM registers the --model path)
        self.endpoint_model = (os.environ.get("QWEN_ENDPOINT_MODEL")
                               or os.environ.get("REWARD_MODEL_NAME")
                               or self.model).strip()
        self.device = os.environ.get("QWEN_DEVICE", "auto").strip()  # auto|cpu|cuda
        self.max_new_tokens = int(os.environ.get("QWEN_MAX_NEW_TOKENS", "512"))
        self.temperature = float(os.environ.get("QWEN_TEMPERATURE", "0.7"))
        self.top_p = float(os.environ.get("QWEN_TOP_P", "0.9"))
        self.timeout = float(os.environ.get("QWEN_TIMEOUT", "120"))
        self.disabled = os.environ.get("LLM_DISABLE", "").strip() in ("1", "true", "True")

    @property
    def use_endpoint(self) -> bool:
        return bool(self.endpoint) and not self.disabled


class LLMClient:
    """LLM wrapper that prefers an OpenAI-compatible vLLM endpoint (Qwen3-8B) and
    falls back to an in-process transformers model when no endpoint is configured."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._model = None
        self._tokenizer = None
        self._load_failed = False
        self._endpoint_ok: Optional[bool] = None

    # ------- endpoint backend -------
    def _endpoint_base(self) -> str:
        base = self.config.endpoint.rstrip("/")
        return base if base.endswith("/v1") else base + "/v1"

    def _check_endpoint(self) -> bool:
        if self._endpoint_ok is not None:
            return self._endpoint_ok
        try:
            import requests
            headers = {}
            key = os.environ.get("OPENAI_API_KEY", "").strip()
            if key:
                headers["Authorization"] = f"Bearer {key}"
            r = requests.get(self._endpoint_base() + "/models", headers=headers,
                             timeout=min(self.config.timeout, 15))
            self._endpoint_ok = (r.status_code == 200)
            if self._endpoint_ok:
                print(f"[llm] endpoint ready: {self.config.endpoint} "
                      f"model={self.config.endpoint_model}")
            else:
                print(f"[llm] endpoint {self.config.endpoint} returned "
                      f"HTTP {r.status_code} -> rule-based fallback.")
        except Exception as e:
            print(f"[llm] endpoint {self.config.endpoint} unreachable: {e} -> rule-based fallback.")
            self._endpoint_ok = False
        return self._endpoint_ok

    def _chat_endpoint(self, messages: list[dict], max_new_tokens: Optional[int],
                       temperature: Optional[float]) -> str:
        import requests
        headers = {"Content-Type": "application/json"}
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        temp = self.config.temperature if temperature is None else temperature
        payload = {
            "model": self.config.endpoint_model,
            "messages": messages,
            "max_tokens": max_new_tokens or self.config.max_new_tokens,
            "temperature": temp,
            "top_p": self.config.top_p,
        }
        try:
            r = requests.post(self._endpoint_base() + "/chat/completions",
                              json=payload, headers=headers, timeout=self.config.timeout)
            r.raise_for_status()
            data = r.json()
            return (data["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:
            print(f"[llm] endpoint generation failed: {e}")
            return ""

    # ------- local transformers backend -------
    def _ensure_loaded(self) -> bool:
        if self.config.disabled or self._load_failed:
            return False
        if self._model is not None:
            return True
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"[llm] loading local {self.config.model} (device={self.config.device}) ...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model, trust_remote_code=True)
            device_map = None if self.config.device == "cpu" else self.config.device
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model,
                torch_dtype="auto",
                device_map=device_map,
                trust_remote_code=True,
            )
            if self.config.device == "cpu":
                self._model = self._model.to("cpu")
            print(f"[llm] local model ready: {self.config.model}")
            return True
        except Exception as e:
            hint = ""
            if "torch" in str(e):
                hint = (" (install torch, or serve the model with vLLM and set "
                        "QWEN_ENDPOINT=http://host:port)")
            print(f"[llm] could not load {self.config.model}: {e}{hint} -> rule-based fallback.")
            self._load_failed = True
            return False

    @property
    def available(self) -> bool:
        """True if some backend is usable (endpoint preferred). Lazy."""
        if self.config.disabled:
            return False
        if self.config.use_endpoint:
            return self._check_endpoint()
        return self._ensure_loaded()

    # ------- inference -------
    def chat(self, messages: list[dict], max_new_tokens: Optional[int] = None,
             temperature: Optional[float] = None) -> str:
        if self.config.disabled:
            return ""
        if self.config.use_endpoint:
            if not self._check_endpoint():
                return ""
            return self._chat_endpoint(messages, max_new_tokens, temperature)
        if not self._ensure_loaded():
            return ""
        try:
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
            temp = self.config.temperature if temperature is None else temperature
            gen = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.config.max_new_tokens,
                do_sample=temp > 0,
                temperature=max(temp, 1e-5),
                top_p=self.config.top_p,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            out_ids = gen[0][inputs.input_ids.shape[1]:]
            return self._tokenizer.decode(out_ids, skip_special_tokens=True).strip()
        except Exception as e:
            print(f"[llm] generation failed: {e}")
            return ""

    def complete(self, prompt: str, system: Optional[str] = None, **kw) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kw)


@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    client = LLMClient()
    if client.config.disabled:
        print("[llm] LLM_DISABLE set -> LLM features off (rule-based fallback).")
    elif client.config.use_endpoint:
        print(f"[llm] backend=endpoint url={client.config.endpoint} "
              f"model={client.config.endpoint_model} (checks on first use)")
    else:
        print(f"[llm] backend=local model={client.config.model} (loads on first use); "
              f"set QWEN_ENDPOINT to use a vLLM server instead")
    return client
