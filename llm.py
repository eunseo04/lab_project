"""
Central LLM client for the closed-loop workflow -- local Qwen2.5-0.5B-Instruct.

Runs a local Hugging Face causal LM (default Qwen/Qwen2.5-0.5B-Instruct) via
transformers. Configuration comes from environment / .env (see .env.example).
No API key is needed. The model is loaded lazily on first use; if it cannot be
loaded (offline, no weights, OOM) the client reports failures as empty strings
and callers fall back to their rule-based path, so the pipeline still runs.

Set LLM_DISABLE=1 to force the rule-based path without touching the model.
The same `.chat()/.complete()/.available` interface is used by pipeline.py.
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


class LLMConfig:
    def __init__(self):
        self.model = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct").strip()
        self.device = os.environ.get("QWEN_DEVICE", "auto").strip()  # auto|cpu|cuda
        self.max_new_tokens = int(os.environ.get("QWEN_MAX_NEW_TOKENS", "512"))
        self.temperature = float(os.environ.get("QWEN_TEMPERATURE", "0.7"))
        self.top_p = float(os.environ.get("QWEN_TOP_P", "0.9"))
        self.disabled = os.environ.get("LLM_DISABLE", "").strip() in ("1", "true", "True")


class LLMClient:
    """Thin wrapper over a local transformers chat model (Qwen2.5-0.5B-Instruct)."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._model = None
        self._tokenizer = None
        self._load_failed = False

    # ------- loading -------
    def _ensure_loaded(self) -> bool:
        if self.config.disabled or self._load_failed:
            return False
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"[llm] loading {self.config.model} (device={self.config.device}) ...")
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
            print(f"[llm] Qwen ready: {self.config.model}")
            return True
        except Exception as e:
            print(f"[llm] could not load {self.config.model}: {e} -> rule-based fallback.")
            self._load_failed = True
            return False

    @property
    def available(self) -> bool:
        """True if the model is usable. Triggers a lazy load on first check."""
        if self.config.disabled:
            return False
        return self._ensure_loaded()

    # ------- inference -------
    def chat(self, messages: list[dict], max_new_tokens: Optional[int] = None,
             temperature: Optional[float] = None) -> str:
        if not self._ensure_loaded():
            return ""
        try:
            import torch
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
    else:
        print(f"[llm] backend=local Qwen model={client.config.model} "
              f"(loads on first use)")
    return client
