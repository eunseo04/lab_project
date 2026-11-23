"""Strict local inference client: errors propagate; a different model is never substituted."""
import hashlib
import json
import math
import urllib.error
import urllib.request


class ModelFailure(RuntimeError):
    pass


class ModelClient:
    is_mock = False

    def __init__(self, config):
        self.config = config
        self.endpoint = config["endpoint"].rstrip("/")
        self.identity = self._request("health")
        for key in ("model", "revision"):
            if self.identity.get(key) != config[key]:
                raise ModelFailure(f"Wrong local model {key}: expected {config[key]}, got {self.identity.get(key)}")
        if self.identity.get("weights_frozen") is not True:
            raise ModelFailure("The policy backend must certify frozen weights")
        self.calls = []
        self.counter = 0

    def _request(self, route, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(self.endpoint + "/" + route, data=data,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.config.get("timeout_seconds", 180)) as response:
                return json.load(response)
        except (OSError, ValueError, urllib.error.HTTPError) as exc:
            detail = exc.read().decode(errors="replace")[:600] if isinstance(exc, urllib.error.HTTPError) else str(exc)
            raise ModelFailure(f"{route} failed: {detail}") from exc

    def generate(self, messages, *, seed=42, purpose="policy", max_tokens=None, return_logprobs=False):
        result = self._request("generate", {"messages": messages, "seed": seed,
            "temperature": self.config.get("temperature", 1.0),
            "max_tokens": max_tokens or self.config.get("max_tokens", 192),
            "return_logprobs": return_logprobs})
        if any(result.get("identity", {}).get(k) != self.config[k] for k in ("model","revision")):
            raise ModelFailure("Model identity changed during generation")
        self.counter += 1
        self.calls.append({"purpose": purpose, "input_tokens": result["prompt_tokens"],
                           "output_tokens": result["output_tokens"], "seconds": result["seconds"]})
        return result

    def score(self, samples):
        if self.identity.get("scoring_mode")!="incremental_teacher_forcing":
            raise ModelFailure("Exact incremental scoring is unavailable; candidate cannot be verified")
        values = []
        for sample in samples:  # bounded memory and strict alignment
            response = self._request("score", {"samples": [sample], "temperature": self.config.get("temperature", 1.0)})
            if any(response.get("identity", {}).get(k) != self.config[k] for k in ("model","revision")):
                raise ModelFailure("Scorer model differs from acting model")
            if len(response.get("scores", [])) != 1:
                raise ModelFailure("Missing exact teacher-forced score")
            value = response["scores"][0]["sum"]
            if not math.isfinite(value):
                raise ModelFailure("Non-finite exact logprob")
            values.append(value)
        self.calls.append({"purpose": "gate_score", "samples": len(samples)})
        return values

    def embed(self, text):
        return self._request("embed", {"texts": [text]})["vectors"][0]

    def embed_many(self, texts):
        vectors = self._request("embed", {"texts": texts})["vectors"]
        if len(vectors) != len(texts): raise ModelFailure("Embedding batch alignment failed")
        return vectors


class TextModel:
    """Skill-Pro callable backed by exactly the same policy model server."""
    def __init__(self, client):
        self.client = client
        self.seed, self.counter = 42, 0
        self.purpose = "skill_selection"

    def __call__(self, prompt, purpose=None):
        self.counter += 1
        seed = int(hashlib.sha256(f"{self.seed}:{self.counter}:{purpose or self.purpose}".encode()).hexdigest()[:8], 16)
        return self.client.generate([{"role": "user", "content": prompt}], seed=seed,
            purpose=purpose or self.purpose,
            max_tokens=768 if self.purpose == "evolution" else 48)["text"]

    def score(self, samples):
        return self.client.score(samples)


def stable_seed(*parts):
    return int(hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:8], 16) % (2**31)
