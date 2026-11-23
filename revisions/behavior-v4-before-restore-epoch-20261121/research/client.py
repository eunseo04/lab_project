"""Strict local inference client: errors propagate; a different model is never substituted."""
import hashlib
import json
import math
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
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
        self._score_cache = OrderedDict()
        self._scoring_endpoints = list(dict.fromkeys([self.endpoint] +
            [url.rstrip("/") for url in config.get("scoring_endpoints", [])]))
        for endpoint in self._scoring_endpoints[1:]:
            identity = self._request("health", endpoint=endpoint)
            self._validate_scorer(identity)

    def _request(self, route, payload=None, endpoint=None):
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request((endpoint or self.endpoint) + "/" + route, data=data,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.config.get("timeout_seconds", 180)) as response:
                return json.load(response)
        except (OSError, ValueError, urllib.error.HTTPError) as exc:
            detail = exc.read().decode(errors="replace")[:600] if isinstance(exc, urllib.error.HTTPError) else str(exc)
            raise ModelFailure(f"{route} failed: {detail}") from exc

    def _validate_scorer(self, identity):
        required = ("model", "revision", "dtype", "chat_template_sha256", "scoring_mode")
        if any(identity.get(k) != self.identity.get(k) for k in required) or identity.get("weights_frozen") is not True:
            raise ModelFailure("Scorer weights, tokenizer, dtype or scoring method differ from the acting model")

    def _score_key(self, sample):
        return hashlib.sha256(json.dumps(sample, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def _remember_score(self, key, value):
        self._score_cache[key] = value
        self._score_cache.move_to_end(key)
        while len(self._score_cache) > self.config.get("score_cache_size", 8192):
            self._score_cache.popitem(last=False)

    def generate(self, messages, *, seed=42, purpose="policy", max_tokens=None, return_logprobs=False):
        result = self._request("generate", {"messages": messages, "seed": seed,
            "temperature": self.config.get("temperature", 1.0),
            "max_tokens": max_tokens or self.config.get("max_tokens", 192),
            "return_logprobs": return_logprobs})
        if any(result.get("identity", {}).get(k) != self.config[k] for k in ("model","revision")):
            raise ModelFailure("Model identity changed during generation")
        self.counter += 1
        call={"purpose": purpose, "input_tokens": result["prompt_tokens"],
              "output_tokens": result["output_tokens"], "seconds": result["seconds"]}
        if purpose in ("skill_selection","termination"):
            call["response"]=result["text"]
        self.calls.append(call)
        value = result.get("generation_logprob")
        if value is not None and math.isfinite(value):
            self._remember_score(self._score_key({"messages": messages, "token_ids": result["token_ids"]}), value)
        return result

    def score(self, samples, *, use_cache=True):
        if self.identity.get("scoring_mode")!="incremental_teacher_forcing":
            raise ModelFailure("Exact incremental scoring is unavailable; candidate cannot be verified")
        keys = [self._score_key(sample) for sample in samples]
        available = self._score_cache if use_cache else {}
        missing = list(dict.fromkeys(key for key in keys if key not in available))
        by_key = dict(zip(keys, samples))
        def score_one(item):
            index, key = item
            endpoint = self._scoring_endpoints[index % len(self._scoring_endpoints)]
            response = self._request("score", {"samples": [by_key[key]],
                "temperature": self.config.get("temperature", 1.0)}, endpoint=endpoint)
            self._validate_scorer(response.get("identity", {}))
            if len(response.get("scores", [])) != 1:
                raise ModelFailure("Missing exact teacher-forced score")
            value = response["scores"][0]["sum"]
            if not math.isfinite(value):
                raise ModelFailure("Non-finite exact logprob")
            return key, value
        # Parallel replicas change only inference placement. Each uses identical
        # frozen weights/template/dtype; result order follows the original batch.
        cached = {key: available[key] for key in keys if key in available}
        with ThreadPoolExecutor(max_workers=len(self._scoring_endpoints)) as executor:
            computed = dict(executor.map(score_one, enumerate(missing)))
        values = [cached[key] if key in cached else computed[key] for key in keys]
        for key, value in computed.items():
            self._remember_score(key, value)
        self.calls.append({"purpose": "gate_score", "samples": len(samples),
            "cache_hits": len(samples)-len(missing), "teacher_forced_samples": len(missing),
            "replicas": len(self._scoring_endpoints)})
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
