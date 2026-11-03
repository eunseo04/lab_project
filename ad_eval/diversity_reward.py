"""History BLEU and embedding cosine against verified successful Red actions."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re

import requests

from .llm import endpoint_base


def normalize_command(command: str) -> str:
    command = re.sub(r"\b(?:172\.32\.10\.(?:10|20)|172\.29\.80\.(?:10|20))\b", "TARGET_IP", command)
    command = re.sub(r"FLAG\{[^\s{}]+\}", "REDACTED_FLAG", command)
    return " ".join(command.split())


def tokens(command: str) -> list[str]:
    return re.findall(r"[\w./:-]+|[^\w\s]", normalize_command(command))


def self_bleu(command: str, references: list[str], max_order: int = 4) -> float:
    """Candidate-to-history BLEU, clipped multi-reference precision, add-one smoothing.

    Effective n-gram order handles short shell commands; this is the per-action
    history Self-BLEU statistic, not a corpus-average diversity estimate.
    """
    candidate = tokens(command)
    refs = [tokens(ref) for ref in references]
    refs = [ref for ref in refs if ref]
    if not candidate or not refs:
        return 0.
    order = min(max_order, len(candidate))
    if order < 1:
        raise ValueError("positive BLEU order required")
    precisions = []
    for n in range(1, order + 1):
        grams = Counter(tuple(candidate[i:i+n]) for i in range(len(candidate) - n + 1))
        maximum = Counter()
        for ref in refs:
            maximum |= Counter(tuple(ref[i:i+n]) for i in range(len(ref) - n + 1))
        matches = sum(min(count, maximum[gram]) for gram, count in grams.items())
        if n == 1 and matches == 0:
            return 0.
        precisions.append((matches + 1.) / (sum(grams.values()) + 1.))
    reference_length = min((len(ref) for ref in refs), key=lambda size: (abs(size - len(candidate)), size))
    brevity = math.exp(min(0., 1. - reference_length / len(candidate)))
    return min(1., brevity * math.exp(sum(math.log(p) for p in precisions) / order))


def cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right) or any(not math.isfinite(v) for v in [*left, *right]):
        raise ValueError("embedding vectors must have equal positive dimensions and finite values")
    norm = math.sqrt(sum(x*x for x in left)) * math.sqrt(sum(x*x for x in right))
    if norm == 0:
        raise ValueError("zero embeddings cannot define cosine similarity")
    return max(0., min(1., sum(a*b for a, b in zip(left, right)) / norm))


class EmbeddingEncoder:
    def __init__(self, model: str, *, endpoint: str | None = None, revision: str | None = None):
        self.model_name, self.revision = model, revision
        self.base = endpoint_base(endpoint) if endpoint else None
        self.model = None
        self.cache = {}
        self.prepared = False

    def prepare(self):
        if self.prepared:
            return
        if self.base:
            response = requests.get(self.base + "/models", timeout=60,
                headers={"Authorization": "Bearer " + os.environ.get(
                    "RED_REWARD_EMBEDDING_API_KEY", os.environ.get("OPENAI_API_KEY", "EMPTY"))})
            response.raise_for_status()
            if self.model_name not in {item["id"] for item in response.json()["data"]}:
                raise RuntimeError("frozen embedding model is not served")
        if self.base is None and self.model is None:
            from sentence_transformers import SentenceTransformer
            # No implicit downloads or trainable embedding parameters.
            self.model = SentenceTransformer(self.model_name, revision=self.revision, device="cpu",
                                             local_files_only=True, trust_remote_code=False)
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
        self.prepared = True

    def encode(self, texts: list[str]) -> list[list[float]]:
        missing = list(dict.fromkeys(text for text in texts if text not in self.cache))
        if missing:
            self.prepare()
            if self.base:
                response = requests.post(self.base + "/embeddings", timeout=60,
                    headers={"Authorization": "Bearer " + os.environ.get(
                        "RED_REWARD_EMBEDDING_API_KEY", os.environ.get("OPENAI_API_KEY", "EMPTY"))},
                    json={"model": self.model_name, "input": missing, "encoding_format": "float"})
                response.raise_for_status()
                data = sorted(response.json()["data"], key=lambda item: item["index"])
                if [item["index"] for item in data] != list(range(len(missing))):
                    raise ValueError("embedding response indices do not match the request")
                vectors = [item["embedding"] for item in data]
            else:
                self.prepare()
                vectors = self.model.encode(missing, convert_to_numpy=True, show_progress_bar=False).tolist()
            if len(vectors) != len(missing):
                raise ValueError("embedding response length mismatch")
            for text, vector in zip(missing, vectors):
                cosine(vector, vector)  # validate even the first cached vector
                self.cache[text] = vector
        return [self.cache[text] for text in texts]


class SuccessMemory:
    def __init__(self, path: Path):
        self.path = path
        self.entries = []
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("schema_version") != 1:
                raise ValueError("unknown successful-action memory schema")
            self.entries = data["entries"]
            if any(not entry.get("verified_success") for entry in self.entries):
                raise ValueError("diversity references must be independently verified successful actions")

    def snapshot(self, limit: int = 0) -> list[dict]:
        entries = self.entries[-limit:] if limit else self.entries
        return [dict(entry) for entry in entries]

    def add(self, entries: list[dict]):
        known = {entry["command"] for entry in self.entries}
        changed = False
        for entry in entries:
            if not entry.get("verified_success"):
                raise ValueError("cannot admit an unverified action to successful history")
            command = normalize_command(entry["command"])
            if command and command not in known:
                self.entries.append({**entry, "command": command})
                known.add(command)
                changed = True
        if changed:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"schema_version": 1, "entries": self.entries}, ensure_ascii=False, indent=2))
            temporary.chmod(0o600)
            temporary.replace(self.path)


class DiversityPenalty:
    def __init__(self, encoder: EmbeddingEncoder | None, cosine_weight: float = .5):
        self.encoder, self.cosine_weight = encoder, cosine_weight

    def score(self, command: str, history: list[dict]) -> tuple[float, dict]:
        references = [entry["command"] for entry in history]
        digest = hashlib.sha256(json.dumps(references, sort_keys=True).encode()).hexdigest()
        if not references:
            return 0., {"self_bleu": 0., "cosine_similarity": 0., "reference_count": 0, "history_sha256": digest}
        command = normalize_command(command)
        bleu = self_bleu(command, references)
        similarity = 0.
        if self.cosine_weight:
            if command in references:
                similarity = 1.
            else:
                if self.encoder is None:
                    raise ValueError("cosine diversity requires a frozen embedding encoder")
                vectors = self.encoder.encode([command, *references])
                similarity = max(cosine(vectors[0], vector) for vector in vectors[1:])
        penalty = (1 - self.cosine_weight) * bleu + self.cosine_weight * similarity
        return penalty, {"self_bleu": bleu, "cosine_similarity": similarity,
                         "cosine_weight": self.cosine_weight, "reference_count": len(references),
                         "embedding_model": self.encoder.model_name if self.encoder else None,
                         "embedding_revision": self.encoder.revision if self.encoder else None,
                         "history_sha256": digest}
