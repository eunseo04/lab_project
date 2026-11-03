"""Research Red reward used only by the proposed on-policy GRPO trainer."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

from .diversity_reward import DiversityPenalty, EmbeddingEncoder, SuccessMemory, normalize_command
from .llm import endpoint_base, served_model
from .semantic_reward import SemanticJudge


@dataclass(frozen=True)
class RedRewardConfig:
    alpha: float
    beta: float
    gamma: float
    semantic_model: str | None = None
    semantic_endpoint: str | None = None
    embedding_model: str | None = None
    embedding_endpoint: str | None = None
    embedding_revision: str | None = None
    cosine_weight: float = .5
    success_threshold: float = .2
    history_limit: int = 0
    version: str = "research_red_grpo_v1"

    def __post_init__(self):
        for name in ("alpha", "beta", "gamma", "cosine_weight", "success_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
        if self.alpha <= 0 or not 0 <= self.cosine_weight <= 1 or not 0 < self.success_threshold <= 1:
            raise ValueError("alpha>0, 0<=cosine_weight<=1, 0<success_threshold<=1 required")
        if type(self.history_limit) is not int or self.history_limit < 0:
            raise ValueError("history_limit must be a nonnegative integer; 0 uses all successful history")
        if self.beta and not (self.semantic_model and self.semantic_endpoint):
            raise ValueError("positive semantic weight requires a frozen judge model and endpoint")
        if self.gamma and self.cosine_weight and not self.embedding_model:
            raise ValueError("cosine diversity requires --red-embedding-model or an explicit cosine-weight=0 ablation")
        if self.semantic_endpoint:
            endpoint_base(self.semantic_endpoint)
        if self.embedding_endpoint:
            endpoint_base(self.embedding_endpoint)
        if self.version != "research_red_grpo_v1":
            raise ValueError("unknown Red GRPO reward version")

    def to_dict(self) -> dict:
        return asdict(self)


def config_from_args(args, *, required=True) -> RedRewardConfig | None:
    coefficients = [getattr(args, "red_reward_" + name, None) for name in ("alpha", "beta", "gamma")]
    if all(value is None for value in coefficients) and not required:
        return None
    if any(value is None for value in coefficients):
        raise ValueError("set --red-reward-alpha, --red-reward-beta, --red-reward-gamma explicitly; weights require experimental tuning")
    judge_model = served_model(getattr(args, "red_semantic_model", None) or args.base_model)
    judge_endpoint = getattr(args, "red_semantic_endpoint", None) or args.endpoint
    student_endpoint = args.trained_endpoint or args.endpoint
    student_model = served_model(args.trained_model)
    if coefficients[1] and endpoint_base(judge_endpoint) == endpoint_base(student_endpoint) and (
            judge_model == student_model or judge_model.startswith(student_model + "-online-")):
        raise ValueError("the semantic judge must be frozen and separate from the updated GRPO policy")
    return RedRewardConfig(*coefficients, semantic_model=judge_model, semantic_endpoint=judge_endpoint,
        embedding_model=getattr(args, "red_embedding_model", None),
        embedding_endpoint=getattr(args, "red_embedding_endpoint", None),
        embedding_revision=getattr(args, "red_embedding_revision", None),
        cosine_weight=getattr(args, "red_diversity_cosine_weight", .5),
        history_limit=getattr(args, "red_diversity_history_limit", 0),
        success_threshold=getattr(args, "red_diversity_success_threshold", .2))


class RedTrainingReward:
    def __init__(self, config: RedRewardConfig, memory: SuccessMemory | None, *, semantic=None, diversity=None):
        if config.gamma and memory is None:
            raise ValueError("positive diversity weight requires a successful-action memory")
        if config.beta and semantic is None:
            raise ValueError("positive semantic weight requires a frozen semantic evaluator")
        if config.gamma and diversity is None:
            raise ValueError("positive diversity weight requires a diversity evaluator")
        self.config, self.memory, self.semantic, self.diversity = config, memory, semantic, diversity

    def prepare(self):
        if self.semantic:
            self.semantic.prepare()
        if self.diversity and self.diversity.encoder:
            self.diversity.encoder.prepare()

    def snapshot(self) -> list[dict]:
        return self.memory.snapshot(self.config.history_limit) if self.memory else []

    def score(self, row: dict, completion: str, attack: float, evidence: dict, history: list[dict]) -> tuple[float, dict]:
        if isinstance(attack, bool) or not math.isfinite(attack) or not -1 <= attack <= 1:
            raise ValueError("Docker attack reward must be finite and in [-1,1]")
        commands = [line.split(":", 1)[1].strip() for line in completion.splitlines()
                    if line.strip().upper().startswith("CMD:")]
        executed = evidence.get("environment_executed") is True
        command = commands[0] if len(commands) == 1 else ""
        if executed and not command:
            raise ValueError("environment result has no single attributable generated command")
        semantic, penalty = 0., 0.
        semantic_info, diversity_info = {"computed": False}, {"computed": False}
        if executed and self.config.beta:
            semantic, semantic_info = self.semantic.score(row, command)
            semantic_info = {"computed": True, **semantic_info}
        if executed and self.config.gamma:
            penalty, diversity_info = self.diversity.score(command, history)
            diversity_info = {"computed": True, **diversity_info}
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in (semantic, penalty)):
            raise ValueError("semantic reward and diversity penalty must lie in [0,1]")
        terms = {"attack": self.config.alpha * attack, "semantic": self.config.beta * semantic,
                 "diversity": -self.config.gamma * penalty}
        total = sum(terms.values())
        if not math.isfinite(total):
            raise ValueError("combined Red reward overflowed; use finite experimental scales")
        return total, {"version": self.config.version, "R_Attack": attack, "R_Semantic": semantic,
                       "P_Diversity": penalty, "weighted_terms": terms,
                       "weights": {"alpha": self.config.alpha, "beta": self.config.beta, "gamma": self.config.gamma},
                       "semantic": semantic_info, "diversity": diversity_info, "command": normalize_command(command)}

    def commit_successes(self, samples: list[dict]):
        if not self.memory:
            return
        successes = []
        for sample in samples:
            evidence, attack = sample["evidence"], sample["components"]["R_Attack"]
            verified = evidence.get("environment_executed") is True and any(
                evidence.get(key) is True for key in ("initial_access", "user_flag", "root_flag", "recon_http_verified"))
            if verified and attack >= self.config.success_threshold:
                successes.append({"command": sample["components"]["command"], "verified_success": True,
                                  "trajectory_id": sample["trajectory_id"], "row_id": sample["row_id"],
                                  "attack_reward": attack,
                                  "evidence": {key: evidence.get(key) for key in
                                               ("initial_access", "user_flag", "root_flag", "recon_http_verified")}})
        self.memory.add(successes)


def build_red_reward(config: RedRewardConfig, history: Path | None) -> RedTrainingReward:
    semantic = SemanticJudge(config.semantic_endpoint, config.semantic_model) if config.beta else None
    encoder = EmbeddingEncoder(config.embedding_model, endpoint=config.embedding_endpoint,
                               revision=config.embedding_revision) if config.gamma and config.cosine_weight else None
    diversity = DiversityPenalty(encoder, config.cosine_weight) if config.gamma else None
    memory = SuccessMemory(history) if config.gamma and history else None
    return RedTrainingReward(config, memory, semantic=semantic, diversity=diversity)
