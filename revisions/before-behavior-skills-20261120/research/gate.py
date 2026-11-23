"""Non-parametric PPO verification math; no neural optimizer or weight updates."""
import math
import numpy as np


def clipped_terms(old_logprobs, new_logprobs, advantages, epsilon=0.2, log_ratio_bound=10):
    old, new, adv = map(lambda x: np.asarray(x, dtype=float), (old_logprobs, new_logprobs, advantages))
    if old.shape != new.shape or old.shape != adv.shape or not old.size:
        raise ValueError("Gate arrays must have identical, nonempty shapes")
    if not all(np.isfinite(a).all() for a in (old, new, adv)):
        raise ValueError("Non-finite logprob/advantage cannot verify a candidate")
    delta = new - old
    if log_ratio_bound is not None:
        delta = np.clip(delta, -log_ratio_bound, log_ratio_bound)
    with np.errstate(over="ignore", under="ignore"):
        ratio = np.exp(delta)
    if not np.isfinite(ratio).all():
        raise ValueError("Importance ratio overflow; candidate cannot be verified")
    surrogate = np.minimum(ratio * adv, np.clip(ratio, 1-epsilon, 1+epsilon) * adv)
    return ratio, surrogate


def gate_score(experiences, old, new, baselines, *, mode="official_code", epsilon=0.2, gamma=0.99):
    """Official dispatcher formula or the paper's distinct return-to-go formula."""
    offset, total, count, trajectory_means = 0, 0.0, 0, []
    diagnostics = []
    for experience in experiences:
        n = len(experience.transitions)
        if not n:
            continue
        baseline = float(baselines.get(experience.env_name, 0.0))
        if mode == "official_code":
            advantages = np.full(n, (experience.reward - baseline) / n)
        elif mode == "paper_rtg":
            # Full episode RTGs are calculated BEFORE filtering to this skill's
            # active transitions, retaining future consequences across skill spans.
            advantages = np.asarray([t["return_to_go"] - baseline for t in experience.transitions])
        else:
            raise ValueError("Gate mode must be official_code or paper_rtg")
        ratio, terms = clipped_terms(old[offset:offset+n], new[offset:offset+n], advantages, epsilon,
                                     log_ratio_bound=10 if mode == "official_code" else None)
        total += float(terms.sum())
        count += n
        trajectory_means.append(float(terms.mean()))
        diagnostics.append({"n": n, "ratio_min": float(ratio.min()), "ratio_max": float(ratio.max()),
                            "old_logprobs": list(map(float, old[offset:offset+n])),
                            "new_logprobs": list(map(float, new[offset:offset+n])),
                            "importance_ratios": ratio.tolist(), "clipped_surrogates": terms.tolist(),
                            "advantages": advantages.tolist(), "surrogate_mean": float(terms.mean()),
                            "transitions": [{k: t.get(k) for k in ("seed", "agent", "step", "skill_id", "skill_version")}
                                            for t in experience.transitions]})
        offset += n
    if offset != len(old) or offset != len(new) or not count:
        raise ValueError("Gate probability/trajectory alignment mismatch")
    value = total / count if mode == "official_code" else float(np.mean(trajectory_means))
    if not math.isfinite(value):
        raise ValueError("Non-finite candidate score")
    return value, diagnostics


def returns_to_go(rewards, gamma=0.99):
    total, result = 0.0, []
    for reward in reversed(rewards):
        total = float(reward) + gamma * total
        result.append(total)
    return list(reversed(result))
