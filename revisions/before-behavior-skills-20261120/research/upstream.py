"""Load unchanged Skill-Pro classes with an explicitly injected encoder dependency."""
import importlib
from pathlib import Path
import sys
import types
import numpy as np
from . import ROOT

_encoder = None


def configure_encoder(callback):
    global _encoder
    _encoder = callback


def encode_text(text, normalize=True):
    if _encoder is None:
        raise RuntimeError("Configure the pinned MiniLM encoder before constructing a SkillPool")
    value = np.asarray(_encoder(text), dtype=float)
    if value.ndim != 1 or not value.size or not np.isfinite(value).all():
        raise RuntimeError("Invalid skill embedding; semantic maintenance cannot be verified")
    return value


source = ROOT / "third_party/Skill-Pro"
sys.path.insert(0, str(source))
# This sole dependency injection avoids importing a second transformer runtime in
# CC4's CPU environment. The real backend serves the original MiniLM embeddings.
injected_encoder = types.ModuleType("utils.encode")
injected_encoder.encode_text = encode_text
injected_encoder.init_text_encoder = lambda *a, **k: None
sys.modules["utils.encode"] = injected_encoder
try:
    from data_structures import Skill, Experience
    from pool_managers import ExperiencePool, GoldenExperiencePool
    from Skills.skill_pool import SkillPool as OriginalSkillPool
    from Skills.skill_evolution import SkillEvolution as OriginalEvolution
    from Skills.skill_evolution import build_success_failure_eval_batch, get_priority
finally:
    sys.path.remove(str(source))
