from copy import deepcopy
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
base = {"arm": "E0", "composition": "hybrid", "red": "finite", "steps": 500,
    "episodes": 100, "train_episodes": 6, "iterations": 3,
    "train_seed": 100, "eval_seed": 10000, "threads": 1,
    "validation_seed":5000,"validation_episodes":1,
    "profile": "controlled-comparison", "scoring": "official",
    "communication": "paper_projected_keep", "skip_busy_policy": True,
    "pool_size": 10, "weights": "third_party/cage-4-submission/weights", "output": "research_outputs/full",
    "model": {"endpoint": "http://127.0.0.1:8018", "model": "Qwen/Qwen3-8B",
        "revision": "b968826d9c46dd6066d109eabc6255188de91218", "temperature": 1.0,
        "max_tokens": 192, "timeout_seconds": 240},
    "selector": {"epsilon": 0.1, "max_skill_age": 8},
    "evolution": {"enabled": True, "gate_mode": "official_code", "ppo_gate": True,
        "semantic_gradient": True, "score_maintenance": True, "threshold": 6,
        "best_of_n": 3, "max_evolutions": 2, "semantic_chunk_steps": 8,
        "clip": 0.2, "gamma": 0.99, "ema_alpha": 0.1, "acceptance_margin": 0.0}}
configs = {"controlled": base}
closed = deepcopy(base)
closed.update(train_episodes=12, iterations=6, output="research_outputs/closed-loop")
configs["closed-loop"] = closed
for arm in ("E0", "E1", "E2", "E3"):
    configs[arm] = {**deepcopy(base), "arm": arm}
paper = deepcopy(base)
paper.update(profile="paper-compatible", skip_busy_policy=False, episodes=2, output="research_outputs/paper-compatible")
configs["paper-compatible"] = paper
rtg = deepcopy(base); rtg["evolution"]["gate_mode"] = "paper_rtg"; rtg["output"] = "research_outputs/paper-rtg"
rtg['evolution']['maintenance_mode']='paper_score'
configs["paper-rtg"] = rtg
for ablation, flag in [("without_semantic_gradient", "semantic_gradient"), ("without_ppo_gate", "ppo_gate"), ("without_score", "score_maintenance")]:
    c = deepcopy(base); c["arm"] = "E2"; c["evolution"][flag] = False
    c["output"] = "research_outputs/ablation-"+ablation
    configs[ablation] = c
pilot = deepcopy(base)
pilot.update(episodes=2, train_episodes=2, iterations=1, steps=12, output="research_outputs/real-pilot")
pilot["evolution"].update(threshold=1, best_of_n=2, max_evolutions=1, semantic_chunk_steps=4)
configs["pilot"] = pilot
for name, config in configs.items():
    (ROOT/"research_configs"/(name+".yaml")).write_text(yaml.safe_dump(config, sort_keys=False))
print("Wrote", len(configs), "explicit research configs")
