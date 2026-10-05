import json

def load_attack_dataset():
    with open(
        "attack_dataset.json",
        "r",
        encoding="utf-8"
        ) as f:
        return json.load(f)