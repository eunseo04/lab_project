from cmath import phase

import requests
import json
from stix2 import MemoryStore, Filter

url = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"

def build_dataset(src):
    
    attack_patterns = src.query([ 
        Filter("type", "=", "attack-pattern")
        ])
    
    strategies = src.query([
        Filter("type", "=", "x-mitre-detection-strategy")
        ])
    
    analytics = src.query([
        Filter("type", "=", "x-mitre-analytic")
        ])
    
    components = src.query([
        Filter("type", "=", "x-mitre-data-component")
        ])
    
    relationships = src.query([
        Filter("type", "=", "relationship")
        ])

    # -------------------------
    # ID → 객체
    # -------------------------

    strategy_map = {
        x["id"]: x
        for x in strategies
    }

    analytic_map = {
        x["id"]: x
        for x in analytics
    }

    component_map = {
        x["id"]: x
        for x in components
    }

    technique_map = {
    attack["id"]: get_attack_id(attack)
    for attack in attack_patterns
}

    # -------------------------
    # Technique 기본 정보
    # -------------------------

    result = {
        get_attack_id(attack): {
            "name": attack.get("name"),
            "description": attack.get("description"),
            "tactics": get_tactics(attack),
            "platforms": attack.get(
            "x_mitre_platforms", []
            ),

            "detection_strategies": [],
            "analytics": [],
            "data_components": []
        }

        for attack in attack_patterns
    }

    # -------------------------
    # Detection Strategy
    # → Technique
    # -------------------------

    for rel in relationships:

        if rel.get("relationship_type") != "detects":
            continue

        strategy_id = rel.get("source_ref")
        technique_stix_id = rel.get("target_ref")

        technique_id = technique_map.get(
            technique_stix_id
        )

        if technique_id not in result:
            continue

        strategy = strategy_map.get(strategy_id)

        if not strategy:
            continue

        result[technique_id]["detection_strategies"].append({
            "stix_id": strategy.get("id"),
            "detection_id": get_attack_id(strategy),
            "name": strategy.get("name"),
            "description": strategy.get("description")
        })

    # -------------------------
    # Detection Strategy
    # → Analytic
    # -------------------------

    for technique in result.values():

        for strategy_data in technique[
            "detection_strategies"
        ]:

            strategy = strategy_map[
                strategy_data["stix_id"]
            ]

            for analytic_id in strategy.get(
                "x_mitre_analytic_refs", []
            ):

                analytic = analytic_map.get(
                    analytic_id
                )

                if not analytic:
                    continue

                technique["analytics"].append({
                    "stix_id": analytic.get("id"),
                    "analytic_id": get_attack_id(
                        analytic
                    ),
                    "name": analytic.get("name"),
                    "description": analytic.get(
                        "description"
                    ),
                    "platforms": analytic.get(
                        "x_mitre_platforms", []
                    ),
                    "mutable_elements": analytic.get(
                        "x_mitre_mutable_elements", []
                    )
                })

    # -------------------------
    # Analytic
    # → Data Component
    # -------------------------

    for technique in result.values():

        for analytic_data in technique[
            "analytics"
        ]:

            analytic = analytic_map[
                analytic_data["stix_id"]
            ]

            for log_ref in analytic.get(
                "x_mitre_log_source_references", []
            ):

                component_id = log_ref.get(
                    "x_mitre_data_component_ref"
                )

                component = component_map.get(
                    component_id
                )

                if not component:
                    continue

                technique["data_components"].append({
                    "stix_id": component.get("id"),
                    "data_component_id": get_attack_id(
                        component
                    ),
                    "name": component.get("name"),
                    "description": component.get(
                        "description"
                    ),
                    "log_source": {
                        "name": log_ref.get("name"),
                        "channel": log_ref.get(
                            "channel"
                        )
                    }
                })

    return result

def get_attack_id(attack):
    for ref in attack.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id")

    return None

def get_tactics(attack):
    return [
        phase.get("phase_name")
        for phase in attack.get("kill_chain_phases", [])
        if phase.get("kill_chain_name") == "mitre-attack"
    ]



if __name__ == "__main__":

    response = requests.get(url)
    response.raise_for_status()

    data = response.json()

    src = MemoryStore(
        stix_data=data["objects"]
    )

    result = build_dataset(src)

    # 생성된 result를 저장
    with open(
        "attack_dataset.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2
        )