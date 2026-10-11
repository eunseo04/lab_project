import json
import re
from pathlib import Path

dataset_path = Path("attack_dataset.json")
rules_path = Path("wazuh-docker/single-node/wazuh_mitre_ids.txt")

with dataset_path.open(encoding="utf-8") as f:
    dataset = json.load(f)

wazuh_ids = set(
    rules_path.read_text(encoding="utf-8").split()
)

attack_ids = {
    technique_id
    for technique_id in dataset
    if re.fullmatch(r"T\d{4}(?:\.\d{3})?", technique_id)
}

matched = attack_ids & wazuh_ids
unmatched = attack_ids - wazuh_ids

print(f"ATT&CK 데이터셋 기법 수: {len(attack_ids)}")
print(f"Wazuh 규칙에 ID가 매핑된 기법 수: {len(matched)}")
print(f"Wazuh 규칙에서 ID를 찾지 못한 기법 수: {len(unmatched)}")

print("\n[매핑된 기법 예시]")
for technique_id in sorted(matched)[:20]:
    print(technique_id, dataset[technique_id].get("name"))

print("\n[ID 매핑을 찾지 못한 기법 예시]")
for technique_id in sorted(unmatched)[:20]:
    print(technique_id, dataset[technique_id].get("name"))