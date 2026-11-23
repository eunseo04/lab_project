"""Summarize the verification actually executed on this machine."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "verification"


def read(path):
    return json.loads((ROOT / path).read_text())


def compact_result(path):
    result = read(path)
    return {key: result[key] for key in ("result_kind", "episodes", "reward", "elapsed_seconds", "invalid_responses")}


def main():
    suites = ET.parse(VERIFY / "pytest.xml").getroot().findall("testsuite")
    tests = {key: sum(int(s.get(key, 0)) for s in suites)
             for key in ("tests", "failures", "errors", "skipped")}
    assert tests["tests"] > 0 and tests["failures"] == tests["errors"] == 0
    tests["passed"] = tests["tests"] - tests["skipped"]
    matrix = read("verification/matrix-smoke/matrix.json")
    assert len(matrix) == 20
    for row in matrix:
        summary = json.loads(Path(row["summary"]).read_text())
        assert summary["episodes"][0]["steps"] == 12
    training = {}
    for protocol in ("paper", "keep"):
        data = read(f"checkpoints/{protocol}-smoke/training.json")
        assert all(data["parameters_changed"]) and all(data["save_reload_matches"])
        training[protocol] = {key: data[key] for key in ("communication", "input_features",
            "episodes_trained", "validation_only", "parameters_changed", "save_reload_matches", "history")}
    full_runs = {"published_keep": compact_result("verification/published-keep-2x500/summary.json"),
                 "mixed_offline": compact_result("verification/mixed-offline-2x500/summary.json")}
    assert all(e["steps"] == e["simulator_steps"] == 500 for r in full_runs.values() for e in r["episodes"])
    assert all(e["mission_phases"] == [0, 1, 2] for r in full_runs.values() for e in r["episodes"])
    source_hashes = {}
    for folder in ("acd", "scripts", "configs"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                source_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "checked_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "workspace": str(ROOT), "doctor": read("verification/doctor.json"),
        "tests": tests,
        "test_scope": ["Original KEEP graph tensors and policy probability parity",
            "Single tick per joint action", "All five agents and three mission phases",
            "Paper communication bit layout", "JSON parsing and invalid-response fallback",
            "Seed repeatability", "Terminal reward inclusion", "Offline socket guard and disabled API"],
        "training_smoke": training, "full_horizon_runs": full_runs,
        "matrix_smoke": {"cases": len(matrix), "steps_per_episode": 12,
            "teams": ["none", "all_rl", "all_llm", "mixed"],
            "reds": ["finite", "aggressive", "stealthy", "impact", "degrade"],
            "source": "verification/matrix-smoke/matrix.json"},
        "native_horizon": compact_result("verification/keep-native-horizon/summary.json"),
        "analysis": {"valid_reasons_exported": len((VERIFY / "reasons-agent4-episode0.jsonl").read_text().splitlines()),
            "reward_png_and_csv": "verification/matrix-smoke/",
            "cluster_cli_tested_with_synthetic_embeddings_only": True,
            "cluster_fixture_result": "verification/synthetic-analysis-output/clusters.json"},
        "dependencies": {"pip_check": "passed", "locked_setup_dry_run": "passed",
                         "fresh_network_reinstall_tested": False},
        "model_api_calls": 0, "model_api_implementation": "commented out",
        "source_hashes": source_hashes,
        "limits": [
            "No LLM or embedding API inference was performed. Mock responses are labeled as pipeline verification.",
            "The 50000-episode training schedules were implemented but not executed; smoke checkpoints do not establish policy quality or convergence.",
            "The public files in both repositories contain identical KEEP weights, not independently identifiable paper-protocol retrained weights.",
            "Paper-mode input features and training collector fixes are documented in README.md; published numeric results have not been reproduced.",
            "CPU runtime tested. GPU training and custom model backend inference were not tested.",
            "Full official CybORG suite was not rerun; local regression tests and actual simulation/training runs are recorded above."
        ],
        "verdict": "Ready for local KEEP evaluation, GCN/PPO training, and offline paper-protocol experiments. Model integration is intentionally disabled and replaceable."
    }
    (VERIFY / "compatibility_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"tests": tests, "matrix_cases": len(matrix), "model_api_calls": 0,
                      "report": str(VERIFY / "compatibility_report.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
