"""Assemble the pinned CC4 base and the paper's extensions, without editing either."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "cage-challenge-4": "https://github.com/cage-challenge/cage-challenge-4",
    "llms-are-acd": "https://github.com/r4wd3r/llms-are-acd",
    "cage-4-submission": "https://github.com/cybermonic/cage-4-submission",
}


def prepare():
    third_party = ROOT / "third_party"
    runtime = ROOT / "runtime"
    runtime.mkdir(exist_ok=True)
    # Only the importable package is needed; exclude the old verification artifacts.
    shutil.copytree(third_party / "cage-challenge-4/CybORG", runtime / "CybORG",
                    dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(third_party / "llms-are-acd/CybORG", runtime / "CybORG",
                    dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # Expose the original graph update/packing to a single simulator step, allowing
    # Action objects for LLMs and integer IDs for KEEP to share that same step.
    p = runtime / "CybORG/Agents/Wrappers/CybermonicWrappers/graph_wrapper.py"
    s = p.read_text()
    s = s.replace("def step(self, action):", "def step(self, action, messages=None):")
    s = s.replace("action_dict=action, messages=self.msg",
                  "action_dict=action, messages=self.msg if messages is None else messages")
    s = s.replace("dict_obs = self.env.environment_controller.get_last_observation(agent).data",
                  "dict_obs = deepcopy(self.env.environment_controller.get_last_observation(agent).data)")
    s = s.replace("def reset(self):", "def reset(self, *args, **kwargs):")
    s = s.replace("obs_tab, action_mask = super().reset()",
                  "obs_tab, action_mask = super().reset(*args, **kwargs)")
    start = s.index("            # Indicates if msg was recieved")
    end = s.index("            # Update the graph based on the raw dictionary", start)
    old = s[start:end]
    helper = "    def _message_features(self, i, msg):\n"
    helper += "\n".join(line[4:] for line in old.rstrip().splitlines())
    helper += "\n        return msg\n\n"
    s = s[:start] + "            msg = self._message_features(i, msg)\n\n" + s[end:]
    s = s.replace("    def step(self, action, messages=None):", helper + "    def step(self, action, messages=None):")
    p.write_text(s)

    # The original formatter prints the full observation each call. Keep its
    # formatting but send traces to local JSONL instead of the terminal.
    p = runtime / "CybORG/Agents/LLMAgents/llm_adapter/obs_formatter.py"
    s = p.read_text().replace('        print(f"{observation}")', "        # Observation is recorded by the local experiment logger.")
    s = s.replace("list(set(range(5)) - {agent_number})", "[i for i in range(5) if i != agent_number]")
    p.write_text(s)

    revisions = {}
    for name, url in SOURCES.items():
        commit = subprocess.check_output(["git", "-C", str(third_party / name), "rev-parse", "HEAD"], text=True).strip()
        revisions[name] = {"url": url, "commit": commit}
    weights = {}
    for path in sorted((third_party / "cage-4-submission/weights").glob("*.pt")):
        weights[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    lock = {"sources": revisions, "published_weights_sha256": weights,
            "paper": "https://arxiv.org/abs/2505.04843v2"}
    (ROOT / "sources.lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    print("Prepared", runtime, "(upstream checkouts unchanged)")


if __name__ == "__main__":
    prepare()
