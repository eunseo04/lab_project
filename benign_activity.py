"""
benign_activity.py -- a background generator of NORMAL (benign) host/network events.

Why this exists (DefenderBench rule): a detector's quality is Macro-F1, which needs
BOTH classes. Without benign traffic you can only ever measure recall ("did we catch
the attack?") and never precision ("did we alarm on normal activity?"). A blue team
that alerts on *everything* would score 100% detection under a recall-only metric.
So the self-play episode interleaves these benign events with the attacker's actions;
the attacker then has to operate *within* this normal baseline (low-and-slow, blending
in) and blue is scored on whether it discriminates malicious from benign -- not just
whether it is noisy.

Each event is shaped like a Wazuh alert `_source` so it flows through the same
extract_wazuh_context() path as real alerts. `malicious=False` is the ground truth;
`noise` (0..1) models how "alarm-like" a benign event looks (a failed login burst is
noisier than a cache hit) and drives the false-positive model in the pipeline.
"""
from __future__ import annotations

import random
from datetime import datetime, timezone

# (rule.description, rule.level, decoder, data template, noise, real_command) -- normal ops.
# `real_command` is the actual benign shell command run ON the target in --live mode so
# Wazuh sees genuine normal activity; we then check whether it (wrongly) alerted -> true FP.
# Commands are deliberately harmless, idempotent, and '|| true'-guarded so they never fail
# the step; they mimic routine admin/monitoring/user activity.
_BENIGN_TEMPLATES = [
    ("Web server 200 response", 2, "web-accesslog", {"srcip": "10.0.0.{n}", "url": "/api/balance"}, 0.01,
     "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1/ 2>/dev/null || true"),
    ("Successful sudo to service account", 3, "sudo", {"user": "deploy", "command": "systemctl status"}, 0.08,
     "sudo -n true 2>/dev/null || id"),
    ("User login session check (sshd)", 3, "sshd", {"srcuser": "analyst", "srcip": "10.0.0.{n}"}, 0.05,
     "who; last -n 1 2>/dev/null || true"),
    ("Scheduled cron job executed", 2, "cron", {"user": "root", "command": "/usr/local/bin/backup.sh"}, 0.04,
     "crontab -l 2>/dev/null || true"),
    ("Package manager query", 3, "yum", {"user": "root", "command": "yum update -y"}, 0.06,
     "(command -v yum >/dev/null && yum --version || dpkg --version) 2>/dev/null | head -1 || true"),
    ("Health-check probe", 1, "web-accesslog", {"srcip": "10.0.0.{n}", "url": "/healthz"}, 0.00,
     "curl -s -o /dev/null http://127.0.0.1/healthz 2>/dev/null || true"),
    ("Failed login (mistyped password)", 5, "sshd", {"srcuser": "analyst", "srcip": "10.0.0.{n}"}, 0.20,
     "ssh -o BatchMode=yes -o ConnectTimeout=1 -o StrictHostKeyChecking=no nouser@127.0.0.1 true 2>/dev/null || true"),
    ("Config file read by monitoring agent", 3, "ossec", {"user": "wazuh", "command": "cat /etc/app.conf"}, 0.07,
     "cat /etc/hostname >/dev/null 2>&1 || true"),
]


def _iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


class BenignActivityGenerator:
    """Deterministic (seeded) stream of normal events, Wazuh-alert-shaped."""

    def __init__(self, rng: random.Random | None = None, agent_name: str = "target"):
        self.rng = rng or random.Random(7)
        self.agent_name = agent_name

    def event(self, ts: datetime | None = None) -> dict:
        desc, level, decoder, data_tpl, noise, real_command = self.rng.choice(_BENIGN_TEMPLATES)
        data = {k: v.replace("{n}", str(self.rng.randint(2, 254))) if isinstance(v, str) else v
                for k, v in data_tpl.items()}
        ts = ts or datetime.now(timezone.utc)
        return {
            "timestamp": _iso(ts),
            "agent": {"name": self.agent_name},
            "rule": {"description": desc, "level": level, "mitre": {"id": []}},
            "decoder": {"name": decoder},
            "data": data,
            "_benign": True,          # ground-truth label (negative class)
            "_noise": noise,          # how alarm-like this normal event looks (0..1)
            "command": real_command,  # actual benign command run on target in --live mode
        }

    def batch(self, n: int, base_ts: datetime | None = None) -> list[dict]:
        return [self.event(base_ts) for _ in range(max(0, n))]
