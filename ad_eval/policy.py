"""Conservative local lab action allowlist for the experimental adapter."""
from __future__ import annotations
import re
from .docker_lab import IPS, OPPONENT

FORBIDDEN = ("rm -rf", "mkfs", "shutdown", "reboot", "hping", "flood", "hydra",
             "ncrack", "medusa", "patator", "chpasswd", "passwd ",
             "systemctl stop", "service stop", "poweroff")


def command_allowed(s: str, role: str, team: str) -> bool:
    s = (s or "").strip()
    low = s.lower()
    if role not in {"red", "blue"} or team not in IPS:
        return False
    if not s or s.upper() == "NODEF" or any(x in low for x in FORBIDDEN):
        return False
    if role == "red":
        target = IPS[OPPONENT[team]]
        if target not in s:
            return False
        found = re.findall(r"https?://([^/\s'\"]+)", s, re.I)
        if any(not h.startswith(target + ":") for h in found):
            return False
        # Disallow secondary hard-coded non-lab IPv4 addresses.
        ipv4 = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", s)
        if any(ip != target and ip != "127.0.0.1" for ip in ipv4):
            return False
    else:
        if any(x in low for x in ("root.txt", "user.txt", "iptables -f", "iptables -p drop",
                                 "systemctl disable", "pkill", "killall", "ufw deny",
                                 "iptables --flush", "truncate ", "policy drop")):
            return False
        if "iptables" in low and ("-j drop" in low or "-j reject" in low) and not (
                "-m string" in low or "--string" in low):
            return False
        if IPS[OPPONENT[team]] in s:
            return False
    return True
