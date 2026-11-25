"""Pinned ACD and historical local encoders for the eight-bit wire."""
import numpy as np
from CybORG.Shared.Enums import TernaryEnum
from CybORG.Shared.Observation import Observation
from CybORG.Agents.LLMAgents.comm_vector import CommVectorGenerator
from .config import LEGACY_ENCODER, ORIGINAL_ENCODER

AGENT_SUBNETS = (
    ("restricted_zone_a_subnet",),
    ("operational_zone_a_subnet",),
    ("restricted_zone_b_subnet",),
    ("operational_zone_b_subnet",),
    ("public_access_zone_subnet", "admin_network_subnet", "office_network_subnet"),
)


def subnet_owner(hostname):
    for i, subnets in enumerate(AGENT_SUBNETS):
        if any(hostname.startswith(s + "_") or hostname == s for s in subnets):
            return i
    return None


def create_message(observation, host_ip_map, encoder=LEGACY_ENCODER):
    if encoder == ORIGINAL_ENCODER:
        if isinstance(observation, dict):
            wrapped = Observation()
            wrapped.data = observation
            observation = wrapped
        # The pinned function does not use its last_action argument.
        message = np.asarray(CommVectorGenerator.create_comm_message(observation, None, host_ip_map))
        if message.shape != (8,):
            raise ValueError(f"Expected eight ACD message bits, got {message.shape}")
        return message.astype(bool)
    if encoder != LEGACY_ENCODER:
        raise ValueError(f"Unknown communication encoder: {encoder}")
    return _legacy_message(observation, host_ip_map)


def _legacy_message(observation, host_ip_map):
    """Only agent-visible events enter this summary; never read red sessions/state.

    Bits 0..4: suspicious remote owners. Bits 5..6: max observed severity
    00 clean, 01 scan, 10 user, 11 admin. Bit 7: action in progress.
    Unknown remote IPs and the undefended contractor network have no owner bit.
    """
    data = getattr(observation, "data", observation)
    message = np.zeros(8, dtype=bool)
    severity = 0
    remote_ips = set()
    for value in data.values():
        if not isinstance(value, dict):
            continue
        seen = set()
        for proc in value.get("Processes", []):
            connections = proc.get("Connections", [])
            if not connections and len(proc) < 2:
                severity = max(severity, 1)
            for conn in connections:
                ip = conn.get("remote_address")
                if ip is not None:
                    ip = str(ip)
                    remote_ips.add(ip)
                    if ip in seen:
                        severity = max(severity, 1)
                    seen.add(ip)
        if len(value.get("Sessions", [])) > 1:
            severity = max(severity, 2)
        for item in value.get("Files", []):
            name = item.get("File Name", "")
            if name in ("cmd.sh", "cmd.exe"):
                severity = max(severity, 2)
            elif name in ("escalate.sh", "escalate.exe"):
                severity = 3
    message[5] = bool(severity & 2)
    message[6] = bool(severity & 1)
    message[7] = data.get("success") == TernaryEnum.IN_PROGRESS
    if severity:
        ip_host = {str(ip): host for host, ip in host_ip_map.items()}
        for ip in remote_ips:
            owner = subnet_owner(ip_host.get(ip, ""))
            if owner is not None:
                message[owner] = True
    return message
