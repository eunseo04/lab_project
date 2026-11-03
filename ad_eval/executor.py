"""Bounded Docker execution; evaluation never falls back to the research host shell."""
from __future__ import annotations

from dataclasses import dataclass
import subprocess

from .docker_lab import HOSTS, KALI


@dataclass(frozen=True)
class CommandResult:
    output: str
    returncode: int | None
    timed_out: bool = False


def execute(command: str, *, container: str, timeout: float = 35) -> CommandResult:
    if container not in {*HOSTS.values(), *KALI.values()}:
        raise ValueError("evaluation requires an explicit local range container")
    if not command.strip() or timeout <= 0:
        raise ValueError("command and positive timeout required")
    try:
        # The container timeout also kills the shell after a host-side docker client timeout.
        result = subprocess.run(
            ["docker", "exec", container, "timeout", "--kill-after=2s", str(timeout),
             "/bin/sh", "-c", command],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            timeout=timeout + 5, check=False,
        )
        return CommandResult(result.stdout, result.returncode, result.returncode in {124, 137})
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return CommandResult(output, None, True)


def run_command(command: str, *, container: str, timeout: float = 35) -> str:
    result = execute(command, container=container, timeout=timeout)
    if result.timed_out:
        return result.output + "\n[executor: command timed out]"
    return result.output
