"""
redtools -- Windows-compatible replacement for the CAI tool stack (tool/).

The original tool/ modules execute commands via tool.common.run_command, which on
Windows fails: it imports termios/pty (POSIX-only, for interactive PTY sessions) and
pulls in the broken sdk / cai packages. This module keeps the SAME command-execution
behavior and the SAME tool signatures (nmap, curl, nc, ...), but:
  * no termios/pty and no interactive session manager (the "대화형" parts are dropped),
  * no sdk @function_tool decorator (plain callables),
  * cross-platform execution via docker exec / subprocess.

Execution target resolution (mirrors CAI's "environment detection"):
  1. ctf  given      -> run inside the CTF target container (ctf.get_shell)
  2. container given -> run inside that Docker container (docker exec)
  3. otherwise       -> run on the local host (subprocess)

These are the attack primitives the red agent calls against the Target.
"""
from __future__ import annotations

import shlex
import subprocess
from typing import Optional

try:
    import docker as _docker
except Exception:  # pragma: no cover
    _docker = None

_DOCKER_CLIENT = None


def _client():
    global _DOCKER_CLIENT
    if _DOCKER_CLIENT is None:
        if _docker is None:
            raise RuntimeError("docker SDK not available")
        _DOCKER_CLIENT = _docker.from_env()
    return _DOCKER_CLIENT


def run_command(command: str, ctf=None, container: Optional[str] = None,
                timeout: int = 100) -> str:
    """Execute a shell command against the resolved target. Non-interactive.

    Returns combined stdout/stderr as text (never raises for tool failures).
    """
    if not command or not command.strip():
        return "Error: No command provided"
    # 1) CTF target container
    if ctf is not None and hasattr(ctf, "get_shell"):
        try:
            return ctf.get_shell(command, timeout=timeout) or ""
        except Exception as e:  # pragma: no cover
            return f"ctf exec error: {e}"
    # 2) named Docker container
    if container:
        try:
            res = _client().containers.get(container).exec_run(["/bin/sh", "-c", command])
            return res.output.decode("utf-8", "ignore")
        except Exception as e:  # pragma: no cover
            return f"container exec error: {e}"
    # 3) local host (Windows-safe: no shell PTY, capture output)
    try:
        proc = subprocess.run(command if isinstance(command, str) else shlex.join(command),
                              shell=True, capture_output=True, text=True, timeout=timeout)
        return (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"
    except Exception as e:  # pragma: no cover
        return f"local exec error: {e}"


# ---- tool wrappers (same names/semantics as tool/, no interactive sessions) ----
def generic_linux_command(command: str = "", ctf=None, container: Optional[str] = None) -> str:
    """Run any command against the target (replacement for tool.generic_linux_command)."""
    return run_command(command, ctf=ctf, container=container)


def nmap(args: str, target: str, ctf=None, container: Optional[str] = None) -> str:
    return run_command(f"nmap {args} {target}", ctf=ctf, container=container)


def curl(args: str, url: str, ctf=None, container: Optional[str] = None) -> str:
    return run_command(f"curl {args} {url}", ctf=ctf, container=container)


def wget(args: str, url: str, ctf=None, container: Optional[str] = None) -> str:
    return run_command(f"wget {args} {url}", ctf=ctf, container=container)


def netcat(host: str, port: str, args: str = "", ctf=None, container: Optional[str] = None) -> str:
    return run_command(f"nc {args} {host} {port}", ctf=ctf, container=container)


def netstat(args: str = "-tulpn", ctf=None, container: Optional[str] = None) -> str:
    return run_command(f"netstat {args}", ctf=ctf, container=container)


# Registry so the LLM/agent can discover available tools by name.
TOOLS = {
    "generic_linux_command": generic_linux_command,
    "nmap": nmap,
    "curl": curl,
    "wget": wget,
    "netcat": netcat,
    "netstat": netstat,
}


def run_tool(name: str, ctf=None, container: Optional[str] = None, **kwargs) -> str:
    """Dispatch a named tool. Unknown name falls back to generic_linux_command."""
    fn = TOOLS.get(name)
    if fn is None:
        return run_command(kwargs.get("command", ""), ctf=ctf, container=container)
    return fn(ctf=ctf, container=container, **kwargs)
