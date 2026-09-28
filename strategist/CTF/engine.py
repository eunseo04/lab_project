from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Optional
import subprocess
import shlex


# ============================================================
# CTF Configuration
# ============================================================

@dataclass
class CTFConfig:
    """
    Configuration for the financial CTF Docker environment.

    This replaces the old GOPSConfig.

    GOPS:
        num_turns
        playing_cards
        score_cards

    CTF:
        target container
        target IP
        target OS
        available tools
        objectives
    """

    container_name: str = "bank-web"
    target_ip: str = "192.168.3.2"
    target_os: str = "linux"
    attacker_ip: str = "192.168.3.10"
    attacker_os: str = "kali"
    available_tools: tuple[str, ...] = (
        "nmap",
        "curl",
        "wget",
        "ping",
        "ssh",
    )

    objectives: tuple[str, ...] = (
        "discover_target",
        "discover_services",
        "identify_vulnerability",
        "obtain_initial_access",
        "escalate_privilege",
        "access_internal_service",
    )

    docker_network: str = "financial_cyber_range"


# ============================================================
# CTF State
# ============================================================

@dataclass(frozen=True)
class CTFState:
    """
    Immutable representation of the current CTF environment.

    This is the state used by MCTS.

    Important:
        Lists are intentionally stored as tuples so that the state
        can safely be used as a tree/search state.
    """
    
    attacker_os: str = "kali"
    attacker_ip: str = "192.168.3.10"
    target_ip: str = "192.168.3.2"
    target_os: str = "linux"
    hosts: tuple[str, ...] = ()
    services: tuple[str, ...] = ()
    vulnerabilities: tuple[str, ...] = ()
    credentials: tuple[str, ...] = ()
    compromised: bool = False
    privilege_level: str = "none"
    internal_access: bool = False
    detected: bool = False
    defensive_controls: tuple[str, ...] = ()
    completed_objectives: tuple[str, ...] = ()
    previous_observations: tuple[str, ...] = ()
    current_objective: str = "discover_target"
    available_tools: tuple[str, ...] = (
        "nmap",
        "curl",
        "wget",
        "ping",
        "ssh",
    )
    done: bool = False
    reward: float = 0.0

    def is_done(self) -> bool:
        return self.done

    def get_reward(self) -> float:
        return self.reward

    def get_state_description(self) -> str:
        """
        Convert the internal state into a textual description.

        This is useful when passing the state to an LLM.
        """

        return f"""
=== CTF STATE ===

Attacker:
- OS: {self.attacker_os}
- IP: {self.attacker_ip}

Target:
- OS: {self.target_os}
- IP: {self.target_ip}

Discovered Hosts:
{self.hosts}

Discovered Services:
{self.services}

Discovered Vulnerabilities:
{self.vulnerabilities}

Credentials:
{self.credentials}

Compromised:
{self.compromised}

Privilege Level:
{self.privilege_level}

Internal Access:
{self.internal_access}

Detected:
{self.detected}

Defensive Controls:
{self.defensive_controls}

Completed Objectives:
{self.completed_objectives}

Current Objective:
{self.current_objective}

Previous Observations:
{self.previous_observations}

Available Tools:
{self.available_tools}

Done:
{self.done}

Reward:
{self.reward}
""".strip()

    @classmethod
    def initial(cls, config: CTFConfig) -> "CTFState":
        """
        Create the initial CTF state.
        """

        return cls(
            attacker_os=config.attacker_os,
            attacker_ip=config.attacker_ip,
            target_ip=config.target_ip,
            target_os=config.target_os,
            available_tools=config.available_tools,
            current_objective=config.objectives[0]
            if config.objectives
            else "",
        )


# ============================================================
# CTF Action
# ============================================================

@dataclass(frozen=True)
class CTFAction:
    """
    Action that an agent can execute.

    Example:

        CTFAction(
            tool="nmap",
            command="nmap -sV 192.168.3.2",
            description="Scan target services"
        )
    """

    tool: str
    command: str
    description: str = ""

    actor: str = "red"

    def __str__(self) -> str:
        return f"{self.tool}: {self.command}"


# ============================================================
# Docker Command Result
# ============================================================

@dataclass
class CommandResult:
    """
    Result returned from Docker/container command execution.
    """

    command: str

    stdout: str

    stderr: str

    return_code: int

    @property
    def success(self) -> bool:
        return self.return_code == 0


# ============================================================
# CTF Environment
# ============================================================

class CTFEnvironment:
    """
    Interface between the agent and the actual CTF Docker target.

    This class is responsible for REAL environment interaction.

    MCTS should normally NOT directly call execute() while expanding
    hypothetical branches.

    Instead:

        MCTS
          ↓
        simulated_transition()
          ↓
        CTFState

    After MCTS selects the final action:

        CTFEnvironment.execute(action)
    """

    def __init__(
        self,
        config: Optional[CTFConfig] = None,
        state: Optional[CTFState] = None,
    ):
        self.config = config or CTFConfig()

        self.state = (
            state
            if state is not None
            else CTFState.initial(self.config)
        )

    # --------------------------------------------------------
    # Docker execution
    # --------------------------------------------------------

    def execute(
        self,
        action: CTFAction,
        timeout: int = 30,
    ) -> CommandResult:
        """
        Execute an action inside the target Docker container.

        Example:

            docker exec bank-web sh -c "..."
        """

        command = [
            "docker",
            "exec",
            self.config.container_name,
            "sh",
            "-c",
            action.command,
        ]

        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            result = CommandResult(
                command=action.command,
                stdout=process.stdout,
                stderr=process.stderr,
                return_code=process.returncode,
            )

        except subprocess.TimeoutExpired as exc:
            result = CommandResult(
                command=action.command,
                stdout=exc.stdout or "",
                stderr="Command timed out.",
                return_code=-1,
            )

        return result

    # --------------------------------------------------------
    # Observe target
    # --------------------------------------------------------

    def observe(self) -> str:
        """
        Obtain a basic observation from the target container.

        This does not expose hidden CTF flags.
        """

        action = CTFAction(
            tool="system",
            command="uname -a && (ip addr || true) && (ss -tuln || true)",
            description="Collect basic target information",
        )

        result = self.execute(action)

        observation = (
            f"return_code={result.return_code}\n"
            f"stdout={result.stdout}\n"
            f"stderr={result.stderr}"
        )

        self.state = replace(
            self.state,
            previous_observations=(
                *self.state.previous_observations,
                observation,
            ),
        )

        return observation

    # --------------------------------------------------------
    # Execute action and update state
    # --------------------------------------------------------

    def step(
        self,
        action: CTFAction,
    ) -> tuple[CTFState, CommandResult]:
        """
        Execute an action on the real Docker target.

        This is the real environment step.
        """

        result = self.execute(action)

        observation = (
            f"[{action.tool}] {action.command}\n"
            f"return_code={result.return_code}\n"
            f"stdout={result.stdout}\n"
            f"stderr={result.stderr}"
        )

        self.state = replace(
            self.state,
            previous_observations=(
                *self.state.previous_observations,
                observation,
            ),
        )

        return self.state, result

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    def reset(self) -> CTFState:
        """
        Reset the logical CTF state.

        Note:
            This does NOT automatically recreate the Docker
            container. Container reset should be implemented
            separately using Docker snapshot/recreation if needed.
        """

        self.state = CTFState.initial(self.config)

        return self.state

    # --------------------------------------------------------
    # MCTS simulation
    # --------------------------------------------------------

    @staticmethod
    def simulated_transition(
        state: CTFState,
        action: CTFAction,
    ) -> tuple[CTFState, float]:
        """
        Simulate an action without touching Docker.

        This is the function used by MCTS.

        It is intentionally conservative:
        the simulator only modifies state based on the action type.

        A more sophisticated version can later be replaced with:

            LLM state transition
            learned world model
            cloned Docker containers
            snapshot/restore
        """

        services = set(state.services)
        vulnerabilities = set(state.vulnerabilities)
        hosts = set(state.hosts)
        credentials = set(state.credentials)
        objectives = set(state.completed_objectives)

        privilege_level = state.privilege_level
        compromised = state.compromised
        internal_access = state.internal_access
        detected = state.detected

        reward = 0.0

        command = action.command.lower()
        tool = action.tool.lower()

        # ----------------------------------------------------
        # Reconnaissance
        # ----------------------------------------------------

        if tool == "nmap" or "nmap" in command:
            hosts.add(state.target_ip)

            # Financial CTF expected services.
            services.update({"22/tcp", "80/tcp",})
            reward += 0.2

            if "discover_target" not in objectives:
                objectives.add("discover_target")

            if "discover_services" not in objectives:
                objectives.add("discover_services")

        # ----------------------------------------------------
        # Web enumeration
        # ----------------------------------------------------

        elif tool in {"curl", "wget"}:
            services.add("80/tcp")
            reward += 0.1

            if (
                "http" in command
                or "curl" in command
                or "wget" in command
            ):
                vulnerabilities.add("web_attack_surface")

                if "identify_vulnerability" not in objectives:
                    objectives.add("identify_vulnerability")

                reward += 0.3

        # ----------------------------------------------------
        # SQL injection / initial access
        # ----------------------------------------------------

        if (
            "sql" in command
            or "union select" in command
            or "or 1=1" in command
            or "sqli" in command
        ):
            vulnerabilities.add("sql_injection")

            compromised = True
            privilege_level = "user"

            objectives.add("identify_vulnerability")
            objectives.add("obtain_initial_access")

            reward += 1.0

        # ----------------------------------------------------
        # SSH / credential usage
        # ----------------------------------------------------

        if tool == "ssh":

            if compromised:
                credentials.add("ssh_credential")

                reward += 0.5

                if privilege_level == "none":
                    privilege_level = "user"

        # ----------------------------------------------------
        # Privilege escalation
        # ----------------------------------------------------

        if (
            "sudo" in command
            or "su " in command
            or "chmod" in command
            or "privilege" in command
            or "escalat" in command
        ):

            if compromised:

                privilege_level = "root"

                objectives.add("escalate_privilege")

                reward += 1.5

        # ----------------------------------------------------
        # Internal service access
        # ----------------------------------------------------

        if (
            "internal" in command
            or "internal-ops" in command
            or "bank-db" in command
        ):

            if compromised:

                internal_access = True

                objectives.add("access_internal_service")

                reward += 2.0

        # ----------------------------------------------------
        # Detection
        # ----------------------------------------------------

        if (
            "scan" in command
            or "nmap" in command
        ):
            # This is only a simple placeholder.
            # Real detection should come from the Blue Team.
            detected = False

        # ----------------------------------------------------
        # Terminal condition
        # ----------------------------------------------------

        done = (
            internal_access
            and privilege_level in {"root", "admin"}
        )

        # ----------------------------------------------------
        # State delta reward
        # ----------------------------------------------------

        old_progress = (
            len(state.services)
            + len(state.vulnerabilities)
            + len(state.credentials)
            + len(state.completed_objectives)
        )

        new_progress = (
            len(services)
            + len(vulnerabilities)
            + len(credentials)
            + len(objectives)
        )

        reward += max(
            0.0,
            (new_progress - old_progress) * 0.1,
        )

        if done and not state.done:
            reward += 5.0

        new_state = replace(
            state,
            hosts=tuple(sorted(hosts)),
            services=tuple(sorted(services)),
            vulnerabilities=tuple(sorted(vulnerabilities)),
            credentials=tuple(sorted(credentials)),
            compromised=compromised,
            privilege_level=privilege_level,
            internal_access=internal_access,
            detected=detected,
            completed_objectives=tuple(sorted(objectives)),
            done=done,
            reward=state.reward + reward,
            previous_observations=(
                *state.previous_observations,
                f"Simulated action: {action}",
            ),
        )

        return new_state, reward


# ============================================================
# Action Generation
# ============================================================

def default_red_actions(
    state: CTFState,
) -> set[CTFAction]:
    """
    Generate basic Red Agent actions from the current CTF state.

    This is intentionally small.

    The actual Red Agent / CAI can later replace this function
    with LLM-generated actions.
    """

    actions: set[CTFAction] = set()

    # --------------------------------------------------------
    # Recon
    # --------------------------------------------------------

    if "nmap" in state.available_tools:

        actions.add(
            CTFAction(
                tool="nmap",
                command=f"nmap -sV {state.target_ip}",
                description="Discover target services",
                actor="red",
            )
        )

    # --------------------------------------------------------
    # HTTP enumeration
    # --------------------------------------------------------

    if "80/tcp" in state.services:

        actions.add(
            CTFAction(
                tool="curl",
                command=f"curl -i http://{state.target_ip}/",
                description="Enumerate web service",
                actor="red",
            )
        )

        actions.add(
            CTFAction(
                tool="curl",
                command=f"curl -i http://{state.target_ip}/login",
                description="Inspect login endpoint",
                actor="red",
            )
        )

    # --------------------------------------------------------
    # SQL injection
    # --------------------------------------------------------

    if "web_attack_surface" in state.vulnerabilities:

        actions.add(
            CTFAction(
                tool="curl",
                command=(
                    f"curl "
                    f"'http://{state.target_ip}/login?username=admin%27%20OR%201%3D1--'"
                ),
                description="Test SQL injection",
                actor="red",
            )
        )

    # --------------------------------------------------------
    # SSH
    # --------------------------------------------------------

    if "22/tcp" in state.services:

        actions.add(
            CTFAction(
                tool="ssh",
                command=f"ssh user@{state.target_ip}",
                description="Attempt SSH access",
                actor="red",
            )
        )

    # --------------------------------------------------------
    # Privilege escalation
    # --------------------------------------------------------

    if state.compromised:

        actions.add(
            CTFAction(
                tool="shell",
                command="sudo -l",
                description="Check privilege escalation opportunities",
                actor="red",
            )
        )

        actions.add(
            CTFAction(
                tool="shell",
                command="id",
                description="Check current privilege level",
                actor="red",
            )
        )

    # --------------------------------------------------------
    # Internal access
    # --------------------------------------------------------

    if state.compromised:

        actions.add(
            CTFAction(
                tool="network",
                command="curl http://internal-ops:8080/",
                description="Attempt access to internal operations service",
                actor="red",
            )
        )

        actions.add(
            CTFAction(
                tool="network",
                command="curl http://bank-db:5432/",
                description="Probe internal database",
                actor="red",
            )
        )

    return actions


# ============================================================
# Prompt Builder
# ============================================================

def build_ctf_simulation_prompt(
    state: CTFState,
    action: CTFAction,
) -> str:
    """
    Build an LLM prompt for predicting the result of an action.

    This can be connected to Searchlight / SearchlightImprove.
    """

    return f"""
You are simulating a cybersecurity CTF environment.

Your task is to predict the state transition caused by the
following action.

CURRENT STATE
-------------
{state.get_state_description()}

ACTION
------
Tool: {action.tool}
Command: {action.command}
Description: {action.description}

Do NOT invent hidden flags.

Predict only information that can reasonably be inferred
from the current state and action.

Return:

1. observation
2. newly discovered hosts
3. newly discovered services
4. newly discovered vulnerabilities
5. newly discovered credentials
6. whether the target becomes compromised
7. privilege level
8. internal access
9. whether the action is detected
10. completed objectives
11. terminal state
12. estimated immediate reward
""".strip()


# ============================================================
# Utility
# ============================================================

def state_from_observation(
    previous_state: CTFState,
    observation: str,
) -> CTFState:
    """
    Basic observation -> CTFState converter.

    This is deliberately simple.

    Later this can be replaced by an LLM parser or
    structured parser for Nmap/Curl/SSH output.
    """

    text = observation.lower()

    services = set(previous_state.services)
    vulnerabilities = set(previous_state.vulnerabilities)

    if "22/tcp" in text or "ssh" in text:
        services.add("22/tcp")

    if "80/tcp" in text or "http" in text:
        services.add("80/tcp")

    if "sql" in text:
        vulnerabilities.add("sql_injection")

    return replace(
        previous_state,
        services=tuple(sorted(services)),
        vulnerabilities=tuple(sorted(vulnerabilities)),
        previous_observations=(
            *previous_state.previous_observations,
            observation,
        ),
    )