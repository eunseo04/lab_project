from dataclasses import dataclass, field
from typing import Callable, Optional, Union
from searchlight.datastructures.adjusters import ForwardTransitor, ActorActionEnumerator
from searchlight.datastructures.estimators import ValueHeuristic


# ============================================================
# 1. CTF State
# ============================================================

@dataclass(frozen=True)
class CTFState:
    """
    실제 Docker CTF Target의 현재 상태를 표현한다.

    Avalon/GOPS처럼 고정된 게임 규칙이나 턴을 저장하지 않는다.
    """

    # --------------------------------------------------------
    # 환경
    # --------------------------------------------------------

    attacker_ip: str
    target_ip: str
    target_os: str

    # --------------------------------------------------------
    # 현재 Target에서 관찰된 정보
    # --------------------------------------------------------

    discovered_services: tuple[str, ...] = ()
    discovered_vulnerabilities: tuple[str, ...] = ()
    discovered_hosts: tuple[str, ...] = ()

    # --------------------------------------------------------
    # 공격 결과 / 접근 상태
    # --------------------------------------------------------

    compromised: bool = False
    privilege_level: str = "none"

    # 예:
    # none
    # user
    # root

    credentials_obtained: tuple[str, ...] = ()

    internal_access: bool = False

    # --------------------------------------------------------
    # 이전 action의 관찰 결과
    # --------------------------------------------------------

    previous_observations: tuple[str, ...] = ()

    # --------------------------------------------------------
    # 현재 목표
    # --------------------------------------------------------

    current_objective: str = ""

    completed_objectives: tuple[str, ...] = ()

    # --------------------------------------------------------
    # 사용 가능한 Tool
    # --------------------------------------------------------

    available_tools: tuple[str, ...] = ()

    # --------------------------------------------------------
    # 실제 환경에서 얻은 reward
    # --------------------------------------------------------

    reward: float = 0.0

    # --------------------------------------------------------
    # 종료 여부
    # --------------------------------------------------------

    done: bool = False

    # ========================================================
    # State utility
    # ========================================================

    def is_done(self) -> bool:
        return self.done

    def get_reward(self) -> float:
        return self.reward

    def get_available_tools(self) -> tuple[str, ...]:
        return self.available_tools

    def get_discovered_services(self) -> tuple[str, ...]:
        return self.discovered_services

    def get_discovered_vulnerabilities(self) -> tuple[str, ...]:
        return self.discovered_vulnerabilities

    def get_previous_observations(self) -> tuple[str, ...]:
        return self.previous_observations

    # ========================================================
    # LLM prompt용 상태 설명
    # ========================================================

    def get_state_description(self) -> str:

        return f"""
Current financial cybersecurity CTF environment:

[Network]
- Attacker IP: {self.attacker_ip}
- Target IP: {self.target_ip}
- Target OS: {self.target_os}

[Discovered Hosts]
{list(self.discovered_hosts)}

[Discovered Services]
{list(self.discovered_services)}

[Discovered Vulnerabilities]
{list(self.discovered_vulnerabilities)}

[Compromise Status]
- Compromised: {self.compromised}
- Privilege level: {self.privilege_level}
- Credentials obtained: {list(self.credentials_obtained)}
- Internal network access: {self.internal_access}

[Previous Observations]
{list(self.previous_observations)}

[Current Objective]
{self.current_objective}

[Completed Objectives]
{list(self.completed_objectives)}

[Available Tools]
{list(self.available_tools)}

[Current Reward]
{self.reward}

[Terminal]
{self.done}
"""

    # ========================================================
    # 초기 상태
    # ========================================================

    @staticmethod
    def init_from_environment(
        attacker_ip: str,
        target_ip: str,
        target_os: str,
        available_tools: list[str],
    ) -> "CTFState":

        return CTFState(
            attacker_ip=attacker_ip,
            target_ip=target_ip,
            target_os=target_os,
            available_tools=tuple(available_tools),
            current_objective="Discover and compromise the target environment",
        )


# ============================================================
# 2. CTF Action
# ============================================================

@dataclass(frozen=True)
class CTFAction:
    """
    MCTS에서 사용하는 하나의 공격 행동.

    Avalon의 카드 번호 대신 실제 Red Agent가 실행할
    tool/action을 표현한다.
    """

    tool: str
    command: str
    description: str = ""

    def __str__(self):
        return self.command


# ============================================================
# 3. State Transition
# ============================================================

class CTFForwardTransitor(ForwardTransitor):
    """
    S + Action -> S'

    주의:
    실제 Docker Target을 직접 여기서 공격하지 않는다.

    MCTS의 각 branch가 독립적으로 탐색될 수 있도록
    simulator가 가상 다음 상태를 반환하도록 한다.
    """

    def __init__(
        self,
        simulator: Optional[
            Callable[[CTFState, CTFAction], tuple[CTFState, float]]
        ] = None,
    ):
        super().__init__()
        self.simulator = simulator

    def _transition(
        self,
        state: CTFState,
        action: CTFAction,
    ) -> tuple[CTFState, dict[int, float]]:

        if state.is_done():
            raise ValueError(
                "Terminal state. No action should be taken."
            )

        if self.simulator is None:
            raise RuntimeError(
                "CTFForwardTransitor requires a simulator."
            )

        # ----------------------------------------------------
        # 가상 환경에서 Action 실행
        # ----------------------------------------------------

        next_state, reward = self.simulator(
            state,
            action,
        )

        # ----------------------------------------------------
        # Strategist의 기존 MCTS 인터페이스에 맞춰
        # actor 0 = Red Agent
        # actor 1 = Blue Agent
        # 로 표현
        # ----------------------------------------------------

        rewards = {
            0: reward,
            1: -reward,
        }

        return next_state, rewards


# ============================================================
# 4. Action Enumerator
# ============================================================

class CTFActorActionEnumerator(ActorActionEnumerator):
    
    def _enumerate(self, state):
        actor = state.current_actor

        if actor is None:
            return None, set()
        
        if actor == 0:
            return 0, self.get_red_actions(state)

        if actor == 1:
            return 1, self.get_blue_actions(state)

        if actor == -1:
            return -1, self.get_environment_actions(state)

        raise ValueError(
            f"Unknown actor: {actor}"
        )

    def get_actions(self, state, actor):
        if actor == 0:
            return self.get_red_actions(state)

        if actor == 1:
            return self.get_blue_actions(state)

        if actor == -1:
            return self.get_environment_actions(state)
        return set()

    @staticmethod
    def parse_str_to_action(action_str: str) -> CTFAction:
        """
        LLM이 반환한 문자열을 CTFAction으로 변환한다.
        """

        return CTFAction(
            tool="shell",
            command=action_str,
        )


# ============================================================
# 5. CTF Value Heuristic
# ============================================================

class CTFValueHeuristic(ValueHeuristic):
    """
    현재 Cybersecurity CTF 상태가 얼마나 유리한지 평가한다.

    GOPS의 카드 점수 계산을 완전히 제거한다.

    평가 기준은 고정된 게임 규칙이 아니라
    현재 Target에서 얻은 실제 상태를 기반으로 한다.
    """

    def __init__(
        self,
        evaluate_function: Optional[
            Callable[[CTFState], float]
        ] = None,
    ):
        self.evaluate_function = evaluate_function

    def _evaluate(
        self,
        state: CTFState,
    ) -> dict:

        if self.evaluate_function is not None:
            value = self.evaluate_function(state)

        else:
            value = self.default_evaluate(state)

        return {
            0: value,
            1: -value,
        }

    @staticmethod
    def default_evaluate(
        state: CTFState,
    ) -> float:

        value = 0.0

        # ----------------------------------------------------
        # 정찰
        # ----------------------------------------------------

        value += len(state.discovered_hosts) * 0.1

        value += len(state.discovered_services) * 0.2

        value += len(state.discovered_vulnerabilities) * 0.5

        # ----------------------------------------------------
        # 공격 성공 정도
        # ----------------------------------------------------

        if state.compromised:
            value += 2.0

        if state.privilege_level == "user":
            value += 1.0

        elif state.privilege_level == "root":
            value += 3.0

        # ----------------------------------------------------
        # credential 획득
        # ----------------------------------------------------

        value += len(state.credentials_obtained) * 0.5

        # ----------------------------------------------------
        # 내부망 접근
        # ----------------------------------------------------

        if state.internal_access:
            value += 2.0

        # ----------------------------------------------------
        # 완료된 목표
        # ----------------------------------------------------

        value += len(state.completed_objectives) * 1.0

        # ----------------------------------------------------
        # 실제 환경에서 계산된 reward
        # ----------------------------------------------------

        value += state.reward

        return value


# ============================================================
# 6. CTF State -> Prompt
# ============================================================

def build_ctf_simulation_prompt(
    state: CTFState,
    action: CTFAction,
) -> str:
    """
    MCTS의 virtual simulation을 위한 LLM prompt.

    Avalon의 게임 규칙 대신
    '이 Action을 수행하면 Target 상태가 어떻게 변하는가?'
    를 예측한다.
    """

    return f"""
You are simulating a financial cybersecurity CTF environment.

Current environment:

{state.get_state_description()}

Candidate action:

Tool:
{action.tool}

Command:
{action.command}

Description:
{action.description}

Predict the result of executing this action.

Return:

1. Observation
2. Newly discovered services
3. Newly discovered vulnerabilities
4. Newly discovered hosts
5. Whether the target became compromised
6. Current privilege level
7. Newly obtained credentials
8. Whether internal network access was obtained
9. Completed objectives
10. Whether the environment is terminal
11. Reward

Do not invent information that cannot reasonably result from
the current environment state and action.
"""


# ============================================================
# 7. Docker Target 관찰 결과 -> CTFState
# ============================================================

def state_from_docker_observation(
    *,
    attacker_ip: str,
    target_ip: str,
    target_os: str,
    observation: dict,
    previous_state: Optional[CTFState] = None,
) -> CTFState:
    """
    실제 Docker Target에서 얻은 observation을
    MCTS가 사용할 CTFState로 변환한다.

    observation 예:

    {
        "services": ["22/tcp", "80/tcp"],
        "vulnerabilities": ["SQL injection"],
        "hosts": ["192.168.20.10"],
        "compromised": True,
        "privilege_level": "user",
        "credentials": ["db_user"],
        "internal_access": True,
        "observation": "SSH access obtained",
        "reward": 2.5,
        "done": False
    }
    """

    previous_state = previous_state or CTFState.init_from_environment(
        attacker_ip=attacker_ip,
        target_ip=target_ip,
        target_os=target_os,
        available_tools=[],
    )

    observations = list(previous_state.previous_observations)

    if observation.get("observation"):
        observations.append(
            observation["observation"]
        )

    return CTFState(
        attacker_ip=attacker_ip,
        target_ip=target_ip,
        target_os=target_os,

        discovered_services=tuple(
            observation.get(
                "services",
                previous_state.discovered_services,
            )
        ),

        discovered_vulnerabilities=tuple(
            observation.get(
                "vulnerabilities",
                previous_state.discovered_vulnerabilities,
            )
        ),

        discovered_hosts=tuple(
            observation.get(
                "hosts",
                previous_state.discovered_hosts,
            )
        ),

        compromised=observation.get(
            "compromised",
            previous_state.compromised,
        ),

        privilege_level=observation.get(
            "privilege_level",
            previous_state.privilege_level,
        ),

        credentials_obtained=tuple(
            observation.get(
                "credentials",
                previous_state.credentials_obtained,
            )
        ),

        internal_access=observation.get(
            "internal_access",
            previous_state.internal_access,
        ),

        previous_observations=tuple(
            observations[-20:]
        ),

        current_objective=observation.get(
            "current_objective",
            previous_state.current_objective,
        ),

        completed_objectives=tuple(
            observation.get(
                "completed_objectives",
                previous_state.completed_objectives,
            )
        ),

        available_tools=tuple(
            observation.get(
                "available_tools",
                previous_state.available_tools,
            )
        ),

        reward=observation.get(
            "reward",
            previous_state.reward,
        ),

        done=observation.get(
            "done",
            False,
        ),
    )