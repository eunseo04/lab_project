from typing import Any, Tuple, Optional
import numpy as np

from .utils import AbstractLogged
from abc import ABC, abstractmethod
import logging
from collections.abc import Hashable


"""
Abstract interfaces for CTF search algorithms.

Actor convention
----------------
-1 : CTF Environment
 0 : Red Team / Attacker
 1 : Blue Team / Defender

The search algorithm itself does not contain CTF-specific rules.
CTF-specific information is provided through:

    State
    InformationFunction
    InformationPrior
    ForwardTransitor
    ActorActionEnumerator
    PolicyPredictor
    ValueHeuristic
"""


# ============================================================
# CTF State
# ============================================================

class StateTemplate(ABC):
    """
    Immutable-like representation of a CTF environment state.

    The state ID is used as the hash key.

    notes can contain CTF environment information such as:

        {
            "hosts": ...,
            "services": ...,
            "vulnerabilities": ...,
            "credentials": ...,
            "network": ...,
            "flags": ...,
            "detected": ...,
        }
    """

    notes: dict

    def __init__(self, id, notes: Optional[dict] = None):
        """
        Args:
            id:
                Unique identifier of the CTF state.

            notes:
                CTF environment information associated with the state.
        """

        self.id = id

        if notes is None:
            notes = {}

        self.notes = notes

        # State ID must be hashable.
        hash(self.id)

    def copy(self):
        """
        Return a copy of the state.
        """
        return StateTemplate(self.id, self.notes.copy())

    def __repr__(self):
        return f"CTFState({self.id}, {self.notes})"

    def __str__(self):
        return f"CTFState({self.id}, {self.notes})"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, StateTemplate):
            return False

        return self.id == other.id


# ============================================================
# Information
# ============================================================

class InformationFunction(AbstractLogged):
    """
    Maps the complete CTF state to the information available
    to a specific actor.

    This is useful for partial-observability CTF environments.

    Example:

        Red may know:
            - discovered hosts
            - discovered services
            - obtained credentials

        Blue may know:
            - alerts
            - logs
            - compromised hosts
    """

    def get_information_set(
        self,
        state: Hashable,
        actor: int
    ) -> Hashable:

        return self._get_information_set(
            state=state,
            actor=actor
        )

    @abstractmethod
    def _get_information_set(
        self,
        state: Hashable,
        actor: int
    ) -> Hashable:

        pass


class InformationPrior(AbstractLogged):
    """
    Generates a possible hidden CTF state from an information set.

    This can be used when an agent does not know the complete
    environment state.
    """

    def get_prior_state(
        self,
        information_set: Hashable
    ) -> Hashable:

        return self._get_prior_state(
            information_set=information_set
        )

    @abstractmethod
    def _get_prior_state(
        self,
        information_set: Hashable
    ) -> Hashable:

        pass


# ============================================================
# Environment Transition
# ============================================================

class ForwardTransitor(ABC):
    """
    Defines how the CTF environment changes after an action.

    Example:

        Red action:
            exploit(host, vulnerability)

        Environment:
            host becomes compromised

        Blue action:
            isolate(host)

        Environment:
            host becomes isolated

    Returns:
        next_state
        rewards
    """

    def transition(
        self,
        state: Hashable,
        action: Hashable,
        actor: int
    ) -> Tuple[Hashable, dict[int, float]]:

        next_state, rewards = self._transition(
            state,
            action,
            actor
        )

        return next_state, rewards

    @abstractmethod
    def _transition(
        self,
        state: Hashable,
        action: Hashable,
        actor: int
    ) -> Tuple[Hashable, dict[int, float]]:

        pass


# ============================================================
# Actor / Action Enumeration
# ============================================================

class ActorActionEnumerator(ABC):
    """
    Determines:

        1. Which actor acts at the current state
        2. Which actions are available

    Actor convention:

        -1 : Environment
         0 : Red Team / Attacker
         1 : Blue Team / Defender

    Example:

        state = attacker_turn

        actor = 0

        actions = {
            Scan(host1),
            Scan(host2),
            Exploit(host1, vuln1),
            Exploit(host2, vuln2)
        }
    """

    def enumerate(
        self,
        state: Hashable
    ) -> tuple[Optional[int], set]:

        return self._enumerate(state)

    @abstractmethod
    def _enumerate(
        self,
        state: Hashable
    ) -> tuple[Optional[int], set]:

        pass


# ============================================================
# Policy Predictor
# ============================================================

class PolicyPredictor(ABC):
    """
    Predicts the probability of each available CTF action.

    The predictor can later be replaced with:

        - Random policy
        - LLM policy
        - RL policy
        - Red Team agent
        - Blue Team agent
    """

    def predict(
        self,
        state: Hashable,
        actions: set[Hashable],
        actor: Optional[int] = None
    ) -> dict:

        return self._predict(
            state,
            actions,
            actor
        )

    def _predict(
        self,
        state: Hashable,
        actions: set[Hashable],
        actor: Optional[int] = None
    ) -> dict:

        # Default: uniform policy.

        if not actions:
            return {}

        probability = 1 / len(actions)

        return {
            action: probability
            for action in actions
        }


# ============================================================
# Value Heuristic
# ============================================================

class ValueHeuristic(ABC):
    """
    Evaluates the current CTF state.

    Example:

        Red:
            +1.0  flag captured
            +0.5  host compromised
            -0.5  attacker detected

        Blue:
            +1.0  attack prevented
            +0.5  host defended
            -1.0  flag compromised

    Returns:

        {
            actor_id: value
        }

        notes
    """

    def evaluate(
        self,
        state: Hashable
    ) -> tuple[dict[int, float], dict]:

        return self._evaluate(state)

    @abstractmethod
    def _evaluate(
        self,
        state: Hashable
    ) -> tuple[dict[int, float], dict]:

        pass


# ============================================================
# Search
# ============================================================

class Search(ABC):
    """
    Abstract class for CTF search algorithms.

    The search algorithm itself does not store the CTF graph.

    All search information should be stored in the datastructure.

    This allows the same search algorithm to be used with:

        Red Team
        Blue Team
        Different CTF environments
    """

    def __init__(self):

        self.total_nodes_expanded = 0
        self.nodes_expanded = 0

        self.logger = logging.getLogger(
            self.__class__.__name__
        )

    def expand(
        self,
        datastructure,
        state: Hashable
    ):

        return self._expand(
            datastructure,
            state
        )

    @abstractmethod
    def _expand(
        self,
        datastructure,
        state: Hashable
    ):

        pass

    def increment_nodes_expanded(self):

        self.nodes_expanded += 1
        self.total_nodes_expanded += 1

    def get_total_nodes_expanded(self):

        return self.total_nodes_expanded

    def reset_total_nodes_expanded(self):

        self.total_nodes_expanded = 0

    def get_nodes_expanded(self):

        return self.nodes_expanded

    def reset(self):

        self.reset_total_nodes_expanded()
        self.nodes_expanded = 0

    def get_best_action(
        self,
        graph,
        state: Hashable,
        actor=0
    ):
        """
        Return the best action for the given actor.

        actor:
            0 -> Red Team
            1 -> Blue Team
        """

        raise NotImplementedError


# ============================================================
# CTF Agent
# ============================================================

class Agent(AbstractLogged):
    """
    Base class for a CTF agent.

    player:

        0 -> Red Team
        1 -> Blue Team
    """

    def __init__(self, player=0):

        self.player = player

        super().__init__()

    def set_player(self, player):

        self.player = player

    def act(
        self,
        state: Hashable,
        actions: set[Hashable]
    ) -> Hashable:

        action = self._act(
            state,
            actions
        )

        if action not in actions:

            raise ValueError(
                f"Action {action} not in actions {actions}"
            )

        return action

    @abstractmethod
    def _act(
        self,
        state: Hashable,
        actions: set[Hashable]
    ) -> Hashable:

        raise NotImplementedError


# ============================================================
# Random CTF Agent
# ============================================================

class RandomAgent(Agent):
    """
    Random CTF agent.

    Useful as a baseline for:

        Red Team
        Blue Team
        Search algorithm testing
    """

    def __init__(
        self,
        player=0,
        rng=None
    ):

        super().__init__(
            player=player
        )

        if rng is None:
            rng = np.random.default_rng()

        self.rng = rng

    def _act(
        self,
        state: Hashable,
        actions: set[Hashable]
    ):

        if not actions:

            raise ValueError(
                "No available actions."
            )

        return self.rng.choice(
            list(actions)
        )


class InitialInferencer2:
    def __init__(self):
        pass

    def predict(self, state):
        raise NotImplementedError


class PackageInitialInferencer(InitialInferencer2):

    def __init__(
        self,
        transitor,
        action_enumerator,
        action_predictor,
        value_heuristic,
    ):
        super().__init__()

        self.transitor = transitor
        self.action_enumerator = action_enumerator
        self.action_predictor = action_predictor
        self.value_heuristic = value_heuristic

    def predict(self, state):

        # 1. 현재 actor와 가능한 action 가져오기
        actor, actions = self.action_enumerator.enumerate(state)

        if actor is None or not actions:
            return {}, {}, {}, {}, {}

        # 2. action별 policy 확률
        policy = self.action_predictor.predict(
            state,
            actions,
            actor
        )

        policies = {
            actor: policy
        }

        next_values = {}
        intermediate_rewards = {}
        transitions = {}
        notes = {}

        # 3. 가능한 모든 action에 대해 다음 상태 계산
        for action in actions:

            joint_action = (
                (actor, action),
            )

            # 다음 상태
            next_state = self.transitor.transition(
                state,
                action,
                actor
            )

            transitions[joint_action] = next_state

            # 즉시 보상
            intermediate_rewards[joint_action] = {
                actor: 0.0
            }

            # 다음 상태 가치 평가
            values, heuristic_notes = \
                self.value_heuristic.evaluate(next_state)

            next_values[next_state] = values

            # heuristic 관련 메모
            if heuristic_notes:
                notes.setdefault(
                    "next_state_to_heuristic_notes",
                    {}
                )

                notes[
                    "next_state_to_heuristic_notes"
                ][next_state] = heuristic_notes

        return (
            policies,
            next_values,
            intermediate_rewards,
            transitions,
            notes
        )