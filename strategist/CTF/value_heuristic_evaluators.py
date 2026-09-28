from __future__ import annotations
from typing import Callable, Optional
import random
from engine import (
    CTFState,
    CTFAction,
    CTFEnvironment,
    default_red_actions,
)

# ============================================================
# CTF Value Heuristic
# ============================================================

class CTFValueHeuristic:
    """
    Evaluate how favorable the current CTF state is.

    This replaces the GOPS-specific value heuristic.

    The value is based on attack progress rather than
    card-game score.
    """

    def __init__(
        self,
        custom_evaluator: Optional[Callable[[CTFState], float]] = None,):
        self.custom_evaluator = custom_evaluator

    def evaluate(self, state: CTFState) -> float:
        if self.custom_evaluator is not None:
            return float(
                self.custom_evaluator(state)
            )

        value = 0.0

        # ----------------------------------------------------
        # Reconnaissance
        # ----------------------------------------------------

        value += len(state.hosts) * 0.1
        value += len(state.services) * 0.2

        # ----------------------------------------------------
        # Vulnerability discovery
        # ----------------------------------------------------

        value += (len(state.vulnerabilities)* 0.5)

        # ----------------------------------------------------
        # Credentials
        # ----------------------------------------------------

        value += (len(state.credentials)* 0.5)

        # ----------------------------------------------------
        # Initial compromise
        # ----------------------------------------------------

        if state.compromised:
            value += 1.0

        # ----------------------------------------------------
        # Privilege
        # ----------------------------------------------------

        if state.privilege_level == "user":
            value += 1.0

        elif state.privilege_level in {"root","admin",}:
            value += 2.5

        # ----------------------------------------------------
        # Internal network access
        # ----------------------------------------------------

        if state.internal_access:
            value += 3.0

        # ----------------------------------------------------
        # Detection penalty
        # ----------------------------------------------------

        if state.detected:
            value -= 1.0

        # ----------------------------------------------------
        # Objective progress
        # ----------------------------------------------------

        value += (
            len(state.completed_objectives)
            * 0.5
        )

        # ----------------------------------------------------
        # Environment reward
        # ----------------------------------------------------

        value += state.reward

        # ----------------------------------------------------
        # Terminal bonus
        # ----------------------------------------------------

        if state.done:
            value += 5.0
        return value


# ============================================================
# Blue Team Value Heuristic
# ============================================================

class BlueCTFValueHeuristic:
    """
    Evaluate a state from the Blue Team perspective.

    Higher value means stronger defensive position.
    """

    def evaluate(self, state: CTFState) -> float:
        value = 0.0

        # No compromise is good for Blue.
        if not state.compromised:
            value += 2.0
        else:
            value -= 2.0

        # Detection gives Blue information.
        if state.detected:
            value += 2.0

        # Defensive controls.
        value += (
            len(state.defensive_controls)
            * 0.5
        )

        # Prevent internal access.
        if not state.internal_access:
            value += 2.0
        else:
            value -= 3.0

        # Prevent privilege escalation.
        if state.privilege_level in {"none","user",}:
            value += 1.0

        elif state.privilege_level in {"root","admin",}:
            value -= 2.0
        return value


# ============================================================
# State Delta Evaluator
# ============================================================

class CTFStateDeltaEvaluator:
    """
    Calculate progress between two CTF states.

    This is useful for reward shaping.
    """

    def __init__(self, red_heuristic: Optional[CTFValueHeuristic] = None,):
        self.red_heuristic = (red_heuristic or CTFValueHeuristic())

    def evaluate(self, previous_state: CTFState, next_state: CTFState,) -> float:
        previous_value = (self.red_heuristic.evaluate(previous_state))
        next_value = (self.red_heuristic.evaluate(next_state))
        return (next_value - previous_value)


# ============================================================
# CTF Rollout
# ============================================================

class CTFRolloutValueHeuristic:
    """
    MCTS-style rollout evaluator.

    IMPORTANT:

    This does NOT execute commands against Docker.

    It uses CTFEnvironment.simulated_transition()
    to explore hypothetical states.
    """

    def __init__(
        self,
        value_heuristic: Optional[
            CTFValueHeuristic
        ] = None,
        rollout_depth: int = 5,
        random_state: Optional[
            random.Random
        ] = None,
    ):

        self.value_heuristic = (value_heuristic or CTFValueHeuristic())
        self.rollout_depth = rollout_depth
        self.random_state = (random_state or random.Random(12))

    # --------------------------------------------------------
    # Generate actions
    # --------------------------------------------------------

    def get_actions(
        self,
        state: CTFState,
    ) -> list[CTFAction]:

        return list(
            default_red_actions(state)
        )

    # --------------------------------------------------------
    # Single rollout
    # --------------------------------------------------------

    def rollout(
        self,
        state: CTFState,
    ) -> float:

        current_state = state

        for _ in range(self.rollout_depth):
            if current_state.done:
                break

            actions = self.get_actions(current_state)

            if not actions:
                break

            action = (self.random_state.choice(actions))
            (current_state, _,) = CTFEnvironment.simulated_transition(current_state,action,)

        return self.value_heuristic.evaluate(
            current_state
        )

    # --------------------------------------------------------
    # Multiple rollouts
    # --------------------------------------------------------

    def evaluate(
        self,
        state: CTFState,
        num_rollouts: int = 10,
    ) -> float:

        if num_rollouts <= 0:
            return self.value_heuristic.evaluate(state)

        total = 0.0
        for _ in range(num_rollouts):
            total += self.rollout(state)
            
        return total / num_rollouts


# ============================================================
# Action Evaluation
# ============================================================

class CTFActionEvaluator:
    """
    Evaluate candidate actions from the current state.

    This is useful for MCTS action selection or for
    comparing candidate Red Agent actions.
    """

    def __init__(self, rollout_heuristic: Optional[CTFRolloutValueHeuristic] = None,):
        self.rollout_heuristic = (rollout_heuristic or CTFRolloutValueHeuristic())

    def evaluate_action(self, state: CTFState, action: CTFAction,) -> float:
        (next_state,immediate_reward,) = CTFEnvironment.simulated_transition(state,action,)
        future_value = (self.rollout_heuristic.evaluate(next_state))

        return (immediate_reward + future_value)

    def rank_actions(self,state: CTFState,actions: Optional[list[CTFAction]] = None,) -> list[tuple[CTFAction, float]]:
        if actions is None:
            actions = list(default_red_actions(state))

        scored_actions = []

        for action in actions:
            score = (self.evaluate_action(state,action,))
            scored_actions.append((action,score,))

        scored_actions.sort(
            key=lambda item: item[1],
            reverse=True,
        )
        return scored_actions


# ============================================================
# Function Evaluator
# ============================================================

class CTFValueFunctionEvaluator:
    """
    Compare a custom value function against
    the built-in CTF heuristic.

    This replaces the old GOPS-specific
    RandomRolloutEvaluator.

    A function must have the form:

        function(state) -> float
    """

    def __init__(self,sample_size: int = 10,random_seed: int = 12,):
        self.sample_size = sample_size
        self.random_state = random.Random(random_seed)
        self.reference_heuristic = (CTFValueHeuristic())

    # --------------------------------------------------------
    # Generate sample states
    # --------------------------------------------------------

    def sample_states(self,initial_state: CTFState,) -> list[CTFState]:
        states = [initial_state]
        current_state = initial_state

        for _ in range(max(self.sample_size * 3,10,)):
            actions = list(default_red_actions(current_state))

            if not actions:
                break

            action = (self.random_state.choice(actions))
            (current_state,_,) = CTFEnvironment.simulated_transition(current_state,action,)
            states.append(current_state)

            if current_state.done:
                current_state = initial_state
        return states

    # --------------------------------------------------------
    # Evaluate function
    # --------------------------------------------------------

    def evaluate_function(
        self,
        function: Callable[[CTFState], float],
        initial_state: Optional[CTFState] = None,) -> float:

        if initial_state is None:
            initial_state = CTFState()

        states = self.sample_states(initial_state)

        if not states:
            return 0.0

        sample_count = min(self.sample_size,len(states),)
        sampled_states = (self.random_state.sample(states,sample_count,))

        error = 0.0
        for state in sampled_states:
            reference_value = (self.reference_heuristic.evaluate(state))
            
            try:
                predicted_value = float(function(state))

            except Exception:
                return 0.0

            error += abs(reference_value - predicted_value)
        mean_error = (error / len(sampled_states))

        # Higher score = closer to reference.
        return -mean_error

    # --------------------------------------------------------
    # Evaluate multiple functions
    # --------------------------------------------------------

    def evaluate(
        self,
        functions: list[Callable[[CTFState], float]],
        initial_state: Optional[CTFState] = None,) -> list[float]:

        scores = []
        for function in functions:
            score = (
                self.evaluate_function(
                    function,
                    initial_state,
                )
            )
            scores.append(score)
        return scores