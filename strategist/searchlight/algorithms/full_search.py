from strategist.searchlight.headers import *
from strategist.searchlight.datastructures.beliefs import *
from strategist.searchlight.datastructures.graphs import ValueGraph2

import itertools
from strategist.searchlight.utils import *
from datetime import datetime


# ============================================================
# CTF Actor Convention
# ============================================================
#
# -1 : Environment
#  0 : Red Team / Attacker
#  1 : Blue Team / Defender
#
# CTF search assumes a two-player zero-sum game:
#
#     Red  -> maximize value
#     Blue -> minimize Red's value
#
# ============================================================


class FullSearch(Search):
    """
    Base class for full-tree search algorithms for CTF.

    The search algorithm does not contain CTF-specific rules.

    CTF-specific behavior is provided through:

        ForwardTransitor
        ValueHeuristic
        ActorActionEnumerator
        PolicyPredictor

    This allows the same search algorithm to work with
    different CTF environments.
    """

    def __init__(
        self,
        forward_transitor: ForwardTransitor,
        value_heuristic: ValueHeuristic,
        actor_action_enumerator: ActorActionEnumerator,
        action_predictor: PolicyPredictor,
    ):

        self.forward_transitor = forward_transitor
        self.value_heuristic = value_heuristic
        self.actor_action_enumerator = actor_action_enumerator
        self.action_predictor = action_predictor

        super().__init__()


class SMMinimax(FullSearch):
    """
    Simultaneous Minimax search for a CTF environment.

    Actor convention:

        0 -> Red Team / Attacker
        1 -> Blue Team / Defender

    Red tries to maximize the value.
    Blue tries to minimize the value.

    The search assumes a zero-sum game.

    Example:

        Red action:
            exploit(host, vulnerability)

        Blue action:
            isolate(host)

        The search evaluates every possible combination:

            exploit + isolate
            exploit + monitor
            scan + isolate
            scan + monitor
            ...

        and selects the Red action with the best
        worst-case Blue response.
    """

    def __init__(
        self,
        forward_transitor: ForwardTransitor,
        value_heuristic: ValueHeuristic,
        actor_action_enumerator: ActorActionEnumerator,
        action_predictor: PolicyPredictor,
        depth=3,
        render=False,
        node_budget=None,
    ):

        super().__init__(
            forward_transitor=forward_transitor,
            value_heuristic=value_heuristic,
            actor_action_enumerator=actor_action_enumerator,
            action_predictor=action_predictor,
        )

        self.depth = depth
        self.render = render
        self.node_budget = node_budget

    # ========================================================
    # Public API
    # ========================================================

    def get_best_action(
        self,
        graph: ValueGraph2,
        state: Hashable,
        actor=0
    ):
        """
        Return the best action found for an actor.

        actor:
            0 -> Red Team
            1 -> Blue Team
        """

        node = graph.get_node(state)

        if node is None:
            raise ValueError(
                "Node not found in graph"
            )

        if actor not in node.actor_to_best_action:
            raise ValueError(
                f"Actor {actor} not found in node"
            )

        return node.actor_to_best_action[actor]

    # ========================================================
    # Expand
    # ========================================================

    def expand(
        self,
        graph: ValueGraph2,
        state: Hashable,
    ) -> None:
        """
        Expand the CTF search tree starting from state.
        """

        self.nodes_expanded = 0

        self._expand(
            graph,
            state,
            prev_node=None,
            depth=self.depth,
        )

    # ========================================================
    # Recursive Search
    # ========================================================

    def _expand(
        self,
        graph: ValueGraph2,
        state: Hashable,
        prev_node=None,
        depth=3,
    ):
        """
        Recursively expand the CTF search tree.

        Returns:
            Red Team value of the state.
        """

        # ----------------------------------------------------
        # Get or create graph node
        # ----------------------------------------------------

        node = graph.get_node(state)

        if node is None:
            node = graph.add_state(state)

        # Connect parent-child relationship.

        if prev_node is not None:
            node.parents.add(prev_node)
            prev_node.children.add(node)

        # ----------------------------------------------------
        # Terminal state
        # ----------------------------------------------------

        if self._is_terminal(state):

            value = self._get_terminal_value(
                state
            )

        # ----------------------------------------------------
        # Depth limit
        # ----------------------------------------------------

        elif depth == 0:

            value = self._get_heuristic_value(
                state
            )

        # ----------------------------------------------------
        # Node budget
        # ----------------------------------------------------

        elif (
            self.node_budget is not None
            and self.nodes_expanded >= self.node_budget
        ):

            value = self._get_heuristic_value(
                state
            )

        # ----------------------------------------------------
        # Expand node
        # ----------------------------------------------------

        else:

            self.increment_nodes_expanded()

            next_state_to_value = {}

            # ------------------------------------------------
            # Determine acting actor
            # ------------------------------------------------

            actor, actions = (
                self.actor_action_enumerator.enumerate(
                    state
                )
            )

            node.actors = (
                set()
                if actor is None
                else {actor}
            )

            # Terminal state returned by enumerator.

            if actor is None:

                value = self._get_terminal_value(
                    state
                )

            # ------------------------------------------------
            # Environment action
            # ------------------------------------------------

            elif actor == -1:

                value = self._expand_environment(
                    graph,
                    state,
                    node,
                    actions,
                    next_state_to_value,
                    depth,
                )

            # ------------------------------------------------
            # Red / Blue simultaneous actions
            # ------------------------------------------------

            elif actor in (0, 1):

                value = self._expand_players(
                    graph,
                    state,
                    node,
                    actor,
                    actions,
                    next_state_to_value,
                    depth,
                )

            else:

                raise NotImplementedError(
                    f"Unsupported CTF actor: {actor}"
                )

        # ----------------------------------------------------
        # Store value
        # ----------------------------------------------------

        self._store_value(
            node,
            value
        )

        return value

    # ========================================================
    # Environment Expansion
    # ========================================================

    def _expand_environment(
        self,
        graph,
        state,
        node,
        actions,
        next_state_to_value,
        depth,
    ):
        """
        Expand environment actions.

        Environment actions are stochastic.

        Example:

            network event
            detection event
            random vulnerability event
            random defender alert
        """

        if not actions:

            return self._get_heuristic_value(
                state
            )

        node.actions = {
            ((-1, action),)
            for action in actions
        }

        # ----------------------------------------------------
        # Predict environment action probabilities
        # ----------------------------------------------------

        probs = self.action_predictor.predict(
            state,
            actions,
            -1
        )

        # ----------------------------------------------------
        # Generate next states
        # ----------------------------------------------------

        for action in actions:

            next_state, rewards = (
                self.forward_transitor.transition(
                    state,
                    action,
                    -1
                )
            )

            joint_action = (
                (-1, action),
            )

            node.action_to_next_state[
                joint_action
            ] = next_state

            if next_state not in next_state_to_value:

                next_state_to_value[
                    next_state
                ] = self._expand(
                    graph,
                    next_state,
                    node,
                    depth,
                )

        # ----------------------------------------------------
        # Expected value
        # ----------------------------------------------------

        value = 0.0

        for action in actions:

            joint_action = (
                (-1, action),
            )

            next_state = (
                node.action_to_next_state[
                    joint_action
                ]
            )

            probability = probs.get(
                action,
                0.0
            )

            value += (
                probability
                * next_state_to_value[next_state]
            )

        node.actor_to_action_to_prob = {
            -1: {
                action: probs[action]
                for action in actions
            }
        }

        return value

    # ========================================================
    # Player Expansion
    # ========================================================

    def _expand_players(
        self,
        graph,
        state,
        node,
        current_actor,
        current_actions,
        next_state_to_value,
        depth,
    ):
        """
        Expand Red and Blue actions.

        For simultaneous CTF:

            Red actions
                x
            Blue actions

        are evaluated as a Cartesian product.
        """

        # ----------------------------------------------------
        # Get actions for Red and Blue
        # ----------------------------------------------------

        red_actions = (
            current_actions
            if current_actor == 0
            else self.actor_action_enumerator
                .enumerate(state)[1]
        )

        blue_actions = (
            self.actor_action_enumerator
            .enumerate(state)[1]
        )

        # ----------------------------------------------------
        # NOTE:
        # The generic ActorActionEnumerator only returns
        # one acting actor.
        #
        # For simultaneous Red/Blue CTF, a concrete
        # implementation should return the available actor
        # and actions, or provide simultaneous actions.
        #
        # Therefore we explicitly request actions through
        # a helper when available.
        # ----------------------------------------------------

        red_actions = self._get_actor_actions(
            state,
            0,
            current_actor,
            current_actions
        )

        blue_actions = self._get_actor_actions(
            state,
            1,
            current_actor,
            current_actions
        )

        if not red_actions or not blue_actions:

            return self._expand_single_actor(
                graph,
                state,
                node,
                current_actor,
                current_actions,
                next_state_to_value,
                depth,
            )

        node.actors = {0, 1}

        node.actions = []

        # ----------------------------------------------------
        # Cartesian product of Red and Blue actions
        # ----------------------------------------------------

        for red_action in red_actions:

            for blue_action in blue_actions:

                joint_action = (
                    (0, red_action),
                    (1, blue_action),
                )

                node.actions.append(
                    joint_action
                )

                next_state, rewards = (
                    self._transition_joint_action(
                        state,
                        joint_action
                    )
                )

                node.action_to_next_state[
                    joint_action
                ] = next_state

                if next_state not in next_state_to_value:

                    next_state_to_value[
                        next_state
                    ] = self._expand(
                        graph,
                        next_state,
                        node,
                        depth - 1,
                    )

        # ----------------------------------------------------
        # Red chooses action that maximizes
        # its worst-case Blue response.
        # ----------------------------------------------------

        red_action_to_value = {}

        red_action_to_blue_response = {}

        for red_action in red_actions:

            worst_value = float("inf")
            best_blue_response = None

            for blue_action in blue_actions:

                joint_action = (
                    (0, red_action),
                    (1, blue_action),
                )

                next_state = (
                    node.action_to_next_state[
                        joint_action
                    ]
                )

                value = (
                    next_state_to_value[
                        next_state
                    ]
                )

                # Blue minimizes Red's value.

                if value < worst_value:

                    worst_value = value
                    best_blue_response = blue_action

            red_action_to_value[
                red_action
            ] = worst_value

            red_action_to_blue_response[
                red_action
            ] = best_blue_response

        # ----------------------------------------------------
        # Red maximizes its minimum value
        # ----------------------------------------------------

        best_red_action, value = max(
            red_action_to_value.items(),
            key=lambda x: x[1]
        )

        node.actor_to_best_action[0] = (
            best_red_action
        )

        # Store Blue's best response.

        best_blue_action = (
            red_action_to_blue_response[
                best_red_action
            ]
        )

        if best_blue_action is not None:

            node.actor_to_best_action[1] = (
                best_blue_action
            )

        # Store action values if graph supports it.

        if hasattr(node, "proaction_to_value"):

            node.proaction_to_value.update(
                red_action_to_value
            )

        return value

    # ========================================================
    # Single Actor Expansion
    # ========================================================

    def _expand_single_actor(
        self,
        graph,
        state,
        node,
        actor,
        actions,
        next_state_to_value,
        depth,
    ):
        """
        Expand a state where only one CTF actor acts.

        This is useful for turn-based CTF environments.
        """

        if not actions:

            return self._get_heuristic_value(
                state
            )

        node.actors = {actor}

        node.actions = [
            ((actor, action),)
            for action in actions
        ]

        action_to_value = {}

        for action in actions:

            joint_action = (
                (actor, action),
            )

            next_state, rewards = (
                self._transition_joint_action(
                    state,
                    joint_action
                )
            )

            node.action_to_next_state[
                joint_action
            ] = next_state

            if next_state not in next_state_to_value:

                next_state_to_value[
                    next_state
                ] = self._expand(
                    graph,
                    next_state,
                    node,
                    depth - 1,
                )

            action_to_value[action] = (
                next_state_to_value[
                    next_state
                ]
            )

        # ----------------------------------------------------
        # Red maximizes.
        # Blue minimizes.
        # ----------------------------------------------------

        if actor == 0:

            best_action, value = max(
                action_to_value.items(),
                key=lambda x: x[1]
            )

        else:

            best_action, value = min(
                action_to_value.items(),
                key=lambda x: x[1]
            )

        node.actor_to_best_action[
            actor
        ] = best_action

        return value

    # ========================================================
    # Transition
    # ========================================================

    def _transition_joint_action(
        self,
        state,
        joint_action
    ):
        """
        Apply a joint CTF action.

        The generic ForwardTransitor interface receives
        one actor/action pair at a time.

        Therefore a concrete CTF transitor should ideally
        provide a joint-action implementation.

        If available:

            transition_joint(...)

        is used.

        Otherwise simultaneous actions are applied
        sequentially.
        """

        if hasattr(
            self.forward_transitor,
            "transition_joint"
        ):

            return (
                self.forward_transitor
                .transition_joint(
                    state,
                    dict(joint_action)
                )
            )

        # ----------------------------------------------------
        # Fallback:
        # Apply Red then Blue sequentially.
        #
        # For truly simultaneous CTF actions, implement
        # transition_joint() in the CTF transitor.
        # ----------------------------------------------------

        current_state = state
        total_rewards = {}

        for actor, action in joint_action:

            current_state, rewards = (
                self.forward_transitor.transition(
                    current_state,
                    action,
                    actor
                )
            )

            for reward_actor, reward in rewards.items():

                total_rewards[reward_actor] = (
                    total_rewards.get(
                        reward_actor,
                        0.0
                    )
                    + reward
                )

        return current_state, total_rewards

    # ========================================================
    # Actor Action Helper
    # ========================================================

    def _get_actor_actions(
        self,
        state,
        actor,
        current_actor,
        current_actions
    ):
        """
        Obtain actions for a specific actor.

        A CTF-specific ActorActionEnumerator can implement:

            get_actions(state, actor)

        for simultaneous Red/Blue environments.

        Otherwise the currently enumerated actions are used.
        """

        if hasattr(
            self.actor_action_enumerator,
            "get_actions"
        ):

            return self.actor_action_enumerator.get_actions(
                state,
                actor
            )

        if actor == current_actor:

            return current_actions

        return set()

    # ========================================================
    # Terminal / Heuristic
    # ========================================================

    def _is_terminal(self, state):
        """
        Check whether the CTF state is terminal.

        A CTF State implementation can provide:

            is_done()

        Otherwise the state is considered non-terminal.
        """

        if hasattr(state, "is_done"):

            return state.is_done()

        if hasattr(state, "done"):

            return state.done

        return False

    def _get_terminal_value(self, state):
        """
        Obtain terminal Red Team value.
        """

        if hasattr(state, "get_reward"):

            return state.get_reward()

        if hasattr(state, "reward"):

            reward = state.reward

            if isinstance(reward, dict):

                return reward.get(
                    0,
                    0.0
                )

            return reward

        # Fall back to heuristic.

        return self._get_heuristic_value(
            state
        )

    def _get_heuristic_value(self, state):
        """
        Evaluate a CTF state.

        ValueHeuristic returns:

            ({actor: value}, notes)

        We use Red Team's value as the minimax value.
        """

        values, notes = (
            self.value_heuristic.evaluate(
                state
            )
        )

        return values.get(
            0,
            0.0
        )

    # ========================================================
    # Graph Value Storage
    # ========================================================

    def _store_value(
        self,
        node,
        value
    ):
        """
        Store Red and Blue values in the graph.

        Zero-sum assumption:

            Blue value = -Red value
        """

        if hasattr(
            node,
            "actor_to_value_estimates"
        ):

            node.actor_to_value_estimates[
                0
            ].append(value)

            node.actor_to_value_estimates[
                1
            ].append(-value)

        elif hasattr(
            node,
            "values_estimates"
        ):

            node.values_estimates.append(
                value
            )
