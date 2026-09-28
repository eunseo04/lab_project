from .headers import *

from searchlight.bandit import MultiarmedBanditLearner
from .llm_utils.llm_api_models import LLMModel
from .prompts.prompt_generators import PromptGenerator

from searchlight.utils import UpdatablePriorityDictionary
from searchlight.gameplay.agents import Agent
from searchlight.headers import State, ValueHeuristic2

from strategist.searchlightimprove.value_heuristic_improve import *
from baseline import *

from strategist.searchlight.datastructures.graphs import (
    ValueGraph2,
    PartialValueGraph,
)

import logging
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ============================================================
# CTF Value Heuristic Evaluator
# ============================================================

class RLValueHeuristicsSSGEvaluator(SimulateSearchGameEvaluator):

    def __init__(
        self,
        simulator: GameSimulator,
        transitor: ForwardTransitor,
        actor_enumerator: ActorActionEnumerator,
        action_enumerator: ActorActionEnumerator,
        num_batch_runs: int = 10,
        rng: np.random.Generator = np.random.default_rng(),
        search_budget: int = 16,
        random_rollouts: int = 16,
        partial_information: bool = False,
        stochastic_combinations: bool = True,
    ):
        super().__init__(
            simulator,
            num_batch_runs,
            players={0, 1},
            rng=rng,
            stochastic_combinations=stochastic_combinations,
        )

        self.transitor = transitor
        self.actor_enumerator = actor_enumerator
        self.action_enumerator = action_enumerator

        self.action_predictor = PolicyPredictor()

        self.search_budget = search_budget
        self.random_rollouts = random_rollouts

        self.players = {0, 1}

        if partial_information:
            self.value_graph_class = PartialValueGraph
        else:
            self.value_graph_class = ValueGraph2

    def evaluate(self, models) -> tuple[list[float], list]:

        agents = self.create_agents(models)
        agents = self.add_filler_agents(agents)

        scores, notes = super().evaluate_agents(agents)

        return scores, notes

    # --------------------------------------------------------
    # Agent 생성
    # --------------------------------------------------------

    def create_agents(self, models) -> list[SearchAgent]:

        graphs = [
            self.value_graph_class(
                adjuster=PUCTAdjuster(),
                utility_estimator=UtilityEstimatorLast(),
                players=self.players,
            )
            for _ in range(len(models))
        ]

        value_heuristics = [
            RLValueHeuristic(model)
            for model in models
        ]

        initial_inferencers = [
            PackageInitialInferencer(
                self.transitor,
                self.action_enumerator,
                self.action_predictor,
                self.actor_enumerator,
                value_heuristic,
            )
            for value_heuristic in value_heuristics
        ]

        search_algorithms = [
            SMMonteCarlo(
                initial_inferencer=initial_inferencer,
                rng=self.random_agent.rng,
                node_budget=self.search_budget,
                num_rollout=self.search_budget,
            )
            for initial_inferencer in initial_inferencers
        ]

        agents = [
            SearchAgent(
                search_algorithm,
                graph,
                self.rng,
            )
            for graph, search_algorithm
            in zip(graphs, search_algorithms)
        ]

        return agents

    # --------------------------------------------------------
    # Random Rollout Benchmark Agent
    # --------------------------------------------------------

    def create_benchmark_agents(self):

        num_agents = 1

        graphs = [
            self.value_graph_class(
                adjuster=PUCTAdjuster(),
                utility_estimator=UtilityEstimatorLast(),
                players=self.players,
            )
            for _ in range(num_agents)
        ]

        value_heuristics = [
            RandomRolloutValueHeuristic(
                actor_enumerator=self.actor_enumerator,
                action_enumerator=self.action_enumerator,
                forward_transitor=self.transitor,
                num_rollouts=self.random_rollouts,
                rng=self.random_agent.rng,
                players=self.players,
            )
            for _ in range(num_agents)
        ]

        initial_inferencers = [
            PackageInitialInferencer(
                self.transitor,
                self.action_enumerator,
                self.action_predictor,
                self.actor_enumerator,
                value_heuristic,
            )
            for value_heuristic in value_heuristics
        ]

        search_algorithms = [
            SMMonteCarlo(
                initial_inferencer=initial_inferencer,
                rng=self.random_agent.rng,
                node_budget=self.search_budget,
                num_rollout=self.search_budget,
            )
            for initial_inferencer in initial_inferencers
        ]

        agents = [
            SearchAgent(
                search_algorithm,
                graph,
                self.rng,
            )
            for graph, search_algorithm
            in zip(graphs, search_algorithms)
        ]

        return ["RandomRolloutValueHeuristicMCTS"], agents

    # --------------------------------------------------------
    # Filler Agent
    # --------------------------------------------------------

    def get_filler_agent(self) -> SearchAgent:

        graph = self.value_graph_class(
            players=self.players,
            adjuster=PUCTAdjuster(),
            utility_estimator=UtilityEstimatorLast(),
        )

        value_heuristic = RandomRolloutValueHeuristic(
            actor_enumerator=self.actor_enumerator,
            action_enumerator=self.action_enumerator,
            forward_transitor=self.transitor,
            num_rollouts=self.random_rollouts,
            rng=self.random_agent.rng,
            players=self.players,
        )

        initial_inferencer = PackageInitialInferencer(
            self.transitor,
            self.action_enumerator,
            self.action_predictor,
            self.actor_enumerator,
            value_heuristic,
        )

        search_algorithm = SMMonteCarlo(
            initial_inferencer=initial_inferencer,
            rng=self.random_agent.rng,
            node_budget=self.search_budget,
            num_rollout=self.search_budget,
        )

        return SearchAgent(
            search_algorithm,
            graph,
            self.rng,
        )

    def add_filler_agents(
        self,
        agents: list[SearchAgent],
    ) -> list[SearchAgent]:

        num_to_add = max(
            0,
            len(self.players) - len(agents)
        )

        filler_agents = [
            self.get_filler_agent()
            for _ in range(num_to_add)
        ]

        return agents + filler_agents


# ============================================================
# RL Value Heuristic
# ============================================================

class RLValueHeuristic(ValueHeuristic2):

    def __init__(self, torch_model):
        self.model = torch_model
        super().__init__()

    def _evaluate(
        self,
        state: CTFState,
    ) -> tuple[dict[int, float], dict]:

        values = self.model.forward(state)

        result = {
            0: float(values[0]),
            1: float(values[1]),
        }

        notes = {}

        return result, notes


# ============================================================
# CTF State Encoder
# ============================================================

class CTFStateEncoder:
    """
    CTFState를 neural network가 사용할 숫자 벡터로 변환한다.

    실제 프로젝트에서는 CTFState의 필드에 맞게
    encode()를 수정한다.
    """

    def __init__(self, input_dims: int):
        self.input_dims = input_dims

    def encode(self, state: CTFState) -> np.ndarray:

        features = state.to_feature_vector()

        features = list(features)

        if len(features) < self.input_dims:
            features.extend(
                [0.0] * (self.input_dims - len(features))
            )

        features = features[:self.input_dims]

        return np.asarray(
            features,
            dtype=np.float32,
        )


# ============================================================
# CTF Value Network
# ============================================================

class VNetwork_CTF(nn.Module):

    def __init__(
        self,
        input_dims: int,
        fc1_dims: int = 128,
        fc2_dims: int = 128,
        num_players: int = 2,
    ):
        super().__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.num_player = num_players

        self.fc1 = nn.Linear(
            input_dims,
            fc1_dims,
        )

        self.fc2 = nn.Linear(
            fc1_dims,
            fc2_dims,
        )

        self.fc3 = nn.Linear(
            fc2_dims,
            num_players,
        )

    def forward(self, observation):

        if isinstance(observation, CTFState):
            state = observation.to_feature_vector()
            state = np.asarray(
                state,
                dtype=np.float32,
            )
            state = T.tensor(state)

        else:
            state = T.tensor(
                observation,
                dtype=T.float32,
            )

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# ============================================================
# RL Evolver
# ============================================================

class RLEvolver(Evolver):

    def __init__(
        self,
        config,
        evaluator: Evaluator,
        batch_size: int = 10,
        rd_seed: int = 0,
    ):

        super().__init__()

        self.config = config
        self.evaluator = evaluator
        self.batch_size = batch_size

        self.num_evolutions = 0
        self.data = []

        self.rd_seed = rd_seed

        # ----------------------------------------------------
        # CTF 설정
        # ----------------------------------------------------

        self.num_players = 2

        self.state_dim = config.env_preset.state_dim

        self.fc1_dims = 128
        self.fc2_dims = 128

        self.lr_V = 5e-4

        self.model = VNetwork_CTF(
            input_dims=self.state_dim,
            fc1_dims=self.fc1_dims,
            fc2_dims=self.fc2_dims,
            num_players=self.num_players,
        )

        self.V_optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.lr_V,
        )

        self.performance_history = []

        # ----------------------------------------------------
        # Initial evaluation
        # ----------------------------------------------------

        scores, notes = self.evaluate(
            [self.model]
        )

        self.data.append(
            notes[0]["trajectory_data"]
        )

        self.logger = logging.getLogger(
            self.__class__.__name__
        )

    def evaluate(
        self,
        models,
    ) -> tuple[list[float], list[dict]]:

        return self.evaluator.evaluate(models)

    # --------------------------------------------------------
    # One evolution
    # --------------------------------------------------------

    def evolve_once(self):

        training_data = self.data[-1]

        for trajectory in training_data:

            data_trajectory = trajectory["trajectory"]

            score_trajectory = trajectory[
                "score_trajectory"
            ]

            for i in range(len(data_trajectory)):

                state = list(
                    data_trajectory[i]
                )[-1]

                score = score_trajectory[i]

                score_lst = [
                    score[0],
                    score[1],
                ]

                # --------------------------------------------
                # Value Network 학습
                # --------------------------------------------

                self.V_optimizer.zero_grad()

                V_pred = self.model.forward(
                    state
                )

                V_true = T.FloatTensor(
                    score_lst
                )

                loss_fn = nn.MSELoss()

                V_loss = loss_fn(
                    V_pred,
                    V_true,
                )

                V_loss.backward()

                self.V_optimizer.step()

        # ----------------------------------------------------
        # 학습 후 재평가
        # ----------------------------------------------------

        scores, notes = self.evaluate(
            [self.model]
        )

        self.performance_history.append(
            scores[0]
        )

        self.num_evolutions += 1

        self.data.append(
            notes[0]["trajectory_data"]
        )

    # --------------------------------------------------------
    # Evolution
    # --------------------------------------------------------

    def evolve(self, num_cycles: int):

        for _ in range(num_cycles):
            self.evolve_once()

        print(
            "Task: CTF, "
            f"Seed: {self.rd_seed}, "
            f"Performance History = "
            f"{self.performance_history}"
        )

        np.save(
            f"performance_history_CTF_seed{self.rd_seed}.npy",
            self.performance_history,
        )

        T.save(
            self.model.state_dict(),
            f"rl_model_CTF_seed{self.rd_seed}.pt",
        )

    # --------------------------------------------------------
    # Final Analysis
    # --------------------------------------------------------

    def produce_analysis(self):

        scores, notes = self.evaluate(
            [self.model]
        )

        self.logger.info(
            f"Final CTF score: {scores[0]}"
        )