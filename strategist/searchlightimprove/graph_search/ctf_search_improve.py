from strategist.searchlight.algorithms.best_first_search import BestFirstSearch
from strategist.searchlight.datastructures.graphs import ValueGraph2
from self_improve_search import SelfImprovementInitialInferencer
#from strategist.llm_api_models import GPT35Multi
from proposers import LLMImprovementProposer
from ..prompts.prompts import IMPROVEMENT_PROMPTS
from strategist.searchlight.datastructures.adjusters import QValueAdjuster
from strategist.searchlight.datastructures.estimators import UtilityEstimatorLast
from strategist.CTF.engine import CTFState
from strategist.searchlight.datastructures.graphs import ValueGraph


def ctf_search_improve(
    seed_functions: list,
    evaluator,
) -> ValueGraph2:
    """
    Improve CTF strategies with BestFirstSearch + LLM self-improvement.

    Args:
        seed_functions:
            Initial CTF strategy candidates.
        evaluator:
            CTF evaluator. It must provide:
                evaluate(improved_functions)
            and return:
                (scores, feedback_notes)

    Returns:
        ValueGraph2 containing the explored/improved CTF strategies.

    Notes:
        Unlike the original search_improve.py, this function does not
        construct any GOPS simulator, GOPS state, or GOPS-specific
        evaluator. The CTF evaluator is injected by the caller.
    """

    # 1. LLM strategy proposer
    gpt = GPT35Multi(
        temperature=0.7,
        num_responses=1,
    )

    proposer = LLMImprovementProposer(
        gpt,
        IMPROVEMENT_PROMPTS,
    )

    # 2. Graph components
    adjuster = QValueAdjuster()
    estimator = UtilityEstimatorLast()

    graph = ValueGraph(
        players={0},
        adjuster=adjuster,
        utility_estimator=estimator
    )

    # 3. Self-improvement inference
    initial_inferencer = SelfImprovementInitialInferencer(
        proposer,
        evaluator,
        graph,
    )

    # 4. Best-first search over strategy candidates
    search_algorithm = BestFirstSearch(
        initial_inferencer
    )

    # 5. Root state
    root_state = CTFState("root")
    root_state.notes = {
        "score": float("-inf"),
        "notes": "root",
        "done": False,
    }

    root_node = graph.add_state(root_state)
    root_node.is_expanded = True

    # 6. Add initial CTF strategies
    for i, seed_function in enumerate(seed_functions):

        seed_state = CTFState(
            f"seed_{i}"
        )

        seed_state.strategy = seed_function
        seed_state.notes = {
            "score": float("-inf"),
            "notes": "seed",
            "done": False,
        }

        graph.add_state(seed_state)
        graph.add_edge(
            root_state,
            seed_state,
        )

    # 7. Search for improved strategies
    search_algorithm.expand(
        graph,
        root_state,
    )

    return graph
