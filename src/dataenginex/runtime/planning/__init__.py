"""Resource estimation and adaptive strategy selection (§15.5, §15.6)."""

from dataenginex.runtime.planning.estimator import (
    EstimateBand,
    ExecutionStrategy,
    StrategyChoice,
    choose_strategy,
    estimate_operation,
)
from dataenginex.runtime.planning.planner import (
    PlanningError,
    build_context,
    build_plan,
    plan_attempt,
)
from dataenginex.runtime.planning.results import (
    ResultNotStorable,
    ResultTooLarge,
    purge_expired_results,
    store_interactive_result,
)

__all__ = [
    "EstimateBand",
    "ExecutionStrategy",
    "PlanningError",
    "ResultNotStorable",
    "ResultTooLarge",
    "StrategyChoice",
    "build_context",
    "build_plan",
    "choose_strategy",
    "estimate_operation",
    "plan_attempt",
    "purge_expired_results",
    "store_interactive_result",
]
