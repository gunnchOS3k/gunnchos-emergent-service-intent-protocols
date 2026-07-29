"""Multi-objective rewards: scalarization, preference, Lagrangian, Pareto."""

from emergent_intent.objectives.rewards import (
    OBJECTIVE_KEYS,
    LagrangianState,
    ObjectiveWeights,
    compute_rewards,
    hypervolume_2d,
    normalize_metrics,
    pareto_front,
    preference_conditioned_scalar,
    weighted_scalarization,
)

__all__ = [
    "OBJECTIVE_KEYS",
    "LagrangianState",
    "ObjectiveWeights",
    "compute_rewards",
    "hypervolume_2d",
    "normalize_metrics",
    "pareto_front",
    "preference_conditioned_scalar",
    "weighted_scalarization",
]
