from __future__ import annotations

from dataclasses import dataclass

import numpy as np


OBJECTIVE_KEYS = (
    "task_success",
    "latency_aoi",
    "energy",
    "message_bits",
    "fairness",
    "spectral_efficiency",
    "violations",
)


@dataclass
class Scalarization:
    weights: dict[str, float]

    def __call__(self, reward_vec: dict[str, float]) -> float:
        return float(sum(self.weights.get(k, 0.0) * reward_vec.get(k, 0.0) for k in OBJECTIVE_KEYS))


class PreferenceConditioned:
    """Maps a preference vector over objectives to weights."""

    def __init__(self, preference: np.ndarray):
        p = np.asarray(preference, dtype=np.float64)
        p = np.clip(p, 1e-6, None)
        p = p / p.sum()
        self.weights = {k: float(p[i]) for i, k in enumerate(OBJECTIVE_KEYS[: len(p)])}

    def scalarize(self, reward_vec: dict[str, float]) -> float:
        return Scalarization(self.weights)(reward_vec)


class LagrangianConstraints:
    def __init__(self, multipliers: dict[str, float] | None = None):
        self.multipliers = multipliers or {"violations": 1.0, "message_bits": 0.1, "energy": 0.1}

    def penalize(self, reward_vec: dict[str, float], base: float) -> float:
        penalty = 0.0
        for k, lam in self.multipliers.items():
            # violations/bits/energy are negative-good in reward_vec; treat magnitude
            penalty += lam * abs(min(0.0, reward_vec.get(k, 0.0)))
        return base - penalty

    def update(self, constraint_violation: float, lr: float = 0.01) -> None:
        self.multipliers["violations"] = max(0.0, self.multipliers["violations"] + lr * constraint_violation)


def pareto_dominates(a: dict[str, float], b: dict[str, float], maximize: set[str] | None = None) -> bool:
    maximize = maximize or {"task_success", "fairness", "spectral_efficiency"}
    better_or_eq = True
    strictly_better = False
    for k in OBJECTIVE_KEYS:
        av, bv = a.get(k, 0.0), b.get(k, 0.0)
        if k in maximize:
            if av < bv:
                better_or_eq = False
                break
            if av > bv:
                strictly_better = True
        else:
            # minimize magnitude of negative objectives: higher (less negative) is better
            if av < bv:
                better_or_eq = False
                break
            if av > bv:
                strictly_better = True
    return better_or_eq and strictly_better


def pareto_front(points: list[dict[str, float]]) -> list[dict[str, float]]:
    front = []
    for p in points:
        if not any(pareto_dominates(q, p) for q in points if q is not p):
            front.append(p)
    return front
