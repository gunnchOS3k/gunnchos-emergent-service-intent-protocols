"""Multi-objective rewards: scalarization, preference, Lagrangian, Pareto."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


OBJECTIVE_KEYS = (
    "task_success",
    "latency",
    "energy",
    "message_bits",
    "fairness",
    "spectral_efficiency",
    "violations",
)


@dataclass
class ObjectiveWeights:
    task_success: float = 1.0
    latency: float = 0.5
    energy: float = 0.2
    message_bits: float = 0.1
    fairness: float = 0.2
    spectral_efficiency: float = 0.3
    violations: float = 1.0

    def as_dict(self) -> dict[str, float]:
        return {
            "task_success": self.task_success,
            "latency": self.latency,
            "energy": self.energy,
            "message_bits": self.message_bits,
            "fairness": self.fairness,
            "spectral_efficiency": self.spectral_efficiency,
            "violations": self.violations,
        }


def normalize_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Map raw metrics to [higher-is-better] utilities in roughly [0,1]."""
    lat = metrics.get("latency_ms", 0.0)
    return {
        "task_success": float(metrics.get("task_success", 0.0)),
        "latency": float(1.0 / (1.0 + lat / 20.0)),
        "energy": float(1.0 / (1.0 + metrics.get("energy", 0.0))),
        "message_bits": float(1.0 / (1.0 + metrics.get("message_bits", 0.0))),
        "fairness": float(metrics.get("fairness", 0.0)),
        "spectral_efficiency": float(np.clip(metrics.get("spectral_efficiency", 0.0), 0.0, 1.0)),
        "violations": float(1.0 / (1.0 + metrics.get("violations", 0.0))),
    }


def weighted_scalarization(
    utilities: dict[str, float], weights: ObjectiveWeights
) -> float:
    w = weights.as_dict()
    return float(sum(w[k] * utilities.get(k, 0.0) for k in w))


def preference_conditioned_scalar(
    utilities: dict[str, float],
    preference: dict[str, float],
) -> float:
    """Preference vector ω over objectives; ||ω||_1 normalized."""
    keys = [k for k in preference if k in utilities]
    s = sum(abs(preference[k]) for k in keys) + 1e-8
    return float(sum((preference[k] / s) * utilities[k] for k in keys))


@dataclass
class LagrangianState:
    multipliers: dict[str, float] = field(
        default_factory=lambda: {"latency": 0.0, "energy": 0.0, "violations": 0.0}
    )
    lr: float = 0.05
    limits: dict[str, float] = field(
        default_factory=lambda: {"latency_ms": 20.0, "energy": 0.5, "violations": 0.5}
    )

    def penalty(self, metrics: dict[str, float]) -> float:
        pen = 0.0
        if "latency" in self.multipliers:
            gap = max(0.0, metrics.get("latency_ms", 0.0) - self.limits.get("latency_ms", 20.0))
            pen += self.multipliers["latency"] * gap
        if "energy" in self.multipliers:
            gap = max(0.0, metrics.get("energy", 0.0) - self.limits.get("energy", 0.5))
            pen += self.multipliers["energy"] * gap
        if "violations" in self.multipliers:
            gap = max(0.0, metrics.get("violations", 0.0) - self.limits.get("violations", 0.5))
            pen += self.multipliers["violations"] * gap
        return float(pen)

    def update(self, metrics: dict[str, float]) -> None:
        if "latency" in self.multipliers:
            gap = metrics.get("latency_ms", 0.0) - self.limits.get("latency_ms", 20.0)
            self.multipliers["latency"] = max(
                0.0, self.multipliers["latency"] + self.lr * gap
            )
        if "energy" in self.multipliers:
            gap = metrics.get("energy", 0.0) - self.limits.get("energy", 0.5)
            self.multipliers["energy"] = max(0.0, self.multipliers["energy"] + self.lr * gap)
        if "violations" in self.multipliers:
            gap = metrics.get("violations", 0.0) - self.limits.get("violations", 0.5)
            self.multipliers["violations"] = max(
                0.0, self.multipliers["violations"] + self.lr * gap
            )


def compute_rewards(
    metrics: dict[str, float],
    weights: ObjectiveWeights,
    team: bool = True,
    preference: dict[str, float] | None = None,
    lagrangian: LagrangianState | None = None,
) -> dict[str, Any]:
    utils = normalize_metrics(metrics)
    if preference:
        scalar = preference_conditioned_scalar(utils, preference)
    else:
        scalar = weighted_scalarization(utils, weights)
    if lagrangian is not None:
        scalar = scalar - lagrangian.penalty(metrics)
        lagrangian.update(metrics)
    return {
        "scalar": scalar,
        "utilities": utils,
        "raw": dict(metrics),
        "team": team,
    }


def pareto_front(points: list[dict[str, float]], keys: tuple[str, ...] | None = None) -> list[int]:
    """Return indices of non-dominated points (all objectives higher-is-better)."""
    keys = keys or ("task_success", "latency", "fairness", "spectral_efficiency")
    n = len(points)
    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ge_all = all(points[j].get(k, 0.0) >= points[i].get(k, 0.0) for k in keys)
            gt_any = any(points[j].get(k, 0.0) > points[i].get(k, 0.0) for k in keys)
            if ge_all and gt_any:
                dominated[i] = True
                break
    return [i for i, d in enumerate(dominated) if not d]


def hypervolume_2d(points: list[tuple[float, float]], ref: tuple[float, float] = (0.0, 0.0)) -> float:
    """Simple 2D hypervolume for maximization relative to ref corner."""
    pts = sorted([(max(p[0], ref[0]), max(p[1], ref[1])) for p in points], key=lambda x: x[0])
    hv = 0.0
    max_y = ref[1]
    prev_x = ref[0]
    for x, y in pts:
        if y > max_y:
            hv += (x - prev_x) * (max_y - ref[1]) if False else 0.0
            # standard: sort by x ascending for max — use descending approach
            max_y = y
            prev_x = x
    # Correct 2D HV for maximization:
    pts2 = sorted(points, key=lambda p: p[0], reverse=True)
    hv = 0.0
    best_y = ref[1]
    prev_x = ref[0]
    # Use ascending x
    pts2 = sorted([(p[0], p[1]) for p in points if p[0] >= ref[0] and p[1] >= ref[1]])
    if not pts2:
        return 0.0
    # filter dominated in 2d
    nd: list[tuple[float, float]] = []
    for p in sorted(pts2, key=lambda t: t[0]):
        if not nd or p[1] > nd[-1][1]:
            nd.append(p)
    prev_x = ref[0]
    for x, y in nd:
        hv += (x - prev_x) * (y - ref[1])
        prev_x = x
    return float(hv)
