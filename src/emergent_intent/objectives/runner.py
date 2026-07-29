"""Multi-objective evaluation helpers producing Pareto result files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from emergent_intent.env import EnvConfig, ScenarioFamily, make_env
from emergent_intent.env.wireless_env import A_ADMISSION, A_MCS, A_POWER, A_PRB, A_PRIORITY
from emergent_intent.objectives.multiobj import (
    OBJECTIVE_KEYS,
    LagrangianConstraints,
    PreferenceConditioned,
    Scalarization,
    pareto_front,
)
from emergent_intent.utils import dump_json


DEFAULT_WEIGHTS = {
    "task_success": 1.0,
    "latency_aoi": 0.2,
    "energy": 0.1,
    "message_bits": 0.1,
    "fairness": 0.2,
    "spectral_efficiency": 0.05,
    "violations": 1.0,
}


def _eval_policy_weights(env, weights: dict[str, float], steps: int, seed: int) -> dict[str, float]:
    env.config.objectives = weights
    obs, _ = env.reset(seed=seed)
    acc = {k: 0.0 for k in OBJECTIVE_KEYS}
    n = 0
    for _ in range(steps):
        if not env.agents:
            break
        actions = {}
        for a in env.agents:
            act = np.zeros(len(env._nvec), dtype=np.int64)
            act[A_POWER] = 3
            act[A_PRB] = 3
            act[A_MCS] = 2
            act[A_ADMISSION] = 1
            act[A_PRIORITY] = 2
            actions[a] = act
        obs, rewards, terms, truncs, infos = env.step(actions)
        rv = next(iter(infos.values()))["reward_vec"]
        for k in OBJECTIVE_KEYS:
            acc[k] += float(rv[k])
        n += 1
        if not env.agents:
            break
    if n == 0:
        return {k: 0.0 for k in OBJECTIVE_KEYS}
    return {k: acc[k] / n for k in OBJECTIVE_KEYS}


def run_multiobjective_pilot(out_dir: str | Path, seeds: int = 3, horizon: int = 16) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    points = []
    methods = {}

    # Fixed weighted scalarization
    for seed in range(seeds):
        env = make_env(
            EnvConfig(
                scenario=ScenarioFamily.education_fairness,
                horizon=horizon,
                n_ue=2,
                seed=seed,
                channel={"mode": "no_comm"},
                objectives=DEFAULT_WEIGHTS,
            )
        )
        vec = _eval_policy_weights(env, DEFAULT_WEIGHTS, horizon, seed)
        sc = Scalarization(DEFAULT_WEIGHTS)(vec)
        points.append(dict(vec))
        methods.setdefault("weighted", []).append({"seed": seed, "scalar": sc, **vec})

    # Preference-conditioned
    prefs = [
        np.array([0.5, 0.1, 0.1, 0.05, 0.15, 0.05, 0.05]),
        np.array([0.2, 0.1, 0.1, 0.05, 0.45, 0.05, 0.05]),
    ]
    for pi, pref in enumerate(prefs):
        pc = PreferenceConditioned(pref)
        for seed in range(seeds):
            env = make_env(
                EnvConfig(
                    scenario=ScenarioFamily.education_fairness,
                    horizon=horizon,
                    n_ue=2,
                    seed=seed + 10 * (pi + 1),
                    channel={"mode": "no_comm"},
                    objectives=pc.weights,
                )
            )
            vec = _eval_policy_weights(env, pc.weights, horizon, seed)
            points.append(dict(vec))
            methods.setdefault(f"preference_{pi}", []).append(
                {"seed": seed, "scalar": pc.scalarize(vec), **vec}
            )

    # Lagrangian
    lag = LagrangianConstraints()
    for seed in range(seeds):
        env = make_env(
            EnvConfig(
                scenario=ScenarioFamily.critical_service,
                horizon=horizon,
                n_ue=2,
                seed=seed + 100,
                channel={"mode": "no_comm"},
                objectives=DEFAULT_WEIGHTS,
            )
        )
        vec = _eval_policy_weights(env, DEFAULT_WEIGHTS, horizon, seed)
        base = Scalarization(DEFAULT_WEIGHTS)(vec)
        penalized = lag.penalize(vec, base)
        lag.update(abs(vec.get("violations", 0.0)))
        points.append(dict(vec))
        methods.setdefault("lagrangian", []).append(
            {"seed": seed, "scalar": penalized, "lambda_violations": lag.multipliers["violations"], **vec}
        )

    front = pareto_front(points)
    payload = {
        "evidence_class": "SYNTHETIC_SIM",
        "label": "PILOT",
        "methods": methods,
        "pareto_front": front,
        "n_points": len(points),
        "n_front": len(front),
        "note": "Pilot-scale heuristic policies; not a final multi-objective claim.",
    }
    dump_json(out_dir / "pareto_front.json", payload)
    dump_json(out_dir / "multiobjective_methods.json", methods)
    return payload
