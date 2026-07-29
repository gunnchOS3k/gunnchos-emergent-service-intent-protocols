#!/usr/bin/env python3
"""Pilot experiment matrix (§6.9) — short runs, labeled PILOT not FINAL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emergent_intent.algorithms import PPOConfig, make_trainer
from emergent_intent.abstraction import run_abstraction_pilot
from emergent_intent.env import EnvConfig, ScenarioFamily, make_env
from emergent_intent.objectives.runner import run_multiobjective_pilot
from emergent_intent.stats import summarize_seed_matrix
from emergent_intent.utils import detect_device, dump_json, git_commit_sha


ROOT = Path(__file__).resolve().parents[1]

ALGOS = [
    "ippo",
    "mappo",
    "vdn",
    "qmix",
    "ppo_discrete_message_entropy_baseline",
    "dial",
    "tarmac",
]
SCENARIOS = [
    "terrestrial_congestion",
    "tn_ntn_failover",
    "critical_service",
    "education_fairness",
    "hidden_blockage_congestion",
    "tn_ntn_continuity",
]


def _run_one(algo: str, scenario: str, seed: int, steps: int, n_ue: int) -> dict:
    comm = "no_comm" if algo in ("ippo", "mappo", "vdn", "qmix") else "discrete_learned"
    env = make_env(
        EnvConfig(
            scenario=ScenarioFamily(scenario),
            horizon=min(16, max(8, steps // 4)),
            n_ue=n_ue,
            seed=seed,
            channel={"mode": comm, "vocab_size": 4, "msg_length": 2},
            include_ntn=scenario.startswith("tn_ntn"),
        )
    )
    ppo = PPOConfig(rollout_steps=16, epochs=1, hidden=16)
    kwargs: dict = {"seed": seed, "prefer_cuda": False}
    if algo in ("ippo", "mappo", "dial", "tarmac", "ppo_discrete_message_entropy_baseline"):
        kwargs["config"] = ppo
    if algo == "dial":
        kwargs["vocab_size"] = 4
        kwargs["msg_length"] = 2
    if algo in ("vdn", "qmix"):
        kwargs["hidden"] = 16
        kwargs["batch_size"] = 8
        kwargs["target_update_interval"] = 10
    trainer = make_trainer(algo, env, **kwargs)
    metrics = trainer.train(total_steps=steps)
    metrics.update(
        {
            "scenario": scenario,
            "seed": seed,
            "n_ue": n_ue,
            "status": "SUCCESS",
            "label": "PILOT",
            "commit": git_commit_sha(),
        }
    )
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "pilot")
    p.add_argument("--quick", action="store_true", help="tiny subset for CI")
    args = p.parse_args()
    out = args.out
    for d in (
        "smoke",
        "pilot",
        "final",
        "ablations",
        "generalization",
        "robustness",
        "interpretability",
    ):
        (ROOT / "results" / d).mkdir(parents=True, exist_ok=True)

    device = detect_device(prefer_cuda=False)
    rows = []
    algos = ALGOS[:3] if args.quick else ALGOS
    scenarios = SCENARIOS[:2] if args.quick else SCENARIOS
    seeds = 2 if args.quick else args.seeds

    # Random / no-comm / fixed-protocol baselines
    for scenario in scenarios:
        for seed in range(seeds):
            env = make_env(
                EnvConfig(
                    scenario=ScenarioFamily(scenario),
                    horizon=12,
                    n_ue=2,
                    seed=seed,
                    channel={"mode": "no_comm"},
                )
            )
            obs, _ = env.reset(seed=seed)
            total = 0.0
            for _ in range(env.config.horizon):
                if not env.agents:
                    break
                actions = {a: env.action_space(a).sample() for a in env.agents}
                obs, rewards, _, _, _ = env.step(actions)
                total += float(sum(rewards.values()) / max(len(rewards), 1))
            rows.append(
                {
                    "algorithm": "random",
                    "scenario": scenario,
                    "seed": seed,
                    "mean_return": total,
                    "status": "SUCCESS",
                    "label": "PILOT",
                    "evidence_class": "SYNTHETIC_SIM",
                }
            )

    for algo in algos:
        for scenario in scenarios:
            for seed in range(seeds):
                try:
                    m = _run_one(algo, scenario, seed, args.steps, n_ue=2)
                    rows.append(m)
                    dump_json(out / f"{algo}_{scenario}_seed{seed}.json", m)
                except Exception as e:  # noqa: BLE001 — record failed runs honestly
                    rows.append(
                        {
                            "algorithm": algo,
                            "scenario": scenario,
                            "seed": seed,
                            "status": "FAILED",
                            "error": str(e),
                            "evidence_class": "SYNTHETIC_SIM",
                            "label": "PILOT",
                        }
                    )

    # Oracle upper bound on one scenario
    env = make_env(
        EnvConfig(
            scenario=ScenarioFamily.hidden_blockage_congestion,
            horizon=12,
            n_ue=1,
            seed=0,
            channel={"mode": "discrete_learned", "vocab_size": 4},
        )
    )
    obs, _ = env.reset(seed=0)
    total = 0.0
    for _ in range(env.config.horizon):
        if not env.agents:
            break
        actions = env.oracle_service_actions()
        actions = {a: actions[a] for a in env.agents}
        obs, rewards, _, _, _ = env.step(actions)
        total += float(sum(rewards.values()) / max(len(rewards), 1))
    rows.append(
        {
            "algorithm": "centralized_oracle",
            "scenario": "hidden_blockage_congestion",
            "seed": 0,
            "mean_return": total,
            "status": "SUCCESS",
            "label": "PILOT",
            "evidence_class": "SYNTHETIC_SIM",
        }
    )

    summary = summarize_seed_matrix(rows)
    dump_json(out / "pilot_summary.json", {"rows": rows, "summary": summary, "device": device.as_dict()})

    # Abstractions
    env = make_env(EnvConfig(horizon=8, n_ue=1, channel={"mode": "no_comm"}))
    abs_reports = run_abstraction_pilot(env, steps=32, seed=0)
    dump_json(ROOT / "results" / "ablations" / "abstraction_pilot.json", abs_reports)

    # Multi-objective / Pareto
    run_multiobjective_pilot(ROOT / "results" / "pilot" / "multiobj", seeds=min(seeds, 5), horizon=12)

    # Placeholders declaring final is NOT earned
    dump_json(
        ROOT / "results" / "final" / "STATUS.json",
        {
            "status": "NOT_RUN",
            "evidence_class": "NONE",
            "reason": "Pilot/smoke steps (64–512) are not final experiments. Final requires dedicated larger budgets and ≥5 seeds per compared method.",
        },
    )
    dump_json(
        ROOT / "results" / "generalization" / "STATUS.json",
        {
            "status": "PILOT_PARTIAL",
            "note": "Held-out UE counts / domain shifts not fully swept; see gaps in paper.",
            "evidence_class": "SYNTHETIC_SIM",
        },
    )
    dump_json(
        ROOT / "results" / "robustness" / "STATUS.json",
        {
            "status": "PILOT_PARTIAL",
            "note": "Unit tests cover erasure/corruption/delay; full robustness matrix pending.",
            "evidence_class": "SYNTHETIC_SIM",
        },
    )
    print(json.dumps({"n_rows": len(rows), "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
