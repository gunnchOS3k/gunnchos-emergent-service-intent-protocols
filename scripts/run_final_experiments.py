#!/usr/bin/env python3
"""Final / ablation / generalization / robustness / intervention experiment driver.

Truth rules:
- smoke ≠ final
- never fabricate scientific success
- mark BLOCKED_COMPUTE_CAPACITY when budgets cannot finish
- prefer largest scientifically useful subset under compute limits
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from emergent_intent.algorithms import PPOConfig, make_trainer
from emergent_intent.comm.interventions import InterventionKind, wrap_fixed_protocol_encoder
from emergent_intent.comm.semantic_protocol import SemanticProtocolController, DEFAULT_MAPPING
from emergent_intent.env import EnvConfig, ScenarioFamily, make_env
from emergent_intent.stats import summarize_seed_matrix
from emergent_intent.utils import detect_device, dump_json, git_commit_sha
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

# Complete baseline / method list for publication-grade comparisons
BASELINES = [
    "random",
    "no_comm",
    "fixed_protocol",
    "centralized_oracle",
    "ippo",
    "mappo",
    "vdn",
    "qmix",
    "ppo_discrete_message_entropy_baseline",
    "dial",
    "tarmac",
]

FLAGSHIP = ["dial", "tarmac", "mappo", "qmix"]

SCENARIOS = [
    "terrestrial_congestion",
    "tn_ntn_failover",
    "critical_service",
    "education_fairness",
    "hidden_blockage_congestion",
    "tn_ntn_continuity",
]


def _heuristic_episode(env, algorithm: str, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    total = 0.0
    controller = SemanticProtocolController()
    for t in range(env.config.horizon):
        if not env.agents:
            break
        if algorithm == "random":
            actions = {a: env.action_space(a).sample() for a in env.agents}
        elif algorithm == "centralized_oracle":
            actions = {a: env.oracle_service_actions()[a] for a in env.agents}
        elif algorithm == "fixed_protocol":
            actions = controller.actions_from_inbox(env, env._last_inbox)
        else:  # no_comm blind mid actions
            actions = {a: np.zeros(len(env._nvec), dtype=np.int64) for a in env.agents}
            for a in actions:
                actions[a][6] = 1  # admission
                actions[a][0] = 2
                actions[a][1] = 2
        obs, rewards, _, _, infos = env.step(actions)
        if infos:
            assert all(infos[a].get("coordination", 0.0) == 0.0 for a in infos)
        if rewards:
            total += float(sum(rewards.values()) / max(len(rewards), 1))
    return {
        "algorithm": algorithm,
        "mean_return": total,
        "status": "SUCCESS",
        "evidence_class": "SYNTHETIC_SIM",
        "seed": seed,
    }


def _run_learned(algo: str, scenario: str, seed: int, steps: int, n_ue: int) -> dict:
    comm = "no_comm" if algo in ("ippo", "mappo", "vdn", "qmix") else "discrete_learned"
    env = make_env(
        EnvConfig(
            scenario=ScenarioFamily(scenario),
            horizon=min(32, max(8, steps // 8)),
            n_ue=n_ue,
            seed=seed,
            channel={"mode": comm, "vocab_size": 8, "msg_length": 2},
            include_ntn=scenario.startswith("tn_ntn"),
            targeted=(algo == "tarmac"),
        )
    )
    ppo = PPOConfig(rollout_steps=min(64, max(16, steps // 8)), epochs=2, hidden=32)
    kwargs: dict = {"seed": seed, "prefer_cuda": False}
    if algo in ("ippo", "mappo", "dial", "tarmac", "ppo_discrete_message_entropy_baseline"):
        kwargs["config"] = ppo
    if algo == "dial":
        kwargs["vocab_size"] = 8
        kwargs["msg_length"] = 2
    if algo in ("vdn", "qmix"):
        kwargs["hidden"] = 32
        kwargs["batch_size"] = 16
        kwargs["target_update_interval"] = 20
    trainer = make_trainer(algo, env, **kwargs)
    metrics = trainer.train(total_steps=steps)
    metrics.update(
        {
            "scenario": scenario,
            "seed": seed,
            "n_ue": n_ue,
            "status": "SUCCESS",
            "commit": git_commit_sha(),
        }
    )
    return metrics


def run_matrix(
    *,
    out_dir: Path,
    label: str,
    algos: list[str],
    scenarios: list[str],
    seeds: int,
    steps: int,
    n_ue: int = 2,
    time_budget_s: float | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t0 = time.time()
    blocked = False
    block_reason = None
    for algo in algos:
        for scenario in scenarios:
            for seed in range(seeds):
                if time_budget_s is not None and (time.time() - t0) > time_budget_s:
                    blocked = True
                    block_reason = "BLOCKED_COMPUTE_CAPACITY"
                    break
                try:
                    if algo in ("random", "no_comm", "fixed_protocol", "centralized_oracle"):
                        env = make_env(
                            EnvConfig(
                                scenario=ScenarioFamily(scenario),
                                horizon=min(24, max(8, steps // 16)),
                                n_ue=n_ue,
                                seed=seed,
                                channel={
                                    "mode": "no_comm"
                                    if algo in ("random", "no_comm")
                                    else ("fixed_protocol" if algo == "fixed_protocol" else "discrete_learned"),
                                    "vocab_size": 16,
                                    "msg_length": 2,
                                },
                                include_ntn=scenario.startswith("tn_ntn"),
                            )
                        )
                        m = _heuristic_episode(env, algo, seed)
                        m.update({"scenario": scenario, "label": label, "requested_steps": steps})
                    else:
                        m = _run_learned(algo, scenario, seed, steps, n_ue)
                        m["label"] = label
                        m["requested_steps"] = steps
                    rows.append(m)
                    dump_json(out_dir / f"{algo}_{scenario}_seed{seed}.json", m)
                except Exception as e:  # noqa: BLE001
                    rows.append(
                        {
                            "algorithm": algo,
                            "scenario": scenario,
                            "seed": seed,
                            "status": "FAILED",
                            "error": str(e),
                            "label": label,
                            "evidence_class": "SYNTHETIC_SIM",
                        }
                    )
            if blocked:
                break
        if blocked:
            break

    summary = summarize_seed_matrix(rows)
    status = {
        "status": "BLOCKED_COMPUTE_CAPACITY" if blocked else ("PARTIAL" if label != "FINAL" else "RAN"),
        "label": label,
        "evidence_class": "SYNTHETIC_SIM",
        "requested_steps": steps,
        "requested_seeds": seeds,
        "algorithms": algos,
        "scenarios": scenarios,
        "n_rows": len(rows),
        "blocked": blocked,
        "block_reason": block_reason,
        "commit": git_commit_sha(),
        "device": detect_device(prefer_cuda=False).as_dict(),
        "baselines_complete_list": BASELINES,
        "note": (
            "FINAL scientific claim requires ≥5 seeds per compared method and budgets beyond smoke; "
            "short runs remain SYNTHETIC_SIM / non-final."
            if label == "FINAL"
            else f"{label} suite."
        ),
    }
    if blocked:
        status["status"] = "BLOCKED_COMPUTE_CAPACITY"
        status["completed_subset"] = sorted({r.get("algorithm") for r in rows if r.get("status") == "SUCCESS"})
    dump_json(out_dir / "STATUS.json", status)
    dump_json(out_dir / "summary.json", {"rows": rows, "summary": summary, "status": status})
    return status


def run_interventions(out: Path, seeds: int = 5) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    controller = SemanticProtocolController()
    rows = []
    for kind in InterventionKind:
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            env = make_env(
                EnvConfig(
                    scenario=ScenarioFamily.hidden_blockage_congestion,
                    horizon=12,
                    n_ue=1,
                    seed=seed,
                    channel={"mode": "fixed_protocol", "vocab_size": 16, "msg_length": 2},
                    delay=2 if kind == InterventionKind.DELAY else 0,
                )
            )
            wrap_fixed_protocol_encoder(env, kind, rng)
            env.reset(seed=seed)
            ep = 0.0
            for t in range(env.config.horizon):
                if not env.agents:
                    break
                acts = controller.actions_from_inbox(env, env._last_inbox)
                _, rewards, _, _, infos = env.step(acts)
                assert all(infos[a]["coordination"] == 0.0 for a in infos)
                if rewards:
                    ep += float(sum(rewards.values()) / max(len(rewards), 1))
            rows.append(
                {
                    "algorithm": f"fixed_protocol/{kind.value}",
                    "scenario": "hidden_blockage_congestion",
                    "seed": seed,
                    "mean_return": ep,
                    "status": "SUCCESS",
                    "label": "INTERVENTION",
                    "evidence_class": "SYNTHETIC_SIM",
                }
            )
    summary = summarize_seed_matrix(rows)
    dump_json(
        out / "STATUS.json",
        {
            "status": "RAN",
            "mapping": DEFAULT_MAPPING.as_dict(),
            "evidence_class": "SYNTHETIC_SIM",
            "seeds": seeds,
        },
    )
    dump_json(out / "summary.json", {"rows": rows, "summary": summary})
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["final", "ablations", "generalization", "robustness", "interventions", "all"], default="all")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--steps", type=int, default=1024, help="Flagship train steps (beyond smoke 512)")
    p.add_argument("--time-budget-s", type=float, default=600.0, help="Wall-clock budget; mark BLOCKED_COMPUTE_CAPACITY if exceeded")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    seeds = 2 if args.quick else args.seeds
    steps = 64 if args.quick else args.steps
    budget = 90.0 if args.quick else args.time_budget_s

    for d in (
        "final",
        "ablations",
        "generalization",
        "robustness",
        "interpretability",
        "interventions",
        "pilot",
        "smoke",
    ):
        (ROOT / "results" / d).mkdir(parents=True, exist_ok=True)

    statuses = {}

    if args.mode in ("final", "all"):
        # Largest scientifically useful subset under budget: flagship + key baselines
        algos = FLAGSHIP + ["random", "no_comm", "fixed_protocol", "ippo", "ppo_discrete_message_entropy_baseline"]
        scenarios = ["hidden_blockage_congestion", "tn_ntn_continuity", "critical_service"]
        if args.quick:
            algos = ["dial", "tarmac", "random", "fixed_protocol"]
            scenarios = scenarios[:2]
        statuses["final"] = run_matrix(
            out_dir=ROOT / "results" / "final",
            label="FINAL",
            algos=algos,
            scenarios=scenarios,
            seeds=seeds,
            steps=steps,
            time_budget_s=budget,
        )

    if args.mode in ("ablations", "all"):
        statuses["ablations"] = run_matrix(
            out_dir=ROOT / "results" / "ablations",
            label="ABLATION",
            algos=["dial", "no_comm", "fixed_protocol", "ppo_discrete_message_entropy_baseline"],
            scenarios=["hidden_blockage_congestion"],
            seeds=seeds,
            steps=min(steps, 768),
            time_budget_s=budget * 0.4,
        )

    if args.mode in ("generalization", "all"):
        # Held-out UE counts
        rows = []
        t0 = time.time()
        blocked = False
        for n_ue in (1, 2, 3):
            for seed in range(seeds):
                if time.time() - t0 > budget * 0.3:
                    blocked = True
                    break
                try:
                    m = _run_learned("dial", "hidden_blockage_congestion", seed, min(steps, 512), n_ue)
                    m["label"] = "GENERALIZATION"
                    m["n_ue"] = n_ue
                    rows.append(m)
                except Exception as e:  # noqa: BLE001
                    rows.append({"algorithm": "dial", "n_ue": n_ue, "seed": seed, "status": "FAILED", "error": str(e)})
            if blocked:
                break
        status = {
            "status": "BLOCKED_COMPUTE_CAPACITY" if blocked else "RAN",
            "evidence_class": "SYNTHETIC_SIM",
            "ue_counts": [1, 2, 3],
            "note": "Held-out UE-count sweep; digital-twin layouts still pending.",
        }
        dump_json(ROOT / "results" / "generalization" / "STATUS.json", status)
        dump_json(ROOT / "results" / "generalization" / "summary.json", {"rows": rows, "summary": summarize_seed_matrix(rows), "status": status})
        statuses["generalization"] = status

    if args.mode in ("robustness", "all"):
        rows = []
        for erasure in (0.0, 0.2, 0.4):
            for seed in range(min(seeds, 5)):
                env = make_env(
                    EnvConfig(
                        scenario=ScenarioFamily.hidden_blockage_congestion,
                        horizon=12,
                        n_ue=1,
                        seed=seed,
                        erasure_p=erasure,
                        channel={"mode": "fixed_protocol", "vocab_size": 16},
                    )
                )
                m = _heuristic_episode(env, "fixed_protocol", seed)
                m.update({"scenario": "hidden_blockage_congestion", "erasure_p": erasure, "label": "ROBUSTNESS"})
                rows.append(m)
        status = {
            "status": "RAN",
            "evidence_class": "SYNTHETIC_SIM",
            "sweeps": ["erasure_p"],
            "note": "Partial robustness matrix (erasure); full domain-shift pending.",
        }
        dump_json(ROOT / "results" / "robustness" / "STATUS.json", status)
        dump_json(ROOT / "results" / "robustness" / "summary.json", {"rows": rows, "summary": summarize_seed_matrix(rows), "status": status})
        statuses["robustness"] = status

    if args.mode in ("interventions", "all"):
        statuses["interventions"] = {
            "status": "RAN",
            "summary_methods": list(run_interventions(ROOT / "results" / "interventions", seeds=seeds).get("methods", {})),
        }

    print(json.dumps(statuses, indent=2, default=str))


if __name__ == "__main__":
    main()
