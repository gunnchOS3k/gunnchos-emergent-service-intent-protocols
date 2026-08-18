"""Repeated CPU runs: mean/std recorded; GPU not invented."""

from __future__ import annotations

import numpy as np

from emergent_intent.comm.semantic_protocol import SemanticProtocolController
from emergent_intent.env import EnvConfig, ScenarioFamily, make_env
from emergent_intent.interpretability.claims import LanguageClaimEvidence, evaluate_language_claim
from emergent_intent.utils.device import detect_device


def _rollout(seed: int) -> float:
    env = make_env(
        EnvConfig(
            scenario=ScenarioFamily.hidden_blockage_congestion,
            horizon=8,
            n_ue=1,
            seed=seed,
            channel={"mode": "fixed_protocol", "vocab_size": 16, "msg_length": 2},
        )
    )
    controller = SemanticProtocolController(max_symbol_age=5.0)
    obs, _ = env.reset(seed=seed)
    total = 0.0
    for _ in range(env.config.horizon):
        acts = controller.actions_from_inbox(env, getattr(env, "_last_inbox", {}))
        obs, rewards, terms, truncs, _ = env.step(acts)
        total += float(sum(rewards.values()) / max(len(rewards), 1))
        if all(terms.values()) or all(truncs.values()):
            break
    return total


def test_three_seeds_record_mean_and_std() -> None:
    seeds = (0, 1, 2)
    returns = np.array([_rollout(s) for s in seeds], dtype=float)
    summary = {
        "seeds": list(seeds),
        "n_runs": int(returns.size),
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std(ddof=1) if returns.size > 1 else 0.0),
        "evidence_class": "SYNTHETIC_SIM",
        "device": detect_device().as_dict(),
    }
    assert summary["n_runs"] == 3
    assert np.isfinite(summary["mean_return"])
    assert summary["device"]["cuda_available"] is False or summary["device"]["device"] in {"cpu", "cuda"}
    ev = LanguageClaimEvidence(
        messages_are_exchanged=True,
        n_repeated_runs=summary["n_runs"],
    )
    claim = evaluate_language_claim(ev)
    assert claim["emergent_language_claimed"] is False
    assert claim["n_repeated_runs"] == 3
