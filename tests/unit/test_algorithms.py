"""Algorithm smoke + checkpoint/resume tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from emergent_intent.algorithms import (
    DialTarmacTrainer,
    IPPOTrainer,
    MAPPOTrainer,
    PPOConfig,
    VDNQMIXTrainer,
)
from emergent_intent.env import EnvConfig, make_env


@pytest.mark.smoke
@pytest.mark.parametrize("algo", ["ippo", "mappo", "vdn", "dial"])
def test_algorithm_smoke(algo: str, tmp_path: Path) -> None:
    cfg = EnvConfig(
        horizon=8,
        seed=0,
        n_ue=1,
        channel={"mode": "discrete_learned" if algo == "dial" else "no_comm", "vocab_size": 4, "msg_length": 2},
    )
    env = make_env(cfg)
    ppo = PPOConfig(rollout_steps=16, epochs=1, hidden=16)
    if algo == "ippo":
        trainer = IPPOTrainer(env, config=ppo, seed=0, prefer_cuda=False)
    elif algo == "mappo":
        trainer = MAPPOTrainer(env, config=ppo, seed=0, prefer_cuda=False)
    elif algo == "vdn":
        trainer = VDNQMIXTrainer(env, method="vdn", seed=0, hidden=16, prefer_cuda=False)
    else:
        trainer = DialTarmacTrainer(
            env, vocab_size=4, msg_length=2, config=ppo, seed=0, prefer_cuda=False
        )
    metrics = trainer.train(total_steps=32)
    assert metrics["steps"] >= 1
    assert metrics["evidence_class"] == "SYNTHETIC_SIM"
    ckpt = tmp_path / f"{algo}.pt"
    trainer.save(ckpt)
    trainer.load(ckpt)


def test_qmix_smoke() -> None:
    env = make_env(EnvConfig(horizon=6, n_ue=1, channel={"mode": "no_comm"}))
    trainer = VDNQMIXTrainer(env, method="qmix", seed=1, hidden=16, prefer_cuda=False)
    out = trainer.train(total_steps=24)
    assert out["algorithm"] == "QMIX"
