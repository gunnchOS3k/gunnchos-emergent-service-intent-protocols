"""Property-based tests (hypothesis)."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings, strategies as st

from emergent_intent.comm import ChannelConfig, make_channel, validate_message_shape
from emergent_intent.env import EnvConfig, make_env
from emergent_intent.objectives import ObjectiveWeights, compute_rewards


@given(
    mode=st.sampled_from(["no_comm", "fixed_protocol", "discrete_learned", "continuous_learned"]),
    seed=st.integers(0, 10_000),
)
@settings(max_examples=20, deadline=None)
def test_env_step_never_crashes(mode: str, seed: int) -> None:
    cfg = EnvConfig(
        horizon=3,
        seed=seed,
        channel={"mode": mode, "vocab_size": 4, "msg_length": 2, "continuous_dim": 4},
    )
    env = make_env(cfg)
    obs, _ = env.reset(seed=seed)
    for _ in range(3):
        if not env.agents:
            break
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rew, term, trunc, info = env.step(actions)
        assert all(np.isfinite(r) for r in rew.values())


@given(
    erasure=st.floats(0, 1),
    corr=st.floats(0, 1),
    delay=st.integers(0, 2),
)
@settings(max_examples=15, deadline=None)
def test_channel_exchange_bits_nonnegative(erasure: float, corr: float, delay: int) -> None:
    cfg = ChannelConfig(
        mode="discrete_learned",
        vocab_size=4,
        msg_length=2,
        erasure_prob=erasure,
        corruption_prob=corr,
        delay_steps=delay,
        bit_cost=0.1,
    )
    ch = make_channel(cfg, ["a", "b"])
    ch.reset(np.random.default_rng(0))
    inbox, bits, _recs = ch.exchange(
        {"a": np.array([1.0, 2.0]), "b": np.array([0.0, 0.0])},
        np.random.default_rng(1),
    )
    assert bits >= 0.0
    assert set(inbox) == {"a", "b"}


@given(
    task=st.floats(0, 1),
    lat=st.floats(0, 100),
    energy=st.floats(0, 5),
)
@settings(max_examples=20, deadline=None)
def test_reward_finite(task: float, lat: float, energy: float) -> None:
    metrics = {
        "task_success": task,
        "latency_ms": lat,
        "energy": energy,
        "message_bits": 1.0,
        "fairness": 0.5,
        "spectral_efficiency": 0.2,
        "violations": 0.0,
    }
    out = compute_rewards(metrics, ObjectiveWeights())
    assert np.isfinite(out["scalar"])


def test_validate_message_accepts_correct_shape() -> None:
    cfg = ChannelConfig(mode="discrete_learned", msg_length=3)
    arr = validate_message_shape(np.array([1, 2, 3]), cfg)
    assert arr.dtype == np.float32
