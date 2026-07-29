"""Unit tests for environment."""

from __future__ import annotations

import numpy as np
import pytest

from emergent_intent.env import EnvConfig, ScenarioFamily, make_env
from emergent_intent.utils import set_global_seed


@pytest.mark.parametrize(
    "scenario",
    [
        "terrestrial_congestion",
        "tn_ntn_failover",
        "critical_service",
        "education_fairness",
    ],
)
def test_env_reset_step_all_scenarios(scenario: str) -> None:
    cfg = EnvConfig(scenario=ScenarioFamily(scenario), horizon=8, seed=0, n_ue=2)
    env = make_env(cfg)
    obs, infos = env.reset(seed=0)
    assert set(obs) == set(env.possible_agents)
    assert all(v.shape == env.observation_space(a).shape for a, v in obs.items())
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs2, rew, term, trunc, infos2 = env.step(actions)
    assert set(rew) == set(obs2)
    assert all(isinstance(r, float) for r in rew.values())


def test_partial_observability_masks_differ() -> None:
    env = make_env(EnvConfig(horizon=4, observation_noise=0.0, seed=1, n_ue=2, channel={"mode": "no_comm"}))
    obs, _ = env.reset(seed=1)
    assert not np.allclose(obs["ue_0"], obs["bs_0"])


def test_deterministic_seed() -> None:
    set_global_seed(123)
    e1 = make_env(EnvConfig(horizon=5, seed=123, n_ue=1, channel={"mode": "no_comm"}))
    o1, _ = e1.reset(seed=123)
    set_global_seed(123)
    e2 = make_env(EnvConfig(horizon=5, seed=123, n_ue=1, channel={"mode": "no_comm"}))
    o2, _ = e2.reset(seed=123)
    for a in o1:
        assert np.allclose(o1[a], o2[a], atol=1e-5)


def test_episode_terminates() -> None:
    env = make_env(EnvConfig(horizon=3, n_ue=1, channel={"mode": "no_comm"}))
    obs, _ = env.reset(seed=0)
    for _ in range(3):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rew, term, trunc, info = env.step(actions)
    assert env.agents == []


def test_comm_modes() -> None:
    for mode in ("no_comm", "fixed_protocol", "discrete_learned", "continuous_learned"):
        env = make_env(
            EnvConfig(horizon=2, n_ue=1, channel={"mode": mode}, continuous_dim=4, msg_len=2)
        )
        obs, _ = env.reset(seed=0)
        actions = {a: env.action_space(a).sample() for a in env.agents}
        env.step(actions)


def test_ntn_agent_present_in_failover() -> None:
    env = make_env(EnvConfig(scenario=ScenarioFamily.tn_ntn_failover, horizon=2, n_ue=1))
    assert "ntn_relay" in env.possible_agents
