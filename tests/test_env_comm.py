"""Unit and property tests for core environment and message channel."""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from emergent_intent.comm.channel import SILENCE, ChannelConfig, MessageChannel
from emergent_intent.env.config import CommMode, EnvConfig, ScenarioFamily
from emergent_intent.env.wireless_env import ServiceIntentEnv
from emergent_intent.intent.schema import rule_based_parse
from emergent_intent.intent.constraints import action_mask, compile_constraints
from emergent_intent.objectives.multiobj import pareto_front
from emergent_intent.utils.seeding import seed_everything


def test_partial_observability_not_global():
    env = ServiceIntentEnv(EnvConfig(n_ue=3, seed=1))
    obs, _ = env.reset(seed=1)
    assert not np.allclose(obs["ue_0"], obs["bs_0"])
    for a, o in obs.items():
        assert o.shape == env.observation_space(a).shape
        assert o.shape[0] > 8  # local + inbox features


def test_deterministic_seed():
    seed_everything(42)
    e1 = ServiceIntentEnv(EnvConfig(seed=42, horizon=5))
    e2 = ServiceIntentEnv(EnvConfig(seed=42, horizon=5))
    o1, _ = e1.reset(seed=42)
    o2, _ = e2.reset(seed=42)
    for a in o1:
        assert np.allclose(o1[a], o2[a])


@pytest.mark.parametrize("scenario", list(ScenarioFamily))
def test_scenarios_run(scenario):
    env = ServiceIntentEnv(EnvConfig(scenario=scenario, horizon=4, n_ue=2, include_ntn=True))
    env.reset(seed=0)
    for _ in range(4):
        if not env.agents:
            break
        actions = {a: env.action_spaces[a].sample() for a in env.agents}
        env.step(actions)


@pytest.mark.parametrize("mode", list(CommMode))
def test_comm_modes(mode):
    env = ServiceIntentEnv(EnvConfig(comm_mode=mode, horizon=3, n_ue=2))
    env.reset(seed=1)
    actions = {a: env.action_space(a).sample() for a in env.agents}
    env.step(actions)


def test_malformed_message_no_crash():
    ch = MessageChannel(ChannelConfig(erasure_p=1.0, corruption_p=1.0, bit_error_p=1.0))
    assert ch.send_discrete(None) is None
    MessageChannel(ChannelConfig(erasure_p=0.0)).send_discrete(np.array([999, -5]))


def test_intent_ambiguity_rejected():
    r = rule_based_parse("maybe somehow do whatever")
    assert not r.ok
    ok = rule_based_parse("critical low latency terrestrial only")
    assert ok.ok and ok.intent is not None
    cons = compile_constraints(ok.intent)
    assert cons.forbid_ntn
    masked = action_mask("ntn_relay", [3], cons, include_ntn=True)
    assert masked[0][0] is True


def test_pareto():
    pts = [
        {
            "task_success": 1.0,
            "fairness": 0.2,
            "spectral_efficiency": 0.1,
            "latency_aoi": -1,
            "energy": -1,
            "message_bits": -1,
            "violations": 0,
        },
        {
            "task_success": 0.5,
            "fairness": 0.9,
            "spectral_efficiency": 0.1,
            "latency_aoi": -1,
            "energy": -1,
            "message_bits": -1,
            "violations": 0,
        },
        {
            "task_success": 0.4,
            "fairness": 0.1,
            "spectral_efficiency": 0.0,
            "latency_aoi": -2,
            "energy": -2,
            "message_bits": -2,
            "violations": -1,
        },
    ]
    front = pareto_front(pts)
    assert len(front) >= 2


@given(st.integers(0, 32), st.floats(0, 1))
@settings(max_examples=25, deadline=None)
def test_channel_bit_cost_nonnegative(msg_len, erasure):
    ch = MessageChannel(ChannelConfig(msg_len=max(int(msg_len), 1), erasure_p=float(erasure)))
    symbols = np.arange(max(int(msg_len), 1)) % 8
    assert ch.bit_cost(symbols) >= 0
