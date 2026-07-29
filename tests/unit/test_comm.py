"""Unit tests for message channel."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from emergent_intent.comm import (
    AttentionTargeter,
    ChannelConfig,
    GraphMessageRouter,
    discrete_message_from_logits,
    gumbel_softmax_sample,
    make_channel,
    validate_message_shape,
)


def test_no_comm_zero_bits() -> None:
    ch = make_channel(ChannelConfig(mode="no_comm", msg_len=2), ["a", "b"])
    ch.reset(np.random.default_rng(0))
    inbox, bits = ch.exchange(
        {"a": ch.empty_message(), "b": ch.empty_message()}, np.random.default_rng(0)
    )
    assert bits == 0.0
    assert set(inbox) == {"a", "b"}


def test_discrete_silence_and_bits() -> None:
    cfg = ChannelConfig(
        mode="discrete_learned", vocab_size=4, msg_len=2, bit_cost=1.0, allow_silence=True
    )
    ch = make_channel(cfg, ["ue_0", "bs_0"])
    ch.reset(np.random.default_rng(0))
    silence = np.array([ch.silence_id(), ch.silence_id()], dtype=np.float32)
    active = np.array([1.0, 2.0], dtype=np.float32)
    _, bits_s = ch.exchange({"ue_0": silence, "bs_0": silence}, np.random.default_rng(1))
    _, bits_a = ch.exchange({"ue_0": active, "bs_0": silence}, np.random.default_rng(1))
    assert bits_s == 0.0
    assert bits_a > 0.0


def test_erasure_and_corruption() -> None:
    cfg = ChannelConfig(
        mode="discrete_learned",
        vocab_size=4,
        msg_len=2,
        erasure_p=1.0,
        corruption_p=0.0,
    )
    ch = make_channel(cfg, ["a", "b"])
    ch.reset(np.random.default_rng(0))
    inbox, _ = ch.exchange(
        {"a": np.array([1.0, 1.0]), "b": np.array([2.0, 2.0])},
        np.random.default_rng(0),
    )
    assert np.all(inbox["a"] == ch.silence_id())


def test_delay_buffer() -> None:
    cfg = ChannelConfig(
        mode="discrete_learned", vocab_size=4, msg_len=2, delay=1, erasure_p=0.0
    )
    ch = make_channel(cfg, ["a", "b"])
    ch.reset(np.random.default_rng(0))
    msg = np.array([1.0, 2.0])
    inbox1, _ = ch.exchange({"a": msg, "b": np.zeros(2)}, np.random.default_rng(0))
    assert inbox1["b"].shape == (2,)
    inbox2, _ = ch.exchange({"a": msg, "b": np.zeros(2)}, np.random.default_rng(0))
    assert inbox2["b"].shape == (2,)


def test_gumbel_and_hard_execution() -> None:
    logits = torch.randn(2, 3, 5)
    soft, ids = discrete_message_from_logits(logits, tau=1.0, hard=True)
    assert soft.shape == (2, 3, 5)
    assert ids.shape == (2, 3)
    y = gumbel_softmax_sample(logits.view(-1, 5), hard=True)
    assert y.shape == (6, 5)


def test_attention_and_graph() -> None:
    attn = AttentionTargeter(8, 3)
    h = torch.randn(2, 8)
    recv = torch.randn(2, 3, 8)
    w = attn(h, recv)
    assert w.shape == (2, 3)
    assert torch.allclose(w.sum(-1), torch.ones(2), atol=1e-5)
    g = GraphMessageRouter(8)
    msgs = torch.randn(2, 3, 8)
    adj = torch.eye(3)
    out = g(msgs, adj)
    assert out.shape == (2, 3, 8)


def test_malformed_message() -> None:
    cfg = ChannelConfig(mode="discrete_learned", msg_len=2)
    with pytest.raises(ValueError):
        validate_message_shape(None, cfg)
    with pytest.raises(ValueError):
        validate_message_shape(np.zeros((2, 2)), cfg)
    with pytest.raises(ValueError):
        validate_message_shape(np.zeros(5), cfg)
    ok = validate_message_shape(np.zeros(2), cfg)
    assert ok.shape == (2,)


def test_targeted_routing() -> None:
    cfg = ChannelConfig(
        mode="discrete_learned", vocab_size=4, msg_len=2, targeted=True
    )
    agents = ["ue_0", "bs_0", "edge_0"]
    ch = make_channel(cfg, agents)
    ch.reset(np.random.default_rng(0))
    outbound = {
        "ue_0": np.array([1.0, 1.0]),
        "bs_0": np.zeros(2),
        "edge_0": np.zeros(2),
    }
    targets = {"ue_0": "bs_0", "bs_0": None, "edge_0": None}
    inbox, bits = ch.exchange(outbound, np.random.default_rng(0), targets=targets)
    assert bits >= 0.0
    assert "bs_0" in inbox
