"""Entropy, MI, topographic similarity, and interventions on CPU."""

from __future__ import annotations

import numpy as np

from emergent_intent.interpretability import (
    analyze_messages,
    message_entropy,
    mutual_information_estimate,
    topographic_similarity,
)
from emergent_intent.interpretability.metrics import symbol_utilization


def test_entropy_uniform_vs_constant() -> None:
    vocab = 8
    uniform = np.tile(np.arange(vocab), 16)
    constant = np.zeros(128, dtype=int)
    h_u = message_entropy(uniform, vocab)
    h_c = message_entropy(constant, vocab)
    assert h_u > 2.5
    assert h_c == 0.0
    assert symbol_utilization(uniform, vocab) == 1.0


def test_mi_and_topographic_similarity_are_estimates() -> None:
    rng = np.random.default_rng(0)
    states = rng.normal(size=(24, 2))
    # Messages that copy a quantized state coordinate → positive structure.
    messages = np.stack(
        [
            (np.abs(np.round(states[:, 0])).astype(int) % 8),
            (np.abs(np.round(states[:, 1])).astype(int) % 8),
        ],
        axis=1,
    )
    topo = topographic_similarity(messages, states)
    mi = mutual_information_estimate(messages[:, 0], states[:, 0])
    assert topo == topo  # not NaN
    assert mi >= 0.0
    report = analyze_messages(messages.astype(int), conditions=states[:, 0], meanings=states, vocab_size=16)
    assert "mi_symbol_condition_bits" in report
    assert "topographic_similarity" in report
    assert report["language_claim"]["emergent_language_claimed"] is False
