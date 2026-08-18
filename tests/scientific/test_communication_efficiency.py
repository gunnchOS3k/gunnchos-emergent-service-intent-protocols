"""Communication-efficiency metrics stay labeled synthetic."""

from __future__ import annotations

import numpy as np
import pytest

from emergent_intent.interpretability.efficiency import (
    bits_from_symbols,
    communication_efficiency,
    silence_fraction,
)


def test_silence_and_bits() -> None:
    symbols = np.array([0, 0, 3, 4, 0])
    assert silence_fraction(symbols, silence_id=0) == pytest.approx(0.6)
    bits = bits_from_symbols(symbols, vocab_size=16, silence_id=0)
    assert bits == pytest.approx(2 * np.log2(16))


def test_efficiency_vs_baselines() -> None:
    out = communication_efficiency(
        message_bits=32.0,
        n_steps=16,
        n_agents=3,
        silence_fraction=0.4,
        task_success=0.8,
        no_comm_success=0.5,
        fixed_protocol_success=0.7,
        vocab_size=16,
        msg_len=2,
    )
    assert out["evidence_class"] == "SYNTHETIC_SIM"
    assert out["delta_success_vs_no_comm"] == pytest.approx(0.3)
    assert out["delta_success_vs_fixed_protocol"] == pytest.approx(0.1)
    assert "not" in out["note"].lower()
    assert "emergent-language claim" in out["note"].lower()
    assert out["bits_per_step"] == pytest.approx(2.0)
