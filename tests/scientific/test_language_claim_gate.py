"""Language-claim gate: exchanging messages is not an emergent language."""

from __future__ import annotations

import numpy as np

from emergent_intent.interpretability import analyze_messages
from emergent_intent.interpretability.claims import (
    LanguageClaimEvidence,
    LanguageClaimGate,
    evaluate_language_claim,
)


def test_messages_alone_do_not_claim_emergent_language() -> None:
    ev = LanguageClaimEvidence(messages_are_exchanged=True, n_repeated_runs=1)
    out = evaluate_language_claim(ev)
    assert out["emergent_language_claimed"] is False
    assert out["descriptive_status"] == "messages_exchanged_only"
    assert "entropy_bits" in out["missing"]
    assert "mi_bits" in out["missing"]
    assert "topographic_similarity" in out["missing"]
    assert "intervention_delta" in out["missing"]


def test_complete_interpretability_still_forbids_language_phrase() -> None:
    ev = LanguageClaimEvidence(
        entropy_bits=1.2,
        mi_bits=0.4,
        topographic_similarity=0.3,
        intervention_delta=-0.2,
        n_repeated_runs=5,
        messages_are_exchanged=True,
    )
    out = evaluate_language_claim(ev)
    assert out["interpretability_complete"] is True
    assert out["emergent_language_claimed"] is False
    assert out["descriptive_status"] == "structured_signaling_candidate"
    assert "emergent language" not in out["descriptive_status"]


def test_analyze_messages_attaches_fail_closed_claim() -> None:
    msgs = np.array([[1, 2], [1, 3], [2, 3], [1, 2]])
    report = analyze_messages(msgs, vocab_size=8)
    assert report["language_claim"]["emergent_language_claimed"] is False
    assert "do not claim emergent language" in report["warning"].lower()


def test_gate_allows_prohibition_text() -> None:
    LanguageClaimGate().assert_not_emergent_language(
        "Do not claim emergent language without interpretability."
    )
