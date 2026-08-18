"""Language-claim gate: messages ≠ emergent language.

A supervisor-ready run may report entropy, MI estimates, topographic
similarity, and intervention deltas. None of those, alone or together,
authorize the phrase "emergent language" in this repository.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


FORBIDDEN_PHRASE = "emergent language"
ALLOWED_WHEN_COMPLETE = "structured_signaling_candidate"
MESSAGES_ONLY = "messages_exchanged_only"


@dataclass
class LanguageClaimEvidence:
    entropy_bits: float | None = None
    mi_bits: float | None = None
    topographic_similarity: float | None = None
    intervention_delta: float | None = None
    n_repeated_runs: int = 0
    messages_are_exchanged: bool = False
    notes: list[str] = field(default_factory=list)


def _present(value: float | None) -> bool:
    return value is not None and value == value  # not None / not NaN


def evaluate_language_claim(ev: LanguageClaimEvidence) -> dict[str, Any]:
    """Return a fail-closed claim record.

    Completeness of interpretability evidence can raise the *descriptive*
    status to ``structured_signaling_candidate``. It never sets
    ``emergent_language_claimed``.
    """
    missing: list[str] = []
    if not _present(ev.entropy_bits):
        missing.append("entropy_bits")
    if not _present(ev.mi_bits):
        missing.append("mi_bits")
    if not _present(ev.topographic_similarity):
        missing.append("topographic_similarity")
    if not _present(ev.intervention_delta):
        missing.append("intervention_delta")
    if ev.n_repeated_runs < 3:
        missing.append("n_repeated_runs>=3")

    complete = not missing
    if ev.messages_are_exchanged and not complete:
        descriptive = MESSAGES_ONLY
    elif complete:
        descriptive = ALLOWED_WHEN_COMPLETE
    else:
        descriptive = "insufficient_evidence"

    return {
        "emergent_language_claimed": False,
        "forbidden_phrase": FORBIDDEN_PHRASE,
        "descriptive_status": descriptive,
        "interpretability_complete": complete,
        "missing": missing,
        "entropy_bits": ev.entropy_bits,
        "mi_bits": ev.mi_bits,
        "topographic_similarity": ev.topographic_similarity,
        "intervention_delta": ev.intervention_delta,
        "n_repeated_runs": ev.n_repeated_runs,
        "messages_are_exchanged": ev.messages_are_exchanged,
        "warning": (
            "Do not claim emergent language because agents send messages. "
            "Require entropy, MI, topographic similarity, interventions, "
            "and repeated runs; even then the allowed wording is "
            f"{ALLOWED_WHEN_COMPLETE}, not {FORBIDDEN_PHRASE}."
        ),
        "notes": ev.notes,
        "evidence_class": "SYNTHETIC_SIM",
    }


class LanguageClaimGate:
    """Convenience wrapper used by tests and the supervisor CPU gate."""

    def evaluate(self, ev: LanguageClaimEvidence) -> dict[str, Any]:
        return evaluate_language_claim(ev)

    def assert_not_emergent_language(self, text: str) -> None:
        lowered = text.lower()
        if FORBIDDEN_PHRASE in lowered and "not claim" not in lowered and "do not claim" not in lowered:
            # Allow documentation that forbids the claim.
            if "without" in lowered or "never" in lowered or "do not" in lowered:
                return
            raise AssertionError(
                f"Text claims '{FORBIDDEN_PHRASE}' without a prohibition: {text[:200]}"
            )
