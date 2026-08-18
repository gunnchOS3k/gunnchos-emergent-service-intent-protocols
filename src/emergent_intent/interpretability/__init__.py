"""Interpretability analyses for learned messages (careful claims)."""

from __future__ import annotations

from typing import Any

import numpy as np


def symbol_counts(messages: np.ndarray, vocab_size: int) -> np.ndarray:
    """messages: (N, L) integer symbols."""
    flat = np.asarray(messages, dtype=np.int64).ravel()
    flat = flat[(flat >= 0) & (flat <= vocab_size)]
    counts = np.bincount(flat, minlength=vocab_size + 1)
    return counts


def message_entropy(messages: np.ndarray, vocab_size: int) -> float:
    counts = symbol_counts(messages, vocab_size).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def symbol_use_fraction(messages: np.ndarray, vocab_size: int) -> float:
    counts = symbol_counts(messages, vocab_size)
    used = int(np.sum(counts[:vocab_size] > 0))
    return float(used / max(1, vocab_size))


def mutual_information_estimate(
    symbols: np.ndarray,
    conditions: np.ndarray,
    n_sym_bins: int | None = None,
    n_cond_bins: int = 8,
) -> float:
    """Discrete MI estimate I(symbol; condition) via histograms (nats->bits)."""
    s = np.asarray(symbols, dtype=np.int64).ravel()
    c = np.asarray(conditions, dtype=np.float64).ravel()
    n = min(s.size, c.size)
    if n == 0:
        return 0.0
    s, c = s[:n], c[:n]
    if n_sym_bins is None:
        n_sym_bins = int(s.max()) + 1 if s.size else 1
    c_bin = np.clip(
        (c - c.min()) / (np.ptp(c) + 1e-8) * (n_cond_bins - 1e-6), 0, n_cond_bins - 1
    ).astype(int)
    joint = np.zeros((n_sym_bins, n_cond_bins), dtype=np.float64)
    for si, ci in zip(s, c_bin, strict=False):
        if 0 <= si < n_sym_bins:
            joint[si, ci] += 1.0
    joint /= joint.sum() + 1e-12
    p_s = joint.sum(1, keepdims=True)
    p_c = joint.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = joint / (p_s @ p_c + 1e-12)
        mi = np.nansum(joint * np.log2(ratio + 1e-12))
    return float(max(0.0, mi))


def topographic_similarity(messages: np.ndarray, meanings: np.ndarray) -> float:
    """Spearman correlation between pairwise message distances and meaning distances.

    Positive values suggest structure; do NOT alone justify an 'emergent language' claim.
    """
    from scipy.stats import spearmanr

    m = np.asarray(messages, dtype=np.float64)
    y = np.asarray(meanings, dtype=np.float64)
    if m.ndim == 1:
        m = m.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    n = min(len(m), len(y))
    m, y = m[:n], y[:n]
    if n < 3:
        return 0.0
    # pairwise distances (upper triangle)
    md, yd = [], []
    for i in range(n):
        for j in range(i + 1, n):
            md.append(np.linalg.norm(m[i] - m[j]))
            yd.append(np.linalg.norm(y[i] - y[j]))
    corr = spearmanr(md, yd).correlation
    return float(0.0 if corr is None or np.isnan(corr) else corr)


def intervene_symbol(
    messages: np.ndarray, position: int, new_symbol: int
) -> np.ndarray:
    out = np.array(messages, copy=True)
    out[..., position] = new_symbol
    return out


def latent_probe_r2(latents: np.ndarray, targets: np.ndarray) -> float:
    """Linear probe R^2 (least squares) for latent interpretability."""
    X = np.asarray(latents, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = min(len(X), len(y))
    X, y = X[:n], y[:n]
    if n < X.shape[1] + 2:
        return 0.0
    X = np.concatenate([X, np.ones((n, 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def symbol_condition_matrix(
    symbols: np.ndarray, conditions: np.ndarray, vocab_size: int, n_cond_bins: int = 5
) -> np.ndarray:
    """Matrix[symbol, condition_bin] counts for visualization."""
    s = np.asarray(symbols, dtype=np.int64).ravel()
    c = np.asarray(conditions, dtype=np.float64).ravel()
    n = min(s.size, c.size)
    mat = np.zeros((vocab_size, n_cond_bins), dtype=np.float64)
    if n == 0:
        return mat
    c_bin = np.clip(
        (c[:n] - c[:n].min()) / (np.ptp(c[:n]) + 1e-8) * (n_cond_bins - 1e-6), 0, n_cond_bins - 1
    ).astype(int)
    for si, ci in zip(s[:n], c_bin, strict=False):
        if 0 <= si < vocab_size:
            mat[si, ci] += 1.0
    return mat


def analyze_messages(
    messages: np.ndarray,
    conditions: np.ndarray | None = None,
    vocab_size: int = 8,
    meanings: np.ndarray | None = None,
) -> dict[str, Any]:
    """Bundle analyses with careful wording for claims."""
    report: dict[str, Any] = {
        "entropy_bits": message_entropy(messages, vocab_size),
        "symbol_use_fraction": symbol_use_fraction(messages, vocab_size),
        "claim_level": "descriptive_statistics_only",
        "warning": (
            "Do not claim emergent language without MI, topographic similarity, "
            "and intervention evidence together."
        ),
    }
    flat = np.asarray(messages, dtype=np.int64).ravel()
    if conditions is not None:
        report["mi_symbol_condition_bits"] = mutual_information_estimate(
            flat, conditions, n_sym_bins=vocab_size
        )
        report["symbol_condition_matrix"] = symbol_condition_matrix(
            flat, conditions, vocab_size
        ).tolist()
    if meanings is not None:
        report["topographic_similarity"] = topographic_similarity(messages, meanings)
    from emergent_intent.interpretability.claims import (
        LanguageClaimEvidence,
        evaluate_language_claim,
    )

    report["language_claim"] = evaluate_language_claim(
        LanguageClaimEvidence(
            entropy_bits=report.get("entropy_bits"),
            mi_bits=report.get("mi_symbol_condition_bits"),
            topographic_similarity=report.get("topographic_similarity"),
            intervention_delta=None,
            n_repeated_runs=0,
            messages_are_exchanged=True,
        )
    )
    return report


from emergent_intent.interpretability.claims import (  # noqa: E402
    LanguageClaimEvidence,
    LanguageClaimGate,
    evaluate_language_claim,
)
from emergent_intent.interpretability.efficiency import (  # noqa: E402
    bits_from_symbols,
    communication_efficiency,
    silence_fraction,
)

__all__ = [
    "LanguageClaimEvidence",
    "LanguageClaimGate",
    "analyze_messages",
    "bits_from_symbols",
    "communication_efficiency",
    "evaluate_language_claim",
    "intervene_symbol",
    "latent_probe_r2",
    "message_entropy",
    "mutual_information_estimate",
    "silence_fraction",
    "symbol_condition_matrix",
    "symbol_counts",
    "symbol_use_fraction",
    "topographic_similarity",
]
