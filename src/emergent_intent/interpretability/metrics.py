from __future__ import annotations

import numpy as np


def message_entropy(symbols: np.ndarray, vocab_size: int) -> float:
    if symbols.size == 0:
        return 0.0
    counts = np.bincount(symbols.astype(int).ravel(), minlength=vocab_size).astype(float)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def symbol_utilization(symbols: np.ndarray, vocab_size: int) -> float:
    if symbols.size == 0:
        return 0.0
    used = len(np.unique(symbols.astype(int)))
    return float(used / max(vocab_size, 1))


def mutual_information_estimate(x: np.ndarray, y: np.ndarray, bins: int = 8) -> float:
    """Histogram MI estimate (bits). Careful wording: estimate only."""
    if x.size == 0 or y.size == 0:
        return 0.0
    x = x.astype(float).ravel()
    y = y.astype(float).ravel()
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    c_xy, _, _ = np.histogram2d(x, y, bins=bins)
    p_xy = c_xy / c_xy.sum()
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
    return float(mi)


def topographic_similarity(messages: np.ndarray, states: np.ndarray) -> float:
    """Spearman-like structural similarity between message and state distances."""
    if len(messages) < 3:
        return float("nan")
    from itertools import combinations

    def dist(a, b):
        return float(np.linalg.norm(a - b))

    md, sd = [], []
    for i, j in combinations(range(len(messages)), 2):
        md.append(dist(messages[i], messages[j]))
        sd.append(dist(states[i], states[j]))
    md = np.asarray(md)
    sd = np.asarray(sd)
    if md.std() == 0 or sd.std() == 0:
        return 0.0
    return float(np.corrcoef(md, sd)[0, 1])


def intervene_message(symbols: np.ndarray, replace_with: int) -> np.ndarray:
    out = symbols.copy()
    out[:] = replace_with
    return out


def symbol_condition_matrix(symbols: np.ndarray, conditions: np.ndarray, vocab_size: int, n_cond: int) -> np.ndarray:
    mat = np.zeros((vocab_size, n_cond), dtype=float)
    for s, c in zip(symbols.ravel(), conditions.ravel()):
        si = int(s) % vocab_size
        ci = int(c) % n_cond
        mat[si, ci] += 1
    row = mat.sum(axis=1, keepdims=True) + 1e-8
    return mat / row
