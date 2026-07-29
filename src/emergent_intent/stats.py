"""Statistics helpers for pilot/smoke summaries (§6.10)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def mean_std(xs: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(list(xs), dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1) if arr.size > 1 else 0.0), "n": int(arr.size)}


def ci95(xs: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(list(xs), dtype=float)
    stats = mean_std(arr)
    if stats["n"] < 2:
        return {**stats, "ci95_low": stats["mean"], "ci95_high": stats["mean"]}
    se = stats["std"] / np.sqrt(stats["n"])
    # normal approx
    return {
        **stats,
        "ci95_low": float(stats["mean"] - 1.96 * se),
        "ci95_high": float(stats["mean"] + 1.96 * se),
    }


def bootstrap_ci(xs: Sequence[float], n_boot: int = 500, seed: int = 0) -> dict[str, float]:
    arr = np.asarray(list(xs), dtype=float)
    if arr.size == 0:
        return {"boot_low": float("nan"), "boot_high": float("nan")}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means.append(sample.mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"boot_low": float(lo), "boot_high": float(hi), "boot_mean": float(np.mean(means))}


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    pooled = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def summarize_seed_matrix(rows: list[dict[str, Any]], value_key: str = "mean_return") -> dict[str, Any]:
    by_method: dict[str, list[float]] = {}
    failed = 0
    for r in rows:
        if r.get("status") not in (None, "SUCCESS", "ok"):
            failed += 1
            continue
        m = r.get("algorithm") or r.get("config") or "unknown"
        if r.get(value_key) is None:
            failed += 1
            continue
        by_method.setdefault(str(m), []).append(float(r[value_key]))
    summary = {}
    for m, vals in by_method.items():
        s = ci95(vals)
        s.update(bootstrap_ci(vals))
        s["seeds"] = len(vals)
        s["final_comparison_eligible"] = len(vals) >= 5
        summary[m] = s
    return {
        "methods": summary,
        "failed_runs": failed,
        "evidence_class": "SYNTHETIC_SIM",
        "note": "Pilot/smoke statistics only; not final scientific evidence unless seeds≥5 and causal gates pass.",
    }
