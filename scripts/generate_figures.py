#!/usr/bin/env python3
"""Generate figures from validated result files (no fabricated curves)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    fig_dir = ROOT / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pilot = ROOT / "results" / "pilot" / "pilot_summary.json"
    if not pilot.exists():
        # minimal placeholder figure stating missing data (not a fake result)
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No pilot_summary.json — run make final-experiments", ha="center")
        ax.axis("off")
        fig.savefig(fig_dir / "missing_pilot.png", dpi=120)
        print("wrote missing_pilot.png")
        return

    payload = json.loads(pilot.read_text())
    rows = [r for r in payload.get("rows", []) if r.get("status") == "SUCCESS" and "mean_return" in r]
    by = {}
    for r in rows:
        key = str(r.get("algorithm"))
        by.setdefault(key, []).append(float(r["mean_return"]))

    methods = sorted(by)
    means = [float(np.mean(by[m])) for m in methods]
    stds = [float(np.std(by[m])) for m in methods]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(methods)), means, yerr=stds, capsize=3, color="#2f5d8a")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean return (pilot)")
    ax.set_title("Pilot returns by algorithm (SYNTHETIC_SIM)")
    fig.tight_layout()
    fig.savefig(fig_dir / "pilot_returns.png", dpi=140)
    plt.close(fig)

    # Pareto scatter if present
    pf = ROOT / "results" / "pilot" / "multiobj" / "pareto_front.json"
    if pf.exists():
        data = json.loads(pf.read_text())
        pts = data.get("pareto_front", [])
        if pts:
            xs = [p.get("task_success", 0) for p in pts]
            ys = [p.get("fairness", 0) for p in pts]
            fig, ax = plt.subplots()
            ax.scatter(xs, ys, c="#b35c1e")
            ax.set_xlabel("task_success")
            ax.set_ylabel("fairness")
            ax.set_title("Pilot Pareto front (subset)")
            fig.tight_layout()
            fig.savefig(fig_dir / "pareto_front.png", dpi=140)
            plt.close(fig)
    print("figures_ok")


if __name__ == "__main__":
    main()
