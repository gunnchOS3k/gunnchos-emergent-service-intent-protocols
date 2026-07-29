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

    # Prefer FINAL summary when present; else pilot.
    sources = [
        ROOT / "results" / "final" / "summary.json",
        ROOT / "results" / "pilot" / "pilot_summary.json",
    ]
    payload = None
    label = "pilot"
    for src in sources:
        if src.exists():
            payload = json.loads(src.read_text())
            label = "final" if "final" in str(src) else "pilot"
            break
    if payload is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No summary.json — run make final-experiments", ha="center")
        ax.axis("off")
        fig.savefig(fig_dir / "missing_pilot.png", dpi=120)
        print("wrote missing_pilot.png")
        return

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
    ax.set_ylabel(f"Mean return ({label})")
    ax.set_title(f"{label.title()} returns by algorithm (SYNTHETIC_SIM)")
    fig.tight_layout()
    fig.savefig(fig_dir / "pilot_returns.png", dpi=140)
    fig.savefig(fig_dir / f"{label}_returns.png", dpi=140)
    plt.close(fig)

    # Intervention figure if present
    inter = ROOT / "results" / "interventions" / "summary.json"
    if inter.exists():
        ip = json.loads(inter.read_text())
        irows = [r for r in ip.get("rows", []) if r.get("status") == "SUCCESS" and "mean_return" in r]
        iby = {}
        for r in irows:
            k = str(r.get("algorithm", "")).split("/")[-1]
            iby.setdefault(k, []).append(float(r["mean_return"]))
        if iby:
            keys = sorted(iby)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(range(len(keys)), [float(np.mean(iby[k])) for k in keys], color="#4a6b4a")
            ax.set_xticks(range(len(keys)))
            ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Mean return")
            ax.set_title("Semantic interventions (SYNTHETIC_SIM)")
            fig.tight_layout()
            fig.savefig(fig_dir / "semantic_interventions.png", dpi=140)
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
