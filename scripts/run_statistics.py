#!/usr/bin/env python3
"""Aggregate statistics + interpretability outputs (§6.10)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from emergent_intent.env import EnvConfig, make_env
from emergent_intent.env.wireless_env import A_MSG
from emergent_intent.interpretability.metrics import (
    message_entropy,
    mutual_information_estimate,
    symbol_condition_matrix,
    symbol_utilization,
    topographic_similarity,
)
from emergent_intent.stats import summarize_seed_matrix
from emergent_intent.utils import dump_json


ROOT = Path(__file__).resolve().parents[1]


def collect_rows(pilot_dir: Path) -> list[dict]:
    rows = []
    summary = pilot_dir / "pilot_summary.json"
    if summary.exists():
        payload = json.loads(summary.read_text())
        rows.extend(payload.get("rows", []))
    for path in pilot_dir.glob("*.json"):
        if path.name in ("pilot_summary.json",):
            continue
        try:
            rows.append(json.loads(path.read_text()))
        except Exception:
            continue
    return rows


def interpretability_probe(out: Path) -> dict:
    env = make_env(
        EnvConfig(
            horizon=20,
            n_ue=1,
            seed=0,
            channel={"mode": "discrete_learned", "vocab_size": 8, "msg_length": 2},
        )
    )
    obs, _ = env.reset(seed=0)
    symbols = []
    states = []
    conditions = []
    for t in range(env.config.horizon):
        if not env.agents:
            break
        actions = {a: env.action_space(a).sample() for a in env.agents}
        actions["ue_0"][A_MSG] = 1 + (t % 4)
        obs, rewards, _, _, infos = env.step(actions)
        for r in env._last_inbox.get("bs_0", []):
            if r.valid > 0:
                symbols.append(r.symbols.copy())
                states.append(env.state()["blockage"].copy())
                conditions.append(int(env.state()["blockage"][0] > 0.5))
    sym = np.asarray(symbols) if symbols else np.zeros((1, 2))
    st = np.asarray(states) if states else np.zeros((1, 1))
    cond = np.asarray(conditions) if conditions else np.zeros(1)
    payload = {
        "evidence_class": "SYNTHETIC_SIM",
        "label": "PILOT",
        "message_entropy": message_entropy(sym.astype(int) % 8, 8),
        "symbol_utilization": symbol_utilization(sym.astype(int) % 8, 8),
        "mi_estimate_bits": mutual_information_estimate(sym.mean(1), cond.astype(float)),
        "topographic_similarity": topographic_similarity(sym, st),
        "symbol_condition_matrix": symbol_condition_matrix(sym.astype(int) % 8, cond, 8, 2).tolist(),
        "note": "Estimates only; do not claim emergent language.",
    }
    dump_json(out / "interpretability_probe.json", payload)
    return payload


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", type=Path, default=ROOT / "results" / "pilot")
    args = p.parse_args()
    rows = collect_rows(args.pilot)
    summary = summarize_seed_matrix(rows)
    dump_json(args.pilot / "statistics_summary.json", summary)
    # seed registry
    registry = [
        {"algorithm": r.get("algorithm"), "scenario": r.get("scenario"), "seed": r.get("seed"), "status": r.get("status")}
        for r in rows
    ]
    dump_json(args.pilot / "seed_registry.json", registry)
    interp_dir = ROOT / "results" / "interpretability"
    interp_dir.mkdir(parents=True, exist_ok=True)
    interpretability_probe(interp_dir)
    # Pareto already under pilot/multiobj; copy pointer
    print(json.dumps({"n_rows": len(rows), "methods": list(summary.get("methods", {}))}, indent=2))


if __name__ == "__main__":
    main()
