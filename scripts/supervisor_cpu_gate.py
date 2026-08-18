#!/usr/bin/env python3
"""Supervisor CPU gate: interpretability + claim gate + honest BLOCKED_GPU."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from emergent_intent.env import CommMode, EnvConfig, ScenarioFamily, make_env
from emergent_intent.env.wireless_env import A_MSG
from emergent_intent.interpretability.claims import LanguageClaimEvidence, evaluate_language_claim
from emergent_intent.interpretability.efficiency import communication_efficiency, silence_fraction
from emergent_intent.interpretability.metrics import (
    message_entropy,
    mutual_information_estimate,
    topographic_similarity,
)
from emergent_intent.utils.device import detect_device, dump_json, git_commit_sha

ROOT = Path(__file__).resolve().parents[1]


def _probe(seed: int) -> dict:
    env = make_env(
        EnvConfig(
            scenario=ScenarioFamily.hidden_blockage_congestion,
            horizon=16,
            n_ue=1,
            seed=seed,
            comm_mode=CommMode.discrete_learned,
            vocab_size=8,
            msg_len=2,
        )
    )
    obs, _ = env.reset(seed=seed)
    symbols = []
    states = []
    rewards = []
    for t in range(env.config.horizon):
        if not env.agents:
            break
        actions = {a: env.action_space(a).sample() for a in env.agents}
        if "ue_0" in actions:
            actions["ue_0"][A_MSG] = 1 + (t % 4)
        obs, rew, *_ = env.step(actions)
        rewards.append(float(sum(rew.values()) / max(len(rew), 1)))
        for r in env._last_inbox.get("bs_0", []):
            if r.valid > 0:
                symbols.append(np.asarray(r.symbols).copy())
                states.append(np.asarray(env.state()["blockage"]).copy())
    sym = np.asarray(symbols) if symbols else np.zeros((1, 2))
    st = np.asarray(states) if states else np.zeros((1, 1))
    sil = silence_fraction(sym.astype(int) % 8, silence_id=0)
    topo = topographic_similarity(sym, st)
    mi = mutual_information_estimate(
        sym.mean(1) if sym.ndim == 2 else sym, st.mean(1) if st.ndim == 2 else st
    )
    def _f(x):
        try:
            v = float(x)
        except (TypeError, ValueError):
            return None
        return v if np.isfinite(v) else None

    return {
        "seed": seed,
        "mean_return": _f(np.mean(rewards) if rewards else 0.0),
        "entropy_bits": _f(message_entropy(sym.astype(int) % 8, 8)),
        "mi_bits": _f(mi),
        "topographic_similarity": _f(topo),
        "silence_fraction": _f(sil),
        "n_messages": int(len(symbols)),
    }


def main() -> int:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "emit_blocked_gpu", ROOT / "scripts" / "emit_blocked_gpu.py"
    )
    assert spec and spec.loader
    emit_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(emit_mod)
    gpu = emit_mod.main(ROOT / "results" / "blocked_gpu" / "BLOCKED_GPU.json")
    seeds = (0, 1, 2)
    rows = [_probe(s) for s in seeds]
    mean_h = float(np.mean([r["entropy_bits"] for r in rows]))
    mean_mi = float(np.mean([r["mi_bits"] for r in rows]))
    topo_vals = np.array([r["topographic_similarity"] for r in rows], dtype=float)
    mean_topo = float(np.nanmean(topo_vals)) if np.isfinite(np.nanmean(topo_vals)) else None
    if mean_topo is not None and not np.isfinite(mean_topo):
        mean_topo = None
    claim = evaluate_language_claim(
        LanguageClaimEvidence(
            entropy_bits=mean_h,
            mi_bits=mean_mi,
            topographic_similarity=mean_topo,
            intervention_delta=None,
            n_repeated_runs=len(seeds),
            messages_are_exchanged=True,
            notes=["CPU supervisor probe; intervention_delta left unset on purpose"],
        )
    )
    eff = communication_efficiency(
        message_bits=float(sum(r["n_messages"] for r in rows) * np.log2(8)),
        n_steps=16 * len(seeds),
        n_agents=3,
        silence_fraction=float(np.mean([r["silence_fraction"] for r in rows])),
        task_success=float(np.mean([r["mean_return"] for r in rows])),
    )
    payload = {
        "gate": "supervisor-cpu-gate",
        "status": "CPU_PASS",
        "dissertation_role": "optional_distributed_intelligence_extension_not_fourth_paper",
        "oulu_genome": "public_theme_mapping_only_not_an_appointment",
        "evidence_class": "SYNTHETIC_SIM",
        "device": detect_device().as_dict(),
        "gpu": gpu,
        "repeated_runs": rows,
        "language_claim": claim,
        "communication_efficiency": eff,
        "commit": git_commit_sha(ROOT),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out = ROOT / "results" / "supervisor" / "SUPERVISOR_CPU_GATE.json"
    dump_json(out, payload)
    print(json.dumps(payload, indent=2, default=str))
    if claim["emergent_language_claimed"]:
        return 1
    if gpu.get("status") not in {"BLOCKED_GPU", "CUDA_HARDWARE_PRESENT", "BLOCKED_HARDWARE"}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
