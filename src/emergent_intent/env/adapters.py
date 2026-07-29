from __future__ import annotations

from pathlib import Path
from typing import Any

REPOS = Path(__file__).resolve().parents[4]


def try_load_edge_io_schema() -> dict[str, Any]:
    p = REPOS / "edge-io-measurement-node"
    return {"available": p.is_dir(), "path": str(p), "role": "radio_network_telemetry"}


def try_load_digital_twin() -> dict[str, Any]:
    p = REPOS / "7gc-digital-twin"
    return {"available": p.is_dir(), "path": str(p), "role": "geometry_mobility"}


def try_load_ntn() -> dict[str, Any]:
    p = REPOS / "ntn-resilience-sim"
    return {"available": p.is_dir(), "path": str(p), "role": "tn_ntn_failures"}


def try_load_spectrumx() -> dict[str, Any]:
    p = REPOS / "spectrumx-ai-ran-gary"
    return {"available": p.is_dir(), "path": str(p), "role": "policy_context"}


def multimodal_adapter_status() -> dict[str, Any]:
    return {
        "radio": try_load_edge_io_schema(),
        "geometry": try_load_digital_twin(),
        "ntn": try_load_ntn(),
        "policy": try_load_spectrumx(),
        "default_public_data": "synthetic_privacy_safe",
    }
