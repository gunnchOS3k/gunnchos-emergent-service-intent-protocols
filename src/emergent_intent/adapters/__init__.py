"""Soft adapters for sibling research repos — never hard-fail if absent."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
SIBLINGS_ROOT = REPO_ROOT.parent

SIBLING_MAP = {
    "edge_io": "edge-io-measurement-node",
    "digital_twin": "7gc-digital-twin",
    "ntn_sim": "ntn-resilience-sim",
    "spectrumx": "spectrumx-ai-ran-gary",
}


@dataclass
class AdapterStatus:
    name: str
    path: str
    available: bool
    detail: str


def sibling_path(key: str) -> Path:
    return SIBLINGS_ROOT / SIBLING_MAP[key]


def probe_siblings() -> dict[str, AdapterStatus]:
    out: dict[str, AdapterStatus] = {}
    for key, dirname in SIBLING_MAP.items():
        p = SIBLINGS_ROOT / dirname
        available = p.is_dir()
        out[key] = AdapterStatus(
            name=key,
            path=str(p),
            available=available,
            detail="present" if available else "absent — soft adapter disabled",
        )
    return out


def _try_import_path(module_name: str, path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception as exc:  # noqa: BLE001 — soft fail
        return {"error": str(exc)}


class EdgeIOAdapter:
    """Optional measurement-node hooks (synthetic fallback)."""

    def __init__(self) -> None:
        self.root = sibling_path("edge_io")
        self.available = self.root.is_dir()

    def sample_rssi(self) -> dict[str, Any]:
        if not self.available:
            return {"source": "fallback_synthetic", "rssi_dbm": -70.0, "available": False}
        return {"source": "sibling_stub", "rssi_dbm": -65.0, "available": True, "path": str(self.root)}


class DigitalTwinAdapter:
    def __init__(self) -> None:
        self.root = sibling_path("digital_twin")
        self.available = self.root.is_dir()

    def twin_state_hint(self) -> dict[str, Any]:
        if not self.available:
            return {"source": "fallback_synthetic", "available": False}
        return {"source": "sibling_stub", "available": True, "path": str(self.root)}


class NTNSimAdapter:
    def __init__(self) -> None:
        self.root = sibling_path("ntn_sim")
        self.available = self.root.is_dir()

    def failover_prior(self) -> dict[str, Any]:
        if not self.available:
            return {"source": "fallback_synthetic", "ntn_availability": 0.5, "available": False}
        return {
            "source": "sibling_stub",
            "ntn_availability": 0.8,
            "available": True,
            "path": str(self.root),
        }


class SpectrumXAdapter:
    def __init__(self) -> None:
        self.root = sibling_path("spectrumx")
        self.available = self.root.is_dir()

    def spectrum_prior(self) -> dict[str, Any]:
        if not self.available:
            return {"source": "fallback_synthetic", "occupancy": 0.4, "available": False}
        return {
            "source": "sibling_stub",
            "occupancy": 0.35,
            "available": True,
            "path": str(self.root),
        }


def load_all_adapters() -> dict[str, Any]:
    return {
        "status": {k: v.__dict__ for k, v in probe_siblings().items()},
        "edge_io": EdgeIOAdapter(),
        "digital_twin": DigitalTwinAdapter(),
        "ntn_sim": NTNSimAdapter(),
        "spectrumx": SpectrumXAdapter(),
    }
