"""Device detection, seeding, config I/O, run manifests."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class DeviceInfo:
    device: str
    cuda_available: bool
    gpu_name: str | None = None
    evidence_label: str = "BLOCKED_HARDWARE"
    torch_version: str = ""
    platform: str = ""

    @property
    def label(self) -> str:
        return self.evidence_label

    def as_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "cuda_available": self.cuda_available,
            "gpu_name": self.gpu_name,
            "evidence_label": self.evidence_label,
            "label": self.label,
            "torch_version": self.torch_version,
            "platform": self.platform,
        }


@dataclass
class RunManifest:
    experiment_id: str = ""
    run_id: str = ""
    config_path: str = ""
    seed: int = 0
    status: str = "RUNNING"
    evidence_class: str = "SYNTHETIC_SIM"
    evidence_label: str = "SYNTHETIC_EXPERIMENT"
    device: dict[str, Any] | DeviceInfo | None = None
    commit: str | None = None
    started_at: str = ""
    created_at: str = ""
    finished_at: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id and self.experiment_id:
            self.run_id = self.experiment_id
        if not self.experiment_id and self.run_id:
            self.experiment_id = self.run_id
        if not self.created_at:
            self.created_at = self.started_at or datetime.now(timezone.utc).isoformat()
        if not self.started_at:
            self.started_at = self.created_at
        if isinstance(self.device, DeviceInfo):
            self.device = self.device.as_dict()

    def finalize(self, status: str, metrics: dict[str, Any] | None = None) -> None:
        self.status = status
        self.finished_at = datetime.now(timezone.utc).isoformat()
        if metrics:
            self.metrics.update(metrics)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        dump_json(path, self.to_dict())


def detect_device(prefer_cuda: bool = False) -> DeviceInfo:
    try:
        import torch

        tv = torch.__version__
        plat = f"{platform.system()} {platform.release()} {platform.machine()}"
        if prefer_cuda and torch.cuda.is_available():
            return DeviceInfo(
                device="cuda",
                cuda_available=True,
                gpu_name=torch.cuda.get_device_name(0),
                evidence_label="CUDA_AVAILABLE",
                torch_version=tv,
                platform=plat,
            )
        system = platform.system().lower()
        machine = platform.machine().lower()
        if system == "darwin" and ("arm" in machine or "aarch" in machine):
            label = "CPU_ONLY_APPLE_SILICON"
        else:
            label = "CPU_ONLY"
        return DeviceInfo(
            device="cpu",
            cuda_available=bool(torch.cuda.is_available()),
            evidence_label=label,
            torch_version=tv,
            platform=plat,
        )
    except Exception:
        return DeviceInfo(device="cpu", cuda_available=False, evidence_label="BLOCKED_HARDWARE")


def torch_device(prefer_cuda: bool = False):
    import torch

    info = detect_device(prefer_cuda=prefer_cuda)
    return torch.device(info.device)


def get_device(prefer_cuda: bool = False):
    return torch_device(prefer_cuda=prefer_cuda)


def cuda_status() -> dict:
    info = detect_device(prefer_cuda=True)
    return {
        "torch_cuda": info.cuda_available,
        "device_count": 1 if info.cuda_available else 0,
        "gpu_name": info.gpu_name,
        "label": info.evidence_label if info.cuda_available else "BLOCKED_HARDWARE",
    }


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def git_commit_sha(cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "UNCOMMITTED"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def load_yaml(path: Path | str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def dump_json(path: Path | str, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def new_manifest(
    experiment_id: str | None = None,
    config_path: str = "",
    seed: int = 0,
    evidence_class: str = "SYNTHETIC_SIM",
    status: str = "RUNNING",
    run_id: str | None = None,
    config: dict | None = None,
) -> RunManifest:
    eid = experiment_id or run_id or "run"
    info = detect_device()
    return RunManifest(
        experiment_id=eid,
        run_id=eid,
        config_path=str(config_path),
        seed=seed,
        status=status,
        evidence_class=evidence_class,
        evidence_label="SYNTHETIC_EXPERIMENT" if evidence_class == "SYNTHETIC_SIM" else evidence_class,
        device=info.as_dict(),
        commit=git_commit_sha(),
        started_at=datetime.now(timezone.utc).isoformat(),
        config=config or {},
        env={
            "python": platform.python_version(),
            "cwd": os.getcwd(),
            "HOSTNAME": os.environ.get("HOSTNAME", platform.node()),
        },
    )


def manifest_to_dict(m: RunManifest) -> dict[str, Any]:
    return m.to_dict()
