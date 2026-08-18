#!/usr/bin/env python3
"""Emit honest BLOCKED_GPU JSON when CUDA is absent. Never invent timings."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from emergent_intent.utils.device import detect_device, git_commit_sha

ROOT = Path(__file__).resolve().parents[1]


def main(out: Path | None = None) -> dict:
    info = detect_device(prefer_cuda=True)
    out = out or (ROOT / "results" / "blocked_gpu" / "BLOCKED_GPU.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    if info.cuda_available:
        payload = {
            "status": "CUDA_HARDWARE_PRESENT",
            "aliases": [],
            "fail_closed": True,
            "gpu_timings_present": False,
            "numeric_gpu_claim": False,
            "device": info.as_dict(),
            "commit": git_commit_sha(ROOT),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "notes": "CUDA visible; this file is not a GPU-measured experiment PASS.",
        }
    else:
        payload = {
            "status": "BLOCKED_GPU",
            "aliases": ["BLOCKED_HARDWARE"],
            "fail_closed": True,
            "gpu_timings_present": False,
            "numeric_gpu_claim": False,
            "device": info.as_dict(),
            "commit": git_commit_sha(ROOT),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "notes": "No CUDA. Default evidence is SYNTHETIC_SIM / CPU-only. Never invent GPU timings.",
            "evidence_class": "SYNTHETIC_SIM",
        }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    main()
