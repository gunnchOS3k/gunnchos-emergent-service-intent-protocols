#!/usr/bin/env python3
"""Run CPU-feasible smoke + medium experiments across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emergent_intent.cli import run_smoke
from emergent_intent.utils import detect_device, dump_json, file_sha256


ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIGS = [
    ROOT / "configs/smoke/cpu_smoke.yaml",
    ROOT / "configs/smoke/cpu_smoke_mappo.yaml",
    ROOT / "configs/smoke/cpu_smoke_vdn.yaml",
    ROOT / "configs/smoke/cpu_smoke_dial.yaml",
    ROOT / "configs/smoke/cpu_smoke_faithful_dial.yaml",
    ROOT / "configs/smoke/medium_cpu.yaml",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--steps", type=int, default=None, help="override total_steps if set")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--out", type=Path, default=ROOT / "results" / "smoke")
    args = p.parse_args()
    info = detect_device(prefer_cuda=(args.device == "cuda"))
    if args.device == "cuda" and not info.cuda_available:
        payload = {
            "status": "BLOCKED_HARDWARE",
            "evidence_class": "BLOCKED",
            "device": info.as_dict(),
        }
        dump_json(args.out / "gate4_gpu_blocked.json", payload)
        print(json.dumps(payload, indent=2))
        raise SystemExit(2)

    args.out.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    # Final smoke table: 5 seeds on primary IPPO config
    primary = ROOT / "configs/smoke/cpu_smoke.yaml"
    for seed in range(args.seeds):
        result = run_smoke(primary, seed=seed, out_dir=args.out)
        summary_rows.append(
            {
                "config": str(primary.name),
                "seed": seed,
                "mean_return": result["metrics"].get("mean_return"),
                "eval_return": result["metrics"].get("eval_return"),
                "status": result["status"],
                "device": result["device"]["label"],
                "evidence_class": result["evidence_class"],
            }
        )

    # Additional algorithm smoke (single seed each) + medium
    for cfg in SMOKE_CONFIGS[1:]:
        result = run_smoke(cfg, seed=0, out_dir=args.out)
        summary_rows.append(
            {
                "config": str(cfg.name),
                "seed": 0,
                "mean_return": result["metrics"].get("mean_return"),
                "eval_return": result["metrics"].get("eval_return"),
                "status": result["status"],
                "device": result["device"]["label"],
                "evidence_class": result["evidence_class"],
            }
        )

    table = {
        "status": "SUCCESS",
        "evidence_class": "SYNTHETIC_SIM",
        "device": info.as_dict(),
        "n_primary_seeds": args.seeds,
        "rows": summary_rows,
    }
    dump_json(args.out / "smoke_summary_table.json", table)

    # checksums
    checksums = {}
    for path in sorted(args.out.glob("*.json")):
        checksums[path.name] = file_sha256(path)
    dump_json(args.out / "checksums.json", checksums)
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
