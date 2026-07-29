"""Build artifact package metadata and checksums."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from emergent_intent.utils import detect_device, dump_json, file_sha256, git_commit_sha


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    art = ROOT / "artifact"
    art.mkdir(parents=True, exist_ok=True)
    results = ROOT / "results" / "smoke"
    checksums = {}
    if results.exists():
        for p in sorted(results.rglob("*")):
            if p.is_file():
                checksums[str(p.relative_to(ROOT))] = file_sha256(p)
    dump_json(art / "checksums.json", checksums)

    expected = {
        "smoke_summary_table": "results/smoke/smoke_summary_table.json",
        "primary_seed_manifests": "results/smoke/smoke_ippo_seed*.json",
        "gpu_gate": "results/smoke/gate4_gpu_blocked.json (when no CUDA)",
        "evidence_class": "SYNTHETIC_SIM",
    }
    dump_json(art / "EXPECTED_OUTPUTS.json", expected)

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "commit": git_commit_sha(ROOT),
        "device": detect_device().as_dict(),
        "statuses": {
            "RELEASE_CANDIDATE_READY": True,
            "DOI_PENDING": True,
            "INDEPENDENT_REPRODUCTION_PENDING": True,
            "PEER_REVIEW_PENDING": True,
        },
    }
    dump_json(art / "BUILD_META.json", meta)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
