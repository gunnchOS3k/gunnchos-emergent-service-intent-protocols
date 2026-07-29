#!/usr/bin/env python3
"""Generate a minimal CycloneDX-like SBOM JSON for this package."""

from __future__ import annotations

import importlib.metadata as md
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKGS = ["numpy", "torch", "gymnasium", "pettingzoo", "pydantic", "pyyaml", "jsonschema", "scipy"]


def main() -> None:
    comps = []
    for name in PKGS:
        try:
            ver = md.version(name)
        except md.PackageNotFoundError:
            ver = "NOT_INSTALLED"
        comps.append({"type": "library", "name": name, "version": ver})
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "component": {"type": "application", "name": "emergent-intent", "version": "0.1.0"},
        },
        "components": comps,
    }
    out = ROOT / "artifact" / "sbom.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sbom, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
