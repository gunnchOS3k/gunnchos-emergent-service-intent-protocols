#!/usr/bin/env python3
"""Validate paper manuscript: fail on placeholders / missing required sections."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "RESOURCE_EFFICIENT_EMERGENT_SERVICE_INTENT_PROTOCOLS.md"

REQUIRED = [
    "Abstract",
    "Introduction",
    "Related work",
    "System model",
    "Doc-POMDP formulation",
    "Communication architecture",
    "State abstraction",
    "Multi-objective method",
    "Algorithms",
    "Experimental setup",
    "Main results",
    "Ablations",
    "Generalization",
    "Robustness",
    "Interpretability",
    "Negative results",
    "Limitations",
    "Ethics and privacy",
    "Reproducibility",
    "Conclusion",
]

PLACEHOLDER_PATTERNS = [
    r"TODO",
    r"TBD",
    r"FIXME",
    r"lorem ipsum",
    r"\[insert[^\]]*\]",
    r"XXX_PLACEHOLDER",
    r"\<placeholder\>",
]


def main() -> int:
    if not PAPER.exists():
        print("PAPER_MISSING", PAPER)
        return 1
    text = PAPER.read_text()
    missing = []
    for sec in REQUIRED:
        # accept "## N. Title" or "## Title"
        if not re.search(rf"^##\s+(\d+\.\s*)?{re.escape(sec)}\b", text, flags=re.I | re.M):
            missing.append(sec)
    bad = []
    for pat in PLACEHOLDER_PATTERNS:
        if re.search(pat, text, flags=re.I):
            bad.append(pat)
    if missing:
        print("MISSING_SECTIONS:", ", ".join(missing))
    if bad:
        print("PLACEHOLDERS_FOUND:", ", ".join(bad))
    if missing or bad:
        return 2
    print("PAPER_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
