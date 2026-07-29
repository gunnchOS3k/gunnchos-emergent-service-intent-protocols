"""Deterministic rule parser and optional LLM stub (never controls radio)."""

from __future__ import annotations

import re

from emergent_intent.intent.schema import ServiceIntent


class RuleBasedIntentParser:
    """Deterministic keyword/rule parser (required path; no LLM)."""

    CLASS_KEYWORDS = {
        "urllc": "URLLC",
        "critical": "URLLC",
        "ultra-reliable": "URLLC",
        "embb": "eMBB",
        "broadband": "eMBB",
        "mmtc": "mMTC",
        "iot": "mMTC",
        "education": "education",
        "fairness": "education",
        "school": "education",
    }

    def parse(self, text: str, service_id: str = "parsed-01") -> ServiceIntent:
        low = text.lower()
        service_class = "best_effort"
        for kw, cls in self.CLASS_KEYWORDS.items():
            if kw in low:
                service_class = cls
                break

        lat = 50.0
        m = re.search(r"(\d+(?:\.\d+)?)\s*ms", low)
        if m:
            lat = float(m.group(1))
        elif "ultra" in low or "critical" in low:
            lat = 5.0

        priority = 3
        if service_class == "URLLC":
            priority = 5
        elif service_class == "education":
            priority = 2

        reliability = 0.999 if service_class == "URLLC" else 0.95
        fairness = 0.7 if service_class == "education" else 0.3

        constraints: list[str] = []
        if "no ntn" in low or "terrestrial only" in low:
            constraints.append("forbid_ntn")
        if "failover" in low:
            constraints.append("allow_ntn_failover")

        return ServiceIntent(
            service_id=service_id,
            service_class=service_class,
            priority=priority,
            max_latency_ms=lat,
            min_reliability=reliability,
            fairness_floor=fairness,
            description=text.strip()[:256],
            constraints=constraints,
        )


class LLMIntentAdapterStub:
    """Optional LLM adapter stub — NEVER directly controls radio.

    Produces a draft natural-language rewrite only; must be re-parsed by
    RuleBasedIntentParser and validated by ServiceIntent before use.
    """

    def draft_rewrite(self, text: str) -> str:
        return f"[LLM_STUB_DRAFT] {text.strip()}"

    def to_intent(self, text: str, parser: RuleBasedIntentParser | None = None) -> ServiceIntent:
        parser = parser or RuleBasedIntentParser()
        draft = self.draft_rewrite(text)
        cleaned = draft.replace("[LLM_STUB_DRAFT]", "").strip()
        return parser.parse(cleaned)
