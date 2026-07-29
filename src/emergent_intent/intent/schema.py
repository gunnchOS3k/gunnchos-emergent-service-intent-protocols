from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError


class ServiceIntent(BaseModel):
    service_id: str = "intent-01"
    service_class: str = Field(..., description="e.g. URLLC, eMBB, education, best_effort")
    priority: int = 1
    min_availability: float = Field(0.0, ge=0.0, le=1.0)
    min_reliability: float = Field(0.0, ge=0.0, le=1.0)
    max_latency_ms: float = 50.0
    max_power: float | None = None
    fairness_floor: float = 0.0
    energy_budget: float = 1e9
    allow_ntn: bool = True
    privacy_restricted: bool = False
    constraints: list[str] = Field(default_factory=list)
    description: str | None = None
    notes: str | None = None


class IntentParseResult(BaseModel):
    ok: bool
    intent: ServiceIntent | None = None
    errors: list[str] = Field(default_factory=list)
    source: str = "rule_based"
    model: str | None = None


def intent_json_schema() -> dict[str, Any]:
    return ServiceIntent.model_json_schema()


def load_intent_schema_file(path: Path | str | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    p = Path(path) if path else root / "schemas" / "service_intent.schema.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return intent_json_schema()


def rule_based_parse(text: str) -> IntentParseResult:
    t = text.lower().strip()
    if not t:
        return IntentParseResult(ok=False, errors=["empty_intent"])
    if any(w in t for w in ("maybe", "whatever", "somehow", "idk")):
        return IntentParseResult(ok=False, errors=["ambiguous_intent_rejected"])

    service = "best_effort"
    if "critical" in t or "public safety" in t or "emergency" in t or "urllc" in t:
        service = "URLLC"
    elif "education" in t or "community" in t or "fair" in t:
        service = "education"
    elif "embb" in t or "broadband" in t:
        service = "eMBB"

    kwargs: dict[str, Any] = {"service_class": service, "constraints": []}
    if "low latency" in t or "urllc" in t:
        kwargs["max_latency_ms"] = 10.0
    if "no ntn" in t or "terrestrial only" in t:
        kwargs["allow_ntn"] = False
        kwargs["constraints"].append("forbid_ntn")
    if "privacy" in t:
        kwargs["privacy_restricted"] = True
    if service == "URLLC":
        kwargs["min_availability"] = 0.99
        kwargs["min_reliability"] = 0.99
        kwargs["priority"] = 5
    if service == "education":
        kwargs["fairness_floor"] = 0.5

    try:
        intent = ServiceIntent(**kwargs)
    except ValidationError as exc:
        return IntentParseResult(ok=False, errors=[str(exc)])
    return IntentParseResult(ok=True, intent=intent, source="rule_based")


def optional_llm_adapter(text: str, enabled: bool = False) -> IntentParseResult:
    if not enabled:
        return rule_based_parse(text)
    base = rule_based_parse(text)
    base.source = "llm_adapter_stub"
    base.model = "DISABLED_OR_UNAVAILABLE"
    return base


def compile_constraints(intent: ServiceIntent) -> dict[str, Any]:
    return {
        "min_availability": intent.min_availability,
        "min_reliability": intent.min_reliability,
        "max_latency_ms": intent.max_latency_ms,
        "max_power": intent.max_power,
        "fairness_floor": intent.fairness_floor,
        "energy_budget": intent.energy_budget,
        "allow_ntn": intent.allow_ntn and "forbid_ntn" not in intent.constraints,
        "privacy_restricted": intent.privacy_restricted,
        "action_mask_rules": [
            "reject_unvalidated_radio_actions",
            "llm_cannot_set_rf_parameters_directly",
        ],
    }


def action_mask(action: list[float] | None, constraints: dict[str, Any]) -> list[float]:
    if action is None:
        return [0.0, 0.0, 0.0, 0.0]
    a = list(action)
    max_power = constraints.get("max_power")
    if max_power is not None and a:
        a[0] = min(a[0], float(max_power))
    if not constraints.get("allow_ntn", True) and len(a) > 3:
        a[3] = 0.0
    return a
