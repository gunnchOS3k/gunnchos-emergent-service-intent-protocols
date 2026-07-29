"""Constraint compiler and action masking for service intents."""

from __future__ import annotations

from emergent_intent.intent.schema import ServiceIntent


class CompiledConstraints:
    """Compiled intent constraints for masking / Lagrangian multipliers."""

    def __init__(self, intent: ServiceIntent):
        self.intent = intent
        self.max_latency_ms = intent.max_latency_ms
        self.min_reliability = intent.min_reliability
        self.fairness_floor = intent.fairness_floor
        self.energy_budget = intent.energy_budget
        self.forbid_ntn = "forbid_ntn" in intent.constraints
        self.allow_ntn_failover = "allow_ntn_failover" in intent.constraints
        self.priority = intent.priority

    def violation(self, metrics: dict[str, float]) -> float:
        v = 0.0
        if metrics.get("latency_ms", 0.0) > self.max_latency_ms:
            v += 1.0
        if metrics.get("fairness", 1.0) < self.fairness_floor:
            v += 1.0
        if metrics.get("energy", 0.0) > self.energy_budget:
            v += 0.5
        return v


def compile_constraints(intent: ServiceIntent) -> CompiledConstraints:
    return CompiledConstraints(intent)


def action_mask(
    agent: str,
    nvec: list[int],
    constraints: CompiledConstraints,
    include_ntn: bool,
) -> list[list[bool]]:
    """Return per-dimension allowed action booleans for MultiDiscrete nvec."""
    masks: list[list[bool]] = [[True] * n for n in nvec]
    if not nvec:
        return masks
    if agent == "ntn_relay" and constraints.forbid_ntn:
        if nvec[0] >= 1:
            masks[0] = [True] + [False] * (nvec[0] - 1)
    if agent in ("orchestrator", "edge_0") and constraints.priority >= 5 and len(nvec) >= 1:
        if nvec[0] >= 2:
            masks[0] = [i == 1 for i in range(nvec[0])]
    if not include_ntn and agent in ("ntn_relay", "ntn_0"):
        masks = [[False] * n for n in nvec]
    return masks


def apply_mask_to_logits(logits, mask: list[bool]):
    import torch

    m = torch.tensor(mask, dtype=torch.bool, device=logits.device)
    return logits.masked_fill(~m, float("-inf"))
