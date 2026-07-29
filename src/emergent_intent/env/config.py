from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ScenarioFamily(str, Enum):
    terrestrial_congestion = "terrestrial_congestion"
    tn_ntn_failover = "tn_ntn_failover"
    critical_service = "critical_service"
    education_fairness = "education_fairness"
    # Communication-necessary scenarios (§6.3)
    hidden_blockage_congestion = "hidden_blockage_congestion"
    tn_ntn_continuity = "tn_ntn_continuity"


class CommMode(str, Enum):
    no_comm = "no_comm"
    fixed_protocol = "fixed_protocol"
    continuous_learned = "continuous_learned"
    discrete_learned = "discrete_learned"


class EnvConfig(BaseModel):
    scenario: ScenarioFamily = ScenarioFamily.terrestrial_congestion
    n_ue: int = 1
    include_ntn: bool = False
    horizon: int = 32
    seed: int = 0
    comm_mode: CommMode = CommMode.discrete_learned
    vocab_size: int = 16
    msg_len: int = 2
    erasure_p: float = 0.0
    bit_error_p: float = 0.0
    corruption_p: float = 0.0
    delay: int = 0
    observation_noise: float = 0.0
    max_power: float = 1.0
    fairness_floor: float = 0.1
    message_bit_budget: float = 64.0
    abstraction: Literal["raw", "engineered", "learned"] = "raw"
    evidence_class: str = "SYNTHETIC_SIM"
    n_ues: int | None = None
    max_steps: int | None = None
    continuous_dim: int = 4
    latency_sla_ms: float = 20.0
    energy_budget: float = 1.0
    fairness_weight: float = 0.1
    bit_cost: float = 0.01
    inbox_capacity: int = 2
    loopback: bool = False
    targeted: bool = False
    stale_threshold: int = 3
    objectives: dict[str, float] = Field(default_factory=dict)
    channel: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnvConfig":
        return cls.model_validate(data)

    @model_validator(mode="after")
    def _apply_channel_and_ntn(self) -> "EnvConfig":
        if self.scenario in (
            ScenarioFamily.tn_ntn_failover,
            ScenarioFamily.tn_ntn_continuity,
        ):
            object.__setattr__(self, "include_ntn", True)
        ch = self.channel or {}
        mapping = {
            "mode": ("comm_mode", lambda v: CommMode(v)),
            "msg_length": ("msg_len", int),
            "msg_len": ("msg_len", int),
            "vocab_size": ("vocab_size", int),
            "erasure_p": ("erasure_p", float),
            "erasure_prob": ("erasure_p", float),
            "bit_error_p": ("bit_error_p", float),
            "corruption_p": ("corruption_p", float),
            "corruption_prob": ("corruption_p", float),
            "delay": ("delay", int),
            "delay_steps": ("delay", int),
            "targeted": ("targeted", bool),
            "inbox_capacity": ("inbox_capacity", int),
            "loopback": ("loopback", bool),
            "stale_threshold": ("stale_threshold", int),
            "bit_cost": ("bit_cost", float),
        }
        for src, (dst, cast) in mapping.items():
            if src in ch:
                object.__setattr__(self, dst, cast(ch[src]))
        if self.n_ues is not None:
            object.__setattr__(self, "n_ue", int(self.n_ues))
        if self.max_steps is not None:
            object.__setattr__(self, "horizon", int(self.max_steps))
        return self
