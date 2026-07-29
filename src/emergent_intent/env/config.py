from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ScenarioFamily(str, Enum):
    terrestrial_congestion = "terrestrial_congestion"
    tn_ntn_failover = "tn_ntn_failover"
    critical_service = "critical_service"
    education_fairness = "education_fairness"


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
    objectives: dict[str, float] = Field(default_factory=dict)
    channel: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnvConfig":
        return cls.model_validate(data)

    @model_validator(mode="after")
    def _apply_channel_and_ntn(self) -> "EnvConfig":
        if self.scenario == ScenarioFamily.tn_ntn_failover:
            object.__setattr__(self, "include_ntn", True)
        ch = self.channel or {}
        if "mode" in ch:
            object.__setattr__(self, "comm_mode", CommMode(ch["mode"]))
        if "msg_length" in ch:
            object.__setattr__(self, "msg_len", int(ch["msg_length"]))
        if "vocab_size" in ch:
            object.__setattr__(self, "vocab_size", int(ch["vocab_size"]))
        if "erasure_p" in ch:
            object.__setattr__(self, "erasure_p", float(ch["erasure_p"]))
        if "bit_error_p" in ch:
            object.__setattr__(self, "bit_error_p", float(ch["bit_error_p"]))
        if "delay" in ch:
            object.__setattr__(self, "delay", int(ch["delay"]))
        if "erasure_prob" in ch:
            object.__setattr__(self, "erasure_p", float(ch["erasure_prob"]))
        if "corruption_prob" in ch:
            object.__setattr__(self, "bit_error_p", float(ch["corruption_prob"]))
        if "delay_steps" in ch:
            object.__setattr__(self, "delay", int(ch["delay_steps"]))
        if self.n_ues is not None:
            object.__setattr__(self, "n_ue", int(self.n_ues))
        if self.max_steps is not None:
            object.__setattr__(self, "horizon", int(self.max_steps))
        return self
