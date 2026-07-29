"""Unit tests for intent, objectives, abstraction, adapters."""

from __future__ import annotations

import numpy as np
import torch

from emergent_intent.abstraction import (
    ContrastiveEncoder,
    IBEncoder,
    VQEncoder,
    abstract_obs,
    engineered_aggregate,
)
from emergent_intent.adapters import load_all_adapters, probe_siblings
from emergent_intent.intent import (
    LLMIntentAdapterStub,
    RuleBasedIntentParser,
    ServiceIntent,
    action_mask,
    compile_constraints,
)
from emergent_intent.objectives import (
    LagrangianState,
    ObjectiveWeights,
    compute_rewards,
    hypervolume_2d,
    pareto_front,
    preference_conditioned_scalar,
)


def test_service_intent_validation() -> None:
    intent = ServiceIntent(
        service_id="urllc-1",
        service_class="URLLC",
        priority=5,
        max_latency_ms=5.0,
        min_reliability=0.999,
    )
    assert intent.priority == 5


def test_rule_parser_and_llm_stub() -> None:
    parser = RuleBasedIntentParser()
    intent = parser.parse("critical URLLC service with 5ms latency and failover")
    assert intent.service_class == "URLLC"
    assert intent.max_latency_ms == 5.0
    assert "allow_ntn_failover" in intent.constraints
    stub = LLMIntentAdapterStub()
    draft = stub.draft_rewrite("education fairness 50ms")
    assert draft.startswith("[LLM_STUB_DRAFT]")
    # Never radio-control: still rule-parsed
    intent2 = stub.to_intent("education school fairness")
    assert intent2.service_class == "education"


def test_constraints_and_masking() -> None:
    intent = RuleBasedIntentParser().parse("critical URLLC 5ms no ntn")
    cc = compile_constraints(intent)
    assert cc.forbid_ntn
    masks = action_mask("ntn_relay", [2, 4], cc, include_ntn=True)
    assert masks[0][0] is True
    assert masks[0][1] is False


def test_objectives() -> None:
    metrics = {
        "task_success": 1.0,
        "latency_ms": 10.0,
        "energy": 0.1,
        "message_bits": 2.0,
        "fairness": 0.8,
        "spectral_efficiency": 0.5,
        "violations": 0.0,
    }
    out = compute_rewards(metrics, ObjectiveWeights())
    assert "scalar" in out
    lag = LagrangianState()
    out2 = compute_rewards(metrics, ObjectiveWeights(), lagrangian=lag)
    assert out2["scalar"] <= out["scalar"] + 1e-6
    pref = preference_conditioned_scalar(out["utilities"], {"task_success": 1.0, "fairness": 1.0})
    assert pref >= 0.0
    pts = [{"task_success": 1.0, "latency": 0.5, "fairness": 0.2, "spectral_efficiency": 0.1},
           {"task_success": 0.5, "latency": 0.9, "fairness": 0.9, "spectral_efficiency": 0.9},
           {"task_success": 0.4, "latency": 0.4, "fairness": 0.4, "spectral_efficiency": 0.4}]
    front = pareto_front(pts)
    assert 0 in front and 1 in front
    hv = hypervolume_2d([(0.5, 0.5), (0.8, 0.2), (0.3, 0.9)])
    assert hv >= 0.0


def test_abstraction_encoders() -> None:
    obs = np.random.randn(16).astype(np.float32)
    assert engineered_aggregate(obs).shape == (8,)
    assert abstract_obs(obs, "raw").shape == (16,)
    ib = IBEncoder(16, 4)
    z, mu, lv = ib(torch.randn(2, 16))
    assert z.shape == (2, 4)
    assert ib.kl(mu, lv).ndim == 0
    vq = VQEncoder(16, 4, codebook=8)
    z, ze, zq = vq(torch.randn(2, 16))
    assert vq.vq_loss(ze, zq).ndim == 0
    c = ContrastiveEncoder(16, 4)
    z1 = c.project(torch.randn(4, 16))
    z2 = c.project(torch.randn(4, 16))
    loss = ContrastiveEncoder.nt_xent(z1, z2)
    assert loss.ndim == 0


def test_soft_adapters_never_fail() -> None:
    status = probe_siblings()
    assert set(status) >= {"edge_io", "digital_twin", "ntn_sim", "spectrumx"}
    adapters = load_all_adapters()
    assert "edge_io" in adapters
    # methods work whether sibling present or not
    assert "available" in adapters["edge_io"].sample_rssi()
    assert "available" in adapters["ntn_sim"].failover_prior()
