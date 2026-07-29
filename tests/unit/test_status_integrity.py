"""Status-integrity: false scientific PASS must be denied."""

from __future__ import annotations

from pathlib import Path

from emergent_intent.algorithms import DialTrainer, PPODiscreteMessageEntropyBaseline, make_trainer
from emergent_intent.algorithms.networks import PPOConfig
from emergent_intent.env import EnvConfig, make_env


def scientific_pass_allowed(
    *,
    causal_tests_ok: bool,
    dial_grad_ok: bool,
    qmix_targets_ok: bool,
    seeds_for_final_method: int,
    generalization_present: bool,
    figures_present: bool,
    paper_placeholders: bool,
    claims_tarmac: bool,
    tarmac_attention_used: bool,
) -> tuple[bool, list[str]]:
    reasons = []
    if not causal_tests_ok:
        reasons.append("causal_tests_missing")
    if not dial_grad_ok:
        reasons.append("dial_gradient_failed")
    if not qmix_targets_ok:
        reasons.append("qmix_targets_failed")
    if seeds_for_final_method < 5:
        reasons.append("insufficient_seeds_for_final")
    if not generalization_present:
        reasons.append("generalization_absent")
    if not figures_present:
        reasons.append("figures_absent")
    if paper_placeholders:
        reasons.append("paper_placeholders")
    if claims_tarmac and not tarmac_attention_used:
        reasons.append("tarmac_attention_unused")
    return (len(reasons) == 0), reasons


def test_smoke_without_causal_denies_scientific_pass() -> None:
    ok, reasons = scientific_pass_allowed(
        causal_tests_ok=False,
        dial_grad_ok=True,
        qmix_targets_ok=True,
        seeds_for_final_method=5,
        generalization_present=True,
        figures_present=True,
        paper_placeholders=False,
        claims_tarmac=False,
        tarmac_attention_used=False,
    )
    assert not ok
    assert "causal_tests_missing" in reasons


def test_false_dial_label_denied_without_gradient() -> None:
    # Baseline trainer is not DIAL
    env = make_env(EnvConfig(horizon=2, n_ue=1, channel={"mode": "discrete_learned", "vocab_size": 4}))
    base = make_trainer("ppo_discrete_message_entropy_baseline", env, prefer_cuda=False)
    assert isinstance(base, PPODiscreteMessageEntropyBaseline)
    ok, reasons = scientific_pass_allowed(
        causal_tests_ok=True,
        dial_grad_ok=False,
        qmix_targets_ok=True,
        seeds_for_final_method=5,
        generalization_present=True,
        figures_present=True,
        paper_placeholders=False,
        claims_tarmac=False,
        tarmac_attention_used=False,
    )
    assert not ok and "dial_gradient_failed" in reasons


def test_tarmac_claim_without_attention_denied() -> None:
    ok, reasons = scientific_pass_allowed(
        causal_tests_ok=True,
        dial_grad_ok=True,
        qmix_targets_ok=True,
        seeds_for_final_method=5,
        generalization_present=True,
        figures_present=True,
        paper_placeholders=False,
        claims_tarmac=True,
        tarmac_attention_used=False,
    )
    assert not ok and "tarmac_attention_unused" in reasons


def test_single_seed_denies_final_evidence() -> None:
    ok, reasons = scientific_pass_allowed(
        causal_tests_ok=True,
        dial_grad_ok=True,
        qmix_targets_ok=True,
        seeds_for_final_method=1,
        generalization_present=True,
        figures_present=True,
        paper_placeholders=False,
        claims_tarmac=False,
        tarmac_attention_used=False,
    )
    assert not ok and "insufficient_seeds_for_final" in reasons


def test_faithful_dial_gradient_supports_dial_gate() -> None:
    import torch

    env = make_env(
        EnvConfig(horizon=3, n_ue=1, channel={"mode": "discrete_learned", "vocab_size": 4, "msg_length": 2})
    )
    trainer = DialTrainer(
        env, vocab_size=4, msg_length=2, config=PPOConfig(hidden=16), seed=0, prefer_cuda=False
    )
    s, r = trainer.agents[0], trainer.agents[1]
    obs, _ = env.reset(seed=0)
    so = torch.as_tensor(obs[s], dtype=torch.float32).unsqueeze(0)
    ro = torch.as_tensor(obs[r], dtype=torch.float32).unsqueeze(0)
    loss, _ = trainer.differentiable_receiver_loss(s, r, so, ro, torch.tensor([0.5]))
    trainer.opts[s].zero_grad()
    loss.backward()
    g = trainer.senders[s].msg_head.weight.grad
    assert g is not None and float(g.abs().sum()) > 0
