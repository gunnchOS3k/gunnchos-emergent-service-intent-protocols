"""§6.4 gradient-flow and naming honesty tests for DIAL / TarMAC / baseline."""

from __future__ import annotations

import torch

from emergent_intent.algorithms import (
    DialTrainer,
    PPODiscreteMessageEntropyBaseline,
    TarMACTrainer,
    make_trainer,
)
from emergent_intent.algorithms.networks import PPOConfig
from emergent_intent.env import EnvConfig, make_env


def test_baseline_is_not_labeled_dial_tarmac() -> None:
    env = make_env(EnvConfig(horizon=4, n_ue=1, channel={"mode": "discrete_learned", "vocab_size": 4}))
    t = PPODiscreteMessageEntropyBaseline(
        env, config=PPOConfig(hidden=16, rollout_steps=8, epochs=1), seed=0, prefer_cuda=False
    )
    out = t.train(total_steps=16)
    assert out["algorithm"] == "ppo_discrete_message_entropy_baseline"
    assert "NOT faithful DIAL" in out["notes"][0] or any("NOT" in n for n in out["notes"])


def test_dial_receiver_loss_grads_reach_sender_message_head() -> None:
    env = make_env(
        EnvConfig(horizon=4, n_ue=1, channel={"mode": "discrete_learned", "vocab_size": 4, "msg_length": 2})
    )
    trainer = DialTrainer(
        env, vocab_size=4, msg_length=2, config=PPOConfig(hidden=16), seed=0, prefer_cuda=False
    )
    sender, receiver = trainer.agents[0], trainer.agents[1]
    obs, _ = env.reset(seed=0)
    so = torch.as_tensor(obs[sender], dtype=torch.float32, device=trainer.device).unsqueeze(0)
    ro = torch.as_tensor(obs[receiver], dtype=torch.float32, device=trainer.device).unsqueeze(0)
    target = torch.tensor([1.0], device=trainer.device)
    loss, msg_logits = trainer.differentiable_receiver_loss(sender, receiver, so, ro, target)
    trainer.opts[sender].zero_grad()
    loss.backward()
    grads = [p.grad for n, p in trainer.senders[sender].named_parameters() if "msg_head" in n]
    assert grads and grads[0] is not None
    assert float(grads[0].abs().sum().item()) > 0.0


def test_tarmac_attention_affects_routing() -> None:
    env = make_env(
        EnvConfig(horizon=4, n_ue=1, targeted=True, channel={"mode": "discrete_learned", "vocab_size": 4})
    )
    trainer = TarMACTrainer(env, config=PPOConfig(hidden=16, rollout_steps=8, epochs=1), seed=0, prefer_cuda=False)
    obs, _ = env.reset(seed=0)
    actions, _, _ = trainer.select_actions(obs)
    assert set(actions) == set(obs)
    diag = trainer.attention_diagnostics()
    assert diag
    out = trainer.train(total_steps=16)
    assert out["algorithm"] == "TARMAC"
    assert "attention" in out


def test_make_trainer_routes_names() -> None:
    env = make_env(EnvConfig(horizon=2, n_ue=1, channel={"mode": "no_comm"}))
    assert isinstance(make_trainer("ppo_discrete_message_entropy_baseline", env, prefer_cuda=False), PPODiscreteMessageEntropyBaseline)
    env2 = make_env(EnvConfig(horizon=2, n_ue=1, channel={"mode": "discrete_learned", "vocab_size": 4}))
    assert isinstance(make_trainer("dial", env2, prefer_cuda=False, vocab_size=4, msg_length=2), DialTrainer)
