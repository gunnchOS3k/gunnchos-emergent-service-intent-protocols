"""End-to-end faithful DIAL validation."""

from __future__ import annotations

import torch

from emergent_intent.algorithms.dial import DialTrainer
from emergent_intent.algorithms.networks import PPOConfig
from emergent_intent.env import EnvConfig, make_env


def _dial_env():
    return make_env(
        EnvConfig(
            horizon=6,
            n_ue=1,
            seed=0,
            channel={"mode": "discrete_learned", "vocab_size": 4, "msg_length": 2},
        )
    )


def test_dial_soft_train_hard_eval_modes() -> None:
    env = _dial_env()
    trainer = DialTrainer(
        env, vocab_size=4, msg_length=2, config=PPOConfig(hidden=16, rollout_steps=8, epochs=1), seed=0, prefer_cuda=False
    )
    obs, _ = env.reset(seed=0)
    soft = trainer.select_actions(obs, hard_messages=False)
    hard = trainer.select_actions(obs, hard_messages=True)
    assert set(soft[0]) == set(obs)
    assert set(hard[0]) == set(obs)
    out = trainer.train(total_steps=32)
    assert out["algorithm"] == "DIAL"
    assert out["dial_task_updates"] > 0
    assert any("PRIMARY objective: task loss" in n for n in out["notes"])
    hard_eval = trainer.evaluate_hard(episodes=2)
    assert "hard_eval_mean_return" in hard_eval


def test_dial_task_loss_grads_reach_sender_message_head() -> None:
    env = _dial_env()
    trainer = DialTrainer(
        env, vocab_size=4, msg_length=2, config=PPOConfig(hidden=16), seed=0, prefer_cuda=False
    )
    sender, receiver = trainer.agents[0], trainer.agents[1]
    obs, _ = env.reset(seed=0)
    so = torch.as_tensor(obs[sender], dtype=torch.float32, device=trainer.device).unsqueeze(0)
    ro = torch.as_tensor(obs[receiver], dtype=torch.float32, device=trainer.device).unsqueeze(0)
    # Task target = realized service proxy, NOT a value bootstrap of the receiver
    task_target = torch.tensor([0.75], device=trainer.device)
    loss, msg_logits, extras = trainer.differentiable_task_loss(
        sender, receiver, so, ro, task_target, hard=False
    )
    assert float(extras["primary_is_task_loss"].item()) == 1.0
    trainer.opts[sender].zero_grad()
    loss.backward()
    grads = [p.grad for n, p in trainer.senders[sender].named_parameters() if "msg_head" in n]
    assert grads and grads[0] is not None
    assert float(grads[0].abs().sum().item()) > 0.0
    # Channel path is soft (Gumbel) during train
    assert msg_logits.requires_grad


def test_dial_primary_is_not_receiver_value_regression() -> None:
    env = _dial_env()
    trainer = DialTrainer(
        env, vocab_size=4, msg_length=2, config=PPOConfig(hidden=16), seed=1, prefer_cuda=False
    )
    s, r = trainer.agents[0], trainer.agents[1]
    obs, _ = env.reset(seed=1)
    so = torch.as_tensor(obs[s], dtype=torch.float32, device=trainer.device).unsqueeze(0)
    ro = torch.as_tensor(obs[r], dtype=torch.float32, device=trainer.device).unsqueeze(0)
    task_target = torch.tensor([1.0], device=trainer.device)
    value_bootstrap = torch.tensor([0.0], device=trainer.device)
    torch.manual_seed(0)
    loss_task, _, extras = trainer.differentiable_task_loss(s, r, so, ro, task_target)
    torch.manual_seed(0)
    loss_via_alias, _ = trainer.differentiable_receiver_loss(s, r, so, ro, task_target)
    assert torch.allclose(loss_task, loss_via_alias, atol=1e-4)
    assert float(extras["primary_is_task_loss"].item()) == 1.0
    assert float(extras["task_loss"].item()) >= 0.0
    # Distinct targets yield distinct primary losses (task ≠ value-bootstrap target)
    torch.manual_seed(0)
    loss_v, _, _ = trainer.differentiable_task_loss(s, r, so, ro, value_bootstrap)
    assert abs(float(loss_task.detach()) - float(loss_v.detach())) > 1e-6
    # Value head exists but is not the primary forward path
    assert hasattr(trainer.receivers[r], "value_head")
    assert hasattr(trainer.receivers[r], "task_head")


def test_dial_end_to_end_short_train() -> None:
    env = _dial_env()
    trainer = DialTrainer(
        env,
        vocab_size=4,
        msg_length=2,
        config=PPOConfig(hidden=16, rollout_steps=8, epochs=1),
        seed=2,
        prefer_cuda=False,
    )
    metrics = trainer.train(total_steps=48)
    assert metrics["steps"] == 48
    assert metrics["evidence_class"] == "SYNTHETIC_SIM"
    assert "Receiver-value regression is auxiliary only" in metrics["notes"][-1] or any(
        "auxiliary" in n.lower() for n in metrics["notes"]
    )
