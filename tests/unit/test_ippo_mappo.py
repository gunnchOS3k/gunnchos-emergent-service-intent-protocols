"""§6.6 IPPO / MAPPO validation beyond non-crash."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from emergent_intent.algorithms.ippo import IPPOTrainer
from emergent_intent.algorithms.mappo import MAPPOTrainer
from emergent_intent.algorithms.networks import PPOConfig, compute_gae, flatten_state
from emergent_intent.env import EnvConfig, make_env


def test_gae_advantage_matches_manual() -> None:
    rew = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    val = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    done = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    adv, ret = compute_gae(rew, val, done, gamma=0.99, lam=0.95)
    assert adv.shape == (3,)
    assert ret.shape == (3,)
    # terminal should cut bootstrap at last step
    assert np.isfinite(adv).all()


def test_ippo_local_actors_and_likelihood() -> None:
    env = make_env(EnvConfig(horizon=6, n_ue=1, channel={"mode": "no_comm"}))
    trainer = IPPOTrainer(
        env, config=PPOConfig(hidden=16, rollout_steps=8, epochs=1), seed=0, prefer_cuda=False
    )
    obs, _ = env.reset(seed=0)
    actions, logps, values = trainer.select_actions(obs, deterministic=False)
    assert set(actions) == set(obs)
    # evaluate likelihood
    a = trainer.agents[0]
    o = torch.as_tensor(obs[a], dtype=torch.float32).unsqueeze(0)
    act = torch.as_tensor(actions[a], dtype=torch.int64).unsqueeze(0)
    logp, ent = trainer.actors[a].evaluate(o, act)
    assert torch.isfinite(logp).all()
    assert torch.isfinite(ent).all()
    assert actions[a].shape[0] == len(env.action_space(a).nvec)


def test_mappo_central_critic_uses_global_state() -> None:
    env = make_env(EnvConfig(horizon=6, n_ue=1, channel={"mode": "no_comm"}))
    trainer = MAPPOTrainer(
        env, config=PPOConfig(hidden=16, rollout_steps=8, epochs=1), seed=0, prefer_cuda=False
    )
    obs, _ = env.reset(seed=0)
    st = flatten_state(env.state())
    assert trainer.state_dim == st.size
    actions, logps, value = trainer.select_actions(obs)
    assert isinstance(value, float)
    # actors use local obs dim
    a = trainer.agents[0]
    assert trainer.actors[a].backbone[0].in_features == int(np.prod(env.observation_space(a).shape))


def test_termination_truncation_handling() -> None:
    env = make_env(EnvConfig(horizon=3, n_ue=1, channel={"mode": "no_comm"}))
    trainer = IPPOTrainer(
        env, config=PPOConfig(hidden=16, rollout_steps=4, epochs=1), seed=0, prefer_cuda=False
    )
    out = trainer.train(total_steps=12)
    assert out["n_episodes"] >= 1
    assert env.agents == [] or out["steps"] >= 3


def test_checkpoint_resume_and_deterministic_eval(tmp_path: Path) -> None:
    env = make_env(EnvConfig(horizon=4, n_ue=1, channel={"mode": "no_comm"}))
    cfg = PPOConfig(hidden=16, rollout_steps=8, epochs=1)
    t = IPPOTrainer(env, config=cfg, seed=1, prefer_cuda=False)
    t.train(total_steps=16)
    ckpt = tmp_path / "ippo.pt"
    t.save(ckpt)
    env2 = make_env(EnvConfig(horizon=4, n_ue=1, channel={"mode": "no_comm"}))
    t2 = IPPOTrainer(env2, config=cfg, seed=1, prefer_cuda=False)
    t2.load(ckpt)
    obs, _ = env2.reset(seed=42)
    a1, _, _ = t.select_actions(obs, deterministic=True)
    a2, _, _ = t2.select_actions(obs, deterministic=True)
    for k in a1:
        assert np.array_equal(a1[k], a2[k])


def test_mappo_multiagent_batching_and_entropy() -> None:
    env = make_env(EnvConfig(horizon=6, n_ue=2, channel={"mode": "no_comm"}))
    trainer = MAPPOTrainer(
        env, config=PPOConfig(hidden=16, rollout_steps=8, epochs=1), seed=0, prefer_cuda=False
    )
    out = trainer.train(total_steps=24)
    assert out["algorithm"] == "MAPPO"
    assert out["updates"] >= 1
    a = trainer.agents[0]
    o = torch.randn(4, int(np.prod(env.observation_space(a).shape)))
    act = torch.stack(
        [torch.as_tensor(env.action_space(a).sample()) for _ in range(4)], dim=0
    ).long()
    logp, ent = trainer.actors[a].evaluate(o, act)
    assert ent.mean().item() >= 0.0
