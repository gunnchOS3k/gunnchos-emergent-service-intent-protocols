"""End-to-end faithful TarMAC validation."""

from __future__ import annotations

import numpy as np
import torch

from emergent_intent.algorithms.networks import PPOConfig
from emergent_intent.algorithms.tarmac import TarMACTrainer
from emergent_intent.env import EnvConfig, make_env


def _tarmac_env():
    return make_env(
        EnvConfig(
            horizon=6,
            n_ue=1,
            seed=0,
            targeted=True,
            channel={"mode": "discrete_learned", "vocab_size": 4, "msg_length": 2},
        )
    )


def test_tarmac_stores_real_peer_kvq_attention() -> None:
    env = _tarmac_env()
    trainer = TarMACTrainer(
        env, config=PPOConfig(hidden=16, rollout_steps=8, epochs=1), seed=0, prefer_cuda=False, d_model=16
    )
    obs, _ = env.reset(seed=0)
    actions, logps, values = trainer.select_actions(obs)
    assert set(actions) == set(obs)
    store = trainer._last_peer_store
    assert store
    for a, ps in store.items():
        assert "peer_h" in ps and "own_key" in ps and "own_value" in ps and "own_query" in ps
        assert "attn" in ps
        # Must not claim self-tile
        assert trainer.last_peer_diagnostics[a]["used_self_tile"] is False
        peer_h = ps["peer_h"]
        own_h = ps["own_h"]
        # First real peer slot should not be an exact copy of own_h when peers exist
        if trainer.n_agents > 1 and peer_h.size(1) > 0:
            # zeros pad allowed; if first peer is a real agent encoding, compare
            first = peer_h[:, 0, :]
            if float(first.abs().sum()) > 1e-6:
                assert not torch.allclose(first, own_h, atol=1e-5)


def test_tarmac_update_refuses_missing_peers_and_not_self_tile() -> None:
    env = _tarmac_env()
    trainer = TarMACTrainer(
        env, config=PPOConfig(hidden=16, rollout_steps=8, epochs=1), seed=1, prefer_cuda=False, d_model=16
    )
    out = trainer.train(total_steps=24)
    assert out["algorithm"] == "TARMAC"
    assert out["updates"] >= 1
    assert any("NOT tile self" in n for n in out["notes"])
    assert "attention" in out


def test_tarmac_joint_logp_includes_control_message_target() -> None:
    env = _tarmac_env()
    trainer = TarMACTrainer(
        env, config=PPOConfig(hidden=16), seed=2, prefer_cuda=False, d_model=16
    )
    obs, _ = env.reset(seed=2)
    actions, logps, _ = trainer.select_actions(obs)
    a = trainer.agents[0]
    o = torch.as_tensor(obs[a], dtype=torch.float32, device=trainer.device).unsqueeze(0)
    act = torch.as_tensor(actions[a], dtype=torch.int64, device=trainer.device).unsqueeze(0)
    h = trainer.policies[a].encode(o)
    # Build peer stack
    hs = {b: trainer.policies[b].encode(
        torch.as_tensor(obs[b], dtype=torch.float32, device=trainer.device).unsqueeze(0)
    ) for b in obs}
    others, _ = trainer._stack_peer_hiddens(hs, a)
    routed, _ = trainer.policies[a].route(h, others)
    joint, ent, _, _, _ = trainer.policies[a].joint_logp_entropy(h, routed, act)
    # Control-only logp (exclude msg/target) should differ from joint when msg dims exist
    x = torch.cat([h, routed], dim=-1)
    ctrl_only = []
    for i, head in enumerate(trainer.policies[a].heads):
        from torch.distributions import Categorical

        dist = Categorical(logits=head(x))
        ctrl_only.append(dist.log_prob(act[:, i]))
    ctrl_logp = torch.stack(ctrl_only, dim=-1).sum(-1)
    assert joint.shape == ctrl_logp.shape
    # Joint includes extra message+target terms → typically different from control-only
    assert float((joint - ctrl_logp).abs().sum().item()) > 0.0 or act.shape[-1] <= len(
        trainer.policies[a].heads
    )
    assert float(logps[a]) == float(logps[a])  # finite
    assert ent is not None


def test_tarmac_end_to_end_short_train() -> None:
    env = _tarmac_env()
    trainer = TarMACTrainer(
        env, config=PPOConfig(hidden=16, rollout_steps=8, epochs=1), seed=3, prefer_cuda=False, d_model=16
    )
    metrics = trainer.train(total_steps=40)
    assert metrics["steps"] == 40
    assert metrics["evidence_class"] == "SYNTHETIC_SIM"
    diag = metrics["attention"]
    assert "_peer" in diag or any(isinstance(v, dict) for v in diag.values())
