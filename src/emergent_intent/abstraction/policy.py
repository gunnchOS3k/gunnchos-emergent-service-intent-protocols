"""Policy wrappers that actually consume state abstractions (§6.7)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from emergent_intent.abstraction.encoders import (
    ContrastiveEncoder,
    IBEncoder,
    VQEncoder,
    engineered_aggregate,
)
from emergent_intent.algorithms.networks import MultiDiscreteActor, nvec_from_env, obs_dim_from_env


@dataclass
class AbstractionReport:
    kind: str
    latent_size: int
    objective: str
    regularization: str
    training_procedure: str
    downstream_utility: float
    message_efficiency: float
    held_out_generalization: float
    representation_stability: float
    nuisance_sensitivity: float
    compute_cost: float
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "kind": self.kind,
            "latent_size": self.latent_size,
            "objective": self.objective,
            "regularization": self.regularization,
            "training_procedure": self.training_procedure,
            "downstream_utility": self.downstream_utility,
            "message_efficiency": self.message_efficiency,
            "held_out_generalization": self.held_out_generalization,
            "representation_stability": self.representation_stability,
            "nuisance_sensitivity": self.nuisance_sensitivity,
            "compute_cost": self.compute_cost,
        }
        d.update(self.extras)
        return d


class AbstractionPolicy(nn.Module):
    """Actor that first maps observations through an abstraction then selects actions."""

    def __init__(self, obs_dim: int, nvec: list[int], kind: str = "raw", latent_dim: int = 8, hidden: int = 32):
        super().__init__()
        self.kind = kind
        self.latent_dim = latent_dim if kind != "raw" else obs_dim
        self.encoder: nn.Module | None
        if kind == "raw":
            self.encoder = None
            in_dim = obs_dim
        elif kind == "engineered":
            self.encoder = None
            in_dim = 8
            self.latent_dim = 8
        elif kind == "ib_encoder":
            self.encoder = IBEncoder(obs_dim, latent_dim=latent_dim, hidden=hidden)
            in_dim = latent_dim
        elif kind == "vq_encoder":
            self.encoder = VQEncoder(obs_dim, latent_dim=latent_dim, hidden=hidden)
            in_dim = latent_dim
        elif kind == "contrastive":
            self.encoder = ContrastiveEncoder(obs_dim, latent_dim=latent_dim, hidden=hidden)
            in_dim = latent_dim
        else:
            raise ValueError(kind)
        self.actor = MultiDiscreteActor(in_dim, nvec, hidden=hidden)

    def encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        aux = torch.tensor(0.0, device=obs.device)
        if self.kind == "raw":
            return obs, aux
        if self.kind == "engineered":
            # batch engineered features
            arr = []
            for i in range(obs.shape[0]):
                arr.append(engineered_aggregate(obs[i].detach().cpu().numpy()))
            z = torch.as_tensor(np.asarray(arr), dtype=torch.float32, device=obs.device)
            return z, aux
        assert self.encoder is not None
        out = self.encoder(obs)
        if self.kind == "ib_encoder":
            z, mu, logvar = out  # type: ignore[misc]
            aux = self.encoder.kl(mu, logvar)  # type: ignore[union-attr]
            return z, aux
        if self.kind == "vq_encoder":
            z, z_e, z_q = out  # type: ignore[misc]
            aux = self.encoder.vq_loss(z_e, z_q)  # type: ignore[union-attr]
            return z, aux
        # contrastive: encode only; contrastive loss applied externally
        return out if not isinstance(out, tuple) else out[0], aux

    def forward(self, obs: torch.Tensor):
        z, aux = self.encode(obs)
        return self.actor(z), z, aux


def run_abstraction_pilot(env, kinds: list[str] | None = None, steps: int = 64, seed: int = 0) -> list[dict]:
    """Train tiny policies on each abstraction and record §6.7 metrics (pilot-scale)."""
    kinds = kinds or ["raw", "engineered", "ib_encoder", "vq_encoder", "contrastive"]
    reports = []
    agent = env.possible_agents[0]
    od = obs_dim_from_env(env, agent)
    nv = nvec_from_env(env, agent)
    for kind in kinds:
        torch.manual_seed(seed)
        policy = AbstractionPolicy(od, nv, kind=kind, latent_dim=8, hidden=32)
        opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
        obs, _ = env.reset(seed=seed)
        returns = []
        ep_ret = 0.0
        zs = []
        import time

        t0 = time.time()
        for t in range(steps):
            o = torch.as_tensor(obs[agent], dtype=torch.float32).unsqueeze(0)
            logits_list, z, aux = policy(o)
            actions = {}
            for a in env.agents:
                if a == agent:
                    acts = [int(logits.argmax(-1).item()) for logits in logits_list]
                    actions[a] = np.asarray(acts + [0] * (len(nv) - len(acts)), dtype=np.int64)[: len(nv)]
                else:
                    actions[a] = env.action_space(a).sample()
            next_obs, rewards, terms, truncs, infos = env.step(actions)
            r = float(rewards.get(agent, 0.0))
            ep_ret += r
            # supervised-ish: maximize reward via REINFORCE on first head
            dist0 = torch.distributions.Categorical(logits=logits_list[0])
            loss = -dist0.log_prob(torch.tensor(actions[agent][0])) * r + 0.01 * aux
            if kind == "contrastive" and t > 0:
                o2 = o + 0.01 * torch.randn_like(o)
                z1 = policy.encoder.project(o)  # type: ignore[union-attr]
                z2 = policy.encoder.project(o2)  # type: ignore[union-attr]
                loss = loss + ContrastiveEncoder.nt_xent(z1, z2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            zs.append(z.detach().cpu().numpy().ravel())
            if (not env.agents) or any(terms.values()) or any(truncs.values()):
                returns.append(ep_ret)
                ep_ret = 0.0
                obs, _ = env.reset(seed=seed + t + 1)
            else:
                obs = next_obs
        cost = time.time() - t0
        zarr = np.asarray(zs) if zs else np.zeros((1, policy.latent_dim))
        stability = float(1.0 / (1.0 + zarr.std()))
        # nuisance: add noise to obs and measure latent shift
        obs, _ = env.reset(seed=seed)
        o = torch.as_tensor(obs[agent], dtype=torch.float32).unsqueeze(0)
        z0, _ = policy.encode(o)
        zn, _ = policy.encode(o + 0.5 * torch.randn_like(o))
        sens = float(torch.norm(z0 - zn).item())
        bits = float(infos[agent]["bits"]) if infos and agent in infos else 0.0
        utility = float(np.mean(returns)) if returns else ep_ret
        # held-out: one eval episode with fixed policy
        obs, _ = env.reset(seed=seed + 999)
        hold = 0.0
        for _ in range(env.config.horizon):
            if not env.agents:
                break
            o = torch.as_tensor(obs[agent], dtype=torch.float32).unsqueeze(0)
            logits_list, _, _ = policy(o)
            actions = {}
            for a in env.agents:
                if a == agent:
                    acts = [int(logits.argmax(-1).item()) for logits in logits_list]
                    actions[a] = np.asarray(acts + [0] * (len(nv) - len(acts)), dtype=np.int64)[: len(nv)]
                else:
                    actions[a] = env.action_space(a).sample()
            obs, rewards, terms, truncs, _ = env.step(actions)
            hold += float(rewards.get(agent, 0.0))
            if not env.agents:
                break
        report = AbstractionReport(
            kind=kind,
            latent_size=int(policy.latent_dim),
            objective={
                "raw": "identity",
                "engineered": "handcrafted_features",
                "ib_encoder": "elbo_kl_bottleneck",
                "vq_encoder": "vq_commitment",
                "contrastive": "nt_xent",
            }[kind],
            regularization={
                "raw": "none",
                "engineered": "none",
                "ib_encoder": "KL(q(z|o)||N(0,I))",
                "vq_encoder": "codebook+commitment",
                "contrastive": "temperature NT-Xent",
            }[kind],
            training_procedure="REINFORCE_pilot_with_aux",
            downstream_utility=utility,
            message_efficiency=1.0 / (1.0 + bits),
            held_out_generalization=hold,
            representation_stability=stability,
            nuisance_sensitivity=sens,
            compute_cost=cost,
        )
        reports.append(report.to_dict())
    return reports
