"""Shared actor-critic building blocks for CPU MARL smoke training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class MultiDiscreteActor(nn.Module):
    """Independent Categorical heads for MultiDiscrete action vectors."""

    def __init__(self, obs_dim: int, nvec: list[int], hidden: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, n) for n in nvec])
        self.nvec = list(nvec)

    def forward(self, obs: torch.Tensor) -> list[torch.Tensor]:
        h = self.backbone(obs)
        return [head(h) for head in self.heads]

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        logits_list = self.forward(obs)
        actions = []
        logps = []
        for logits in logits_list:
            dist = Categorical(logits=logits)
            a = logits.argmax(-1) if deterministic else dist.sample()
            actions.append(a)
            logps.append(dist.log_prob(a))
        action = torch.stack(actions, dim=-1)
        logp = torch.stack(logps, dim=-1).sum(-1)
        return action, logp, logits_list

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits_list = self.forward(obs)
        logps = []
        ents = []
        for i, logits in enumerate(logits_list):
            dist = Categorical(logits=logits)
            a = actions[..., i]
            logps.append(dist.log_prob(a))
            ents.append(dist.entropy())
        return torch.stack(logps, dim=-1).sum(-1), torch.stack(ents, dim=-1).sum(-1)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class CentralCritic(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    epochs: int = 4
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout_steps: int = 128
    hidden: int = 64


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        next_v = values[t + 1] if t + 1 < T else 0.0
        next_nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_v * next_nonterminal - values[t]
        last = delta + gamma * lam * next_nonterminal * last
        adv[t] = last
    returns = adv + values[:T]
    return adv, returns


def obs_dim_from_env(env, agent: str) -> int:
    return int(np.prod(env.observation_space(agent).shape))


def nvec_from_env(env, agent: str) -> list[int]:
    space = env.action_space(agent)
    return [int(x) for x in space.nvec]


def flatten_state(state: dict[str, Any]) -> np.ndarray:
    keys = sorted(state.keys())
    parts = []
    for k in keys:
        v = state[k]
        if isinstance(v, dict):
            continue
        arr = np.asarray(v, dtype=np.float32).ravel()
        parts.append(arr)
    if not parts:
        return np.zeros(1, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32)
