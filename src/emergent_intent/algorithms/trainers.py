from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from emergent_intent.env.config import EnvConfig
from emergent_intent.env.wireless_env import ServiceIntentEnv
from emergent_intent.utils.seeding import seed_everything


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.mu = nn.Linear(64, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor):
        h = self.net(obs)
        mu = torch.sigmoid(self.mu(h))
        std = self.log_std.exp().expand_as(mu)
        return Normal(mu, std)


class Critic(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


@dataclass
class TrainResult:
    algorithm: str
    seed: int
    episode_returns: list[float] = field(default_factory=list)
    mean_return: float = 0.0
    evidence_label: str = "SYNTHETIC_EXPERIMENT"
    checkpoint_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _rollout(env: ServiceIntentEnv, policy_fn, episodes: int) -> list[float]:
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        total = 0.0
        done = False
        while not done and env.agents:
            actions = {a: policy_fn(a, obs[a]) for a in env.agents}
            obs, rewards, terms, truncs, _ = env.step(actions)
            total += float(np.mean(list(rewards.values()))) if rewards else 0.0
            done = all(terms.values()) if terms else True
        returns.append(total)
    return returns


def run_random(config: EnvConfig, episodes: int = 5) -> TrainResult:
    seed_everything(config.seed)
    env = ServiceIntentEnv(config)
    rng = np.random.default_rng(config.seed)

    def policy(_a, _o):
        return rng.random(env._act_dim).astype(np.float32)

    rets = _rollout(env, policy, episodes)
    return TrainResult("random", config.seed, rets, float(np.mean(rets)))


def run_ippo(config: EnvConfig, episodes: int = 8, updates: int = 4) -> TrainResult:
    """Independent PPO-style smoke trainer (CPU-friendly)."""
    seed_everything(config.seed)
    env = ServiceIntentEnv(config)
    obs_dim, act_dim = env._obs_dim, env._act_dim
    actors = {a: Actor(obs_dim, act_dim) for a in env.possible_agents}
    critics = {a: Critic(obs_dim) for a in env.possible_agents}
    opt_a = {a: torch.optim.Adam(actors[a].parameters(), lr=3e-3) for a in actors}
    opt_c = {a: torch.optim.Adam(critics[a].parameters(), lr=3e-3) for a in critics}
    episode_returns: list[float] = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=config.seed + ep)
        traj: dict[str, list] = {a: [] for a in env.possible_agents}
        total = 0.0
        while env.agents:
            actions = {}
            for a in list(env.agents):
                o = torch.tensor(obs[a], dtype=torch.float32)
                dist = actors[a](o)
                act = dist.sample().clamp(0, 1)
                logp = dist.log_prob(act).sum()
                actions[a] = act.detach().numpy()
                traj[a].append((o, act.detach(), logp, None))
            obs, rewards, terms, truncs, infos = env.step(actions)
            for a in rewards:
                if traj[a]:
                    o, act, logp, _ = traj[a][-1]
                    traj[a][-1] = (o, act, logp, rewards[a])
            total += float(np.mean(list(rewards.values()))) if rewards else 0.0
            if all(terms.get(a, True) for a in terms):
                break
        episode_returns.append(total)
        # lightweight update
        if ep % max(episodes // updates, 1) == 0:
            for a, steps in traj.items():
                if not steps or steps[0][3] is None:
                    continue
                obs_b = torch.stack([s[0] for s in steps])
                act_b = torch.stack([s[1] for s in steps])
                rew_b = torch.tensor([s[3] for s in steps], dtype=torch.float32)
                values = critics[a](obs_b).squeeze(-1)
                adv = rew_b - values.detach()
                dist = actors[a](obs_b)
                logp = dist.log_prob(act_b).sum(-1)
                loss_a = -(logp * adv).mean()
                loss_c = torch.nn.functional.mse_loss(values, rew_b)
                opt_a[a].zero_grad()
                loss_a.backward()
                opt_a[a].step()
                opt_c[a].zero_grad()
                loss_c.backward()
                opt_c[a].step()

    return TrainResult("ippo", config.seed, episode_returns, float(np.mean(episode_returns)))


def run_mappo(config: EnvConfig, episodes: int = 8) -> TrainResult:
    """MAPPO-style: centralized critic on concatenated observations."""
    seed_everything(config.seed)
    env = ServiceIntentEnv(config)
    n_agents = len(env.possible_agents)
    obs_dim, act_dim = env._obs_dim, env._act_dim
    actors = {a: Actor(obs_dim, act_dim) for a in env.possible_agents}
    critic = Critic(obs_dim * n_agents)
    opt_a = {a: torch.optim.Adam(actors[a].parameters(), lr=3e-3) for a in actors}
    opt_c = torch.optim.Adam(critic.parameters(), lr=3e-3)
    returns: list[float] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=config.seed + ep)
        total = 0.0
        batch_obs = []
        batch_rew = []
        while env.agents:
            actions = {}
            for a in list(env.agents):
                o = torch.tensor(obs[a], dtype=torch.float32)
                dist = actors[a](o)
                act = dist.sample().clamp(0, 1)
                actions[a] = act.detach().numpy()
            concat = torch.cat([torch.tensor(obs[a], dtype=torch.float32) for a in env.possible_agents])
            obs, rewards, terms, truncs, _ = env.step(actions)
            total += float(np.mean(list(rewards.values()))) if rewards else 0.0
            batch_obs.append(concat)
            batch_rew.append(total)
            if all(terms.get(a, True) for a in terms):
                break
        returns.append(total)
        if batch_obs:
            ob = torch.stack(batch_obs)
            rw = torch.tensor(batch_rew, dtype=torch.float32)
            val = critic(ob).squeeze(-1)
            loss = torch.nn.functional.mse_loss(val, rw)
            opt_c.zero_grad()
            loss.backward()
            opt_c.step()
            for a in actors:
                # tiny entropy bonus step
                o0 = ob[:, :obs_dim]
                dist = actors[a](o0)
                ent = dist.entropy().mean()
                opt_a[a].zero_grad()
                (-0.01 * ent).backward()
                opt_a[a].step()
    return TrainResult("mappo", config.seed, returns, float(np.mean(returns)))


class VDNMixer(nn.Module):
    def forward(self, agent_qs: torch.Tensor) -> torch.Tensor:
        return agent_qs.sum(dim=-1)


def run_vdn(config: EnvConfig, episodes: int = 8) -> TrainResult:
    seed_everything(config.seed)
    env = ServiceIntentEnv(config)
    obs_dim = env._obs_dim
    n_act = 5  # discretized
    qnets = {a: nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, n_act)) for a in env.possible_agents}
    opts = {a: torch.optim.Adam(qnets[a].parameters(), lr=3e-3) for a in qnets}
    mixer = VDNMixer()
    returns: list[float] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=config.seed + ep)
        total = 0.0
        while env.agents:
            actions = {}
            qs = []
            for a in list(env.agents):
                o = torch.tensor(obs[a], dtype=torch.float32)
                q = qnets[a](o)
                ai = int(torch.argmax(q).item())
                act = np.zeros(env._act_dim, dtype=np.float32)
                act[0] = ai / max(n_act - 1, 1)
                act[2] = ai / max(n_act - 1, 1)
                actions[a] = act
                qs.append(q.max())
            obs, rewards, terms, truncs, _ = env.step(actions)
            r = float(np.mean(list(rewards.values()))) if rewards else 0.0
            total += r
            if qs:
                qsum = mixer(torch.stack(qs).unsqueeze(0))
                loss = (qsum - r) ** 2
                for a in list(env.possible_agents):
                    opts[a].zero_grad()
                loss.mean().backward()
                for a in opts:
                    opts[a].step()
            if all(terms.get(a, True) for a in terms):
                break
        returns.append(total)
    return TrainResult("vdn", config.seed, returns, float(np.mean(returns)))


def run_comm_dial(config: EnvConfig, episodes: int = 8) -> TrainResult:
    """Communication-aware baseline: differentiable discrete messages via Gumbel-Softmax."""
    from emergent_intent.comm.gumbel import discrete_symbols_from_onehot, gumbel_softmax_sample

    seed_everything(config.seed)
    cfg = config.model_copy(update={"comm_mode": config.comm_mode})
    env = ServiceIntentEnv(cfg)
    obs_dim = env._obs_dim
    vocab = config.vocab_size
    enc = nn.Sequential(nn.Linear(obs_dim, 32), nn.Tanh(), nn.Linear(32, vocab))
    opt = torch.optim.Adam(enc.parameters(), lr=3e-3)
    returns: list[float] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=config.seed + ep)
        total = 0.0
        while env.agents:
            actions = {}
            loss_terms = []
            for a in list(env.agents):
                o = torch.tensor(obs[a], dtype=torch.float32)
                logits = enc(o)
                onehot = gumbel_softmax_sample(logits, tau=1.0, hard=True)
                sym = int(discrete_symbols_from_onehot(onehot.unsqueeze(0)).item())
                act = np.array([0.5, 0.5, (sym + 1) / (vocab + 1), 0.0], dtype=np.float32)
                actions[a] = act
                loss_terms.append(onehot.pow(2).mean())
            obs, rewards, terms, truncs, _ = env.step(actions)
            r = float(np.mean(list(rewards.values()))) if rewards else 0.0
            total += r
            if loss_terms:
                loss = sum(loss_terms) / len(loss_terms) - r
                opt.zero_grad()
                loss.backward()
                opt.step()
            if all(terms.get(a, True) for a in terms):
                break
        returns.append(total)
    return TrainResult(
        "comm_dial",
        config.seed,
        returns,
        float(np.mean(returns)),
        extra={"note": "Gumbel-Softmax discrete messaging; not an emergent-language claim"},
    )


def save_checkpoint(result: TrainResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.__dict__, indent=2) + "\n", encoding="utf-8")
    result.checkpoint_path = str(path)
