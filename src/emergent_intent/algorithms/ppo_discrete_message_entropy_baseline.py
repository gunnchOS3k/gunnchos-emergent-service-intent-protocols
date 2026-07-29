"""Truthful rename of the previous DIAL/TarMAC-labeled trainer.

This baseline uses PPO with discrete message action dims and a mild message-entropy
bonus. It does NOT implement differentiable communication through a channel, nor
attention-based routing. Formerly mislabeled as DIAL/TarMAC.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from emergent_intent.algorithms.networks import Critic, PPOConfig, compute_gae, nvec_from_env, obs_dim_from_env
from emergent_intent.utils import detect_device, set_global_seed

ALGORITHM_NAME = "ppo_discrete_message_entropy_baseline"


class DiscreteMessageEntropyActor(nn.Module):
    def __init__(self, obs_dim: int, nvec: list[int], hidden: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, n) for n in nvec])
        self.nvec = list(nvec)

    def forward(self, obs: torch.Tensor) -> list[torch.Tensor]:
        h = self.backbone(obs)
        return [head(h) for head in self.heads]


class PPODiscreteMessageEntropyBaseline:
    """Independent PPO with entropy bonus on the message action dimension."""

    def __init__(
        self,
        env,
        config: PPOConfig | None = None,
        seed: int = 0,
        prefer_cuda: bool = True,
        **_ignored,
    ):
        self.env = env
        self.config = config or PPOConfig()
        self.seed = seed
        set_global_seed(seed)
        info = detect_device(prefer_cuda=prefer_cuda)
        self.device = torch.device(info.device)
        self.device_label = info.label
        self.agents = list(env.possible_agents)
        self.actors: dict[str, DiscreteMessageEntropyActor] = {}
        self.critics: dict[str, Critic] = {}
        self.opts: dict[str, torch.optim.Optimizer] = {}
        # message factor is second-to-last in env nvec layout
        self.msg_dim_index = max(len(nvec_from_env(env, self.agents[0])) - 2, 0)
        for a in self.agents:
            od = obs_dim_from_env(env, a)
            nv = nvec_from_env(env, a)
            actor = DiscreteMessageEntropyActor(od, nv, hidden=self.config.hidden).to(self.device)
            critic = Critic(od, hidden=self.config.hidden).to(self.device)
            self.actors[a] = actor
            self.critics[a] = critic
            self.opts[a] = torch.optim.Adam(
                list(actor.parameters()) + list(critic.parameters()), lr=self.config.lr
            )

    def select_actions(self, obs, deterministic: bool = False, **_kw):
        actions, logps, values = {}, {}, {}
        for a, o in obs.items():
            t = torch.as_tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits_list = self.actors[a](t)
            acts, lp = [], 0.0
            for logits in logits_list:
                dist = Categorical(logits=logits)
                act = logits.argmax(-1) if deterministic else dist.sample()
                acts.append(act)
                lp = lp + dist.log_prob(act)
            actions[a] = torch.stack(acts, dim=-1).squeeze(0).cpu().numpy().astype(np.int64)
            logps[a] = float(lp.item())
            with torch.no_grad():
                values[a] = float(self.critics[a](t).item())
        return actions, logps, values

    def train(self, total_steps: int = 512) -> dict[str, Any]:
        cfg = self.config
        buf: dict[str, dict[str, list]] = {
            a: {k: [] for k in ("obs", "act", "rew", "done", "logp", "val")} for a in self.agents
        }
        obs, _ = self.env.reset(seed=self.seed)
        ep_returns: list[float] = []
        ep_ret = 0.0
        steps = 0
        updates = 0
        while steps < total_steps:
            actions, logps, values = self.select_actions(obs)
            next_obs, rewards, terms, truncs, _ = self.env.step(actions)
            done_env = (not self.env.agents) or any(truncs.values()) or any(terms.values())
            for a in list(obs.keys()):
                if a not in rewards:
                    continue
                buf[a]["obs"].append(obs[a])
                buf[a]["act"].append(actions[a])
                buf[a]["rew"].append(float(rewards[a]))
                buf[a]["done"].append(float(done_env))
                buf[a]["logp"].append(logps[a])
                buf[a]["val"].append(values[a])
            if rewards:
                ep_ret += float(np.mean(list(rewards.values())))
            steps += 1
            if done_env or steps >= total_steps:
                ep_returns.append(ep_ret)
                ep_ret = 0.0
                if len(buf[self.agents[0]]["obs"]) >= min(cfg.rollout_steps, 16) or done_env:
                    self._update(buf)
                    updates += 1
                    for a in self.agents:
                        for k in buf[a]:
                            buf[a][k] = []
                obs, _ = self.env.reset(seed=self.seed + steps)
            else:
                obs = next_obs
        return {
            "algorithm": ALGORITHM_NAME,
            "steps": steps,
            "updates": updates,
            "mean_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
            "n_episodes": len(ep_returns),
            "device": self.device_label,
            "evidence_class": "SYNTHETIC_SIM",
            "notes": [
                "NOT faithful DIAL",
                "NOT TarMAC",
                "PPO on MultiDiscrete including message dims + message entropy bonus",
            ],
        }

    def _update(self, buf: dict[str, dict[str, list]]) -> None:
        cfg = self.config
        for a in self.agents:
            if len(buf[a]["obs"]) < 2:
                continue
            obs = torch.as_tensor(np.asarray(buf[a]["obs"]), dtype=torch.float32, device=self.device)
            act = torch.as_tensor(np.asarray(buf[a]["act"]), dtype=torch.int64, device=self.device)
            old_logp = torch.as_tensor(buf[a]["logp"], dtype=torch.float32, device=self.device)
            rew = np.asarray(buf[a]["rew"], dtype=np.float32)
            done = np.asarray(buf[a]["done"], dtype=np.float32)
            val = np.asarray(buf[a]["val"], dtype=np.float32)
            adv, ret = compute_gae(rew, val, done, cfg.gamma, cfg.gae_lambda)
            adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
            ret_t = torch.as_tensor(ret, dtype=torch.float32, device=self.device)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            for _ in range(cfg.epochs):
                logits_list = self.actors[a](obs)
                logps, ents, msg_ent = [], [], None
                for i, logits in enumerate(logits_list):
                    dist = Categorical(logits=logits)
                    if act.shape[-1] > i:
                        logps.append(dist.log_prob(act[:, i]))
                        ents.append(dist.entropy())
                        if i == self.msg_dim_index:
                            msg_ent = dist.entropy().mean()
                logp = torch.stack(logps, dim=-1).sum(-1)
                ent = torch.stack(ents, dim=-1).sum(-1)
                ratio = torch.exp(logp - old_logp)
                surr1 = ratio * adv_t
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_t
                policy_loss = -torch.min(surr1, surr2).mean() - cfg.ent_coef * ent.mean()
                if msg_ent is not None:
                    policy_loss = policy_loss - 0.001 * msg_ent
                v = self.critics[a](obs)
                value_loss = nn.functional.mse_loss(v, ret_t)
                loss = policy_loss + cfg.vf_coef * value_loss
                self.opts[a].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actors[a].parameters()) + list(self.critics[a].parameters()),
                    cfg.max_grad_norm,
                )
                self.opts[a].step()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": ALGORITHM_NAME,
                "actors": {a: self.actors[a].state_dict() for a in self.agents},
                "critics": {a: self.critics[a].state_dict() for a in self.agents},
                "seed": self.seed,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=True)
        for a in self.agents:
            self.actors[a].load_state_dict(payload["actors"][a])
            self.critics[a].load_state_dict(payload["critics"][a])


# Back-compat alias — intentionally points to the truthful baseline name.
DialTarmacTrainer = PPODiscreteMessageEntropyBaseline
