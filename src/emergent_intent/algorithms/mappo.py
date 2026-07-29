"""Multi-Agent PPO with centralized critic (MAPPO)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from emergent_intent.algorithms.networks import (
    CentralCritic,
    MultiDiscreteActor,
    PPOConfig,
    compute_gae,
    flatten_state,
    nvec_from_env,
    obs_dim_from_env,
)
from emergent_intent.utils import detect_device, set_global_seed


class MAPPOTrainer:
    def __init__(self, env, config: PPOConfig | None = None, seed: int = 0, prefer_cuda: bool = True):
        self.env = env
        self.config = config or PPOConfig()
        self.seed = seed
        set_global_seed(seed)
        info = detect_device(prefer_cuda=prefer_cuda)
        self.device = torch.device(info.device)
        self.device_label = info.label
        self.agents = list(env.possible_agents)
        self.actors = {
            a: MultiDiscreteActor(
                obs_dim_from_env(env, a), nvec_from_env(env, a), hidden=self.config.hidden
            ).to(self.device)
            for a in self.agents
        }
        # probe state dim
        env.reset(seed=seed)
        st = flatten_state(env.state())
        self.state_dim = int(st.size)
        self.critic = CentralCritic(self.state_dim, hidden=self.config.hidden).to(self.device)
        params = list(self.critic.parameters())
        for a in self.agents:
            params += list(self.actors[a].parameters())
        self.opt = torch.optim.Adam(params, lr=self.config.lr)

    def select_actions(self, obs, deterministic: bool = False):
        actions, logps = {}, {}
        for a in obs:
            o = torch.as_tensor(obs[a], dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                act, logp, _ = self.actors[a].act(o, deterministic=deterministic)
            actions[a] = act.squeeze(0).cpu().numpy().astype(np.int64)
            logps[a] = float(logp.item())
        st = torch.as_tensor(flatten_state(self.env.state()), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            v = float(self.critic(st.unsqueeze(0)).item())
        return actions, logps, v

    def train(self, total_steps: int = 512) -> dict[str, Any]:
        cfg = self.config
        buf = {k: [] for k in ("obs", "act", "rew", "done", "logp", "val", "state", "agent")}
        obs, _ = self.env.reset(seed=self.seed)
        ep_returns: list[float] = []
        ep_ret = 0.0
        steps = 0
        updates = 0
        while steps < total_steps:
            actions, logps, value = self.select_actions(obs)
            state = flatten_state(self.env.state())
            next_obs, rewards, terms, truncs, _ = self.env.step(actions)
            done_env = (not self.env.agents) or any(truncs.values()) or any(terms.values())
            team_r = float(np.mean(list(rewards.values()))) if rewards else 0.0
            for a in list(obs.keys()):
                if a not in rewards:
                    continue
                buf["obs"].append(obs[a])
                buf["act"].append(actions[a])
                buf["rew"].append(team_r)
                buf["done"].append(float(done_env))
                buf["logp"].append(logps[a])
                buf["val"].append(value)
                buf["state"].append(state)
                buf["agent"].append(a)
            ep_ret += team_r
            steps += 1
            if done_env or steps >= total_steps:
                ep_returns.append(ep_ret)
                ep_ret = 0.0
                if len(buf["rew"]) >= cfg.rollout_steps or done_env:
                    self._update(buf)
                    updates += 1
                    for k in buf:
                        buf[k] = []
                obs, _ = self.env.reset(seed=self.seed + steps)
            else:
                obs = next_obs
        return {
            "algorithm": "MAPPO",
            "steps": steps,
            "updates": updates,
            "mean_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
            "n_episodes": len(ep_returns),
            "device": self.device_label,
            "evidence_class": "SYNTHETIC_SIM",
        }

    def _update(self, buf: dict[str, list]) -> None:
        cfg = self.config
        if len(buf["rew"]) < 2:
            return
        # Update per agent slices
        agents_in_buf = buf["agent"]
        for a in self.agents:
            idx = [i for i, x in enumerate(agents_in_buf) if x == a]
            if len(idx) < 2:
                continue
            obs = torch.as_tensor(np.asarray([buf["obs"][i] for i in idx]), dtype=torch.float32, device=self.device)
            act = torch.as_tensor(np.asarray([buf["act"][i] for i in idx]), dtype=torch.int64, device=self.device)
            old_logp = torch.as_tensor([buf["logp"][i] for i in idx], dtype=torch.float32, device=self.device)
            rew = np.asarray([buf["rew"][i] for i in idx], dtype=np.float32)
            done = np.asarray([buf["done"][i] for i in idx], dtype=np.float32)
            val = np.asarray([buf["val"][i] for i in idx], dtype=np.float32)
            st = torch.as_tensor(np.asarray([buf["state"][i] for i in idx]), dtype=torch.float32, device=self.device)
            adv, ret = compute_gae(rew, val, done, cfg.gamma, cfg.gae_lambda)
            adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
            ret_t = torch.as_tensor(ret, dtype=torch.float32, device=self.device)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            for _ in range(cfg.epochs):
                logp, ent = self.actors[a].evaluate(obs, act)
                ratio = torch.exp(logp - old_logp)
                surr1 = ratio * adv_t
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_t
                policy_loss = -torch.min(surr1, surr2).mean() - cfg.ent_coef * ent.mean()
                v = self.critic(st)
                value_loss = nn.functional.mse_loss(v, ret_t)
                loss = policy_loss + cfg.vf_coef * value_loss
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actors[a].parameters()) + list(self.critic.parameters()),
                    cfg.max_grad_norm,
                )
                self.opt.step()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": "MAPPO",
                "actors": {a: self.actors[a].state_dict() for a in self.agents},
                "critic": self.critic.state_dict(),
                "seed": self.seed,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=True)
        for a in self.agents:
            self.actors[a].load_state_dict(payload["actors"][a])
        self.critic.load_state_dict(payload["critic"])
