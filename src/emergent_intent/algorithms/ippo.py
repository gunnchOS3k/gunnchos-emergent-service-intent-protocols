"""Independent PPO (IPPO) for PettingZoo ParallelEnv."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from emergent_intent.algorithms.networks import (
    Critic,
    MultiDiscreteActor,
    PPOConfig,
    compute_gae,
    nvec_from_env,
    obs_dim_from_env,
)
from emergent_intent.utils import detect_device, set_global_seed


class IPPOTrainer:
    def __init__(self, env, config: PPOConfig | None = None, seed: int = 0, prefer_cuda: bool = True):
        self.env = env
        self.config = config or PPOConfig()
        self.seed = seed
        set_global_seed(seed)
        info = detect_device(prefer_cuda=prefer_cuda)
        self.device = torch.device(info.device)
        self.device_label = info.label
        self.agents = list(env.possible_agents)
        self.actors: dict[str, MultiDiscreteActor] = {}
        self.critics: dict[str, Critic] = {}
        self.opt_actors: dict[str, torch.optim.Optimizer] = {}
        self.opt_critics: dict[str, torch.optim.Optimizer] = {}
        for a in self.agents:
            od = obs_dim_from_env(env, a)
            nv = nvec_from_env(env, a)
            actor = MultiDiscreteActor(od, nv, hidden=self.config.hidden).to(self.device)
            critic = Critic(od, hidden=self.config.hidden).to(self.device)
            self.actors[a] = actor
            self.critics[a] = critic
            self.opt_actors[a] = torch.optim.Adam(actor.parameters(), lr=self.config.lr)
            self.opt_critics[a] = torch.optim.Adam(critic.parameters(), lr=self.config.lr)

    def select_actions(
        self, obs: dict[str, np.ndarray], deterministic: bool = False
    ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, float]]:
        actions: dict[str, np.ndarray] = {}
        logps: dict[str, float] = {}
        values: dict[str, float] = {}
        for a in obs:
            o = torch.as_tensor(obs[a], dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                act, logp, _ = self.actors[a].act(o, deterministic=deterministic)
                v = self.critics[a](o)
            actions[a] = act.squeeze(0).cpu().numpy().astype(np.int64)
            logps[a] = float(logp.item())
            values[a] = float(v.item())
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
                if len(buf[self.agents[0]]["obs"]) >= cfg.rollout_steps or done_env:
                    self._update(buf)
                    updates += 1
                    for a in self.agents:
                        for k in buf[a]:
                            buf[a][k] = []
                obs, _ = self.env.reset(seed=self.seed + steps)
            else:
                obs = next_obs

        return {
            "algorithm": "IPPO",
            "steps": steps,
            "updates": updates,
            "mean_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
            "n_episodes": len(ep_returns),
            "device": self.device_label,
            "evidence_class": "SYNTHETIC_SIM",
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
                logp, ent = self.actors[a].evaluate(obs, act)
                ratio = torch.exp(logp - old_logp)
                surr1 = ratio * adv_t
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_t
                policy_loss = -torch.min(surr1, surr2).mean() - cfg.ent_coef * ent.mean()
                v = self.critics[a](obs)
                value_loss = nn.functional.mse_loss(v, ret_t)
                self.opt_actors[a].zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.actors[a].parameters(), cfg.max_grad_norm)
                self.opt_actors[a].step()
                self.opt_critics[a].zero_grad()
                (cfg.vf_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critics[a].parameters(), cfg.max_grad_norm)
                self.opt_critics[a].step()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "algorithm": "IPPO",
            "seed": self.seed,
            "actors": {a: self.actors[a].state_dict() for a in self.agents},
            "critics": {a: self.critics[a].state_dict() for a in self.agents},
        }
        torch.save(payload, path)

    def load(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=True)
        for a in self.agents:
            self.actors[a].load_state_dict(payload["actors"][a])
            self.critics[a].load_state_dict(payload["critics"][a])
