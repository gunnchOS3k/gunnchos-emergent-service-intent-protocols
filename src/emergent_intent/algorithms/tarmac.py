"""TarMAC-style targeted communication with learned attention routing.

Sender emits key/value; receivers emit queries; attention weights route messages into
the receiving policy. Attention diagnostics are recorded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from emergent_intent.algorithms.networks import Critic, PPOConfig, compute_gae, nvec_from_env, obs_dim_from_env
from emergent_intent.comm.attention import TargetedMessage
from emergent_intent.utils import detect_device, set_global_seed


class TarMACPolicy(nn.Module):
    def __init__(self, obs_dim: int, nvec: list[int], d_model: int = 32, n_agents: int = 2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(obs_dim, d_model), nn.Tanh())
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.ctrl_nvec = list(nvec[:-2]) if len(nvec) >= 2 else list(nvec)
        self.heads = nn.ModuleList([nn.Linear(d_model * 2, n) for n in self.ctrl_nvec])
        self.msg_token = nn.Linear(d_model * 2, nvec[-2] if len(nvec) >= 2 else 2)
        self.target_head = nn.Linear(d_model, n_agents)
        self.attn = TargetedMessage(d_model)
        self.d_model = d_model
        self.n_agents = n_agents

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.enc(obs)

    def route(
        self, own_h: torch.Tensor, others_h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # own_h: [B,D], others_h: [B,N,D]
        msg, attn = self.attn(own_h, others_h)
        return msg, attn

    def act_from(self, h: torch.Tensor, routed: torch.Tensor, deterministic: bool = False):
        x = torch.cat([h, routed], dim=-1)
        acts, logps = [], []
        for head in self.heads:
            logits = head(x)
            dist = Categorical(logits=logits)
            a = logits.argmax(-1) if deterministic else dist.sample()
            acts.append(a)
            logps.append(dist.log_prob(a))
        msg_logits = self.msg_token(x)
        msg_dist = Categorical(logits=msg_logits)
        msg_a = msg_logits.argmax(-1) if deterministic else msg_dist.sample()
        tgt_logits = self.target_head(h)
        tgt_dist = Categorical(logits=tgt_logits)
        tgt_a = tgt_logits.argmax(-1) if deterministic else tgt_dist.sample()
        logp = torch.stack(logps, dim=-1).sum(-1) + msg_dist.log_prob(msg_a) + tgt_dist.log_prob(tgt_a)
        return acts, msg_a, tgt_a, logp, x


class TarMACTrainer:
    def __init__(
        self,
        env,
        config: PPOConfig | None = None,
        seed: int = 0,
        prefer_cuda: bool = True,
        d_model: int = 32,
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
        self.n_agents = len(self.agents)
        self.policies = {
            a: TarMACPolicy(
                obs_dim_from_env(env, a),
                nvec_from_env(env, a),
                d_model=d_model,
                n_agents=self.n_agents,
            ).to(self.device)
            for a in self.agents
        }
        self.critics = {
            a: Critic(obs_dim_from_env(env, a) + d_model, hidden=self.config.hidden).to(self.device)
            for a in self.agents
        }
        self.opts = {
            a: torch.optim.Adam(
                list(self.policies[a].parameters()) + list(self.critics[a].parameters()),
                lr=self.config.lr,
            )
            for a in self.agents
        }
        self.last_attention: dict[str, np.ndarray] = {}

    def select_actions(self, obs, deterministic: bool = False):
        hs = {}
        for a, o in obs.items():
            t = torch.as_tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0)
            hs[a] = self.policies[a].encode(t)
        # stack others for each agent
        actions, logps, values = {}, {}, {}
        for a in obs:
            others = []
            for b in self.agents:
                if b == a:
                    continue
                if b in hs:
                    others.append(hs[b])
                else:
                    others.append(torch.zeros_like(hs[a]))
            if not others:
                others = [hs[a]]
            others_h = torch.stack(others, dim=1)  # [B,N-1,D]
            # pad to n_agents for attention module consistency
            if others_h.size(1) < self.n_agents:
                pad = self.n_agents - others_h.size(1)
                others_h = F.pad(others_h, (0, 0, 0, pad))
            routed, attn = self.policies[a].route(hs[a], others_h)
            self.last_attention[a] = attn.detach().cpu().numpy()
            acts, msg_a, tgt_a, logp, fused = self.policies[a].act_from(
                hs[a], routed, deterministic=deterministic
            )
            nvec = nvec_from_env(self.env, a)
            full = np.zeros(len(nvec), dtype=np.int64)
            ctrl = torch.stack(acts, dim=-1).squeeze(0).cpu().numpy().astype(np.int64)
            full[: ctrl.size] = ctrl
            if len(nvec) >= 2:
                full[-2] = int(msg_a.item()) % nvec[-2]
                full[-1] = int(tgt_a.item()) % nvec[-1]
            actions[a] = full
            logps[a] = float(logp.item())
            o = torch.as_tensor(obs[a], dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                values[a] = float(self.critics[a](torch.cat([o, routed], dim=-1)).item())
        return actions, logps, values

    def attention_diagnostics(self) -> dict[str, Any]:
        out = {}
        for a, attn in self.last_attention.items():
            out[a] = {
                "mean": float(np.mean(attn)),
                "entropy": float(-(attn * np.log(attn + 1e-8)).sum(-1).mean()),
                "argmax": int(np.argmax(attn.ravel())),
            }
        return out

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
            "algorithm": "TARMAC",
            "steps": steps,
            "updates": updates,
            "mean_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
            "n_episodes": len(ep_returns),
            "device": self.device_label,
            "evidence_class": "SYNTHETIC_SIM",
            "attention": self.attention_diagnostics(),
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
                h = self.policies[a].encode(obs)
                # self-attention surrogate using tiled hidden as candidates
                others = h.unsqueeze(1).expand(-1, self.n_agents, -1)
                routed, _ = self.policies[a].route(h, others)
                acts, msg_a, tgt_a, logp, fused = self.policies[a].act_from(h, routed)
                # recompute logp for stored actions approximately via heads
                x = fused
                logps = []
                ents = []
                for i, head in enumerate(self.policies[a].heads):
                    dist = Categorical(logits=head(x))
                    if act.shape[-1] > i:
                        logps.append(dist.log_prob(act[:, i]))
                        ents.append(dist.entropy())
                if not logps:
                    continue
                logp = torch.stack(logps, dim=-1).sum(-1)
                ent = torch.stack(ents, dim=-1).sum(-1)
                ratio = torch.exp(logp - old_logp)
                surr1 = ratio * adv_t
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_t
                policy_loss = -torch.min(surr1, surr2).mean() - cfg.ent_coef * ent.mean()
                v = self.critics[a](torch.cat([obs, routed.detach()], dim=-1))
                value_loss = F.mse_loss(v, ret_t)
                loss = policy_loss + cfg.vf_coef * value_loss
                self.opts[a].zero_grad()
                loss.backward()
                self.opts[a].step()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": "TARMAC",
                "policies": {a: self.policies[a].state_dict() for a in self.agents},
                "critics": {a: self.critics[a].state_dict() for a in self.agents},
                "seed": self.seed,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=True)
        for a in self.agents:
            self.policies[a].load_state_dict(payload["policies"][a])
            self.critics[a].load_state_dict(payload["critics"][a])
