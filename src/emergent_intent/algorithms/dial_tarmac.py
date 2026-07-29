"""Communication-aware baseline: DIAL + TarMAC-style targeted messaging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from emergent_intent.algorithms.networks import Critic, PPOConfig, compute_gae, nvec_from_env, obs_dim_from_env
from emergent_intent.comm.channel import AttentionTargeter, gumbel_softmax_sample
from emergent_intent.utils import detect_device, set_global_seed


class DialTarmacActor(nn.Module):
    """Policy that emits environment actions + discrete message logits + attention query."""

    def __init__(
        self,
        obs_dim: int,
        nvec: list[int],
        vocab_size: int,
        msg_length: int,
        n_agents: int,
        hidden: int = 64,
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
        )
        # control actions only (exclude message dims already in nvec if present)
        self.ctrl_nvec = list(nvec)
        self.heads = nn.ModuleList([nn.Linear(hidden, n) for n in self.ctrl_nvec])
        self.msg_head = nn.Linear(hidden, msg_length * vocab_size)
        self.vocab_size = vocab_size
        self.msg_length = msg_length
        self.targeter = AttentionTargeter(hidden, n_agents)
        self.hidden = hidden

    def forward(self, obs: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        ctrl_logits = [head(h) for head in self.heads]
        msg_logits = self.msg_head(h).view(-1, self.msg_length, self.vocab_size)
        return ctrl_logits, msg_logits, h


class DialTarmacTrainer:
    """DIAL-style differentiable discrete messages + TarMAC attention targeting.

    During training: Gumbel-Softmax relaxation on message logits.
    During execution: hard argmax symbols (actual discrete).
    """

    def __init__(
        self,
        env,
        vocab_size: int = 8,
        msg_length: int = 2,
        config: PPOConfig | None = None,
        seed: int = 0,
        prefer_cuda: bool = True,
        tau: float = 1.0,
    ):
        self.env = env
        self.config = config or PPOConfig()
        self.seed = seed
        self.tau = tau
        self.vocab_size = vocab_size
        self.msg_length = msg_length
        set_global_seed(seed)
        info = detect_device(prefer_cuda=prefer_cuda)
        self.device = torch.device(info.device)
        self.device_label = info.label
        self.agents = list(env.possible_agents)
        self.actors: dict[str, DialTarmacActor] = {}
        self.critics: dict[str, Critic] = {}
        self.opts: dict[str, torch.optim.Optimizer] = {}
        for a in self.agents:
            od = obs_dim_from_env(env, a)
            nv = nvec_from_env(env, a)
            actor = DialTarmacActor(
                od, nv, vocab_size, msg_length, n_agents=len(self.agents), hidden=self.config.hidden
            ).to(self.device)
            critic = Critic(od, hidden=self.config.hidden).to(self.device)
            self.actors[a] = actor
            self.critics[a] = critic
            self.opts[a] = torch.optim.Adam(
                list(actor.parameters()) + list(critic.parameters()), lr=self.config.lr
            )

    def select_actions(self, obs, deterministic: bool = False, hard_messages: bool = True):
        actions = {}
        logps = {}
        values = {}
        msg_soft = {}
        for a, o in obs.items():
            t = torch.as_tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0)
            ctrl_logits, msg_logits, h = self.actors[a](t)
            acts = []
            lp = 0.0
            for logits in ctrl_logits:
                dist = Categorical(logits=logits)
                act = logits.argmax(-1) if deterministic else dist.sample()
                acts.append(act)
                lp = lp + dist.log_prob(act)
            action = torch.stack(acts, dim=-1).squeeze(0).cpu().numpy().astype(np.int64)
            # messages: Gumbel for training signal stored separately; env uses hard symbols in action dims
            y = gumbel_softmax_sample(msg_logits.view(-1, self.vocab_size), tau=self.tau, hard=hard_messages)
            y = y.view(1, self.msg_length, self.vocab_size)
            symbols = y.argmax(-1).squeeze(0).cpu().numpy().astype(np.int64)
            # If action space includes message dims, overwrite trailing dims
            n_ctrl = len(ctrl_logits)
            if action.size >= n_ctrl + self.msg_length:
                action[n_ctrl : n_ctrl + self.msg_length] = symbols
            actions[a] = action
            logps[a] = float(lp.item() if torch.is_tensor(lp) else lp)
            with torch.no_grad():
                values[a] = float(self.critics[a](t).item())
            msg_soft[a] = y.detach()
        return actions, logps, values, msg_soft

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
            actions, logps, values, _ = self.select_actions(obs, hard_messages=True)
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
            "algorithm": "DIAL_TARMAC",
            "steps": steps,
            "updates": updates,
            "mean_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
            "n_episodes": len(ep_returns),
            "device": self.device_label,
            "evidence_class": "SYNTHETIC_SIM",
            "notes": [
                "Gumbel-Softmax used for discrete message training path",
                "Hard symbols used at env execution",
                "AttentionTargeter available for targeted routing",
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
                ctrl_logits, msg_logits, h = self.actors[a](obs)
                logps = []
                ents = []
                for i, logits in enumerate(ctrl_logits):
                    dist = Categorical(logits=logits)
                    if act.shape[-1] > i:
                        logps.append(dist.log_prob(act[:, i]))
                        ents.append(dist.entropy())
                logp = torch.stack(logps, dim=-1).sum(-1)
                ent = torch.stack(ents, dim=-1).sum(-1)
                # DIAL: encourage informative messages via entropy of msg distribution
                msg_probs = torch.softmax(msg_logits, dim=-1)
                msg_ent = -(msg_probs * (msg_probs + 1e-8).log()).sum(-1).mean()
                ratio = torch.exp(logp - old_logp)
                surr1 = ratio * adv_t
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_t
                policy_loss = -torch.min(surr1, surr2).mean() - cfg.ent_coef * ent.mean()
                # mild message entropy bonus (communication exploration)
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
                "algorithm": "DIAL_TARMAC",
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
