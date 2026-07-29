"""Faithful DIAL-style differentiable discrete communication.

Training: Gumbel-Softmax relaxed messages consumed by a receiver head; receiver task
loss propagates gradients through the channel soft message into the sender message-head.
Evaluation: hard argmax symbols.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from emergent_intent.algorithms.networks import PPOConfig, compute_gae, nvec_from_env, obs_dim_from_env
from emergent_intent.comm.channel import gumbel_softmax_sample
from emergent_intent.utils import detect_device, set_global_seed


class DialSender(nn.Module):
    def __init__(self, obs_dim: int, nvec: list[int], vocab: int, msg_len: int, hidden: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
        )
        # control actions exclude message + target trailing dims when present
        self.ctrl_nvec = list(nvec[:-2]) if len(nvec) >= 2 else list(nvec)
        self.heads = nn.ModuleList([nn.Linear(hidden, n) for n in self.ctrl_nvec])
        self.msg_head = nn.Linear(hidden, msg_len * vocab)
        self.vocab = vocab
        self.msg_len = msg_len
        self.hidden = hidden

    def forward(self, obs: torch.Tensor):
        h = self.backbone(obs)
        ctrl = [head(h) for head in self.heads]
        msg_logits = self.msg_head(h).view(-1, self.msg_len, self.vocab)
        return ctrl, msg_logits, h


class DialReceiver(nn.Module):
    """Consumes local obs + soft/hard message embedding and predicts task value/logits."""

    def __init__(self, obs_dim: int, msg_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + msg_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, msg_embed: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, msg_embed], dim=-1)).squeeze(-1)


class DialTrainer:
    """Faithful DIAL: relaxed train / hard eval, gradient through channel to sender."""

    def __init__(
        self,
        env,
        vocab_size: int = 8,
        msg_length: int = 2,
        config: PPOConfig | None = None,
        seed: int = 0,
        prefer_cuda: bool = True,
        tau: float = 1.0,
        comm_cost_coef: float = 0.01,
    ):
        self.env = env
        self.config = config or PPOConfig()
        self.seed = seed
        self.tau = tau
        self.comm_cost_coef = comm_cost_coef
        self.vocab_size = vocab_size
        self.msg_length = msg_length
        set_global_seed(seed)
        info = detect_device(prefer_cuda=prefer_cuda)
        self.device = torch.device(info.device)
        self.device_label = info.label
        self.agents = list(env.possible_agents)
        self.senders: dict[str, DialSender] = {}
        self.receivers: dict[str, DialReceiver] = {}
        self.opts: dict[str, torch.optim.Optimizer] = {}
        msg_dim = msg_length * vocab_size
        for a in self.agents:
            od = obs_dim_from_env(env, a)
            nv = nvec_from_env(env, a)
            sender = DialSender(od, nv, vocab_size, msg_length, hidden=self.config.hidden).to(
                self.device
            )
            receiver = DialReceiver(od, msg_dim, hidden=self.config.hidden).to(self.device)
            self.senders[a] = sender
            self.receivers[a] = receiver
            self.opts[a] = torch.optim.Adam(
                list(sender.parameters()) + list(receiver.parameters()), lr=self.config.lr
            )

    def _msg_embed(self, msg_logits: torch.Tensor, hard: bool) -> tuple[torch.Tensor, torch.Tensor]:
        # msg_logits: [B, L, V]
        B, L, V = msg_logits.shape
        flat = msg_logits.reshape(B * L, V)
        onehot = gumbel_softmax_sample(flat, tau=self.tau, hard=hard)
        onehot = onehot.view(B, L, V)
        embed = onehot.reshape(B, L * V)
        symbols = onehot.argmax(-1)
        return embed, symbols

    def select_actions(self, obs, deterministic: bool = False, hard_messages: bool = True):
        actions, logps, values, soft_msgs = {}, {}, {}, {}
        embeds = {}
        for a, o in obs.items():
            t = torch.as_tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0)
            ctrl, msg_logits, _ = self.senders[a](t)
            acts, lp = [], 0.0
            for logits in ctrl:
                dist = Categorical(logits=logits)
                act = logits.argmax(-1) if deterministic else dist.sample()
                acts.append(act)
                lp = lp + dist.log_prob(act)
            embed, symbols = self._msg_embed(msg_logits, hard=hard_messages)
            # build MultiDiscrete env action: ctrl + msg_token + target0
            ctrl_np = torch.stack(acts, dim=-1).squeeze(0).cpu().numpy().astype(np.int64)
            msg_token = int(symbols[0, 0].item()) + 1  # 0 reserved silence; shift
            nvec = nvec_from_env(self.env, a)
            full = np.zeros(len(nvec), dtype=np.int64)
            full[: ctrl_np.size] = ctrl_np
            if len(nvec) >= 2:
                full[-2] = min(msg_token, nvec[-2] - 1)
                full[-1] = 0
            actions[a] = full
            logps[a] = float(lp.item())
            with torch.no_grad():
                values[a] = float(self.receivers[a](t, embed).item())
            soft_msgs[a] = msg_logits
            embeds[a] = embed
        return actions, logps, values, soft_msgs, embeds

    def differentiable_receiver_loss(
        self, sender: str, receiver: str, sender_obs: torch.Tensor, receiver_obs: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Task loss at receiver that backprops into sender message-head."""
        _, msg_logits, _ = self.senders[sender](sender_obs)
        embed, _ = self._msg_embed(msg_logits, hard=False)
        # Channel noise (straight-through friendly): mix uniform noise
        noise = torch.rand_like(embed) * 0.01
        noisy = embed + noise
        # bit-cost regularizer: encourage sparse soft messages
        probs = torch.softmax(msg_logits, dim=-1)
        bit_cost = (probs * torch.log(probs.clamp_min(1e-8))).sum() * -self.comm_cost_coef
        pred = self.receivers[receiver](receiver_obs, noisy)
        task_loss = F.mse_loss(pred, target)
        return task_loss + bit_cost, msg_logits

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
            actions, logps, values, soft_msgs, embeds = self.select_actions(obs, hard_messages=True)
            # Differentiable communication update between first two agents when available
            if len(self.agents) >= 2 and steps % 2 == 0:
                s_a, r_a = self.agents[0], self.agents[1]
                if s_a in obs and r_a in obs:
                    so = torch.as_tensor(obs[s_a], dtype=torch.float32, device=self.device).unsqueeze(0)
                    ro = torch.as_tensor(obs[r_a], dtype=torch.float32, device=self.device).unsqueeze(0)
                    # target = reward bootstrapping proxy from current value estimate
                    tgt = torch.as_tensor([values.get(r_a, 0.0)], dtype=torch.float32, device=self.device)
                    loss, _ = self.differentiable_receiver_loss(s_a, r_a, so, ro, tgt)
                    self.opts[s_a].zero_grad()
                    self.opts[r_a].zero_grad()
                    loss.backward()
                    self.opts[s_a].step()
                    self.opts[r_a].step()

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
                    self._ppo_update(buf)
                    updates += 1
                    for a in self.agents:
                        for k in buf[a]:
                            buf[a][k] = []
                obs, _ = self.env.reset(seed=self.seed + steps)
            else:
                obs = next_obs
        return {
            "algorithm": "DIAL",
            "steps": steps,
            "updates": updates,
            "mean_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
            "n_episodes": len(ep_returns),
            "device": self.device_label,
            "evidence_class": "SYNTHETIC_SIM",
            "notes": [
                "Gumbel-Softmax relaxed messages during differentiable path",
                "Hard symbols at env execution / evaluation",
                "Receiver loss backprops to sender message-head",
            ],
        }

    def _ppo_update(self, buf: dict[str, dict[str, list]]) -> None:
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
                ctrl, msg_logits, _ = self.senders[a](obs)
                logps, ents = [], []
                for i, logits in enumerate(ctrl):
                    dist = Categorical(logits=logits)
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
                embed, _ = self._msg_embed(msg_logits, hard=False)
                v = self.receivers[a](obs, embed)
                value_loss = F.mse_loss(v, ret_t)
                # communication cost on soft distribution
                probs = torch.softmax(msg_logits, dim=-1)
                comm_cost = -self.comm_cost_coef * (probs * torch.log(probs.clamp_min(1e-8))).sum(-1).mean()
                loss = policy_loss + cfg.vf_coef * value_loss + comm_cost
                self.opts[a].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.senders[a].parameters()) + list(self.receivers[a].parameters()),
                    cfg.max_grad_norm,
                )
                self.opts[a].step()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": "DIAL",
                "senders": {a: self.senders[a].state_dict() for a in self.agents},
                "receivers": {a: self.receivers[a].state_dict() for a in self.agents},
                "seed": self.seed,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=True)
        for a in self.agents:
            self.senders[a].load_state_dict(payload["senders"][a])
            self.receivers[a].load_state_dict(payload["receivers"][a])
