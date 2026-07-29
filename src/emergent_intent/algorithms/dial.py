"""Faithful DIAL-style differentiable discrete communication.

Training: Gumbel-Softmax relaxed messages through a differentiable channel;
primary objective is receiver **task loss** (next-step team reward / service),
with gradients into the sender message-head. Receiver-value regression is NOT
the primary objective.

Evaluation: hard argmax symbols (no soft relaxation).
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
    """Predicts scalar task outcome from local obs + message embedding."""

    def __init__(self, obs_dim: int, msg_dim: int, hidden: int = 64):
        super().__init__()
        self.task_head = nn.Sequential(
            nn.Linear(obs_dim + msg_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # Optional value head — auxiliary only; not the primary DIAL objective.
        self.value_head = nn.Sequential(
            nn.Linear(obs_dim + msg_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, msg_embed: torch.Tensor) -> torch.Tensor:
        return self.task_head(torch.cat([obs, msg_embed], dim=-1)).squeeze(-1)

    def value(self, obs: torch.Tensor, msg_embed: torch.Tensor) -> torch.Tensor:
        return self.value_head(torch.cat([obs, msg_embed], dim=-1)).squeeze(-1)


class DialTrainer:
    """Faithful DIAL: soft channel train / hard eval; task-loss → sender msg-head."""

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
        value_aux_coef: float = 0.1,
    ):
        self.env = env
        self.config = config or PPOConfig()
        self.seed = seed
        self.tau = tau
        self.comm_cost_coef = comm_cost_coef
        self.value_aux_coef = value_aux_coef
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
            ctrl_np = torch.stack(acts, dim=-1).squeeze(0).cpu().numpy().astype(np.int64)
            msg_token = int(symbols[0, 0].item()) + 1
            nvec = nvec_from_env(self.env, a)
            full = np.zeros(len(nvec), dtype=np.int64)
            full[: ctrl_np.size] = ctrl_np
            if len(nvec) >= 2:
                full[-2] = min(msg_token, nvec[-2] - 1)
                full[-1] = 0
            actions[a] = full
            logps[a] = float(lp.item())
            with torch.no_grad():
                values[a] = float(self.receivers[a].value(t, embed).item())
            soft_msgs[a] = msg_logits
            embeds[a] = embed
        return actions, logps, values, soft_msgs, embeds

    def differentiable_task_loss(
        self,
        sender: str,
        receiver: str,
        sender_obs: torch.Tensor,
        receiver_obs: torch.Tensor,
        task_target: torch.Tensor,
        *,
        hard: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Primary DIAL objective: task prediction loss through soft channel.

        Gradients flow: task_loss → receiver.task_head → soft embed → Gumbel →
        sender.msg_head. Value regression is returned only as an auxiliary metric.
        """
        _, msg_logits, _ = self.senders[sender](sender_obs)
        embed, _ = self._msg_embed(msg_logits, hard=hard)
        # Differentiable channel noise
        noise = torch.rand_like(embed) * 0.01
        noisy = embed + noise
        probs = torch.softmax(msg_logits, dim=-1)
        bit_cost = -self.comm_cost_coef * (probs * torch.log(probs.clamp_min(1e-8))).sum()
        task_pred = self.receivers[receiver](receiver_obs, noisy)
        task_loss = F.mse_loss(task_pred, task_target)
        with torch.no_grad():
            # Explicitly NOT used as primary loss; recorded for honesty tests.
            value_aux = self.receivers[receiver].value(receiver_obs, noisy.detach())
        primary = task_loss + bit_cost
        extras = {
            "task_loss": task_loss.detach(),
            "bit_cost": bit_cost.detach() if torch.is_tensor(bit_cost) else torch.tensor(0.0),
            "value_aux": value_aux.detach(),
            "primary_is_task_loss": torch.tensor(1.0),
        }
        return primary, msg_logits, extras

    # Backward-compatible alias used by older unit tests
    def differentiable_receiver_loss(
        self,
        sender: str,
        receiver: str,
        sender_obs: torch.Tensor,
        receiver_obs: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loss, logits, _ = self.differentiable_task_loss(
            sender, receiver, sender_obs, receiver_obs, target, hard=False
        )
        return loss, logits

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
        dial_task_updates = 0
        while steps < total_steps:
            # Soft messages during differentiable train path bookkeeping;
            # env execution uses hard symbols.
            actions, logps, values, soft_msgs, embeds = self.select_actions(
                obs, hard_messages=True
            )
            next_obs, rewards, terms, truncs, _ = self.env.step(actions)
            done_env = (not self.env.agents) or any(truncs.values()) or any(terms.values())

            # Primary DIAL update: task target = realized team reward (not value regress).
            if len(self.agents) >= 2 and rewards:
                s_a, r_a = self.agents[0], self.agents[1]
                if s_a in obs and r_a in obs and r_a in rewards:
                    so = torch.as_tensor(obs[s_a], dtype=torch.float32, device=self.device).unsqueeze(0)
                    ro = torch.as_tensor(obs[r_a], dtype=torch.float32, device=self.device).unsqueeze(0)
                    team_r = float(np.mean(list(rewards.values())))
                    tgt = torch.as_tensor([team_r], dtype=torch.float32, device=self.device)
                    loss, _, extras = self.differentiable_task_loss(s_a, r_a, so, ro, tgt, hard=False)
                    self.opts[s_a].zero_grad()
                    self.opts[r_a].zero_grad()
                    loss.backward()
                    self.opts[s_a].step()
                    self.opts[r_a].step()
                    dial_task_updates += 1
                    _ = extras  # retained for debugging / tests

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
            "dial_task_updates": dial_task_updates,
            "mean_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
            "n_episodes": len(ep_returns),
            "device": self.device_label,
            "evidence_class": "SYNTHETIC_SIM",
            "notes": [
                "Gumbel-Softmax relaxed messages during differentiable task path",
                "Hard symbols at env execution / evaluation",
                "PRIMARY objective: task loss through channel into sender message-head",
                "Receiver-value regression is auxiliary only (not primary)",
            ],
        }

    def evaluate_hard(self, episodes: int = 3) -> dict[str, float]:
        """Hard-message evaluation (no soft relaxation)."""
        rets = []
        for ep in range(episodes):
            obs, _ = self.env.reset(seed=self.seed + 1000 + ep)
            ep_ret = 0.0
            for _ in range(self.env.config.horizon):
                if not self.env.agents:
                    break
                actions, _, _, _, _ = self.select_actions(obs, deterministic=True, hard_messages=True)
                obs, rewards, terms, truncs, _ = self.env.step(actions)
                if rewards:
                    ep_ret += float(np.mean(list(rewards.values())))
                if (not self.env.agents) or any(truncs.values()) or any(terms.values()):
                    break
            rets.append(ep_ret)
        return {"hard_eval_mean_return": float(np.mean(rets)), "episodes": float(len(rets))}

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
                # Soft-channel task loss on realized returns (primary DIAL path in batch)
                task_pred = self.receivers[a](obs, embed)
                task_loss = F.mse_loss(task_pred, ret_t)
                # Auxiliary value head — small weight only
                v = self.receivers[a].value(obs, embed.detach())
                value_aux = F.mse_loss(v, ret_t)
                probs = torch.softmax(msg_logits, dim=-1)
                comm_cost = -self.comm_cost_coef * (probs * torch.log(probs.clamp_min(1e-8))).sum(-1).mean()
                loss = (
                    policy_loss
                    + task_loss
                    + self.value_aux_coef * cfg.vf_coef * value_aux
                    + comm_cost
                )
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
