"""Value decomposition: VDN and QMIX with replay, targets, Double Q, terminal masking."""

from __future__ import annotations

import copy
from collections import deque
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from emergent_intent.algorithms.networks import flatten_state, nvec_from_env, obs_dim_from_env
from emergent_intent.utils import detect_device, set_global_seed


class AgentQNet(nn.Module):
    def __init__(self, obs_dim: int, nvec: list[int], hidden: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, n) for n in nvec])
        self.nvec = nvec

    def forward(self, obs: torch.Tensor) -> list[torch.Tensor]:
        h = self.backbone(obs)
        return [head(h) for head in self.heads]

    def q_selected(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        logits = self.forward(obs)
        q = 0.0
        for i, head in enumerate(logits):
            q = q + head.gather(1, actions[:, i : i + 1]).squeeze(1)
        return q  # type: ignore[return-value]

    def greedy(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.forward(obs)
        return torch.stack([h.argmax(-1) for h in logits], dim=-1)


class QMIXMixer(nn.Module):
    def __init__(self, n_agents: int, state_dim: int, embed: int = 32):
        super().__init__()
        self.n_agents = n_agents
        self.hyper_w1 = nn.Linear(state_dim, n_agents * embed)
        self.hyper_b1 = nn.Linear(state_dim, embed)
        self.hyper_w2 = nn.Linear(state_dim, embed)
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, embed), nn.ReLU(), nn.Linear(embed, 1))
        self.embed = embed

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        B, N = agent_qs.shape
        w1 = torch.abs(self.hyper_w1(state)).view(B, N, self.embed)
        b1 = self.hyper_b1(state).view(B, 1, self.embed)
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
        w2 = torch.abs(self.hyper_w2(state)).view(B, self.embed, 1)
        b2 = self.hyper_b2(state).view(B, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.squeeze(-1).squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self._buf: deque[dict[str, Any]] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buf)

    def add(self, transition: dict[str, Any]) -> None:
        self._buf.append(transition)

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[dict[str, Any]]:
        n = len(self._buf)
        if n == 0:
            return []
        idx = rng.choice(n, size=min(batch_size, n), replace=False)
        return [self._buf[i] for i in idx]

    def state_dict(self) -> dict[str, Any]:
        return {"capacity": self.capacity, "data": list(self._buf)}

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.capacity = int(payload.get("capacity", self.capacity))
        self._buf = deque(payload.get("data", []), maxlen=self.capacity)


class VDNQMIXTrainer:
    def __init__(
        self,
        env,
        method: Literal["vdn", "qmix"] = "vdn",
        seed: int = 0,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 500,
        hidden: int = 64,
        prefer_cuda: bool = True,
        buffer_size: int = 2000,
        batch_size: int = 32,
        target_update_interval: int = 50,
        polyak_tau: float | None = None,
        double_q: bool = True,
    ):
        self.env = env
        self.method = method
        self.seed = seed
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.polyak_tau = polyak_tau
        self.double_q = double_q
        set_global_seed(seed)
        self.rng = np.random.default_rng(seed)
        info = detect_device(prefer_cuda=prefer_cuda)
        self.device = torch.device(info.device)
        self.device_label = info.label
        self.agents = list(env.possible_agents)
        self.qnets = {
            a: AgentQNet(obs_dim_from_env(env, a), nvec_from_env(env, a), hidden=hidden).to(
                self.device
            )
            for a in self.agents
        }
        self.target_qnets = {a: copy.deepcopy(self.qnets[a]).to(self.device) for a in self.agents}
        for a in self.agents:
            for p in self.target_qnets[a].parameters():
                p.requires_grad_(False)
        env.reset(seed=seed)
        self._flatten_state = flatten_state
        self.state_dim = int(flatten_state(env.state()).size)
        self.mixer = (
            QMIXMixer(len(self.agents), self.state_dim).to(self.device) if method == "qmix" else None
        )
        self.target_mixer = copy.deepcopy(self.mixer).to(self.device) if self.mixer is not None else None
        if self.target_mixer is not None:
            for p in self.target_mixer.parameters():
                p.requires_grad_(False)
        params: list[nn.Parameter] = []
        for a in self.agents:
            params += list(self.qnets[a].parameters())
        if self.mixer is not None:
            params += list(self.mixer.parameters())
        self.opt = torch.optim.Adam(params, lr=lr)
        self.replay = ReplayBuffer(buffer_size)
        self._step = 0
        self._updates = 0
        self.diagnostics: dict[str, float] = {}

    def _epsilon(self) -> float:
        t = min(1.0, self._step / max(1, self.epsilon_decay))
        return self.epsilon_start + t * (self.epsilon_end - self.epsilon_start)

    def select_actions(self, obs: dict[str, np.ndarray], deterministic: bool = False) -> dict[str, np.ndarray]:
        eps = 0.0 if deterministic else self._epsilon()
        actions = {}
        for a, o in obs.items():
            if self.rng.random() < eps:
                actions[a] = self.env.action_space(a).sample()
            else:
                t = torch.as_tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    actions[a] = self.qnets[a].greedy(t).squeeze(0).cpu().numpy().astype(np.int64)
        return actions

    def _soft_update(self) -> None:
        if self.polyak_tau is None:
            for a in self.agents:
                self.target_qnets[a].load_state_dict(self.qnets[a].state_dict())
            if self.mixer is not None and self.target_mixer is not None:
                self.target_mixer.load_state_dict(self.mixer.state_dict())
        else:
            tau = self.polyak_tau
            for a in self.agents:
                for tp, p in zip(self.target_qnets[a].parameters(), self.qnets[a].parameters()):
                    tp.data.mul_(1 - tau).add_(p.data, alpha=tau)
            if self.mixer is not None and self.target_mixer is not None:
                for tp, p in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                    tp.data.mul_(1 - tau).add_(p.data, alpha=tau)

    def update_targets(self) -> None:
        self._soft_update()

    def target_param_snapshot(self) -> dict[str, torch.Tensor]:
        out = {}
        for a in self.agents:
            for name, p in self.target_qnets[a].named_parameters():
                out[f"{a}.{name}"] = p.detach().cpu().clone()
        if self.target_mixer is not None:
            for name, p in self.target_mixer.named_parameters():
                out[f"mixer.{name}"] = p.detach().cpu().clone()
        return out

    def train(self, total_steps: int = 512, batch_hint: int | None = None) -> dict[str, Any]:
        batch_size = batch_hint or self.batch_size
        obs, _ = self.env.reset(seed=self.seed)
        ep_returns: list[float] = []
        ep_ret = 0.0
        losses: list[float] = []
        for step in range(total_steps):
            self._step = step
            state = self._flatten_state(self.env.state())
            actions = self.select_actions(obs)
            next_obs, rewards, terms, truncs, _ = self.env.step(actions)
            done = (not self.env.agents) or any(truncs.values()) or any(terms.values())
            team_r = float(np.mean(list(rewards.values()))) if rewards else 0.0
            next_state = (
                self._flatten_state(self.env.state())
                if self.env.agents
                else np.zeros(self.state_dim, dtype=np.float32)
            )
            self.replay.add(
                {
                    "obs": {a: np.asarray(obs[a], dtype=np.float32) for a in obs},
                    "act": {a: np.asarray(actions[a], dtype=np.int64) for a in actions},
                    "rew": team_r,
                    "next_obs": {
                        a: np.asarray(next_obs.get(a, obs[a]), dtype=np.float32) for a in obs
                    }
                    if next_obs
                    else {a: np.asarray(obs[a], dtype=np.float32) for a in obs},
                    "done": float(done),
                    "state": np.asarray(state, dtype=np.float32),
                    "next_state": np.asarray(next_state, dtype=np.float32),
                }
            )
            ep_ret += team_r
            if len(self.replay) >= batch_size:
                losses.append(self._update(self.replay.sample(batch_size, self.rng)))
                self._updates += 1
                if self._updates % self.target_update_interval == 0:
                    self.update_targets()
            if done:
                ep_returns.append(ep_ret)
                ep_ret = 0.0
                obs, _ = self.env.reset(seed=self.seed + step + 1)
            else:
                obs = next_obs
        self.diagnostics["mean_loss"] = float(np.mean(losses)) if losses else 0.0
        return {
            "algorithm": self.method.upper(),
            "steps": total_steps,
            "mean_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
            "n_episodes": len(ep_returns),
            "mean_loss": self.diagnostics["mean_loss"],
            "updates": self._updates,
            "replay_size": len(self.replay),
            "device": self.device_label,
            "evidence_class": "SYNTHETIC_SIM",
            "double_q": self.double_q,
        }

    def _update(self, batch: list[dict[str, Any]]) -> float:
        if not batch:
            return 0.0
        states = torch.as_tensor(
            np.asarray([b["state"] for b in batch]), dtype=torch.float32, device=self.device
        )
        next_states = torch.as_tensor(
            np.asarray([b["next_state"] for b in batch]), dtype=torch.float32, device=self.device
        )
        rewards = torch.as_tensor([b["rew"] for b in batch], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([b["done"] for b in batch], dtype=torch.float32, device=self.device)
        agent_qs = []
        target_qs = []
        for a in self.agents:
            o = torch.as_tensor(
                np.asarray([b["obs"][a] for b in batch]), dtype=torch.float32, device=self.device
            )
            no = torch.as_tensor(
                np.asarray([b["next_obs"][a] for b in batch]), dtype=torch.float32, device=self.device
            )
            act = torch.as_tensor(
                np.asarray([b["act"][a] for b in batch]), dtype=torch.int64, device=self.device
            )
            agent_qs.append(self.qnets[a].q_selected(o, act))
            with torch.no_grad():
                if self.double_q:
                    # Double Q: online selects, target evaluates
                    greedy = self.qnets[a].greedy(no)
                    target_qs.append(self.target_qnets[a].q_selected(no, greedy))
                else:
                    greedy = self.target_qnets[a].greedy(no)
                    target_qs.append(self.target_qnets[a].q_selected(no, greedy))
        q_stack = torch.stack(agent_qs, dim=1)
        t_stack = torch.stack(target_qs, dim=1)
        if self.mixer is None:
            q_tot = q_stack.sum(1)
            with torch.no_grad():
                # Terminal masking: (1 - done)
                target_tot = rewards + self.gamma * (1.0 - dones) * t_stack.sum(1)
        else:
            q_tot = self.mixer(q_stack, states)
            with torch.no_grad():
                mixer_t = self.target_mixer if self.target_mixer is not None else self.mixer
                target_tot = rewards + self.gamma * (1.0 - dones) * mixer_t(t_stack, next_states)
        loss = F.mse_loss(q_tot, target_tot)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.qnets[self.agents[0]].parameters()), 10.0
        )
        self.opt.step()
        return float(loss.item())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "algorithm": self.method,
            "qnets": {a: self.qnets[a].state_dict() for a in self.agents},
            "target_qnets": {a: self.target_qnets[a].state_dict() for a in self.agents},
            "seed": self.seed,
            "step": self._step,
            "updates": self._updates,
            "replay": self.replay.state_dict(),
            "double_q": self.double_q,
        }
        if self.mixer is not None:
            payload["mixer"] = self.mixer.state_dict()
        if self.target_mixer is not None:
            payload["target_mixer"] = self.target_mixer.state_dict()
        torch.save(payload, path)

    def load(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        for a in self.agents:
            self.qnets[a].load_state_dict(payload["qnets"][a])
            if "target_qnets" in payload:
                self.target_qnets[a].load_state_dict(payload["target_qnets"][a])
        if self.mixer is not None and "mixer" in payload:
            self.mixer.load_state_dict(payload["mixer"])
        if self.target_mixer is not None and "target_mixer" in payload:
            self.target_mixer.load_state_dict(payload["target_mixer"])
        if "replay" in payload:
            self.replay.load_state_dict(payload["replay"])
        self._step = int(payload.get("step", 0))
        self._updates = int(payload.get("updates", 0))
