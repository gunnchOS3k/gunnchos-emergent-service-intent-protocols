"""Value decomposition: VDN and QMIX for cooperative MultiDiscrete (factored)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from emergent_intent.algorithms.networks import nvec_from_env, obs_dim_from_env
from emergent_intent.utils import detect_device, set_global_seed


class AgentQNet(nn.Module):
    """Q-network emitting per-dimension logits; total Q = sum of selected logits."""

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
        # agent_qs: (B, N), state: (B, S)
        B, N = agent_qs.shape
        w1 = torch.abs(self.hyper_w1(state)).view(B, N, self.embed)
        b1 = self.hyper_b1(state).view(B, 1, self.embed)
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
        w2 = torch.abs(self.hyper_w2(state)).view(B, self.embed, 1)
        b2 = self.hyper_b2(state).view(B, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.squeeze(-1).squeeze(-1)


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
    ):
        self.env = env
        self.method = method
        self.seed = seed
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        set_global_seed(seed)
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
        env.reset(seed=seed)
        from emergent_intent.algorithms.networks import flatten_state

        self._flatten_state = flatten_state
        self.state_dim = int(flatten_state(env.state()).size)
        self.mixer = (
            QMIXMixer(len(self.agents), self.state_dim).to(self.device) if method == "qmix" else None
        )
        params: list[nn.Parameter] = []
        for a in self.agents:
            params += list(self.qnets[a].parameters())
        if self.mixer is not None:
            params += list(self.mixer.parameters())
        self.opt = torch.optim.Adam(params, lr=lr)
        self._step = 0

    def _epsilon(self) -> float:
        t = min(1.0, self._step / max(1, self.epsilon_decay))
        return self.epsilon_start + t * (self.epsilon_end - self.epsilon_start)

    def select_actions(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        eps = self._epsilon()
        actions = {}
        for a, o in obs.items():
            if np.random.random() < eps:
                actions[a] = self.env.action_space(a).sample()
            else:
                t = torch.as_tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    actions[a] = self.qnets[a].greedy(t).squeeze(0).cpu().numpy().astype(np.int64)
        return actions

    def train(self, total_steps: int = 512, batch_hint: int = 32) -> dict[str, Any]:
        replay: list[dict[str, Any]] = []
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
            replay.append(
                {
                    "obs": {a: obs[a] for a in obs},
                    "act": {a: actions[a] for a in actions},
                    "rew": team_r,
                    "next_obs": {a: next_obs.get(a, obs[a]) for a in obs} if next_obs else obs,
                    "done": float(done),
                    "state": state,
                    "next_state": next_state,
                }
            )
            ep_ret += team_r
            if len(replay) >= batch_hint:
                losses.append(self._update(replay[-batch_hint:]))
            if done:
                ep_returns.append(ep_ret)
                ep_ret = 0.0
                obs, _ = self.env.reset(seed=self.seed + step + 1)
            else:
                obs = next_obs
        return {
            "algorithm": self.method.upper(),
            "steps": total_steps,
            "mean_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
            "n_episodes": len(ep_returns),
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
            "device": self.device_label,
            "evidence_class": "SYNTHETIC_SIM",
        }

    def _update(self, batch: list[dict[str, Any]]) -> float:
        B = len(batch)
        agent_qs = []
        target_qs = []
        states = torch.as_tensor(
            np.asarray([b["state"] for b in batch]), dtype=torch.float32, device=self.device
        )
        next_states = torch.as_tensor(
            np.asarray([b["next_state"] for b in batch]), dtype=torch.float32, device=self.device
        )
        rewards = torch.as_tensor([b["rew"] for b in batch], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([b["done"] for b in batch], dtype=torch.float32, device=self.device)
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
                greedy = self.qnets[a].greedy(no)
                target_qs.append(self.qnets[a].q_selected(no, greedy))
        q_stack = torch.stack(agent_qs, dim=1)  # (B,N)
        t_stack = torch.stack(target_qs, dim=1)
        if self.mixer is None:
            q_tot = q_stack.sum(1)
            with torch.no_grad():
                target_tot = rewards + self.gamma * (1 - dones) * t_stack.sum(1)
        else:
            q_tot = self.mixer(q_stack, states)
            with torch.no_grad():
                target_tot = rewards + self.gamma * (1 - dones) * self.mixer(t_stack, next_states)
        loss = F.mse_loss(q_tot, target_tot)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.item())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "algorithm": self.method,
            "qnets": {a: self.qnets[a].state_dict() for a in self.agents},
            "seed": self.seed,
        }
        if self.mixer is not None:
            payload["mixer"] = self.mixer.state_dict()
        torch.save(payload, path)

    def load(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=True)
        for a in self.agents:
            self.qnets[a].load_state_dict(payload["qnets"][a])
        if self.mixer is not None and "mixer" in payload:
            self.mixer.load_state_dict(payload["mixer"])
