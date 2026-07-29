from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SILENCE = -1


@dataclass
class ChannelConfig:
    mode: str = "discrete_learned"
    vocab_size: int = 16
    msg_len: int = 4
    msg_length: int | None = None
    erasure_p: float = 0.0
    bit_error_p: float = 0.0
    erasure_prob: float | None = None
    corruption_prob: float | None = None
    delay: int = 0
    delay_steps: int | None = None
    corruption_p: float = 0.0
    allow_silence: bool = True
    bit_cost: float = 1.0
    continuous_dim: int = 4
    targeted: bool = False

    def __post_init__(self) -> None:
        if self.msg_length is not None:
            self.msg_len = int(self.msg_length)
        else:
            self.msg_length = self.msg_len
        if self.erasure_prob is not None:
            self.erasure_p = float(self.erasure_prob)
        else:
            self.erasure_prob = self.erasure_p
        if self.corruption_prob is not None:
            self.corruption_p = float(self.corruption_prob)
        else:
            self.corruption_prob = self.corruption_p
        if self.delay_steps is not None:
            self.delay = int(self.delay_steps)
        else:
            self.delay_steps = self.delay


def gumbel_softmax_sample(logits: torch.Tensor, tau: float = 1.0, hard: bool = True) -> torch.Tensor:
    return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)


def discrete_message_from_logits(
    logits: torch.Tensor, tau: float = 1.0, hard: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    # logits: [..., L, V] or [..., V]
    onehot = gumbel_softmax_sample(logits, tau=tau, hard=hard)
    symbols = onehot.argmax(dim=-1)
    return onehot, symbols


def validate_message_shape(msg, cfg: ChannelConfig) -> np.ndarray:
    if msg is None:
        raise ValueError("malformed message: None")
    arr = np.asarray(msg)
    if arr.ndim > 1:
        raise ValueError(f"malformed message: expected 1D, got shape {arr.shape}")
    expect = cfg.continuous_dim if cfg.mode == "continuous_learned" else cfg.msg_len
    if arr.size != expect:
        raise ValueError(f"malformed message: length {arr.size} != {expect}")
    return arr.astype(np.float32)


def make_channel(cfg: ChannelConfig | None = None, agents: list[str] | None = None, seed: int = 0) -> "MessageChannel":
    return MessageChannel(cfg or ChannelConfig(), agents=agents or ["a", "b"], rng=np.random.default_rng(seed))


class AttentionTargeter(nn.Module):
    def __init__(self, d_model: int, n_agents: int):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.n_agents = n_agents

    def forward(self, h: torch.Tensor, others: torch.Tensor) -> torch.Tensor:
        q = self.query(h).unsqueeze(1)
        k = self.key(others)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (k.size(-1) ** 0.5)
        return F.softmax(scores, dim=-1).squeeze(1)


class GraphMessageRouter(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.lin = nn.Linear(d_model, d_model)

    def forward(self, node_h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(node_h.size(0), -1, -1)
        return torch.matmul(adj, self.lin(node_h))


class MessageChannel:
    """Multi-agent message exchange with erasures, delay, corruption, targeting."""

    def __init__(
        self,
        cfg: ChannelConfig,
        agents: list[str] | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.cfg = cfg
        self.config = cfg
        self.agents = list(agents or [])
        self.rng = rng or np.random.default_rng(0)
        self._delay_queues: dict[str, list[np.ndarray]] = {}

    def silence_id(self) -> int:
        return int(self.cfg.vocab_size)  # reserved silence token id

    def empty_message(self) -> np.ndarray:
        if self.cfg.mode == "continuous_learned":
            return np.zeros(self.cfg.continuous_dim, dtype=np.float32)
        return np.full(self.cfg.msg_len, self.silence_id(), dtype=np.float32)

    def encode_action_symbols(self, symbols: np.ndarray) -> np.ndarray:
        s = np.asarray(symbols).ravel()
        L = self.cfg.msg_len
        if self.cfg.mode == "continuous_learned":
            dim = self.cfg.continuous_dim
            if s.size < dim:
                s = np.pad(s.astype(np.float32), (0, dim - s.size))
            return ((s[:dim].astype(np.float32) / 2.0) - 1.0)
        fill = self.silence_id()
        if s.size < L:
            s = np.pad(s.astype(np.int64), (0, L - s.size), constant_values=fill)
        return s[:L].astype(np.float32)

    def bits_for(self, msg: np.ndarray | None) -> float:
        return self.bit_cost(msg)

    def reset(self, rng: np.random.Generator | None = None) -> None:
        if rng is not None:
            self.rng = rng
        self._delay_queues = {a: [] for a in self.agents}

    def bit_cost(self, symbols: np.ndarray | None) -> float:
        if symbols is None:
            return 0.0
        arr = np.asarray(symbols).ravel()
        if arr.size == 0:
            return 0.0
        if np.all(arr == self.silence_id()) or np.all(arr == SILENCE):
            return 0.0
        return float(arr.size * self.cfg.bit_cost)

    def send_discrete(self, symbols: np.ndarray | None) -> Optional[np.ndarray]:
        # Compatibility API used by ServiceIntentEnv
        if self.cfg.mode == "no_comm" or symbols is None:
            return None
        sym = np.asarray(symbols, dtype=np.int64).copy()
        if self.cfg.allow_silence and sym.size and (
            int(sym.ravel()[0]) == SILENCE or int(sym.ravel()[0]) == self.silence_id()
        ):
            out = sym
        else:
            if self.rng.random() < self.cfg.erasure_p:
                out = None
            else:
                if self.rng.random() < self.cfg.corruption_p:
                    sym = self.rng.integers(0, self.cfg.vocab_size, size=sym.shape)
                mask = self.rng.random(sym.shape) < self.cfg.bit_error_p
                noise = self.rng.integers(0, self.cfg.vocab_size, size=sym.shape)
                sym = np.where(mask, noise, np.mod(sym, max(self.cfg.vocab_size, 1)))
                out = sym
        if self.cfg.delay <= 0:
            return out
        key = "_single"
        self._delay_queues.setdefault(key, []).append(out if out is not None else self.empty_message())
        if len(self._delay_queues[key]) <= self.cfg.delay:
            return None
        return self._delay_queues[key].pop(0)

    def send_continuous(self, vec: np.ndarray | None) -> Optional[np.ndarray]:
        if vec is None or self.cfg.mode == "no_comm":
            return None
        v = np.asarray(vec, dtype=np.float64).copy()
        if self.rng.random() < self.cfg.erasure_p:
            return None
        if self.cfg.bit_error_p > 0:
            v = v + self.rng.normal(0, self.cfg.bit_error_p, size=v.shape)
        return v

    def exchange(
        self,
        outbound: dict[str, np.ndarray],
        rng: np.random.Generator,
        targets: dict[str, str | None] | None = None,
    ) -> tuple[dict[str, np.ndarray], float]:
        self.rng = rng
        if self.cfg.mode == "no_comm":
            return {a: np.zeros(self.cfg.msg_len, dtype=np.float32) for a in self.agents}, 0.0

        bits = 0.0
        delivered: dict[str, np.ndarray] = {a: self.empty_message() for a in self.agents}
        targets = targets or {}

        for sender, msg in outbound.items():
            msg_arr = np.asarray(msg, dtype=np.float32).ravel()
            if msg_arr.size != self.cfg.msg_len:
                msg_arr = self.empty_message()
            cost = self.bit_cost(msg_arr)
            bits += cost
            if cost == 0.0 and self.cfg.allow_silence:
                payload = self.empty_message()
            else:
                payload = msg_arr.copy()
                if rng.random() < self.cfg.erasure_p:
                    payload = self.empty_message()
                elif rng.random() < self.cfg.corruption_p:
                    payload = rng.integers(0, self.cfg.vocab_size, size=self.cfg.msg_len).astype(np.float32)

            receivers = self.agents
            if self.cfg.targeted:
                tgt = targets.get(sender)
                receivers = [tgt] if tgt else []

            for recv in receivers:
                if recv is None or recv == sender:
                    continue
                q = self._delay_queues.setdefault(recv, [])
                q.append(payload)
                if self.cfg.delay <= 0:
                    delivered[recv] = q.pop(0)
                elif len(q) > self.cfg.delay:
                    delivered[recv] = q.pop(0)
                else:
                    delivered[recv] = self.empty_message()

        return delivered, float(bits)
