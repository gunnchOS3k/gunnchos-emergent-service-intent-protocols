from __future__ import annotations

from copy import deepcopy
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
    loopback: bool = False
    inbox_capacity: int = 2
    stale_threshold: int = 3

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


@dataclass
class InboxRecord:
    """One delivered (or empty) inbox slot visible to a receiver."""

    sender_id: int = -1
    symbols: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    age: float = 0.0
    stale: float = 0.0
    confidence: float = 0.0
    erased: float = 0.0
    silence: float = 1.0
    valid: float = 0.0

    def as_vector(self, msg_len: int, n_agents: int) -> np.ndarray:
        # sender embedding: one-hot + pad/truncate to fixed agent count embedding dim=4
        emb = np.zeros(4, dtype=np.float32)
        if 0 <= self.sender_id < n_agents:
            emb[self.sender_id % 4] = 1.0
            emb[min(3, self.sender_id // 4)] += 0.25
        sym = np.asarray(self.symbols, dtype=np.float32).ravel()
        if sym.size < msg_len:
            sym = np.pad(sym, (0, msg_len - sym.size))
        else:
            sym = sym[:msg_len]
        meta = np.array(
            [self.age, self.stale, self.confidence, self.erased, self.silence, self.valid],
            dtype=np.float32,
        )
        return np.concatenate([emb, sym, meta])


SLOT_META_DIM = 4 + 6  # sender emb + age/stale/conf/erased/silence/valid


def slot_dim(msg_len: int) -> int:
    return SLOT_META_DIM + msg_len


def gumbel_softmax_sample(logits: torch.Tensor, tau: float = 1.0, hard: bool = True) -> torch.Tensor:
    return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)


def discrete_message_from_logits(
    logits: torch.Tensor, tau: float = 1.0, hard: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
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


def make_channel(
    cfg: ChannelConfig | None = None, agents: list[str] | None = None, seed: int = 0
) -> "MessageChannel":
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
    """Multi-agent message exchange with erasures, delay, corruption, targeting, aging."""

    def __init__(
        self,
        cfg: ChannelConfig,
        agents: list[str] | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.cfg = cfg
        self.config = cfg
        self.agents = list(agents or [])
        self.agent_index = {a: i for i, a in enumerate(self.agents)}
        self.rng = rng or np.random.default_rng(0)
        self._pending: dict[str, list[tuple[int, InboxRecord]]] = {}
        self._inbox: dict[str, list[InboxRecord]] = {}

    def silence_id(self) -> int:
        return int(self.cfg.vocab_size)

    def empty_message(self) -> np.ndarray:
        if self.cfg.mode == "continuous_learned":
            return np.zeros(self.cfg.continuous_dim, dtype=np.float32)
        return np.full(self.cfg.msg_len, self.silence_id(), dtype=np.float32)

    def empty_record(self) -> InboxRecord:
        msg_len = self.cfg.msg_len if self.cfg.mode != "continuous_learned" else self.cfg.continuous_dim
        return InboxRecord(
            sender_id=-1,
            symbols=np.zeros(msg_len, dtype=np.float32),
            age=0.0,
            stale=0.0,
            confidence=0.0,
            erased=0.0,
            silence=0.0,
            valid=0.0,
        )

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
        self._pending = {a: [] for a in self.agents}
        self._inbox = {a: [] for a in self.agents}

    def bit_cost(self, symbols: np.ndarray | None) -> float:
        if symbols is None:
            return 0.0
        arr = np.asarray(symbols).ravel()
        if arr.size == 0:
            return 0.0
        if (
            np.all(arr == self.silence_id())
            or np.all(arr == SILENCE)
            or np.allclose(arr, 0.0)
        ):
            return 0.0
        return float(arr.size * self.cfg.bit_cost)

    def _is_silence(self, arr: np.ndarray) -> bool:
        a = np.asarray(arr).ravel()
        if a.size == 0:
            return True
        return bool(np.all(a == self.silence_id()) or np.all(a == SILENCE) or np.allclose(a, 0.0))

    def _corrupt(self, payload: np.ndarray) -> np.ndarray:
        sym = np.asarray(payload, dtype=np.float32).copy()
        if self.cfg.mode == "continuous_learned":
            if self.cfg.bit_error_p > 0:
                sym = sym + self.rng.normal(0, self.cfg.bit_error_p, size=sym.shape).astype(np.float32)
            return sym
        if self.rng.random() < self.cfg.corruption_p:
            sym = self.rng.integers(0, self.cfg.vocab_size, size=sym.shape).astype(np.float32)
        mask = self.rng.random(sym.shape) < self.cfg.bit_error_p
        noise = self.rng.integers(0, self.cfg.vocab_size, size=sym.shape).astype(np.float32)
        return np.where(mask, noise, np.mod(sym, max(self.cfg.vocab_size, 1))).astype(np.float32)

    def send_discrete(self, symbols: np.ndarray | None) -> Optional[np.ndarray]:
        """Compatibility single-shot API (does not populate multi-agent inbox)."""
        if self.cfg.mode == "no_comm" or symbols is None:
            return None
        sym = np.asarray(symbols, dtype=np.int64).copy()
        if self.cfg.allow_silence and sym.size and (
            int(sym.ravel()[0]) == SILENCE or int(sym.ravel()[0]) == self.silence_id()
        ):
            out: Optional[np.ndarray] = sym.astype(np.float32)
        else:
            if self.rng.random() < self.cfg.erasure_p:
                out = None
            else:
                out = self._corrupt(sym.astype(np.float32))
        return out

    def send_continuous(self, vec: np.ndarray | None) -> Optional[np.ndarray]:
        if vec is None or self.cfg.mode == "no_comm":
            return None
        v = np.asarray(vec, dtype=np.float32).copy()
        if self.rng.random() < self.cfg.erasure_p:
            return None
        return self._corrupt(v)

    def tick_ages(self) -> None:
        """Increment age of held inbox records once per environment step."""
        for a, slots in self._inbox.items():
            for rec in slots:
                rec.age += 1.0
                rec.stale = 1.0 if rec.age >= self.cfg.stale_threshold else 0.0
                if rec.valid > 0:
                    rec.confidence = max(0.0, rec.confidence - 0.05)
        # Advance pending due counters without mutating record ages until delivery
        for a, pending in list(self._pending.items()):
            self._pending[a] = [(max(0, d), r) for d, r in pending]

    def exchange(
        self,
        outbound: dict[str, np.ndarray],
        rng: np.random.Generator | None = None,
        targets: dict[str, str | None] | None = None,
    ) -> tuple[dict[str, np.ndarray], float, dict[str, list[InboxRecord]]]:
        if rng is not None:
            self.rng = rng
        if self.cfg.mode == "no_comm":
            empty = {a: self.empty_message() for a in self.agents}
            self._inbox = {a: [] for a in self.agents}
            return empty, 0.0, {a: [] for a in self.agents}

        bits = 0.0
        targets = targets or {}
        # Age existing inbox contents before new deliveries
        self.tick_ages()

        # Deliver matured pending messages first
        arriving: dict[str, list[InboxRecord]] = {a: [] for a in self.agents}
        for recv, queue in list(self._pending.items()):
            remain = []
            for due, rec in queue:
                if due <= 0:
                    arriving[recv].append(rec)
                else:
                    remain.append((due - 1, rec))
            self._pending[recv] = remain

        for sender, msg in outbound.items():
            if sender not in self.agent_index:
                continue
            try:
                msg_arr = validate_message_shape(msg, self.cfg) if msg is not None else self.empty_message()
            except ValueError:
                msg_arr = self.empty_message()
            cost = self.bit_cost(msg_arr)
            bits += cost
            is_sil = self._is_silence(msg_arr) or cost == 0.0
            erased = False
            payload = msg_arr.copy()
            confidence = 1.0
            if is_sil:
                payload = self.empty_message()
                confidence = 0.0
            elif self.rng.random() < self.cfg.erasure_p:
                erased = True
                payload = self.empty_message()
                confidence = 0.0
            else:
                before = payload.copy()
                payload = self._corrupt(payload)
                if not np.allclose(before, payload):
                    confidence = 0.4

            # Pure silence does not occupy inbox slots (avoids displacing valid mail);
            # erased events are still delivered so receivers can observe erasure.
            if is_sil and not erased:
                continue

            rec = InboxRecord(
                sender_id=self.agent_index[sender],
                symbols=payload.astype(np.float32),
                age=0.0,
                stale=0.0,
                confidence=confidence,
                erased=1.0 if erased else 0.0,
                silence=1.0 if is_sil else 0.0,
                valid=0.0 if (erased or is_sil) else 1.0,
            )

            receivers = list(self.agents)
            if self.cfg.targeted:
                tgt = targets.get(sender)
                receivers = [tgt] if tgt else []

            for recv in receivers:
                if recv is None or recv not in self.agent_index:
                    continue
                if recv == sender and not self.cfg.loopback:
                    continue
                # Per-receiver copy so ageing one inbox does not mutate another.
                delivered = deepcopy(rec)
                if self.cfg.delay <= 0:
                    arriving[recv].append(delivered)
                else:
                    self._pending.setdefault(recv, []).append((self.cfg.delay - 1, delivered))

        # Bound inbox capacity: keep valid mail, newest first
        for a in self.agents:
            combined = arriving.get(a, []) + self._inbox.get(a, [])
            combined = sorted(combined, key=lambda r: (r.valid, -r.age), reverse=True)
            combined = combined[: self.cfg.inbox_capacity]
            self._inbox[a] = combined

        flat = {
            a: (
                self._inbox[a][0].symbols
                if self._inbox[a]
                else self.empty_message()
            )
            for a in self.agents
        }
        return flat, float(bits), {a: list(self._inbox[a]) for a in self.agents}

    def inbox_vector(self, agent: str, n_agents: int | None = None) -> np.ndarray:
        n_agents = n_agents or max(len(self.agents), 1)
        cap = self.cfg.inbox_capacity
        msg_len = self.cfg.msg_len if self.cfg.mode != "continuous_learned" else self.cfg.continuous_dim
        sd = slot_dim(msg_len)
        out = np.zeros(cap * sd, dtype=np.float32)
        slots = self._inbox.get(agent, [])
        for i in range(cap):
            rec = slots[i] if i < len(slots) else self.empty_record()
            if rec.symbols.size == 0:
                rec.symbols = self.empty_message()
            vec = rec.as_vector(msg_len, n_agents)
            out[i * sd : (i + 1) * sd] = vec
        return out
