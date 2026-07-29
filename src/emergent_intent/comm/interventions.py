"""Message interventions for semantic causality experiments.

Interventions corrupt or replace outbound / inbox content. Rewards never get a
presence bonus; performance changes only if controllers read altered messages.
"""

from __future__ import annotations

from copy import deepcopy
from enum import Enum
from typing import Callable

import numpy as np

from emergent_intent.comm.channel import InboxRecord, SILENCE
from emergent_intent.comm.semantic_protocol import SYMBOL, encode_fixed_protocol_message

# Avoid importing wireless_env (circular); message action index is stable.
A_MSG = 9


class InterventionKind(str, Enum):
    CORRECT = "correct"
    RANDOM = "random"
    CONSTANT = "constant"
    PERMUTE = "permute"
    SILENCE = "silence"
    DELAY = "delay"
    STALE = "stale"
    CORRUPT = "corrupt"
    ADVERSARIAL = "adversarial"


ADVERSARIAL_FLIP = {
    SYMBOL["blockage_high"]: SYMBOL["blockage_clear"],
    SYMBOL["blockage_clear"]: SYMBOL["blockage_high"],
    SYMBOL["congestion_high"]: SYMBOL["congestion_low"],
    SYMBOL["congestion_low"]: SYMBOL["congestion_high"],
    SYMBOL["tn_down"]: SYMBOL["tn_ok"],
    SYMBOL["tn_ok"]: SYMBOL["tn_down"],
    SYMBOL["ntn_ok"]: SYMBOL["ntn_costly"],
    SYMBOL["ntn_costly"]: SYMBOL["ntn_ok"],
    SYMBOL["priority_high"]: SYMBOL["priority_low"],
    SYMBOL["priority_low"]: SYMBOL["priority_high"],
    SYMBOL["handover_needed"]: SYMBOL["handover_hold"],
    SYMBOL["handover_hold"]: SYMBOL["handover_needed"],
}


def intervene_symbols(
    symbols: np.ndarray,
    kind: InterventionKind,
    rng: np.random.Generator,
    *,
    vocab_size: int = 16,
    constant_symbol: int = 0,
) -> np.ndarray:
    arr = np.asarray(symbols, dtype=np.float32).copy()
    if kind == InterventionKind.CORRECT:
        return arr
    if kind == InterventionKind.SILENCE:
        return np.full_like(arr, SILENCE if SILENCE >= 0 else vocab_size)
    if kind == InterventionKind.CONSTANT:
        return np.full_like(arr, float(constant_symbol))
    if kind == InterventionKind.RANDOM:
        return rng.integers(0, max(vocab_size, 1), size=arr.shape).astype(np.float32)
    if kind == InterventionKind.PERMUTE:
        flat = arr.ravel().copy()
        rng.shuffle(flat)
        return flat.reshape(arr.shape)
    if kind == InterventionKind.CORRUPT:
        out = arr.copy()
        mask = rng.random(out.shape) < 0.5
        out[mask] = rng.integers(0, max(vocab_size, 1), size=int(mask.sum()))
        return out
    if kind == InterventionKind.ADVERSARIAL:
        out = arr.copy()
        for i, v in enumerate(out.ravel()):
            iv = int(v)
            out.ravel()[i] = float(ADVERSARIAL_FLIP.get(iv, (iv + 1) % max(vocab_size, 1)))
        return out
    if kind in (InterventionKind.DELAY, InterventionKind.STALE):
        # Symbol identity unchanged; timing handled by inbox age / channel delay.
        return arr
    return arr


def patch_inbox(
    inbox_map: dict[str, list],
    kind: InterventionKind,
    rng: np.random.Generator,
    *,
    vocab_size: int = 16,
    stale_age: float = 10.0,
) -> dict[str, list]:
    """Return a deep-copied inbox map after applying an intervention."""
    out: dict[str, list] = {}
    for agent, slots in inbox_map.items():
        new_slots = []
        for rec in slots:
            r = deepcopy(rec)
            if kind == InterventionKind.SILENCE:
                r.symbols = np.full_like(r.symbols, float(vocab_size))
                r.silence = 1.0
                r.valid = 0.0
            elif kind == InterventionKind.STALE:
                r.age = float(stale_age)
                r.stale = 1.0
            elif kind != InterventionKind.CORRECT:
                r.symbols = intervene_symbols(
                    r.symbols, kind, rng, vocab_size=vocab_size
                )
                if kind == InterventionKind.ADVERSARIAL:
                    r.confidence = 0.1
            new_slots.append(r)
        out[agent] = new_slots
    return out


def make_action_intervention(
    kind: InterventionKind,
    rng: np.random.Generator,
    *,
    vocab_size: int = 16,
    delay_steps: int = 2,
) -> Callable:
    """Build env-step intervention that mutates outbound message tokens / inbox.

    For DELAY: zero message tokens for the first ``delay_steps`` steps.
    For STALE/CORRUPT/...: after step, patch env._last_inbox before next controller read
    by wrapping — here we mutate A_MSG / inject via channel config where possible.
    """
    held: dict[str, list] = {"t": [0], "pending": []}

    def _fn(env, acts: dict[str, np.ndarray], t: int) -> dict[str, np.ndarray]:
        held["t"][0] = t
        out = {a: np.asarray(v).copy() for a, v in acts.items()}
        if kind == InterventionKind.SILENCE:
            for a in out:
                out[a][A_MSG] = 0
            return out
        if kind == InterventionKind.DELAY and t < delay_steps:
            for a in out:
                out[a][A_MSG] = 0
            return out
        if kind == InterventionKind.RANDOM:
            for a in out:
                out[a][A_MSG] = 1 + int(rng.integers(0, max(vocab_size, 1)))
            return out
        if kind == InterventionKind.CONSTANT:
            for a in out:
                out[a][A_MSG] = 1  # always same non-silence token
            return out
        # For CORRECT / PERMUTE / CORRUPT / ADVERSARIAL / STALE under fixed_protocol,
        # mutate the channel's last inbox that the controller will read next step.
        # Also monkey-patch outbound by temporarily overriding fixed protocol encode.
        return out

    return _fn


def apply_post_step_inbox_intervention(
    env,
    kind: InterventionKind,
    rng: np.random.Generator,
    *,
    vocab_size: int = 16,
) -> None:
    """Mutate env._last_inbox in-place after env.step for next-step obs→act path."""
    if kind == InterventionKind.CORRECT:
        return
    if kind == InterventionKind.DELAY:
        # Age-only: force high age so controller max_symbol_age filters them.
        for slots in env._last_inbox.values():
            for r in slots:
                r.age = float(getattr(env.channel.cfg, "stale_threshold", 3) + 2)
                r.stale = 1.0
        return
    env._last_inbox = patch_inbox(
        env._last_inbox, kind, rng, vocab_size=vocab_size
    )


def wrap_fixed_protocol_encoder(env, kind: InterventionKind, rng: np.random.Generator):
    """Monkey-patch env._fixed_protocol_message with intervened symbols."""
    original = env._fixed_protocol_message
    vocab = int(env.config.vocab_size)

    def _intervened(agent: str) -> np.ndarray:
        raw = encode_fixed_protocol_message(agent, env._state, msg_len=env.config.msg_len)
        if kind == InterventionKind.CORRECT:
            return raw
        if kind == InterventionKind.SILENCE:
            return np.full(env.config.msg_len, float(env.channel.silence_id()), dtype=np.float32)
        return intervene_symbols(raw, kind, rng, vocab_size=vocab)

    env._fixed_protocol_message = _intervened  # type: ignore[method-assign]
    return original
