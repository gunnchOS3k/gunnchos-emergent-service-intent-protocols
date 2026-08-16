"""Fixed-protocol semantic lexicon and observation→action controllers.

Messages never enter the reward directly. They help only when a controller
reads inbox observations and changes radio/control actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Action factor indices (duplicated to avoid circular import with wireless_env).
A_POWER = 0
A_PRB = 1
A_MCS = 2
A_ACCESS = 3
A_HANDOVER = 4
A_ROUTING = 5
A_ADMISSION = 6
A_PRIORITY = 7
A_OFFLOAD = 8
A_MSG = 9
A_TARGET = 10


# Canonical discrete symbol meanings for fixed-protocol / intervention tests.
# Ids start at 1 so all-zero payloads are not mistaken for silence by the channel.
SYMBOL = {
    "blockage_clear": 1,
    "blockage_high": 2,
    "congestion_low": 3,
    "congestion_high": 4,
    "tn_ok": 5,
    "tn_down": 6,
    "ntn_ok": 7,
    "ntn_costly": 8,
    "priority_low": 9,
    "priority_high": 10,
    "handover_hold": 11,
    "handover_needed": 12,
}


@dataclass(frozen=True)
class ProtocolMapping:
    """Documents the fixed-protocol symbol → control mapping."""

    blockage: dict[str, int]
    congestion: dict[str, int]
    tn_ntn: dict[str, int]
    priority: dict[str, int]
    handover: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "blockage": self.blockage,
            "congestion": self.congestion,
            "tn_ntn": self.tn_ntn,
            "priority": self.priority,
            "handover": self.handover,
            "note": "Messages affect outcomes only via observation→action mappings.",
        }


DEFAULT_MAPPING = ProtocolMapping(
    blockage={
        "clear_symbol": SYMBOL["blockage_clear"],
        "high_symbol": SYMBOL["blockage_high"],
        "on_high_prb": 4,
        "on_high_power": 4,
        "on_clear_prb": 1,
        "on_clear_power": 1,
    },
    congestion={
        "low_symbol": SYMBOL["congestion_low"],
        "high_symbol": SYMBOL["congestion_high"],
        "on_high_mcs": 1,
        "on_low_mcs": 3,
    },
    tn_ntn={
        "tn_ok": SYMBOL["tn_ok"],
        "tn_down": SYMBOL["tn_down"],
        "ntn_ok": SYMBOL["ntn_ok"],
        "ntn_costly": SYMBOL["ntn_costly"],
        "on_tn_down_access": 1,  # NTN
        "on_tn_ok_access": 0,  # TN
    },
    priority={
        "low": SYMBOL["priority_low"],
        "high": SYMBOL["priority_high"],
        "on_high_priority": 2,
        "on_low_priority": 0,
    },
    handover={
        "hold": SYMBOL["handover_hold"],
        "needed": SYMBOL["handover_needed"],
        "on_needed": 2,
        "on_hold": 0,
    },
)


def encode_fixed_protocol_message(agent: str, state: dict[str, Any], msg_len: int = 2) -> np.ndarray:
    """Emit canonical semantic symbols from global state (sender side)."""
    if agent.startswith("ue_"):
        i = int(agent.split("_")[1])
        blk = SYMBOL["blockage_high"] if float(state["blockage"][i]) > 0.5 else SYMBOL["blockage_clear"]
        q = SYMBOL["congestion_high"] if float(state["queue"][i]) > 1.0 else SYMBOL["congestion_low"]
        vals = [blk, q]
    elif agent.startswith("bs_"):
        cong = (
            SYMBOL["congestion_high"]
            if float(state["congestion"]) > 0.5
            else SYMBOL["congestion_low"]
        )
        tn = SYMBOL["tn_down"] if float(state["tn_available"]) < 0.5 else SYMBOL["tn_ok"]
        vals = [cong, tn]
    elif agent == "edge_0":
        prio = (
            SYMBOL["priority_high"]
            if float(state["service_critical"].sum()) > 0
            else SYMBOL["priority_low"]
        )
        hand = (
            SYMBOL["handover_needed"]
            if float(state["tn_available"]) < 0.4
            else SYMBOL["handover_hold"]
        )
        vals = [prio, hand]
    elif agent == "ntn_relay":
        ntn = SYMBOL["ntn_ok"] if float(state["ntn_available"]) > 0.5 else SYMBOL["ntn_costly"]
        cost = SYMBOL["ntn_costly"] if float(state["ntn_cost"]) >= 0.5 else SYMBOL["ntn_ok"]
        vals = [ntn, cost]
    else:
        vals = [0, 0]
    out = np.asarray(vals[:msg_len], dtype=np.float32)
    if out.size < msg_len:
        out = np.pad(out, (0, msg_len - out.size))
    return out


def _valid_symbols_from_inbox(slots: list, *, max_age: float | None = None) -> list[int]:
    syms: list[int] = []
    for r in slots:
        if getattr(r, "valid", 0) <= 0:
            continue
        if getattr(r, "silence", 0) >= 1.0:
            continue
        if max_age is not None and float(getattr(r, "age", 0.0)) > max_age:
            continue
        for s in np.asarray(r.symbols).ravel():
            syms.append(int(s))
    return syms


def apply_semantic_mapping_to_action(
    action: np.ndarray,
    symbols: list[int],
    mapping: ProtocolMapping = DEFAULT_MAPPING,
    nvec: list[int] | None = None,
) -> np.ndarray:
    """Mutate a MultiDiscrete action from decoded inbox symbols (obs→act path)."""
    act = np.asarray(action, dtype=np.int64).copy()
    nvec = nvec or [5, 5, 4, 4, 3, 3, 2, 3, 2, 8, 4]
    act[A_ADMISSION] = 1
    # Safe mid defaults first — unknown/corrupt symbols must not leave power=0
    # (zero energy makes spectral_efficiency explode in the reward).
    act[A_POWER] = min(2, nvec[A_POWER] - 1)
    act[A_PRB] = min(2, nvec[A_PRB] - 1)
    act[A_MCS] = min(1, nvec[A_MCS] - 1)
    act[A_PRIORITY] = 0
    if not symbols:
        act[A_POWER] = min(1, nvec[A_POWER] - 1)
        act[A_PRB] = min(1, nvec[A_PRB] - 1)
        return act

    if mapping.blockage["high_symbol"] in symbols:
        act[A_POWER] = min(mapping.blockage["on_high_power"], nvec[A_POWER] - 1)
        act[A_PRB] = min(mapping.blockage["on_high_prb"], nvec[A_PRB] - 1)
    elif mapping.blockage["clear_symbol"] in symbols:
        act[A_POWER] = min(mapping.blockage["on_clear_power"], nvec[A_POWER] - 1)
        act[A_PRB] = min(mapping.blockage["on_clear_prb"], nvec[A_PRB] - 1)

    if mapping.congestion["high_symbol"] in symbols:
        act[A_MCS] = min(mapping.congestion["on_high_mcs"], nvec[A_MCS] - 1)
        act[A_ROUTING] = min(2, nvec[A_ROUTING] - 1)
    elif mapping.congestion["low_symbol"] in symbols:
        act[A_MCS] = min(mapping.congestion["on_low_mcs"], nvec[A_MCS] - 1)

    if mapping.tn_ntn["tn_down"] in symbols or mapping.tn_ntn["ntn_ok"] in symbols:
        act[A_ACCESS] = mapping.tn_ntn["on_tn_down_access"]
        act[A_HANDOVER] = min(mapping.handover["on_needed"], nvec[A_HANDOVER] - 1)
    elif mapping.tn_ntn["tn_ok"] in symbols:
        act[A_ACCESS] = mapping.tn_ntn["on_tn_ok_access"]
        act[A_HANDOVER] = mapping.handover["on_hold"]

    if mapping.priority["high"] in symbols:
        act[A_PRIORITY] = min(mapping.priority["on_high_priority"], nvec[A_PRIORITY] - 1)
    elif mapping.priority["low"] in symbols:
        act[A_PRIORITY] = mapping.priority["on_low_priority"]

    if mapping.handover["needed"] in symbols:
        act[A_HANDOVER] = min(mapping.handover["on_needed"], nvec[A_HANDOVER] - 1)
        act[A_ACCESS] = mapping.tn_ntn["on_tn_down_access"]
    elif mapping.handover["hold"] in symbols:
        act[A_HANDOVER] = mapping.handover["on_hold"]

    return act


class SemanticProtocolController:
    """Closed-loop controller: inbox observations → control actions."""

    def __init__(self, mapping: ProtocolMapping | None = None, max_symbol_age: float | None = 2.0):
        self.mapping = mapping or DEFAULT_MAPPING
        self.max_symbol_age = max_symbol_age

    def actions_from_inbox(
        self,
        env,
        inbox_map: dict[str, list] | None = None,
        base: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        inbox_map = inbox_map if inbox_map is not None else getattr(env, "_last_inbox", {})
        agents = list(env.agents)
        nvec = list(env._nvec)
        out: dict[str, np.ndarray] = {}
        for a in agents:
            act = (
                np.asarray(base[a], dtype=np.int64).copy()
                if base and a in base
                else np.zeros(len(nvec), dtype=np.int64)
            )
            act[A_ADMISSION] = 1
            # Prefer reading from inbox (observation path). Also emit protocol msg for next step.
            symbols = _valid_symbols_from_inbox(
                inbox_map.get(a, []), max_age=self.max_symbol_age
            )
            act = apply_semantic_mapping_to_action(act, symbols, self.mapping, nvec)
            # Keep sending fixed-protocol content so peers can observe
            if a.startswith("ue_"):
                act[A_TARGET] = env.possible_agents.index("bs_0") if "bs_0" in env.possible_agents else 0
                act[A_MSG] = 1  # non-silence; payload overridden in fixed_protocol mode
            elif a == "edge_0" and "bs_0" in env.possible_agents:
                act[A_TARGET] = env.possible_agents.index("bs_0")
                act[A_MSG] = 1
            elif a == "ntn_relay":
                act[A_MSG] = 1
            out[a] = act
        return out


def rollout_semantic_return(
    env,
    controller: SemanticProtocolController,
    *,
    seeds: tuple[int, ...] = tuple(range(8)),
    steps: int | None = None,
    intervene_fn=None,
) -> float:
    """Mean episodic team return under semantic controller (± optional intervention)."""
    totals = []
    horizon = steps or env.config.horizon
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        # Warm-start: one fixed-protocol exchange so inboxes are non-empty
        if env.config.comm_mode.value == "fixed_protocol" or env.config.comm_mode.name == "fixed_protocol":
            pass
        ep = 0.0
        for t in range(horizon):
            if not env.agents:
                break
            acts = controller.actions_from_inbox(env, getattr(env, "_last_inbox", {}))
            if intervene_fn is not None:
                acts = intervene_fn(env, acts, t)
            obs, rewards, _, _, _ = env.step(acts)
            if rewards:
                ep += float(sum(rewards.values()) / max(len(rewards), 1))
        totals.append(ep)
    return float(np.mean(totals))
