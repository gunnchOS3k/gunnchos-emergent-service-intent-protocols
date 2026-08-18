"""Communication-efficiency metrics (CPU / synthetic).

Bits sent are a cost, not evidence of language. Compare against silence
and fixed-protocol baselines; never treat a non-zero message rate as
emergent communication.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def communication_efficiency(
    *,
    message_bits: float,
    n_steps: int,
    n_agents: int,
    silence_fraction: float,
    task_success: float,
    no_comm_success: float | None = None,
    fixed_protocol_success: float | None = None,
    vocab_size: int = 16,
    msg_len: int = 4,
) -> dict[str, Any]:
    steps = max(int(n_steps), 1)
    agents = max(int(n_agents), 1)
    bits = float(message_bits)
    bits_per_step = bits / steps
    bits_per_agent_step = bits / (steps * agents)
    capacity_bits = float(agents * steps * msg_len * np.log2(max(vocab_size, 2)))
    utilization = bits / capacity_bits if capacity_bits > 0 else 0.0
    success = float(task_success)
    bits_per_success = bits / max(success, 1e-8)
    delta_vs_silence = None
    delta_vs_fixed = None
    if no_comm_success is not None:
        delta_vs_silence = success - float(no_comm_success)
    if fixed_protocol_success is not None:
        delta_vs_fixed = success - float(fixed_protocol_success)
    return {
        "message_bits": bits,
        "bits_per_step": bits_per_step,
        "bits_per_agent_step": bits_per_agent_step,
        "silence_fraction": float(silence_fraction),
        "channel_utilization": float(utilization),
        "task_success": success,
        "bits_per_success": float(bits_per_success),
        "delta_success_vs_no_comm": delta_vs_silence,
        "delta_success_vs_fixed_protocol": delta_vs_fixed,
        "evidence_class": "SYNTHETIC_SIM",
        "note": (
            "Efficiency statistics only. Non-zero bits or utilization is not "
            "an emergent-language claim."
        ),
    }


def silence_fraction(symbols: np.ndarray, silence_id: int = 0) -> float:
    arr = np.asarray(symbols).ravel()
    if arr.size == 0:
        return 1.0
    return float(np.mean(arr == silence_id))


def bits_from_symbols(symbols: np.ndarray, vocab_size: int, silence_id: int = 0) -> float:
    arr = np.asarray(symbols).ravel()
    if arr.size == 0:
        return 0.0
    active = arr[arr != silence_id]
    if active.size == 0:
        return 0.0
    return float(active.size * np.log2(max(int(vocab_size), 2)))
