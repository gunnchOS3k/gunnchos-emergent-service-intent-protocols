from __future__ import annotations

from emergent_intent.utils.device import set_global_seed


def seed_everything(seed: int) -> None:
    set_global_seed(seed)
