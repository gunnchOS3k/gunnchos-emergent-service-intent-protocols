"""MARL algorithms for emergent service-intent protocols."""

from emergent_intent.algorithms.dial_tarmac import DialTarmacTrainer
from emergent_intent.algorithms.ippo import IPPOTrainer
from emergent_intent.algorithms.mappo import MAPPOTrainer
from emergent_intent.algorithms.networks import PPOConfig
from emergent_intent.algorithms.value_decomp import VDNQMIXTrainer

__all__ = [
    "DialTarmacTrainer",
    "IPPOTrainer",
    "MAPPOTrainer",
    "PPOConfig",
    "VDNQMIXTrainer",
]


def make_trainer(name: str, env, **kwargs):
    key = name.lower()
    if key == "ippo":
        return IPPOTrainer(env, **kwargs)
    if key == "mappo":
        return MAPPOTrainer(env, **kwargs)
    if key in ("vdn", "qmix"):
        return VDNQMIXTrainer(env, method=key, **kwargs)  # type: ignore[arg-type]
    if key in ("dial", "tarmac", "dial_tarmac"):
        return DialTarmacTrainer(env, **kwargs)
    raise ValueError(f"Unknown algorithm: {name}")
