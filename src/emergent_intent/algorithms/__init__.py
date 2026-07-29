"""MARL algorithms for emergent service-intent protocols."""

from emergent_intent.algorithms.dial import DialTrainer
from emergent_intent.algorithms.ippo import IPPOTrainer
from emergent_intent.algorithms.mappo import MAPPOTrainer
from emergent_intent.algorithms.networks import PPOConfig
from emergent_intent.algorithms.ppo_discrete_message_entropy_baseline import (
    ALGORITHM_NAME as PPO_MSG_ENTROPY_NAME,
    PPODiscreteMessageEntropyBaseline,
)
from emergent_intent.algorithms.tarmac import TarMACTrainer
from emergent_intent.algorithms.value_decomp import VDNQMIXTrainer

# Truthful alias: former DialTarmacTrainer was not DIAL/TarMAC.
DialTarmacTrainer = PPODiscreteMessageEntropyBaseline

__all__ = [
    "DialTrainer",
    "DialTarmacTrainer",
    "IPPOTrainer",
    "MAPPOTrainer",
    "PPOConfig",
    "PPODiscreteMessageEntropyBaseline",
    "PPO_MSG_ENTROPY_NAME",
    "TarMACTrainer",
    "VDNQMIXTrainer",
]


def make_trainer(name: str, env, **kwargs):
    key = name.lower().replace("-", "_")
    if key == "ippo":
        return IPPOTrainer(env, **kwargs)
    if key == "mappo":
        return MAPPOTrainer(env, **kwargs)
    if key in ("vdn", "qmix"):
        return VDNQMIXTrainer(env, method=key, **kwargs)  # type: ignore[arg-type]
    if key in ("dial", "faithful_dial"):
        return DialTrainer(env, **kwargs)
    if key in ("tarmac", "tarmac_attn"):
        return TarMACTrainer(env, **kwargs)
    if key in (
        "dial_tarmac",
        "ppo_discrete_message_entropy_baseline",
        "ppo_msg_entropy",
        "simple_discrete_message_ppo",
    ):
        return PPODiscreteMessageEntropyBaseline(env, **kwargs)
    raise ValueError(f"Unknown algorithm: {name}")
