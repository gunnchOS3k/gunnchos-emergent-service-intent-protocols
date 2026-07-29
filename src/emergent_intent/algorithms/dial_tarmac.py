"""Deprecated module path.

Historically claimed DIAL/TarMAC but only implemented PPO + message entropy.
Re-exports the truthful baseline. Prefer:
  - emergent_intent.algorithms.dial.DialTrainer for faithful DIAL
  - emergent_intent.algorithms.tarmac.TarMACTrainer for attention routing
  - PPODiscreteMessageEntropyBaseline for the simple discrete-message PPO
"""

from emergent_intent.algorithms.ppo_discrete_message_entropy_baseline import (
    ALGORITHM_NAME,
    DialTarmacTrainer,
    PPODiscreteMessageEntropyBaseline,
)

__all__ = ["ALGORITHM_NAME", "DialTarmacTrainer", "PPODiscreteMessageEntropyBaseline"]
