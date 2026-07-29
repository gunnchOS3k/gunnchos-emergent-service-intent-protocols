"""Communication package."""

from emergent_intent.comm.attention import TargetedMessage
from emergent_intent.comm.channel import (
    SILENCE,
    AttentionTargeter,
    ChannelConfig,
    GraphMessageRouter,
    MessageChannel,
    discrete_message_from_logits,
    gumbel_softmax_sample,
    make_channel,
    validate_message_shape,
)
from emergent_intent.comm.gumbel import discrete_symbols_from_onehot

__all__ = [
    "SILENCE",
    "AttentionTargeter",
    "ChannelConfig",
    "GraphMessageRouter",
    "MessageChannel",
    "TargetedMessage",
    "discrete_message_from_logits",
    "discrete_symbols_from_onehot",
    "gumbel_softmax_sample",
    "make_channel",
    "validate_message_shape",
]
