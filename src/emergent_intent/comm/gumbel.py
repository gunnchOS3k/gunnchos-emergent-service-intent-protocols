from __future__ import annotations
import torch
import torch.nn.functional as F

def gumbel_softmax_sample(logits: torch.Tensor, tau: float = 1.0, hard: bool = True) -> torch.Tensor:
    """Differentiable discrete sampling; execution can use hard one-hot/argmax."""
    return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)

def discrete_symbols_from_onehot(onehot: torch.Tensor) -> torch.Tensor:
    return onehot.argmax(dim=-1)
