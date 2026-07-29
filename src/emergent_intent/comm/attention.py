from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class TargetedMessage(nn.Module):
    """TarMAC-style attention over candidate receivers."""

    def __init__(self, d_model: int, n_heads: int = 2):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.n_heads = n_heads

    def forward(self, sender_h: torch.Tensor, receiver_hs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # sender_h: [B,D], receiver_hs: [B,N,D]
        q = self.query(sender_h).unsqueeze(1)  # [B,1,D]
        k = self.key(receiver_hs)
        v = self.value(receiver_hs)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (k.size(-1) ** 0.5)
        attn = F.softmax(scores, dim=-1)
        msg = torch.matmul(attn, v).squeeze(1)
        return msg, attn.squeeze(1)
