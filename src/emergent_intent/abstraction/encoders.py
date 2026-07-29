from __future__ import annotations

import torch
import torch.nn as nn


class RawAbstraction(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class EngineeredAggregation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mean/std/max pooling style aggregate features
        if x.dim() == 1:
            x = x.unsqueeze(0)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        mx = x.max(dim=-1, keepdim=True).values
        return torch.cat([mean, std, mx], dim=-1)


class InformationBottleneckEncoder(nn.Module):
    """Learned encoder with a justified bottleneck (low-dim Gaussian latent)."""

    def __init__(self, in_dim: int, latent_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, 2 * latent_dim))
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mu, logvar = h[..., : self.latent_dim], h[..., self.latent_dim :]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        # KL to N(0,I) as bottleneck regularizer
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return z, kl
