"""State abstraction: raw, engineered, learned (IB / VQ / contrastive)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


AbstractionKind = Literal["raw", "engineered", "ib_encoder", "vq_encoder", "contrastive"]


def engineered_aggregate(obs: np.ndarray) -> np.ndarray:
    """Hand-engineered compression of local observation vector."""
    o = np.asarray(obs, dtype=np.float32).ravel()
    if o.size < 8:
        o = np.pad(o, (0, 8 - o.size))
    local, inbox = o[:8], o[8:]
    feats = np.array(
        [
            local[0],  # load
            local[2],  # normalized delay
            local[3],  # energy
            local[7],  # tn_link
            float(np.mean(inbox)) if inbox.size else 0.0,
            float(np.std(inbox)) if inbox.size else 0.0,
            float(np.max(np.abs(inbox))) if inbox.size else 0.0,
            float(np.linalg.norm(local)),
        ],
        dtype=np.float32,
    )
    return feats


class IBEncoder(nn.Module):
    """Information-bottleneck style stochastic encoder q(z|o)."""

    def __init__(self, obs_dim: int, latent_dim: int = 8, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        mu, logvar = self.mu(h), self.logvar(h).clamp(-10, 10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    def kl(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


class VQEncoder(nn.Module):
    """Vector-quantized latent codebook encoder."""

    def __init__(self, obs_dim: int, latent_dim: int = 8, codebook: int = 16, hidden: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, latent_dim)
        )
        self.codebook = nn.Embedding(codebook, latent_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook, 1.0 / codebook)
        self.latent_dim = latent_dim

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e = self.encoder(obs)
        # distances to codebook
        dist = (
            z_e.pow(2).sum(1, keepdim=True)
            - 2 * z_e @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(1)
        )
        idx = dist.argmin(1)
        z_q = self.codebook(idx)
        # straight-through
        z = z_e + (z_q - z_e).detach()
        return z, z_e, z_q

    def vq_loss(self, z_e: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(z_q.detach(), z_e) + 0.25 * F.mse_loss(z_q, z_e.detach())


class ContrastiveEncoder(nn.Module):
    """Simple SimCLR-style projector for observation augmentations."""

    def __init__(self, obs_dim: int, latent_dim: int = 8, hidden: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.proj = nn.Sequential(nn.Linear(latent_dim, hidden), nn.ReLU(), nn.Linear(hidden, latent_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def project(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.encoder(obs)
        return F.normalize(self.proj(z), dim=-1)

    @staticmethod
    def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
        B = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)
        sim = z @ z.t() / temperature
        mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, -1e9)
        pos = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
        loss = F.cross_entropy(sim, pos)
        return loss


def abstract_obs(
    obs: np.ndarray,
    kind: AbstractionKind = "raw",
    encoder: nn.Module | None = None,
) -> np.ndarray:
    if kind == "raw":
        return np.asarray(obs, dtype=np.float32)
    if kind == "engineered":
        return engineered_aggregate(obs)
    if encoder is None:
        raise ValueError(f"encoder required for kind={kind}")
    with torch.no_grad():
        t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        out = encoder(t)
        if isinstance(out, tuple):
            out = out[0]
        return out.squeeze(0).cpu().numpy().astype(np.float32)
