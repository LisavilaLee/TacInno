"""Per-slot vector quantizer of the tactile innovation code."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PerSlotVectorQuantizer"]


class PerSlotVectorQuantizer(nn.Module):
    """One codebook per code slot: ``codebook`` (num_slots, codebook_size, code_dim).

    Each slot's latent vector is replaced by its Euclidean nearest codebook entry (straight-through gradient, as in
    VQ-VAE). Returns the codebook loss ``||sg(z) - e||^2`` and the commitment loss ``||z - sg(e)||^2`` (weighted by the
    caller). The codebook is initialised from encoder outputs at the first training step (:meth:`init_from`), and
    entries that were not selected since the last reset are re-initialised from encoder outputs (:meth:`reset_dead`).
    """

    def __init__(self, num_slots: int = 4, codebook_size: int = 16, code_dim: int = 32):
        super().__init__()
        self.n_q, self.n_codes = num_slots, codebook_size
        self.codebook = nn.Parameter(torch.randn(num_slots, codebook_size, code_dim))
        self.register_buffer('usage', torch.zeros(num_slots, codebook_size))
        self.register_buffer('cb_initialised', torch.zeros((), dtype=torch.bool))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """z (B, slots, code_dim) fp32 -> (z_q straight-through, indices (B, slots), codebook loss, commitment loss)."""
        e = self.codebook
        dist = z.pow(2).sum(-1, keepdim=True) - 2 * torch.einsum('bqd,qkd->bqk', z, e) + e.pow(2).sum(-1)[None]
        idx = dist.argmin(-1)
        q = e[torch.arange(self.n_q, device=z.device)[None], idx]
        return z + (q - z).detach(), idx, F.mse_loss(q, z.detach()), F.mse_loss(z, q.detach())

    def lookup(self, idx: torch.Tensor) -> torch.Tensor:
        """indices (B, slots) -> codebook vectors (B, slots, code_dim)."""
        return self.codebook[torch.arange(self.n_q, device=idx.device)[None], idx]

    @torch.no_grad()
    def init_from(self, z: torch.Tensor) -> None:
        for s in range(self.n_q):
            sel = torch.randperm(z.shape[0], device=z.device)[:self.n_codes]
            self.codebook.data[s].copy_(z[sel, s])
        self.cb_initialised.fill_(True)

    @torch.no_grad()
    def count(self, idx: torch.Tensor) -> None:
        self.usage += F.one_hot(idx, self.n_codes).float().sum(0)

    @torch.no_grad()
    def reset_dead(self, z: torch.Tensor, gen: Optional[torch.Generator] = None) -> int:
        """Replace entries unused since the last reset by random encoder outputs of this batch; returns #replaced."""
        n = 0
        for s in range(self.n_q):
            dead = torch.nonzero(self.usage[s] == 0)[:, 0]
            if len(dead):
                sel = torch.randint(0, z.shape[0], (len(dead),), device=z.device, generator=gen)
                self.codebook.data[s, dead] = z[sel, s]
            n += len(dead)
        self.usage.zero_()
        return int(n)
