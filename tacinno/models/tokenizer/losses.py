"""Reconstruction targets and loss of the TacInno tokenizer."""
from __future__ import annotations

from typing import List, Tuple

import torch

from tacinno.datasets.tactile_fields import CONTACT_THR, DEPTH_REF, PRESS_SCALE

__all__ = ["W_CONTACT", "change_targets", "reconstruction_loss"]

# Value of the indentation channel at the contact threshold (depth < 28.3 mm).
W_CONTACT = (DEPTH_REF - CONTACT_THR) / PRESS_SCALE


def change_targets(fields: torch.Tensor, mode: str = 'cumulative') -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """fields (B, F, gels, C, H, W) at the window frames -> (target (B, F-1, ...), frame pairs whose contact masks
    weight the loss). ``cumulative``: X_k - X_0; ``incremental``: X_k - X_{k-1}."""
    n = fields.shape[1]
    if mode == 'cumulative':
        return fields[:, 1:] - fields[:, :1], [(0, k) for k in range(1, n)]
    if mode == 'incremental':
        return fields[:, 1:] - fields[:, :-1], [(k - 1, k) for k in range(1, n)]
    raise ValueError(mode)


def reconstruction_loss(pred: torch.Tensor, fields: torch.Tensor, mode: str = 'cumulative',
                        contact_weight: float = 4.0) -> torch.Tensor:
    """Contact-weighted L1 per sample and horizon, (B, F-1).

    pred (B, F-1, gels, C, H, W) fp32; fields (B, F, gels, C, H, W). Per pixel the weight is
    1 + contact_weight * [indentation above the contact threshold at either frame of the pair], shared by the C channels;
    the loss is the weighted mean absolute error over gels, channels and pixels."""
    tgt, pairs = change_targets(fields, mode)
    losses = []
    for h, (a, b) in enumerate(pairs):
        con = ((fields[:, a, :, 0] > W_CONTACT) | (fields[:, b, :, 0] > W_CONTACT)).float()
        w = (1.0 + contact_weight * con)[:, :, None].expand_as(pred[:, h])
        losses.append(((pred[:, h] - tgt[:, h]).abs() * w).flatten(1).sum(1) / w.flatten(1).sum(1))
    return torch.stack(losses, 1)
