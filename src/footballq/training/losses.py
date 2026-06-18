"""Loss functions for deterministic trajectory prediction."""

from __future__ import annotations

import torch


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over visible future entities only."""

    valid = mask.unsqueeze(-1).to(dtype=pred.dtype)
    squared = (pred - target).pow(2) * valid
    denom = valid.sum().clamp_min(1.0) * pred.shape[-1]
    return squared.sum() / denom
