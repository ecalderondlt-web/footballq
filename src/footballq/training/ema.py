"""EMA target-network helpers."""

from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def update_ema(target: nn.Module, online: nn.Module, momentum: float = 0.996) -> None:
    """Update target parameters toward online parameters."""

    for target_param, online_param in zip(target.parameters(), online.parameters(), strict=True):
        target_param.data.mul_(momentum).add_(online_param.data, alpha=1.0 - momentum)
