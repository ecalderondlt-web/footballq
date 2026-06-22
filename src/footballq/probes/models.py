"""Small probe heads trained on frozen features."""

from __future__ import annotations

import torch
from torch import nn


class LinearProbe(nn.Module):
    """A single linear probe head."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.head = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class MLPProbe(nn.Module):
    """A small MLP probe head."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def create_probe_model(
    probe_type: str,
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 128,
    dropout: float = 0.1,
) -> nn.Module:
    """Create a linear or MLP probe head."""

    if probe_type == "linear":
        return LinearProbe(input_dim=input_dim, output_dim=output_dim)
    if probe_type == "mlp":
        return MLPProbe(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    raise ValueError(f"Unknown probe_type {probe_type!r}. Expected 'linear' or 'mlp'.")
