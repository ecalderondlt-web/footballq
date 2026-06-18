"""Simple MLP deterministic trajectory baseline."""

from __future__ import annotations

import torch
from torch import nn


class MLPBaseline(nn.Module):
    """Flattened-window MLP that predicts all entities jointly."""

    def __init__(
        self,
        history_steps: int,
        horizon_steps: int,
        n_entities: int,
        n_features: int,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_sizes = hidden_sizes or [256, 256]
        input_dim = history_steps * n_entities * n_features
        output_dim = horizon_steps * n_entities * 2
        layers: list[nn.Module] = []
        current = input_dim
        for hidden in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(current, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current = hidden
        layers.append(nn.Linear(current, output_dim))
        self.net = nn.Sequential(*layers)
        self.horizon_steps = int(horizon_steps)
        self.n_entities = int(n_entities)

    def forward(self, past: torch.Tensor) -> torch.Tensor:
        batch = past.shape[0]
        flat = torch.nan_to_num(past, nan=0.0).reshape(batch, -1)
        out = self.net(flat)
        return out.reshape(batch, self.horizon_steps, self.n_entities, 2)
