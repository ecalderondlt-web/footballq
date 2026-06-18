"""Lightweight spatiotemporal transformer baseline."""

from __future__ import annotations

import torch
from torch import nn


class SpatioTemporalTransformerBaseline(nn.Module):
    """FootBots-style deterministic predictor with temporal and social attention."""

    def __init__(
        self,
        history_steps: int,
        horizon_steps: int,
        n_entities: int,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers_temporal: int = 2,
        n_layers_social: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.history_steps = int(history_steps)
        self.horizon_steps = int(horizon_steps)
        self.n_entities = int(n_entities)
        self.input_projection = nn.Linear(n_features, d_model)
        self.temporal_pos = nn.Parameter(torch.zeros(1, history_steps, 1, d_model))
        self.entity_pos = nn.Parameter(torch.zeros(1, 1, n_entities, d_model))

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        social_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal = nn.TransformerEncoder(temporal_layer, num_layers=n_layers_temporal)
        self.social = nn.TransformerEncoder(social_layer, num_layers=n_layers_social)
        self.decoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon_steps * 2),
        )

    def forward(self, past: torch.Tensor) -> torch.Tensor:
        batch = past.shape[0]
        x = torch.nan_to_num(past, nan=0.0)
        x = self.input_projection(x)
        x = x + self.temporal_pos + self.entity_pos
        x = x.permute(0, 2, 1, 3).reshape(batch * self.n_entities, self.history_steps, -1)
        temporal_state = self.temporal(x)[:, -1, :]
        temporal_state = temporal_state.reshape(batch, self.n_entities, -1)
        social_state = self.social(temporal_state)
        out = self.decoder(social_state)
        out = out.reshape(batch, self.n_entities, self.horizon_steps, 2)
        return out.permute(0, 2, 1, 3).contiguous()
