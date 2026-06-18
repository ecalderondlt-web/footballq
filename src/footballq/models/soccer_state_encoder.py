"""Reusable soccer state encoder for latent representation learning."""

from __future__ import annotations

import torch
from torch import nn


class SoccerStateEncoder(nn.Module):
    """Encode ``[B, C, 23, F]`` soccer state tensors into ``[B, z_dim]`` latents."""

    def __init__(
        self,
        context_steps: int,
        n_entities: int,
        n_features: int,
        z_dim: int = 128,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.context_steps = int(context_steps)
        self.n_entities = int(n_entities)
        self.feature_projection = nn.Linear(n_features, d_model)
        self.temporal_pos = nn.Parameter(torch.zeros(1, context_steps, 1, d_model))
        self.entity_pos = nn.Parameter(torch.zeros(1, 1, n_entities, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, z_dim),
        )

    def forward(self, state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        x = torch.nan_to_num(state, nan=0.0)
        x = self.feature_projection(x) + self.temporal_pos + self.entity_pos
        x = x.reshape(batch, self.context_steps * self.n_entities, -1)
        valid = mask.reshape(batch, self.context_steps * self.n_entities)
        valid_safe = valid.clone()
        empty = valid_safe.sum(dim=1) == 0
        if bool(empty.any()):
            valid_safe[empty, 0] = True
        encoded = self.encoder(x, src_key_padding_mask=~valid_safe)
        weights = valid_safe.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.projection(pooled)
