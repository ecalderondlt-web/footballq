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
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.context_steps = int(context_steps)
        self.n_entities = int(n_entities)
        self.d_model = int(d_model)
        self.pooling = str(pooling)
        if self.pooling not in {"mean", "cls", "temporal_gru"}:
            raise ValueError(
                "SoccerStateEncoder pooling must be 'mean', 'cls', or 'temporal_gru'."
            )
        self.feature_projection = nn.Linear(n_features, d_model)
        self.temporal_pos = nn.Parameter(torch.zeros(1, context_steps, 1, d_model))
        self.entity_pos = nn.Parameter(torch.zeros(1, 1, n_entities, d_model))
        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, d_model)) if self.pooling == "cls" else None
        )
        self.temporal_gru = (
            nn.GRU(d_model, d_model, batch_first=True)
            if self.pooling == "temporal_gru"
            else None
        )
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

    def _encode_sequence(
        self,
        state: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = state.shape[0]
        x = torch.nan_to_num(state, nan=0.0)
        x = self.feature_projection(x) + self.temporal_pos + self.entity_pos
        x = x.reshape(batch, self.context_steps * self.n_entities, -1)
        valid = mask.reshape(batch, self.context_steps * self.n_entities)
        if self.pooling == "cls":
            cls = self.cls_token.expand(batch, -1, -1)
            x = torch.cat([cls, x], dim=1)
            cls_valid = torch.ones(batch, 1, dtype=torch.bool, device=valid.device)
            valid = torch.cat([cls_valid, valid], dim=1)
        valid_safe = valid.clone()
        empty = valid_safe.sum(dim=1) == 0
        if bool(empty.any()):
            valid_safe[empty, 0] = True
        encoded = self.encoder(x, src_key_padding_mask=~valid_safe)
        return encoded, valid_safe

    def encode_entity_tokens(self, state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return one temporally pooled contextual token per entity."""

        batch = state.shape[0]
        encoded, _valid_safe = self._encode_sequence(state, mask)
        if self.pooling == "cls":
            encoded = encoded[:, 1:]
        encoded = encoded.view(batch, self.context_steps, self.n_entities, self.d_model)
        weights = mask.unsqueeze(-1).to(encoded.dtype)
        entity_tokens = (encoded * weights).sum(dim=1)
        entity_tokens = entity_tokens / weights.sum(dim=1).clamp_min(1.0)
        return entity_tokens

    def forward(self, state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        encoded, valid_safe = self._encode_sequence(state, mask)
        if self.pooling == "cls":
            return self.projection(encoded[:, 0])
        if self.pooling == "temporal_gru":
            encoded_frames = encoded.view(
                batch,
                self.context_steps,
                self.n_entities,
                -1,
            )
            frame_valid = valid_safe.view(batch, self.context_steps, self.n_entities)
            frame_weights = frame_valid.unsqueeze(-1).to(encoded.dtype)
            frame_summary = (encoded_frames * frame_weights).sum(dim=2)
            frame_summary = frame_summary / frame_weights.sum(dim=2).clamp_min(1.0)
            _sequence, hidden = self.temporal_gru(frame_summary)
            return self.projection(hidden[-1])
        weights = valid_safe.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.projection(pooled)
