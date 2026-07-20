"""Causal StatsBomb event encoder with optional sparse 360 geometry."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class FreezeFrameSetEncoder(nn.Module):
    """Encode an unordered, masked StatsBomb 360 freeze frame."""

    def __init__(self, n_features: int, d_model: int) -> None:
        super().__init__()
        self.player_encoder = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        self.availability_embedding = nn.Embedding(2, d_model)

    def forward(
        self,
        freeze_frame: torch.Tensor,
        freeze_mask: torch.Tensor,
        has_360: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.player_encoder(torch.nan_to_num(freeze_frame, nan=0.0))
        weights = freeze_mask.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * weights).sum(dim=-2) / weights.sum(dim=-2).clamp_min(1.0)
        return pooled + self.availability_embedding(has_360.long())


class StatsBombEventEncoder(nn.Module):
    """Encode causal event sequences and predict the next event and location."""

    def __init__(
        self,
        categorical_vocab_sizes: list[int],
        n_continuous_features: int,
        n_freeze_frame_features: int,
        *,
        event_type_feature_index: int = 0,
        geometry_continuous_indices: tuple[int, ...] = (15, 16),
        use_360: bool = True,
        categorical_dim: int = 24,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        max_sequence_length: int = 64,
    ) -> None:
        super().__init__()
        if not categorical_vocab_sizes:
            raise ValueError("StatsBombEventEncoder requires categorical vocabularies.")
        if not 0 <= event_type_feature_index < len(categorical_vocab_sizes):
            raise ValueError("event_type_feature_index is out of range.")
        self.use_360 = bool(use_360)
        self.max_sequence_length = int(max_sequence_length)
        self.event_type_feature_index = int(event_type_feature_index)
        self.categorical_embeddings = nn.ModuleList(
            nn.Embedding(int(size), categorical_dim) for size in categorical_vocab_sizes
        )
        self.continuous_projection = nn.Sequential(
            nn.Linear(n_continuous_features, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        continuous_mask = torch.ones(n_continuous_features, dtype=torch.float32)
        if not self.use_360:
            for index in geometry_continuous_indices:
                continuous_mask[int(index)] = 0.0
        self.register_buffer("continuous_feature_mask", continuous_mask, persistent=True)
        self.freeze_frame_encoder = FreezeFrameSetEncoder(n_freeze_frame_features, d_model)
        input_dim = len(categorical_vocab_sizes) * categorical_dim + d_model * 2
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.max_sequence_length, d_model)
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
        event_type_size = int(categorical_vocab_sizes[self.event_type_feature_index])
        self.event_type_head = nn.Linear(d_model, event_type_size)
        self.location_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
            nn.Sigmoid(),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        categorical = batch["categorical"]
        continuous = torch.nan_to_num(batch["continuous"], nan=0.0)
        event_mask = batch["event_mask"].bool()
        sequence_length = int(categorical.shape[1])
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"Event sequence length {sequence_length} exceeds {self.max_sequence_length}."
            )
        categorical_parts = [
            embedding(categorical[..., index])
            for index, embedding in enumerate(self.categorical_embeddings)
        ]
        parts = [
            *categorical_parts,
            self.continuous_projection(continuous * self.continuous_feature_mask),
        ]
        freeze_frame = batch["freeze_frame"]
        freeze_mask = batch["freeze_mask"]
        has_360 = batch["has_360"]
        if not self.use_360:
            freeze_frame = torch.zeros_like(freeze_frame)
            freeze_mask = torch.zeros_like(freeze_mask)
            has_360 = torch.zeros_like(has_360)
        parts.append(
            self.freeze_frame_encoder(
                freeze_frame,
                freeze_mask,
                has_360,
            )
        )
        encoded_input = self.input_projection(torch.cat(parts, dim=-1))
        encoded_input = encoded_input + self.position_embedding[:, :sequence_length]
        safe_mask = event_mask.clone()
        empty = safe_mask.sum(dim=1) == 0
        if bool(empty.any()):
            safe_mask[empty, 0] = True
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=encoded_input.device,
            ),
            diagonal=1,
        )
        sequence = self.encoder(
            encoded_input,
            mask=causal_mask,
            src_key_padding_mask=~safe_mask,
        )
        weights = safe_mask.unsqueeze(-1).to(sequence.dtype)
        pooled = (sequence * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return {
            "sequence": sequence,
            "pooled": pooled,
            "next_event_type_logits": self.event_type_head(sequence),
            "next_location": self.location_head(sequence),
        }


def statsbomb_event_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    location_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute masked next-event classification and next-location regression losses."""

    valid = batch["event_mask"].bool()
    logits = outputs["next_event_type_logits"]
    target_type = batch["target_event_type"]
    event_type_loss = F.cross_entropy(logits[valid], target_type[valid])
    event_type_accuracy = (logits.argmax(dim=-1)[valid] == target_type[valid]).float().mean()

    location_valid = valid & batch["target_location_mask"].bool()
    if bool(location_valid.any()):
        predicted_location = outputs["next_location"][location_valid]
        target_location = batch["target_location"][location_valid]
        location_loss = F.smooth_l1_loss(predicted_location, target_location)
        location_mae = F.l1_loss(predicted_location, target_location)
    else:
        location_loss = logits.sum() * 0.0
        location_mae = logits.sum() * 0.0
    total_loss = event_type_loss + float(location_weight) * location_loss
    return {
        "total_loss": total_loss,
        "event_type_loss": event_type_loss,
        "event_type_accuracy": event_type_accuracy,
        "location_loss": location_loss,
        "location_mae": location_mae,
        "location_target_count": location_valid.sum().to(logits.dtype),
    }


def detached_event_metrics(losses: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Convert scalar event losses to JSON-compatible values."""

    return {name: float(value.detach().cpu()) for name, value in losses.items()}
