"""Lightweight coordinate decoder models."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from footballq.decoding.dataset import DecoderDatasetData


class MLPDecoder(nn.Module):
    """Decode a flat latent feature vector to coordinate trajectories."""

    def __init__(
        self,
        input_dim: int,
        output_steps: int,
        n_entities: int = 23,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        hidden_sizes = hidden_sizes or []
        output_dim = int(output_steps) * int(n_entities) * 2
        layers: list[nn.Module] = []
        current = int(input_dim)
        activation_layer: type[nn.Module] = nn.GELU if activation == "gelu" else nn.ReLU
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(current, int(hidden)), activation_layer(), nn.Dropout(dropout)])
            current = int(hidden)
        layers.append(nn.Linear(current, output_dim))
        self.net = nn.Sequential(*layers)
        self.output_steps = int(output_steps)
        self.n_entities = int(n_entities)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = torch.nan_to_num(x.float(), nan=0.0).flatten(start_dim=1)
        out = self.net(flat)
        return out.view(x.shape[0], self.output_steps, self.n_entities, 2)


class SequenceMLPDecoder(nn.Module):
    """Pool or flatten latent sequences before coordinate decoding."""

    def __init__(
        self,
        latent_dim: int,
        input_steps: int,
        output_steps: int,
        n_entities: int = 23,
        pooling: str = "mean",
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.pooling = str(pooling)
        if self.pooling == "flatten":
            input_dim = int(latent_dim) * int(input_steps)
        elif self.pooling in {"mean", "last", "max"}:
            input_dim = int(latent_dim)
        else:
            raise ValueError("pooling must be one of: flatten, mean, last, max")
        self.decoder = MLPDecoder(
            input_dim=input_dim,
            output_steps=output_steps,
            n_entities=n_entities,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x.float(), nan=0.0)
        if self.pooling == "flatten":
            features = x.flatten(start_dim=1)
        elif self.pooling == "last":
            features = x[:, -1]
        elif self.pooling == "max":
            features = x.max(dim=1).values
        else:
            features = x.mean(dim=1)
        return self.decoder(features)


def decoder_output_steps(mode: str, data: DecoderDatasetData) -> int:
    if mode == "reconstruct_current":
        return 1
    if mode == "rollout_from_latents":
        return data.rollout_steps
    return data.horizon_steps


def create_coordinate_decoder(
    config: dict[str, Any],
    data: DecoderDatasetData,
) -> nn.Module:
    """Create a coordinate decoder from a config dictionary."""

    target_cfg = config.get("target", {})
    model_cfg = config.get("model", {})
    mode = str(target_cfg.get("mode", model_cfg.get("input_type", "future_from_z")))
    name = str(model_cfg.get("name", "linear"))
    hidden_sizes = [int(value) for value in model_cfg.get("hidden_sizes", [])]
    if name == "mlp" and not hidden_sizes:
        hidden_sizes = [256, 256]
    output_steps = decoder_output_steps(mode, data)
    dropout = float(model_cfg.get("dropout", 0.0))
    activation = str(model_cfg.get("activation", "relu"))
    if name in {"linear", "mlp"}:
        return MLPDecoder(
            input_dim=data.latent_dim,
            output_steps=output_steps,
            n_entities=data.n_entities,
            hidden_sizes=[] if name == "linear" else hidden_sizes,
            dropout=dropout,
            activation=activation,
        )
    if name in {"context_mlp", "rollout_mlp", "sequence_mlp"}:
        input_steps = data.context_z_steps if mode == "future_from_context" else data.rollout_steps
        return SequenceMLPDecoder(
            latent_dim=data.latent_dim,
            input_steps=input_steps,
            output_steps=output_steps,
            n_entities=data.n_entities,
            pooling=str(model_cfg.get("pooling", "mean" if mode == "future_from_context" else "flatten")),
            hidden_sizes=hidden_sizes or [256],
            dropout=dropout,
            activation=activation,
        )
    raise ValueError(f"Unknown decoder model name: {name!r}")
