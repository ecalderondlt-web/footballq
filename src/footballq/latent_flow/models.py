"""Neural models for latent flow matching."""

from __future__ import annotations

import math

import torch
from torch import nn

from footballq.latent_flow.baselines import LatentMLPPredictor


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Return sinusoidal embeddings for scalar times in [0, 1]."""

    t = t.float().view(-1, 1)
    half = max(1, dim // 2)
    frequencies = torch.exp(
        torch.linspace(
            math.log(1.0),
            math.log(1000.0),
            half,
            device=t.device,
            dtype=t.dtype,
        )
    ).view(1, -1)
    angles = t * frequencies
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if emb.shape[1] < dim:
        emb = torch.nn.functional.pad(emb, (0, dim - emb.shape[1]))
    return emb[:, :dim]


class LatentFlowMLP(nn.Module):
    """Conditional flow model over flattened future latent sequences."""

    def __init__(
        self,
        latent_dim: int,
        context_steps: int,
        horizon_steps: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        time_embed_dim: int = 64,
        conditioning: str = "past_z_gru",
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.context_steps = int(context_steps)
        self.horizon_steps = int(horizon_steps)
        self.time_embed_dim = int(time_embed_dim)
        self.conditioning = str(conditioning)
        if self.conditioning == "past_z_flat":
            self.context_encoder = nn.Sequential(
                nn.Linear(self.latent_dim * self.context_steps, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        elif self.conditioning == "past_z_gru":
            self.context_encoder = nn.GRU(
                input_size=self.latent_dim,
                hidden_size=hidden_dim,
                batch_first=True,
            )
        else:
            raise ValueError(
                "conditioning must be 'past_z_gru' or 'past_z_flat', "
                f"got {self.conditioning!r}"
            )
        flat_future_dim = self.horizon_steps * self.latent_dim
        input_dim = flat_future_dim + hidden_dim + self.time_embed_dim
        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(max(1, int(num_layers) - 1)):
            layers.extend(
                [
                    nn.Linear(current, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            current = hidden_dim
        layers.append(nn.Linear(current, flat_future_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, past_z: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.conditioning == "past_z_flat":
            context = self.context_encoder(past_z.flatten(start_dim=1))
        else:
            _, hidden = self.context_encoder(past_z)
            context = hidden[-1]
        t_emb = sinusoidal_time_embedding(t, self.time_embed_dim)
        features = torch.cat([x_t.flatten(start_dim=1), context, t_emb], dim=1)
        out = self.net(features)
        return out.view(x_t.shape[0], self.horizon_steps, self.latent_dim)


def create_latent_model(
    config: dict[str, object],
    latent_dim: int,
    context_steps: int,
    horizon_steps: int,
) -> nn.Module:
    """Create a latent flow or deterministic MLP model from config."""

    model_cfg = dict(config.get("model", {})) if isinstance(config, dict) else {}
    name = str(model_cfg.get("name", "latent_flow_mlp"))
    hidden_dim = int(model_cfg.get("hidden_dim", 256))
    num_layers = int(model_cfg.get("num_layers", 4))
    dropout = float(model_cfg.get("dropout", 0.1))
    if name in {"latent_flow_mlp", "residual_latent_flow_mlp"}:
        return LatentFlowMLP(
            latent_dim=latent_dim,
            context_steps=context_steps,
            horizon_steps=horizon_steps,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            time_embed_dim=int(model_cfg.get("time_embed_dim", 64)),
            conditioning=str(model_cfg.get("conditioning", "past_z_gru")),
        )
    if name == "mlp_latent":
        return LatentMLPPredictor(
            latent_dim=latent_dim,
            context_steps=context_steps,
            horizon_steps=horizon_steps,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    raise ValueError(f"Unknown latent model name: {name}")
