"""Soccer temporal-difference JEPA model."""

from __future__ import annotations

import copy

import torch
from torch import nn

from footballq.models.soccer_state_encoder import SoccerStateEncoder


class MotionEncoder(nn.Module):
    """Encode the observed motion chunk into a latent delta."""

    def __init__(
        self,
        delta_steps: int,
        n_entities: int,
        n_features: int,
        z_dim: int = 128,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.delta_steps = int(delta_steps)
        self.n_entities = int(n_entities)
        input_dim = delta_steps * n_entities * n_features + z_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(
        self,
        delta_state: torch.Tensor,
        delta_mask: torch.Tensor,
        z_t: torch.Tensor,
    ) -> torch.Tensor:
        masked = torch.nan_to_num(delta_state, nan=0.0) * delta_mask.unsqueeze(-1).to(
            delta_state.dtype
        )
        flat = masked.reshape(masked.shape[0], -1)
        return self.net(torch.cat([flat, z_t], dim=-1))


class SoccerTDJEPA(nn.Module):
    """Online/target TD-JEPA model with additive latent transition."""

    def __init__(
        self,
        context_steps: int,
        delta_steps: int,
        n_entities: int,
        n_features: int,
        z_dim: int = 128,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        motion_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.online_encoder = SoccerStateEncoder(
            context_steps=context_steps,
            n_entities=n_entities,
            n_features=n_features,
            z_dim=z_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False
        self.motion_encoder = MotionEncoder(
            delta_steps=delta_steps,
            n_entities=n_entities,
            n_features=n_features,
            z_dim=z_dim,
            hidden_dim=motion_hidden_dim,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        z_t = self.online_encoder(batch["state_t"], batch["mask_t"])
        delta_z = self.motion_encoder(batch["delta_state"], batch["delta_mask"], z_t)
        z_pred = z_t + delta_z
        with torch.no_grad():
            z_target = self.target_encoder(
                batch["state_t_plus_delta"],
                batch["mask_t_plus_delta"],
            )
        return {
            "z_t": z_t,
            "delta_z": delta_z,
            "z_pred": z_pred,
            "z_target": z_target,
        }
