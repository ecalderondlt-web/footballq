"""Simple latent rollout baselines."""

from __future__ import annotations

import torch
from torch import nn


def last_latent_predict(past_z: torch.Tensor, horizon_steps: int) -> torch.Tensor:
    """Repeat the most recent latent state for the full horizon."""

    return past_z[:, -1:, :].expand(-1, int(horizon_steps), -1).contiguous()


def constant_latent_velocity_predict(past_z: torch.Tensor, horizon_steps: int) -> torch.Tensor:
    """Roll out with a constant difference in latent space."""

    if past_z.shape[1] < 2:
        return last_latent_predict(past_z, horizon_steps)
    dz = past_z[:, -1, :] - past_z[:, -2, :]
    steps = torch.arange(
        1,
        int(horizon_steps) + 1,
        dtype=past_z.dtype,
        device=past_z.device,
    ).view(1, -1, 1)
    return past_z[:, -1:, :] + steps * dz.unsqueeze(1)


def latent_baseline_predict(
    past_z: torch.Tensor,
    horizon_steps: int,
    residual_mode: str,
) -> torch.Tensor:
    """Dispatch a named latent baseline used for residual targets."""

    if residual_mode == "last_latent":
        return last_latent_predict(past_z, horizon_steps)
    if residual_mode in {"constant_latent_velocity", "constant_velocity", "cv"}:
        return constant_latent_velocity_predict(past_z, horizon_steps)
    raise ValueError(
        f"Unknown residual_mode {residual_mode!r}. "
        "Expected last_latent or constant_latent_velocity."
    )


def residual_future(
    past_z: torch.Tensor,
    future_z: torch.Tensor,
    residual_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(baseline_future_z, residual_future_z)`` for a residual mode."""

    baseline = latent_baseline_predict(past_z, future_z.shape[1], residual_mode)
    return baseline, future_z - baseline


def normalize_residual(residual: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Normalize residuals with broadcastable train-set statistics."""

    return (residual - mean.to(residual.device)) / std.to(residual.device).clamp_min(1e-6)


def denormalize_residual(
    residual_norm: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Invert residual normalization."""

    return residual_norm * std.to(residual_norm.device).clamp_min(1e-6) + mean.to(
        residual_norm.device
    )


class LatentMLPPredictor(nn.Module):
    """Deterministic MLP baseline from flattened context to future latents."""

    def __init__(
        self,
        latent_dim: int,
        context_steps: int,
        horizon_steps: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.context_steps = int(context_steps)
        self.horizon_steps = int(horizon_steps)
        layers: list[nn.Module] = []
        input_dim = self.latent_dim * self.context_steps
        output_dim = self.latent_dim * self.horizon_steps
        current = input_dim
        for _ in range(max(1, int(num_layers) - 1)):
            layers.extend(
                [
                    nn.Linear(current, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current = hidden_dim
        layers.append(nn.Linear(current, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, past_z: torch.Tensor) -> torch.Tensor:
        out = self.net(past_z.flatten(start_dim=1))
        return out.view(past_z.shape[0], self.horizon_steps, self.latent_dim)
