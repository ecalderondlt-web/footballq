"""Conditional flow-matching objective and sampler for latent futures."""

from __future__ import annotations

import torch
from torch import nn


def flow_matching_loss(
    model: nn.Module,
    past_z: torch.Tensor,
    future_z: torch.Tensor,
    future_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute straight-line conditional flow-matching loss."""

    noise = torch.randn_like(future_z)
    t = torch.rand(future_z.shape[0], device=future_z.device, dtype=future_z.dtype)
    view_t = t.view(-1, 1, 1)
    x_t = (1.0 - view_t) * noise + view_t * future_z
    target_velocity = future_z - noise
    pred_velocity = model(past_z, x_t, t)
    error = (pred_velocity - target_velocity).square()
    if future_mask is not None:
        mask = future_mask.to(error.device).bool().unsqueeze(-1)
        denom = mask.sum().clamp_min(1).to(error.dtype) * error.shape[-1]
        return (error * mask.to(error.dtype)).sum() / denom
    return error.mean()


@torch.no_grad()
def sample_latent_flow(
    model: nn.Module,
    past_z: torch.Tensor,
    horizon_steps: int,
    latent_dim: int,
    num_samples: int = 8,
    num_steps: int = 20,
) -> torch.Tensor:
    """Sample future latent sequences with fixed-step Euler integration."""

    model.eval()
    batch_size = past_z.shape[0]
    num_samples = int(num_samples)
    num_steps = max(1, int(num_steps))
    x = torch.randn(
        batch_size * num_samples,
        int(horizon_steps),
        int(latent_dim),
        device=past_z.device,
        dtype=past_z.dtype,
    )
    repeated_past = past_z.repeat_interleave(num_samples, dim=0)
    dt = 1.0 / float(num_steps)
    for step in range(num_steps):
        t = torch.full(
            (batch_size * num_samples,),
            float(step) / float(num_steps),
            device=past_z.device,
            dtype=past_z.dtype,
        )
        x = x + dt * model(repeated_past, x, t)
    return x.view(batch_size, num_samples, int(horizon_steps), int(latent_dim))
