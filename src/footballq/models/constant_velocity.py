"""Constant-velocity deterministic baseline."""

from __future__ import annotations

import torch
from torch import nn


def _last_valid_values(
    past: torch.Tensor,
    past_mask: torch.Tensor,
    feature_indices: list[int],
) -> torch.Tensor:
    """Return last visible feature values per entity."""

    values = past[..., feature_indices]
    batch, history, entities, features = values.shape
    output = torch.zeros((batch, entities, features), dtype=values.dtype, device=values.device)
    for step in range(history):
        mask = past_mask[:, step, :].unsqueeze(-1)
        output = torch.where(mask, values[:, step, :, :], output)
    return output


def predict_constant_velocity(
    past: torch.Tensor,
    past_mask: torch.Tensor,
    horizon_steps: int,
    dt: float,
    feature_names: list[str],
    clip: bool = True,
) -> torch.Tensor:
    """Roll future normalized x/y positions forward from the last visible state."""

    x_idx = feature_names.index("x_norm")
    y_idx = feature_names.index("y_norm")
    vx_idx = feature_names.index("vx_norm")
    vy_idx = feature_names.index("vy_norm")
    last_xy = _last_valid_values(past, past_mask, [x_idx, y_idx])
    last_v = _last_valid_values(past, past_mask, [vx_idx, vy_idx])
    steps = torch.arange(
        1,
        horizon_steps + 1,
        dtype=past.dtype,
        device=past.device,
    ).view(1, horizon_steps, 1, 1)
    pred = last_xy.unsqueeze(1) + steps * float(dt) * last_v.unsqueeze(1)
    if clip:
        pred = pred.clamp(min=-1.25, max=1.25)
    return pred


class ConstantVelocityBaseline(nn.Module):
    """Non-learned baseline that predicts straight-line motion."""

    def __init__(self, horizon_steps: int, fps: float, feature_names: list[str]) -> None:
        super().__init__()
        self.horizon_steps = int(horizon_steps)
        self.dt = 1.0 / float(fps)
        self.feature_names = list(feature_names)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return predict_constant_velocity(
            batch["past"],
            batch["past_mask"],
            horizon_steps=self.horizon_steps,
            dt=self.dt,
            feature_names=self.feature_names,
        )
