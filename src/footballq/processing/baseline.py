"""Simple trajectory baselines."""

from __future__ import annotations

import numpy as np

from footballq.constants import PITCH_LENGTH_M, PITCH_WIDTH_M


def constant_velocity_rollout(
    last_positions_xy: np.ndarray,
    last_velocities_xy: np.ndarray,
    future_steps: int,
    dt: float,
    clip_players: bool = True,
) -> np.ndarray:
    """Roll positions forward with constant velocity.

    Parameters
    ----------
    last_positions_xy:
        Array shaped `[agents, 2]`.
    last_velocities_xy:
        Array shaped `[agents, 2]`.
    future_steps:
        Number of future steps to predict.
    dt:
        Time delta between steps in seconds.
    clip_players:
        Clip all predicted positions to pitch bounds. This baseline does not know which
        agent is the ball, so callers can disable clipping if needed.
    """

    positions = np.asarray(last_positions_xy, dtype=float)
    velocities = np.asarray(last_velocities_xy, dtype=float)
    steps = np.arange(1, future_steps + 1, dtype=float)[:, None, None]
    rollout = positions[None, :, :] + steps * dt * velocities[None, :, :]
    if clip_players:
        rollout[..., 0] = np.clip(rollout[..., 0], 0.0, PITCH_LENGTH_M)
        rollout[..., 1] = np.clip(rollout[..., 1], 0.0, PITCH_WIDTH_M)
    return rollout

