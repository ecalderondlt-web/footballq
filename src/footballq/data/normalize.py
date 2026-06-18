"""Coordinate normalization for Phase 1 trajectory baselines."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from footballq.constants import PITCH_CENTER_X_M, PITCH_CENTER_Y_M

XY_SCALE_M = np.array([PITCH_CENTER_X_M, PITCH_CENTER_Y_M], dtype=np.float32)


def normalize_xy_from_meters(xy_m: Any) -> Any:
    """Convert top-left-origin pitch meters to centered normalized coordinates.

    The project stores canonical coordinates as meters in ``[0, 105] x [0, 68]``.
    The learning pipeline uses approximately ``[-1, 1]`` coordinates by centering
    the pitch first and then dividing by half-pitch dimensions.
    """

    if isinstance(xy_m, torch.Tensor):
        scale = torch.as_tensor(XY_SCALE_M, dtype=xy_m.dtype, device=xy_m.device)
        return (xy_m - scale) / scale
    arr = np.asarray(xy_m, dtype=np.float32)
    return (arr - XY_SCALE_M) / XY_SCALE_M


def denormalize_xy_to_meters(xy_norm: Any) -> Any:
    """Convert centered normalized coordinates back to canonical pitch meters."""

    if isinstance(xy_norm, torch.Tensor):
        scale = torch.as_tensor(XY_SCALE_M, dtype=xy_norm.dtype, device=xy_norm.device)
        return xy_norm * scale + scale
    arr = np.asarray(xy_norm, dtype=np.float32)
    return arr * XY_SCALE_M + XY_SCALE_M


def normalize_velocity_from_mps(vxy_mps: Any) -> Any:
    """Convert meters-per-second velocity to normalized-coordinate velocity."""

    if isinstance(vxy_mps, torch.Tensor):
        scale = torch.as_tensor(XY_SCALE_M, dtype=vxy_mps.dtype, device=vxy_mps.device)
        return vxy_mps / scale
    arr = np.asarray(vxy_mps, dtype=np.float32)
    return arr / XY_SCALE_M


def denormalize_velocity_to_mps(vxy_norm: Any) -> Any:
    """Convert normalized-coordinate velocity to meters per second."""

    if isinstance(vxy_norm, torch.Tensor):
        scale = torch.as_tensor(XY_SCALE_M, dtype=vxy_norm.dtype, device=vxy_norm.device)
        return vxy_norm * scale
    arr = np.asarray(vxy_norm, dtype=np.float32)
    return arr * XY_SCALE_M
