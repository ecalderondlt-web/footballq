"""Stress-slice labels for grouped coordinate-decoder evaluation."""

from __future__ import annotations

import math
from typing import Any

import torch

from footballq.data.normalize import denormalize_xy_to_meters
from footballq.data.windows import BALL_INDEX, ENTITY_PLAYER, TEAM_AWAY, TEAM_HOME
from footballq.decoding.dataset import DecoderDatasetData

STRESS_SLICE_NAMES = [
    "all_windows",
    "high_future_ball_displacement",
    "high_ball_acceleration",
    "high_ball_direction_change",
    "high_team_shape_change",
    "high_team_width_change",
    "high_team_length_change",
    "high_stretch_index_change",
    "possession_change",
    "event_near_window",
]


def _nan_quantile(values: torch.Tensor, q: float) -> float:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return float("nan")
    return float(torch.quantile(finite.float(), float(q)).item())


def _high_mask(values: torch.Tensor, percentile: float) -> tuple[torch.Tensor, float]:
    threshold = _nan_quantile(values, percentile)
    if not math.isfinite(threshold):
        return torch.zeros_like(values, dtype=torch.bool), threshold
    return torch.isfinite(values) & (values >= threshold), threshold


def _safe_norm_delta(a: torch.Tensor, b: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    values = torch.linalg.vector_norm(a - b, dim=-1)
    return values.masked_fill(~valid, float("nan"))


def _ball_values(data: DecoderDatasetData) -> dict[str, torch.Tensor]:
    current = denormalize_xy_to_meters(data.examples["current_xy"].float())[:, BALL_INDEX]
    future = denormalize_xy_to_meters(data.examples["future_xy"].float())[:, :, BALL_INDEX]
    current_mask = data.examples["current_mask"].bool()[:, BALL_INDEX]
    future_mask = data.examples["future_mask"].bool()[:, :, BALL_INDEX]
    last_valid = future_mask[:, -1] & current_mask
    displacement = _safe_norm_delta(future[:, -1], current, last_valid)

    if future.shape[1] >= 3:
        fps = float(data.metadata.get("fps", 10.0))
        velocities = (future[:, 1:] - future[:, :-1]) * fps
        velocity_mask = future_mask[:, 1:] & future_mask[:, :-1]
        acceleration = (velocities[:, 1:] - velocities[:, :-1]) * fps
        acceleration_mask = velocity_mask[:, 1:] & velocity_mask[:, :-1]
        acceleration_mag = torch.linalg.vector_norm(acceleration, dim=-1)
        acceleration_mag = acceleration_mag.masked_fill(~acceleration_mask, float("nan"))
        ball_acceleration = torch.nan_to_num(acceleration_mag, nan=-float("inf")).max(dim=1).values
        ball_acceleration = ball_acceleration.masked_fill(~torch.isfinite(ball_acceleration), float("nan"))

        v1 = velocities[:, :-1]
        v2 = velocities[:, 1:]
        direction_mask = velocity_mask[:, :-1] & velocity_mask[:, 1:]
        denom = (
            torch.linalg.vector_norm(v1, dim=-1) * torch.linalg.vector_norm(v2, dim=-1)
        ).clamp_min(1e-6)
        cos = ((v1 * v2).sum(dim=-1) / denom).clamp(-1.0, 1.0)
        angles = torch.acos(cos).masked_fill(~direction_mask, float("nan"))
        direction_change = torch.nan_to_num(angles, nan=-float("inf")).max(dim=1).values
        direction_change = direction_change.masked_fill(
            ~torch.isfinite(direction_change),
            float("nan"),
        )
    else:
        n = data.num_examples
        ball_acceleration = torch.full((n,), float("nan"))
        direction_change = torch.full((n,), float("nan"))

    return {
        "future_ball_displacement_m": displacement,
        "ball_acceleration_mps2": ball_acceleration,
        "ball_direction_change_rad": direction_change,
    }


def _team_selector(data: DecoderDatasetData, team_code: int) -> torch.Tensor:
    return (data.examples["entity_type"].long() == ENTITY_PLAYER) & (
        data.examples["team_id"].long() == team_code
    )


def _team_stat(xy_m: torch.Tensor, mask: torch.Tensor, selector: torch.Tensor, stat: str) -> torch.Tensor:
    valid = mask & selector
    valid_f = valid.unsqueeze(-1).float()
    count = valid_f.sum(dim=1).clamp_min(1.0)
    centroid = (xy_m * valid_f).sum(dim=1) / count
    if stat == "centroid":
        return centroid
    if stat == "width":
        values = xy_m[..., 1]
    elif stat == "length":
        values = xy_m[..., 0]
    elif stat == "stretch":
        distance = torch.linalg.vector_norm(xy_m - centroid.unsqueeze(1), dim=-1)
        out = (distance * valid.float()).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return out.masked_fill(valid.sum(dim=1) == 0, float("nan"))
    else:
        raise ValueError(f"Unknown team stat: {stat}")
    max_values = values.masked_fill(~valid, float("-inf")).max(dim=1).values
    min_values = values.masked_fill(~valid, float("inf")).min(dim=1).values
    out = max_values - min_values
    return out.masked_fill(valid.sum(dim=1) == 0, float("nan"))


def _team_shape_values(data: DecoderDatasetData) -> dict[str, torch.Tensor]:
    current = denormalize_xy_to_meters(data.examples["current_xy"].float())
    future_last = denormalize_xy_to_meters(data.examples["future_xy"].float())[:, -1]
    current_mask = data.examples["current_mask"].bool()
    future_mask = data.examples["future_mask"].bool()[:, -1]
    width_parts: list[torch.Tensor] = []
    length_parts: list[torch.Tensor] = []
    stretch_parts: list[torch.Tensor] = []
    centroid_parts: list[torch.Tensor] = []
    for team_code in [TEAM_HOME, TEAM_AWAY]:
        selector = _team_selector(data, team_code)
        if not bool(selector.any()):
            continue
        current_centroid = _team_stat(current, current_mask, selector, "centroid")
        future_centroid = _team_stat(future_last, future_mask, selector, "centroid")
        centroid_parts.append(torch.linalg.vector_norm(future_centroid - current_centroid, dim=-1))
        for stat, bucket in [
            ("width", width_parts),
            ("length", length_parts),
            ("stretch", stretch_parts),
        ]:
            current_stat = _team_stat(current, current_mask, selector, stat)
            future_stat = _team_stat(future_last, future_mask, selector, stat)
            bucket.append((future_stat - current_stat).abs())

    def _mean(parts: list[torch.Tensor]) -> torch.Tensor:
        if not parts:
            return torch.full((data.num_examples,), float("nan"))
        stacked = torch.stack(parts, dim=1)
        return torch.nanmean(stacked, dim=1)

    width_change = _mean(width_parts)
    length_change = _mean(length_parts)
    stretch_change = _mean(stretch_parts)
    centroid_change = _mean(centroid_parts)
    return {
        "team_shape_change_m": centroid_change + width_change + length_change + stretch_change,
        "team_width_change_m": width_change,
        "team_length_change_m": length_change,
        "stretch_index_change_m": stretch_change,
    }


def _metadata_masks(data: DecoderDatasetData) -> dict[str, torch.Tensor]:
    n = data.num_examples
    event_type = [str(value).strip().lower() for value in data.examples.get("event_type", [])]
    event_available = torch.tensor(
        [value not in {"", "unknown", "none", "nan", "<na>"} for value in event_type],
        dtype=torch.bool,
    )
    if event_available.numel() != n:
        event_available = torch.zeros(n, dtype=torch.bool)

    possession_change = data.examples.get("possession_change")
    if isinstance(possession_change, torch.Tensor) and possession_change.numel() == n:
        possession_mask = possession_change.bool()
    else:
        possession_mask = torch.zeros(n, dtype=torch.bool)
    return {"possession_change": possession_mask, "event_near_window": event_available}


def compute_stress_slices(
    data: DecoderDatasetData,
    percentile: float = 0.75,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Compute evaluation-only stress-slice masks from ground-truth futures.

    The returned masks are not used by decoder datasets as input features. They are intended only
    for grouped evaluation and summary reporting.
    """

    scores = {**_ball_values(data), **_team_shape_values(data)}
    thresholds: dict[str, Any] = {
        "percentile": float(percentile),
        "definitions": {
            "high_future_ball_displacement": "top percentile of final future ball displacement from current ball position",
            "high_ball_acceleration": "top percentile of max future ball acceleration magnitude",
            "high_ball_direction_change": "top percentile of max future ball velocity direction change",
            "high_team_shape_change": "top percentile of final team centroid/width/length/stretch change sum",
            "high_team_width_change": "top percentile of final team width change",
            "high_team_length_change": "top percentile of final team length change",
            "high_stretch_index_change": "top percentile of final team stretch-index change",
            "possession_change": "uses explicit possession_change examples field when present",
            "event_near_window": "label-frame event_type is known/nonempty",
        },
    }
    masks: dict[str, torch.Tensor] = {
        "all_windows": torch.ones(data.num_examples, dtype=torch.bool),
    }
    mapping = {
        "high_future_ball_displacement": "future_ball_displacement_m",
        "high_ball_acceleration": "ball_acceleration_mps2",
        "high_ball_direction_change": "ball_direction_change_rad",
        "high_team_shape_change": "team_shape_change_m",
        "high_team_width_change": "team_width_change_m",
        "high_team_length_change": "team_length_change_m",
        "high_stretch_index_change": "stretch_index_change_m",
    }
    for slice_name, score_name in mapping.items():
        mask, threshold = _high_mask(scores[score_name], percentile)
        masks[slice_name] = mask
        thresholds[f"{slice_name}_threshold"] = threshold
        thresholds[f"{slice_name}_score"] = score_name
    masks.update(_metadata_masks(data))
    thresholds["possession_change_available"] = bool(masks["possession_change"].any())
    thresholds["event_near_window_available"] = bool(masks["event_near_window"].any())
    thresholds["slice_counts_all_examples"] = {
        name: int(mask.sum().item()) for name, mask in masks.items()
    }
    return masks, thresholds

