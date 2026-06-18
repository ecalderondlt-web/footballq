"""Evaluation metrics reported in meters."""

from __future__ import annotations

import math

import torch

from footballq.data.normalize import denormalize_xy_to_meters
from footballq.data.windows import ENTITY_BALL, ENTITY_PLAYER, TEAM_AWAY, TEAM_HOME


def _mean_or_nan(values: torch.Tensor) -> float:
    finite = torch.isfinite(values)
    if not bool(finite.any()):
        return float("nan")
    return float(values[finite].mean().item())


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    if selected.numel() == 0:
        return float("nan")
    return float(selected.mean().item())


def _distance_m(pred_norm: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
    pred_m = denormalize_xy_to_meters(pred_norm)
    target_m = denormalize_xy_to_meters(target_norm)
    return torch.linalg.vector_norm(pred_m - target_m, dim=-1)


def _entity_selector(entity_type: torch.Tensor, team_id: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "player":
        return entity_type == ENTITY_PLAYER
    if kind == "ball":
        return entity_type == ENTITY_BALL
    if kind == "home":
        return (entity_type == ENTITY_PLAYER) & (team_id == TEAM_HOME)
    if kind == "away":
        return (entity_type == ENTITY_PLAYER) & (team_id == TEAM_AWAY)
    return torch.ones_like(entity_type, dtype=torch.bool)


def _ade_fde(
    distances: torch.Tensor,
    future_mask: torch.Tensor,
    entity_type: torch.Tensor,
    team_id: torch.Tensor,
    kind: str,
) -> tuple[float, float]:
    selector = _entity_selector(entity_type, team_id, kind).unsqueeze(1)
    valid = future_mask & selector
    ade = _masked_mean(distances, valid)
    fde = _masked_mean(distances[:, -1, :], valid[:, -1, :])
    return ade, fde


def _team_stat(
    xy_m: torch.Tensor,
    mask: torch.Tensor,
    team_selector: torch.Tensor,
    stat: str,
) -> torch.Tensor:
    valid = mask & team_selector.unsqueeze(1)
    valid_f = valid.unsqueeze(-1).to(dtype=xy_m.dtype)
    count = valid_f.sum(dim=2).clamp_min(1.0)
    centroid = (xy_m * valid_f).sum(dim=2) / count
    if stat == "centroid":
        return centroid

    if stat == "width":
        values = xy_m[..., 1]
        max_values = values.masked_fill(~valid, float("-inf")).max(dim=2).values
        min_values = values.masked_fill(~valid, float("inf")).min(dim=2).values
        out = max_values - min_values
        return out.masked_fill(valid.sum(dim=2) == 0, float("nan"))
    if stat == "length":
        values = xy_m[..., 0]
        max_values = values.masked_fill(~valid, float("-inf")).max(dim=2).values
        min_values = values.masked_fill(~valid, float("inf")).min(dim=2).values
        out = max_values - min_values
        return out.masked_fill(valid.sum(dim=2) == 0, float("nan"))
    if stat == "stretch":
        distance = torch.linalg.vector_norm(xy_m - centroid.unsqueeze(2), dim=-1)
        distance = distance.masked_fill(~valid, 0.0)
        count_flat = valid.sum(dim=2)
        out = distance.sum(dim=2) / count_flat.clamp_min(1)
        return out.masked_fill(count_flat == 0, float("nan"))
    raise ValueError(f"Unknown team stat: {stat}")


def _team_shape_errors(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    future_mask: torch.Tensor,
    entity_type: torch.Tensor,
    team_id: torch.Tensor,
) -> dict[str, float]:
    pred_m = denormalize_xy_to_meters(pred_norm)
    target_m = denormalize_xy_to_meters(target_norm)
    teams = [
        (entity_type == ENTITY_PLAYER) & (team_id == TEAM_HOME),
        (entity_type == ENTITY_PLAYER) & (team_id == TEAM_AWAY),
    ]
    centroid_errors: list[torch.Tensor] = []
    width_errors: list[torch.Tensor] = []
    length_errors: list[torch.Tensor] = []
    stretch_errors: list[torch.Tensor] = []
    for selector in teams:
        if not bool(selector.any()):
            continue
        pred_centroid = _team_stat(pred_m, future_mask, selector, "centroid")
        target_centroid = _team_stat(target_m, future_mask, selector, "centroid")
        centroid_errors.append(torch.linalg.vector_norm(pred_centroid - target_centroid, dim=-1))
        for stat, bucket in [
            ("width", width_errors),
            ("length", length_errors),
            ("stretch", stretch_errors),
        ]:
            pred_stat = _team_stat(pred_m, future_mask, selector, stat)
            target_stat = _team_stat(target_m, future_mask, selector, stat)
            bucket.append((pred_stat - target_stat).abs())

    def flatten_mean(parts: list[torch.Tensor]) -> float:
        if not parts:
            return float("nan")
        return _mean_or_nan(torch.cat([part.reshape(-1) for part in parts]))

    return {
        "team_centroid_error_m": flatten_mean(centroid_errors),
        "team_width_error_m": flatten_mean(width_errors),
        "team_length_error_m": flatten_mean(length_errors),
        "team_stretch_index_error_m": flatten_mean(stretch_errors),
    }


def compute_metrics(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    future_mask: torch.Tensor,
    entity_type: torch.Tensor,
    team_id: torch.Tensor,
) -> dict[str, float]:
    """Compute ADE/FDE and team-shape errors in meters."""

    pred_norm = pred_norm.detach().cpu()
    target_norm = target_norm.detach().cpu()
    future_mask = future_mask.detach().cpu().bool()
    entity_type = entity_type.detach().cpu().long()
    team_id = team_id.detach().cpu().long()
    distances = _distance_m(pred_norm, target_norm)
    metrics: dict[str, float] = {}
    for prefix, kind in [
        ("player", "player"),
        ("ball", "ball"),
        ("all_entity", "all"),
        ("home_team", "home"),
        ("away_team", "away"),
    ]:
        ade, fde = _ade_fde(distances, future_mask, entity_type, team_id, kind)
        metrics[f"{prefix}_ADE_m"] = ade
        metrics[f"{prefix}_FDE_m"] = fde
    metrics.update(_team_shape_errors(pred_norm, target_norm, future_mask, entity_type, team_id))

    # Possession-aware splits are emitted as NaN placeholders until real datasets supply possession.
    metrics.setdefault("possession_team_ADE_m", math.nan)
    metrics.setdefault("non_possession_team_ADE_m", math.nan)
    metrics.setdefault("offensive_team_ADE_m", math.nan)
    metrics.setdefault("defensive_team_ADE_m", math.nan)
    return metrics
