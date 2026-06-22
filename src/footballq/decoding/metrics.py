"""Coordinate decoder metrics reported in meters."""

from __future__ import annotations

import math

import torch

from footballq.training.metrics import compute_metrics


def _finite_or_nan(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else float("nan")


def _with_stretch_alias(metrics: dict[str, float]) -> dict[str, float]:
    if "team_stretch_index_error_m" in metrics:
        metrics.setdefault("stretch_index_error_m", metrics["team_stretch_index_error_m"])
    return metrics


def compute_future_coordinate_metrics(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    mask: torch.Tensor,
    entity_type: torch.Tensor,
    team_id: torch.Tensor,
) -> dict[str, float]:
    """Compute future rollout metrics in meters for normalized coordinate tensors."""

    if pred_norm.ndim == 5:
        sample_metrics = compute_sampled_coordinate_metrics(
            pred_norm,
            target_norm,
            mask,
            entity_type,
            team_id,
        )
        return sample_metrics
    metrics = compute_metrics(pred_norm, target_norm, mask, entity_type, team_id)
    return _with_stretch_alias(metrics)


def compute_reconstruction_metrics(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    mask: torch.Tensor,
    entity_type: torch.Tensor,
    team_id: torch.Tensor,
) -> dict[str, float]:
    """Compute current-state reconstruction metrics in meters."""

    metrics = compute_metrics(
        pred_norm.unsqueeze(1),
        target_norm.unsqueeze(1),
        mask.unsqueeze(1),
        entity_type,
        team_id,
    )
    return {
        "current_player_error_m": _finite_or_nan(metrics["player_ADE_m"]),
        "current_ball_error_m": _finite_or_nan(metrics["ball_ADE_m"]),
        "current_all_entity_error_m": _finite_or_nan(metrics["all_entity_ADE_m"]),
        "current_team_centroid_error_m": _finite_or_nan(metrics["team_centroid_error_m"]),
        "current_team_width_error_m": _finite_or_nan(metrics["team_width_error_m"]),
        "current_team_length_error_m": _finite_or_nan(metrics["team_length_error_m"]),
        "current_stretch_index_error_m": _finite_or_nan(
            metrics["team_stretch_index_error_m"]
        ),
    }


def _ade_fde_per_sample(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from footballq.data.normalize import denormalize_xy_to_meters

    pred_m = denormalize_xy_to_meters(pred_norm)
    target_m = denormalize_xy_to_meters(target_norm).unsqueeze(1)
    expanded_mask = mask.bool().unsqueeze(1)
    distances = torch.linalg.vector_norm(pred_m - target_m, dim=-1)
    valid = expanded_mask.float()
    ade = (distances * valid).sum(dim=(2, 3)) / valid.sum(dim=(2, 3)).clamp_min(1.0)
    last_dist = distances[:, :, -1]
    last_mask = expanded_mask[:, :, -1]
    fde = (last_dist * last_mask.float()).sum(dim=2) / last_mask.float().sum(dim=2).clamp_min(1.0)
    return ade, fde


def coordinate_diversity_m(pred_norm: torch.Tensor) -> float:
    """Mean pairwise trajectory distance for predictions [B, K, H, E, 2]."""

    if pred_norm.ndim != 5 or pred_norm.shape[1] < 2:
        return 0.0
    from footballq.data.normalize import denormalize_xy_to_meters

    flat = denormalize_xy_to_meters(pred_norm).flatten(start_dim=2)
    distances = torch.cdist(flat, flat)
    k = int(pred_norm.shape[1])
    upper = torch.triu(torch.ones((k, k), dtype=torch.bool, device=pred_norm.device), diagonal=1)
    return float(distances[:, upper].mean().item())


def compute_sampled_coordinate_metrics(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    mask: torch.Tensor,
    entity_type: torch.Tensor,
    team_id: torch.Tensor,
) -> dict[str, float]:
    """Compute mean and best-of-k coordinate rollout metrics for sampled predictions."""

    mean_pred = pred_norm.mean(dim=1)
    metrics = compute_metrics(mean_pred, target_norm, mask, entity_type, team_id)
    metrics = _with_stretch_alias(metrics)
    ade, fde = _ade_fde_per_sample(pred_norm, target_norm, mask)
    k = int(pred_norm.shape[1])
    metrics.update(
        {
            "mean_ADE_m": float(ade.mean().item()),
            "mean_FDE_m": float(fde.mean().item()),
            "minADE_k_m": float(ade.min(dim=1).values.mean().item()),
            "minFDE_k_m": float(fde.min(dim=1).values.mean().item()),
            f"minADE_{k}_m": float(ade.min(dim=1).values.mean().item()),
            f"minFDE_{k}_m": float(fde.min(dim=1).values.mean().item()),
            "coordinate_diversity_m": coordinate_diversity_m(pred_norm),
        }
    )
    return metrics


def decoder_metrics(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    mask: torch.Tensor,
    entity_type: torch.Tensor,
    team_id: torch.Tensor,
    mode: str,
) -> dict[str, float]:
    if mode == "reconstruct_current":
        if pred_norm.ndim == 4:
            pred_norm = pred_norm[:, 0]
        return compute_reconstruction_metrics(pred_norm, target_norm, mask, entity_type, team_id)
    return compute_future_coordinate_metrics(pred_norm, target_norm, mask, entity_type, team_id)
