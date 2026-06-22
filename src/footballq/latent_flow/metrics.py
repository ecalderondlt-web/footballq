"""Latent rollout metrics for Experiment 4A."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(values.dtype)
    return (values * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def _ade_per_sample(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    distances = torch.linalg.norm(pred - target, dim=-1)
    return (distances * mask.float()).sum(dim=-1) / mask.float().sum(dim=-1).clamp_min(1.0)


def _fde_per_sample(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    distances = torch.linalg.norm(pred - target, dim=-1)
    last_indices = mask.long().sum(dim=-1).clamp_min(1) - 1
    return distances.gather(dim=-1, index=last_indices.unsqueeze(-1)).squeeze(-1)


def diversity_mean_pairwise_distance(predictions: torch.Tensor) -> float:
    """Mean pairwise distance across samples for predictions [B, K, H, D]."""

    if predictions.ndim != 4 or predictions.shape[1] < 2:
        return math.nan
    flat = predictions.flatten(start_dim=2)
    distances = torch.cdist(flat, flat)
    k = int(predictions.shape[1])
    upper = torch.triu(torch.ones((k, k), device=predictions.device, dtype=torch.bool), diagonal=1)
    return float(distances[:, upper].mean().item())


def compute_latent_rollout_metrics(
    predictions: torch.Tensor,
    target: torch.Tensor,
    future_mask: torch.Tensor,
) -> dict[str, float]:
    """Compute latent-space metrics for deterministic or sampled predictions."""

    target = target.float()
    future_mask = future_mask.bool()
    if predictions.ndim == 3:
        pred = predictions.float()
        distances = torch.linalg.norm(pred - target, dim=-1)
        mse = (pred - target).square().mean(dim=-1)
        valid_pred = pred[future_mask]
        valid_target = target[future_mask]
        cosine = F.cosine_similarity(valid_pred, valid_target, dim=-1).mean()
        ade = _masked_mean(distances, future_mask)
        fde = _fde_per_sample(pred, target, future_mask).mean()
        step_mse = _masked_mean(mse, future_mask)
        one_step = distances[:, 0].mean()
        return {
            "latent_ADE": float(ade.item()),
            "latent_FDE": float(fde.item()),
            "latent_RMSE": float(torch.sqrt(step_mse).item()),
            "latent_step_mse": float(step_mse.item()),
            "latent_cosine_similarity": float(cosine.item()),
            "cosine_similarity": float(cosine.item()),
            "one_step_error": float(one_step.item()),
            "multi_step_rollout_error": float(ade.item()),
            "minADE": float(ade.item()),
            "minFDE": float(fde.item()),
            "minADE_4": float(ade.item()),
            "minFDE_4": float(fde.item()),
            "minADE_8": float(ade.item()),
            "minFDE_8": float(fde.item()),
            "diversity_mean_pairwise_distance": 0.0,
            "sample_std_mean": 0.0,
            "num_examples": int(target.shape[0]),
        }

    if predictions.ndim != 4:
        raise ValueError(
            "Predictions must have shape [B, H, D] or [B, K, H, D], "
            f"got {tuple(predictions.shape)}"
        )
    pred = predictions.float()
    expanded_target = target.unsqueeze(1).expand_as(pred)
    expanded_mask = future_mask.unsqueeze(1).expand(pred.shape[0], pred.shape[1], pred.shape[2])
    distances = torch.linalg.norm(pred - expanded_target, dim=-1)
    mse = (pred - expanded_target).square().mean(dim=-1)
    valid_pred = pred[expanded_mask]
    valid_target = expanded_target[expanded_mask]
    cosine = F.cosine_similarity(valid_pred, valid_target, dim=-1).mean()
    ade_per = _ade_per_sample(pred, expanded_target, expanded_mask)
    fde_per = _fde_per_sample(pred, expanded_target, expanded_mask)
    k4 = min(4, pred.shape[1])
    k8 = min(8, pred.shape[1])
    kall = pred.shape[1]
    sample_1 = pred[:, 0]
    sample_1_ade = _ade_per_sample(sample_1, target, future_mask).mean()
    sample_1_fde = _fde_per_sample(sample_1, target, future_mask).mean()
    one_step = distances[:, :, 0].mean()
    step_mse = _masked_mean(mse, expanded_mask)
    return {
        "latent_ADE": float(ade_per.mean().item()),
        "latent_FDE": float(fde_per.mean().item()),
        "latent_RMSE": float(torch.sqrt(step_mse).item()),
        "latent_step_mse": float(step_mse.item()),
        "latent_cosine_similarity": float(cosine.item()),
        "cosine_similarity": float(cosine.item()),
        "one_step_error": float(one_step.item()),
        "multi_step_rollout_error": float(ade_per.mean().item()),
        "sample_1_latent_ADE": float(sample_1_ade.item()),
        "sample_1_latent_FDE": float(sample_1_fde.item()),
        "minADE": float(ade_per.min(dim=1).values.mean().item()),
        "minFDE": float(fde_per.min(dim=1).values.mean().item()),
        "minADE_4": float(ade_per[:, :k4].min(dim=1).values.mean().item()),
        "minFDE_4": float(fde_per[:, :k4].min(dim=1).values.mean().item()),
        "minADE_8": float(ade_per[:, :k8].min(dim=1).values.mean().item()),
        "minFDE_8": float(fde_per[:, :k8].min(dim=1).values.mean().item()),
        f"minADE_{kall}": float(ade_per.min(dim=1).values.mean().item()),
        f"minFDE_{kall}": float(fde_per.min(dim=1).values.mean().item()),
        "diversity_mean_pairwise_distance": diversity_mean_pairwise_distance(pred),
        "sample_std_mean": float(pred.std(dim=1, unbiased=False).mean().item()),
        "num_examples": int(target.shape[0]),
    }
