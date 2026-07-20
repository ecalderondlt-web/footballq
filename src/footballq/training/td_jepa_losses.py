"""TD-JEPA losses and collapse diagnostics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def variance_loss(z: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    """VICReg-style variance floor for anti-collapse pressure."""

    if z.shape[0] < 2:
        return z.new_tensor(0.0)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(threshold - std).mean()


def temporal_motion_reconstruction_loss(
    ordered_prediction: torch.Tensor,
    reversed_prediction: torch.Tensor,
    displacement: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct signed endpoint displacement from both temporal directions."""

    if ordered_prediction.shape != displacement.shape:
        raise ValueError("ordered_prediction and displacement must have matching shapes.")
    if reversed_prediction.shape != displacement.shape:
        raise ValueError("reversed_prediction and displacement must have matching shapes.")
    if valid_mask.shape != displacement.shape[:-1]:
        raise ValueError("valid_mask must match the displacement batch and entity axes.")

    weights = valid_mask.unsqueeze(-1).to(displacement.dtype)
    denominator = (weights.sum() * displacement.shape[-1]).clamp_min(1.0)
    ordered_error = ((ordered_prediction - displacement.detach()).pow(2) * weights).sum()
    reversed_error = ((reversed_prediction + displacement.detach()).pow(2) * weights).sum()
    loss = 0.5 * (ordered_error + reversed_error) / denominator

    predictions = torch.cat([ordered_prediction, reversed_prediction], dim=0)
    targets = torch.cat([displacement, -displacement], dim=0)
    combined_mask = torch.cat([valid_mask, valid_mask], dim=0)
    prediction_vectors = predictions[combined_mask]
    target_vectors = targets[combined_mask]
    moving = target_vectors.norm(dim=-1) > 1e-6
    if bool(moving.any()):
        cosine = F.cosine_similarity(
            prediction_vectors[moving],
            target_vectors[moving],
            dim=-1,
        ).mean()
    else:
        cosine = loss.new_tensor(0.0)
    return loss, cosine


def match_mean_invariance_loss(
    z: torch.Tensor,
    match_ids: list[str] | tuple[str, ...],
) -> tuple[torch.Tensor, int]:
    """Penalize differences between normalized batch means for distinct matches."""

    if len(match_ids) != int(z.shape[0]):
        raise ValueError("match_ids must have one value per embedding row.")
    groups: dict[str, list[int]] = {}
    for index, match_id in enumerate(match_ids):
        groups.setdefault(str(match_id), []).append(index)
    if len(groups) < 2:
        return z.new_tensor(0.0), len(groups)
    normalized = F.normalize(z, dim=-1)
    means = torch.stack(
        [normalized[indices].mean(dim=0) for indices in groups.values()],
        dim=0,
    )
    centered = means - means.mean(dim=0, keepdim=True)
    return centered.pow(2).mean(), len(groups)


def _masked_state_mse(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(reconstruction.dtype)
    diff_sq = (reconstruction - target.detach()).pow(2) * weights
    denom = weights.sum().clamp_min(1.0) * reconstruction.shape[-1]
    return diff_sq.sum() / denom


def td_jepa_loss(
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
    z_online: torch.Tensor,
    variance_weight: float = 0.1,
    variance_threshold: float = 1.0,
    state_reconstruction: torch.Tensor | None = None,
    state_target: torch.Tensor | None = None,
    state_mask: torch.Tensor | None = None,
    slot_reconstruction_weight: float = 0.0,
    context_reconstruction: torch.Tensor | None = None,
    context_target: torch.Tensor | None = None,
    context_mask: torch.Tensor | None = None,
    context_reconstruction_weight: float = 0.0,
    no_motion_margin_weight: float = 0.0,
    no_motion_margin: float = 0.01,
    transition_reconstruction: torch.Tensor | None = None,
    transition_target: torch.Tensor | None = None,
    transition_mask: torch.Tensor | None = None,
    transition_reconstruction_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Compute normalized TD prediction loss and diagnostics."""

    pred_norm = F.normalize(z_pred, dim=-1)
    target_norm = F.normalize(z_target.detach(), dim=-1)
    online_norm = F.normalize(z_online, dim=-1)
    pred_dist = (pred_norm - target_norm).pow(2).mean(dim=-1)
    online_dist = (online_norm - target_norm).pow(2).mean(dim=-1)
    td_loss = pred_dist.mean()
    no_motion_td_loss = online_dist.mean()
    no_motion_margin_loss = F.relu(
        float(no_motion_margin) + pred_dist - online_dist
    ).mean()
    online_var = variance_loss(z_online, threshold=variance_threshold)
    target_var = variance_loss(z_target.detach(), threshold=variance_threshold)
    anti_collapse = online_var + target_var
    cosine = F.cosine_similarity(pred_norm, target_norm, dim=-1).mean()
    online_std = z_online.std(dim=0, unbiased=False)
    target_std = z_target.detach().std(dim=0, unbiased=False)
    slot_reconstruction = z_pred.new_tensor(0.0)
    if slot_reconstruction_weight > 0.0:
        if state_reconstruction is None or state_target is None or state_mask is None:
            raise ValueError(
                "slot_reconstruction_weight requires state_reconstruction, "
                "state_target, and state_mask."
            )
        slot_reconstruction = _masked_state_mse(
            state_reconstruction,
            state_target,
            state_mask,
        )
    context_reconstruction_loss = z_pred.new_tensor(0.0)
    if context_reconstruction_weight > 0.0:
        if context_reconstruction is None or context_target is None or context_mask is None:
            raise ValueError(
                "context_reconstruction_weight requires context_reconstruction, "
                "context_target, and context_mask."
            )
        context_reconstruction_loss = _masked_state_mse(
            context_reconstruction,
            context_target,
            context_mask,
        )
    transition_reconstruction_loss = z_pred.new_tensor(0.0)
    if transition_reconstruction_weight > 0.0:
        if (
            transition_reconstruction is None
            or transition_target is None
            or transition_mask is None
        ):
            raise ValueError(
                "transition_reconstruction_weight requires transition_reconstruction, "
                "transition_target, and transition_mask."
            )
        transition_reconstruction_loss = _masked_state_mse(
            transition_reconstruction,
            transition_target,
            transition_mask,
        )
    ball_dynamic_reconstruction_loss = z_pred.new_tensor(0.0)
    if state_reconstruction is not None and state_target is not None and state_mask is not None:
        dynamic_features = min(4, state_reconstruction.shape[-1])
        ball_dynamic_reconstruction_loss = _masked_state_mse(
            state_reconstruction[:, :, :1, :dynamic_features],
            state_target[:, :, :1, :dynamic_features],
            state_mask[:, :, :1],
        )
    total_loss = td_loss + variance_weight * anti_collapse
    total_loss = total_loss + float(slot_reconstruction_weight) * slot_reconstruction
    total_loss = (
        total_loss
        + float(context_reconstruction_weight) * context_reconstruction_loss
    )
    total_loss = total_loss + float(no_motion_margin_weight) * no_motion_margin_loss
    total_loss = (
        total_loss
        + float(transition_reconstruction_weight) * transition_reconstruction_loss
    )
    return {
        "td_loss": td_loss,
        "no_motion_td_loss": no_motion_td_loss,
        "no_motion_margin_loss": no_motion_margin_loss,
        "anti_collapse_loss": anti_collapse,
        "slot_reconstruction_loss": slot_reconstruction,
        "context_reconstruction_loss": context_reconstruction_loss,
        "transition_reconstruction_loss": transition_reconstruction_loss,
        "ball_dynamic_reconstruction_loss": ball_dynamic_reconstruction_loss,
        "total_loss": total_loss,
        "cosine_similarity": cosine,
        "z_online_std_mean": online_std.mean(),
        "z_online_std_min": online_std.min(),
        "z_target_std_mean": target_std.mean(),
        "z_target_std_min": target_std.min(),
    }
