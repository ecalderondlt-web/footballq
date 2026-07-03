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
    total_loss = td_loss + variance_weight * anti_collapse
    total_loss = total_loss + float(slot_reconstruction_weight) * slot_reconstruction
    total_loss = (
        total_loss
        + float(context_reconstruction_weight) * context_reconstruction_loss
    )
    total_loss = total_loss + float(no_motion_margin_weight) * no_motion_margin_loss
    return {
        "td_loss": td_loss,
        "no_motion_td_loss": no_motion_td_loss,
        "no_motion_margin_loss": no_motion_margin_loss,
        "anti_collapse_loss": anti_collapse,
        "slot_reconstruction_loss": slot_reconstruction,
        "context_reconstruction_loss": context_reconstruction_loss,
        "total_loss": total_loss,
        "cosine_similarity": cosine,
        "z_online_std_mean": online_std.mean(),
        "z_online_std_min": online_std.min(),
        "z_target_std_mean": target_std.mean(),
        "z_target_std_min": target_std.min(),
    }
