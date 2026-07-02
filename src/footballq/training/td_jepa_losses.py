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
) -> dict[str, torch.Tensor]:
    """Compute normalized TD prediction loss and diagnostics."""

    pred_norm = F.normalize(z_pred, dim=-1)
    target_norm = F.normalize(z_target.detach(), dim=-1)
    td_loss = F.mse_loss(pred_norm, target_norm)
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
        weights = state_mask.unsqueeze(-1).to(state_reconstruction.dtype)
        diff_sq = (state_reconstruction - state_target.detach()).pow(2) * weights
        denom = weights.sum().clamp_min(1.0) * state_reconstruction.shape[-1]
        slot_reconstruction = diff_sq.sum() / denom
    total_loss = td_loss + variance_weight * anti_collapse
    total_loss = total_loss + float(slot_reconstruction_weight) * slot_reconstruction
    return {
        "td_loss": td_loss,
        "anti_collapse_loss": anti_collapse,
        "slot_reconstruction_loss": slot_reconstruction,
        "total_loss": total_loss,
        "cosine_similarity": cosine,
        "z_online_std_mean": online_std.mean(),
        "z_online_std_min": online_std.min(),
        "z_target_std_mean": target_std.mean(),
        "z_target_std_min": target_std.min(),
    }
