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
    return {
        "td_loss": td_loss,
        "anti_collapse_loss": anti_collapse,
        "total_loss": td_loss + variance_weight * anti_collapse,
        "cosine_similarity": cosine,
        "z_online_std_mean": online_std.mean(),
        "z_online_std_min": online_std.min(),
        "z_target_std_mean": target_std.mean(),
        "z_target_std_min": target_std.min(),
    }
