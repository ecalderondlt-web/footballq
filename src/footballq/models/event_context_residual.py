"""Frozen tracking prediction with a matched event-context residual head."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from footballq.models.statsbomb_event_encoder import StatsBombEventEncoder
from footballq.models.td_jepa import SoccerTDJEPA

EVENT_CONTEXT_FAMILIES = {"tracking", "raw", "random", "pretrained"}


class FrozenTrackingEventResidual(nn.Module):
    """Train a small correction while tracking and event encoders remain frozen."""

    def __init__(
        self,
        tracking_model: SoccerTDJEPA,
        *,
        family: str,
        z_dim: int,
        event_encoder: StatsBombEventEncoder | None = None,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if family not in EVENT_CONTEXT_FAMILIES:
            raise ValueError(f"Unknown event-context family {family!r}.")
        if family in {"random", "pretrained"} and event_encoder is None:
            raise ValueError(f"Event encoder is required for family {family!r}.")
        self.tracking_model = tracking_model
        self.event_encoder = event_encoder
        self.family = family
        self.z_dim = int(z_dim)
        self.correction_head = nn.Sequential(
            nn.Linear(self.z_dim, int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), self.z_dim),
        )
        for parameter in self.tracking_model.parameters():
            parameter.requires_grad = False
        if self.event_encoder is not None:
            for parameter in self.event_encoder.parameters():
                parameter.requires_grad = False
        self.tracking_model.eval()
        if self.event_encoder is not None:
            self.event_encoder.eval()

    def train(self, mode: bool = True) -> FrozenTrackingEventResidual:
        super().train(mode)
        self.tracking_model.eval()
        if self.event_encoder is not None:
            self.event_encoder.eval()
        self.correction_head.train(mode)
        return self

    def _event_context(
        self,
        batch: dict[str, torch.Tensor],
        reference: torch.Tensor,
        *,
        ablate_event: bool,
    ) -> torch.Tensor:
        if self.family == "tracking" or ablate_event:
            return torch.zeros_like(reference)
        has_history = batch["event_mask"].any(dim=1).unsqueeze(-1).to(reference.dtype)
        if self.family == "raw":
            context = batch["raw_event_context"].to(reference.dtype)
            if int(context.shape[-1]) != self.z_dim:
                raise ValueError("Raw event context dimension does not match tracking z_dim.")
            return context * has_history
        if self.event_encoder is None:
            raise RuntimeError("Event encoder is unavailable for encoded context.")
        batch_size, sequence_length = batch["event_mask"].shape
        event_batch = {
            "categorical": batch["event_categorical"],
            "continuous": batch["event_continuous"],
            "event_mask": batch["event_mask"],
            "freeze_frame": reference.new_zeros(
                (batch_size, sequence_length, 22, 6)
            ),
            "freeze_mask": torch.zeros(
                (batch_size, sequence_length, 22),
                dtype=torch.bool,
                device=reference.device,
            ),
            "has_360": torch.zeros(
                (batch_size, sequence_length),
                dtype=torch.bool,
                device=reference.device,
            ),
        }
        context = self.event_encoder(event_batch)["pooled"]
        if int(context.shape[-1]) != self.z_dim:
            raise ValueError("Encoded event context dimension does not match tracking z_dim.")
        return context * has_history

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        ablate_event: bool = False,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            z_t = self.tracking_model.online_encoder(batch["state_t"], batch["mask_t"])
            delta_z = self.tracking_model.motion_encoder(
                batch["delta_state"],
                batch["delta_mask"],
                z_t,
            )
            base_prediction = z_t + delta_z
            z_target = self.tracking_model.target_encoder(
                batch["state_t_plus_delta"],
                batch["mask_t_plus_delta"],
            )
            context = self._event_context(
                batch,
                z_t,
                ablate_event=ablate_event,
            )
        correction = self.correction_head(context)
        return {
            "z_pred": base_prediction + correction,
            "z_base": base_prediction,
            "z_target": z_target,
            "event_context": context,
            "event_correction": correction,
        }


def event_context_residual_loss(
    outputs: dict[str, torch.Tensor],
    event_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute frozen-target latent prediction metrics with event-coverage slices."""

    target = F.normalize(outputs["z_target"].detach(), dim=-1)
    prediction = F.normalize(outputs["z_pred"], dim=-1)
    base = F.normalize(outputs["z_base"].detach(), dim=-1)
    per_example = (prediction - target).pow(2).mean(dim=-1)
    base_per_example = (base - target).pow(2).mean(dim=-1)
    has_history = event_mask.any(dim=1)
    no_history = ~has_history
    zero = per_example.sum() * 0.0
    event_loss = per_example[has_history].mean() if bool(has_history.any()) else zero
    no_event_loss = per_example[no_history].mean() if bool(no_history.any()) else zero
    return {
        "td_loss": per_example.mean(),
        "base_td_loss": base_per_example.mean(),
        "event_history_td_loss": event_loss,
        "no_event_history_td_loss": no_event_loss,
        "cosine_similarity": F.cosine_similarity(prediction, target, dim=-1).mean(),
        "correction_norm": outputs["event_correction"].norm(dim=-1).mean(),
        "event_history_examples": has_history.sum().to(per_example.dtype),
        "no_event_history_examples": no_history.sum().to(per_example.dtype),
    }
