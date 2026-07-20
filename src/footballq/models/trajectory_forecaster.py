"""Matched multi-horizon trajectory forecasters for downstream representation tests."""

from __future__ import annotations

import torch
from torch import nn

from footballq.models.soccer_state_encoder import SoccerStateEncoder

FORECAST_FAMILIES = ("raw", "frozen", "finetuned")
FORECAST_REPRESENTATION_MODES = ("global", "entity_tokens")
FORECAST_DECODER_MODES = ("shared", "player_ball", "player_global_ball")


def last_observed_kinematics(
    state: torch.Tensor,
    mask: torch.Tensor,
    *,
    fps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return last position, last-two-point velocity, and last-observation mask."""

    xy = state[..., :2]
    batch, steps, entities, _ = xy.shape
    indices = torch.arange(steps, device=state.device).view(1, steps, 1)
    valid_indices = torch.where(mask, indices, torch.full_like(indices, -1))
    last_index = valid_indices.max(dim=1).values
    last_safe = last_index.clamp_min(0)
    gather_xy = last_safe.view(batch, 1, entities, 1).expand(-1, 1, -1, 2)
    last_xy = xy.gather(1, gather_xy).squeeze(1)
    has_last = last_index >= 0
    prior_valid = mask & (indices < last_safe.unsqueeze(1))
    prior_indices = torch.where(prior_valid, indices, torch.full_like(indices, -1))
    prior_index = prior_indices.max(dim=1).values
    prior_safe = prior_index.clamp_min(0)
    gather_prior = prior_safe.view(batch, 1, entities, 1).expand(-1, 1, -1, 2)
    prior_xy = xy.gather(1, gather_prior).squeeze(1)
    elapsed = ((last_safe - prior_safe).float() / float(fps)).clamp_min(1.0 / float(fps))
    has_velocity = (prior_index >= 0) & has_last
    velocity = (last_xy - prior_xy) / elapsed.unsqueeze(-1)
    velocity = torch.where(has_velocity.unsqueeze(-1), velocity, torch.zeros_like(velocity))
    last_xy = torch.where(has_last.unsqueeze(-1), last_xy, torch.zeros_like(last_xy))
    return last_xy, velocity, has_last


def predict_last_position(
    state: torch.Tensor,
    mask: torch.Tensor,
    horizons_seconds: tuple[float, ...],
    *,
    fps: float,
) -> torch.Tensor:
    last_xy, _velocity, _has_last = last_observed_kinematics(state, mask, fps=fps)
    return last_xy.unsqueeze(1).expand(-1, len(horizons_seconds), -1, -1).contiguous()


def predict_constant_velocity(
    state: torch.Tensor,
    mask: torch.Tensor,
    horizons_seconds: tuple[float, ...],
    *,
    fps: float,
) -> torch.Tensor:
    last_xy, velocity, _has_last = last_observed_kinematics(state, mask, fps=fps)
    horizon = torch.tensor(horizons_seconds, dtype=state.dtype, device=state.device)
    return last_xy.unsqueeze(1) + horizon.view(1, -1, 1, 1) * velocity.unsqueeze(1)


class MultiHorizonTrajectoryForecaster(nn.Module):
    """Predict endpoint residuals over a shared kinematic baseline and state encoder."""

    def __init__(
        self,
        encoder: SoccerStateEncoder,
        *,
        family: str,
        z_dim: int,
        n_entities: int,
        horizons_seconds: tuple[float, ...],
        fps: float,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        representation_mode: str = "global",
        token_dim: int | None = None,
        decoder_mode: str = "shared",
        ball_index: int = 0,
    ) -> None:
        super().__init__()
        if family not in FORECAST_FAMILIES:
            raise ValueError(f"Unknown forecast family {family!r}.")
        if representation_mode not in FORECAST_REPRESENTATION_MODES:
            raise ValueError(f"Unknown forecast representation mode {representation_mode!r}.")
        if decoder_mode not in FORECAST_DECODER_MODES:
            raise ValueError(f"Unknown forecast decoder mode {decoder_mode!r}.")
        if decoder_mode != "shared" and representation_mode != "entity_tokens":
            raise ValueError("Type-conditioned decoding requires entity_tokens representations.")
        if not 0 <= int(ball_index) < int(n_entities):
            raise ValueError("Forecast ball_index must identify one canonical entity slot.")
        self.encoder = encoder
        self.family = family
        self.representation_mode = representation_mode
        self.decoder_mode = decoder_mode
        self.ball_index = int(ball_index)
        self.n_entities = int(n_entities)
        self.horizons_seconds = tuple(float(value) for value in horizons_seconds)
        self.fps = float(fps)
        if representation_mode == "global":
            decoder_input_dim = int(z_dim) + self.n_entities * 5
            ball_decoder_input_dim = decoder_input_dim
            decoder_output_dim = len(self.horizons_seconds) * self.n_entities * 2
        else:
            if token_dim is None:
                raise ValueError("entity_tokens forecasting requires token_dim.")
            decoder_input_dim = int(token_dim) + 5
            ball_decoder_input_dim = (
                int(token_dim) + self.n_entities * 5
                if decoder_mode == "player_global_ball"
                else decoder_input_dim
            )
            decoder_output_dim = len(self.horizons_seconds) * 2

        def decoder_block(input_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, decoder_output_dim),
            )

        if decoder_mode == "shared":
            self.decoder = decoder_block(decoder_input_dim)
        else:
            self.decoder = nn.ModuleDict(
                {
                    "player": decoder_block(decoder_input_dim),
                    "ball": decoder_block(ball_decoder_input_dim),
                }
            )
        if family == "frozen":
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def forward(self, state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.family == "frozen":
            self.encoder.eval()
            with torch.no_grad():
                representation = self._encode(state, mask)
        else:
            representation = self._encode(state, mask)
        last_xy, velocity, has_last = last_observed_kinematics(state, mask, fps=self.fps)
        kinematics = torch.cat(
            [last_xy, velocity, has_last.unsqueeze(-1).to(state.dtype)], dim=-1
        )
        if self.representation_mode == "global":
            global_input = torch.cat(
                [representation, kinematics.reshape(state.shape[0], -1)], dim=-1
            )
            correction = self.decoder(global_input).view(
                state.shape[0], len(self.horizons_seconds), self.n_entities, 2
            )
        else:
            entity_input = torch.cat([representation, kinematics], dim=-1)
            if self.decoder_mode == "shared":
                correction_flat = self.decoder(entity_input)
            else:
                player_input = torch.cat(
                    [
                        entity_input[:, : self.ball_index],
                        entity_input[:, self.ball_index + 1 :],
                    ],
                    dim=1,
                )
                player_correction = self.decoder["player"](player_input)
                if self.decoder_mode == "player_global_ball":
                    ball_input = torch.cat(
                        [
                            representation[:, self.ball_index],
                            kinematics.reshape(state.shape[0], -1),
                        ],
                        dim=-1,
                    ).unsqueeze(1)
                else:
                    ball_input = entity_input[
                        :, self.ball_index : self.ball_index + 1
                    ]
                ball_correction = self.decoder["ball"](ball_input)
                correction_flat = torch.cat(
                    [
                        player_correction[:, : self.ball_index],
                        ball_correction,
                        player_correction[:, self.ball_index :],
                    ],
                    dim=1,
                )
            correction = correction_flat.view(
                state.shape[0], self.n_entities, len(self.horizons_seconds), 2
            )
            correction = correction.permute(0, 2, 1, 3).contiguous()
        baseline = predict_constant_velocity(
            state,
            mask,
            self.horizons_seconds,
            fps=self.fps,
        )
        return baseline + correction

    def _encode(self, state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.representation_mode == "entity_tokens":
            return self.encoder.encode_entity_tokens(state, mask)
        return self.encoder(state, mask)
