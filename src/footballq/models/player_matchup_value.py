"""Profile-conditioned critical-value model for RLCS V2."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from footballq.data.rlcs_player_profiles import PROFILE_DIMENSION
from footballq.data.rlcs_value_windows import (
    N_ENTITIES,
    N_FEATURES,
    N_PLAYERS,
    PAIR_GEOMETRY_DIMENSION,
    SCALAR_CONTEXT_DIMENSION,
    TEAM_FORM_DIMENSION,
    TIME_STEPS,
)

ValueCondition = Literal[
    "state",
    "team_form",
    "actor_profile",
    "additive_profiles",
    "full_matchup",
]
VALUE_CONDITIONS: tuple[ValueCondition, ...] = (
    "state",
    "team_form",
    "actor_profile",
    "additive_profiles",
    "full_matchup",
)


class PlayerMatchupValueModel(nn.Module):
    """Small matched-capacity model with explicit actor-opponent interactions."""

    def __init__(
        self,
        *,
        input_features: int = N_FEATURES,
        time_steps: int = TIME_STEPS,
        entities: int = N_ENTITIES,
        width: int = 192,
        layers: int = 3,
        attention_heads: int = 6,
        feed_forward_width: int = 768,
        dropout: float = 0.10,
        profile_dimension: int = PROFILE_DIMENSION,
        profile_projection: int = 64,
        pair_geometry_dimension: int = PAIR_GEOMETRY_DIMENSION,
        pair_output_dimension: int = 64,
        team_form_dimension: int = TEAM_FORM_DIMENSION,
        scalar_context_dimension: int = SCALAR_CONTEXT_DIMENSION,
        outcome_classes: int = 3,
    ) -> None:
        super().__init__()
        if width % attention_heads:
            raise ValueError("Transformer width must be divisible by the attention-head count.")
        self.input_features = int(input_features)
        self.time_steps = int(time_steps)
        self.entities = int(entities)
        self.profile_dimension = int(profile_dimension)
        self.pair_geometry_dimension = int(pair_geometry_dimension)
        self.profile_projection_width = int(profile_projection)
        self.pair_output_dimension = int(pair_output_dimension)
        self.input_projection = nn.Linear(input_features, width)
        self.time_embedding = nn.Parameter(torch.zeros(time_steps, width))
        self.entity_embedding = nn.Parameter(torch.zeros(entities, width))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=attention_heads,
            dim_feedforward=feed_forward_width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.geometry_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=layers, norm=nn.LayerNorm(width)
        )
        self.scalar_projection = nn.Sequential(
            nn.Linear(scalar_context_dimension, 64), nn.GELU(), nn.LayerNorm(64)
        )
        self.team_form_projection = nn.Sequential(
            nn.Linear(team_form_dimension, 64), nn.GELU(), nn.LayerNorm(64)
        )
        profile_input = 2 * profile_dimension + 1
        self.profile_projection = nn.Sequential(
            nn.Linear(profile_input, profile_projection),
            nn.GELU(),
            nn.LayerNorm(profile_projection),
        )
        self.additive_projection = nn.Sequential(
            nn.Linear(3 * profile_projection, profile_projection),
            nn.GELU(),
            nn.LayerNorm(profile_projection),
        )
        pair_input = 3 * profile_dimension + pair_geometry_dimension
        self.opponent_pair_mlp = nn.Sequential(
            nn.Linear(pair_input, 128),
            nn.GELU(),
            nn.Linear(128, pair_output_dimension),
            nn.GELU(),
            nn.LayerNorm(pair_output_dimension),
        )
        self.opponent_attention = nn.Linear(pair_output_dimension, 1)
        self.teammate_pair_mlp = nn.Sequential(
            nn.Linear(pair_input, 128),
            nn.GELU(),
            nn.Linear(128, pair_output_dimension),
            nn.GELU(),
            nn.LayerNorm(pair_output_dimension),
        )
        head_input = (
            width
            + 64
            + 64
            + profile_projection
            + profile_projection
            + 2 * pair_output_dimension
            + pair_output_dimension
        )
        self.outcome_head = nn.Sequential(
            nn.Linear(head_input, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, outcome_classes),
        )
        nn.init.normal_(self.time_embedding, std=0.02)
        nn.init.normal_(self.entity_embedding, std=0.02)

    def _validate_shapes(
        self,
        state: torch.Tensor,
        state_mask: torch.Tensor,
        scalar_context: torch.Tensor,
        team_form: torch.Tensor,
        profiles: torch.Tensor,
        profile_uncertainty: torch.Tensor,
        profile_effective_sample_size: torch.Tensor,
        pair_geometry: torch.Tensor,
        teammate_geometry: torch.Tensor,
    ) -> None:
        batch = state.shape[0]
        expected = (batch, self.time_steps, self.entities, self.input_features)
        if tuple(state.shape) != expected:
            raise ValueError(f"state must have shape {expected}, got {tuple(state.shape)}")
        if tuple(state_mask.shape) != expected[:-1]:
            raise ValueError("state_mask has the wrong shape.")
        if tuple(scalar_context.shape) != (batch, SCALAR_CONTEXT_DIMENSION):
            raise ValueError("scalar_context has the wrong shape.")
        if tuple(team_form.shape) != (batch, TEAM_FORM_DIMENSION):
            raise ValueError("team_form has the wrong shape.")
        profile_shape = (batch, N_PLAYERS, self.profile_dimension)
        wrong_profile = tuple(profiles.shape) != profile_shape
        wrong_uncertainty = tuple(profile_uncertainty.shape) != profile_shape
        if wrong_profile or wrong_uncertainty:
            raise ValueError("profiles and uncertainty must have shape [batch, 6, profile_dim].")
        if tuple(profile_effective_sample_size.shape) != (batch, N_PLAYERS):
            raise ValueError("profile effective sample size must have shape [batch, 6].")
        if tuple(pair_geometry.shape) != (batch, 3, self.pair_geometry_dimension):
            raise ValueError("opponent pair geometry must have shape [batch, 3, pair_dim].")
        if tuple(teammate_geometry.shape) != (batch, 2, self.pair_geometry_dimension):
            raise ValueError("teammate geometry must have shape [batch, 2, pair_dim].")

    @staticmethod
    def _zeros(reference: torch.Tensor, width: int) -> torch.Tensor:
        return reference.new_zeros((reference.shape[0], int(width)))

    def forward(
        self,
        *,
        state: torch.Tensor,
        state_mask: torch.Tensor,
        scalar_context: torch.Tensor,
        team_form: torch.Tensor,
        profiles: torch.Tensor,
        profile_uncertainty: torch.Tensor,
        profile_effective_sample_size: torch.Tensor,
        pair_geometry: torch.Tensor,
        teammate_geometry: torch.Tensor,
        condition: ValueCondition,
    ) -> dict[str, torch.Tensor]:
        if condition not in VALUE_CONDITIONS:
            raise ValueError(f"Unknown V2 condition {condition!r}.")
        self._validate_shapes(
            state,
            state_mask,
            scalar_context,
            team_form,
            profiles,
            profile_uncertainty,
            profile_effective_sample_size,
            pair_geometry,
            teammate_geometry,
        )
        batch = state.shape[0]
        tokens = self.input_projection(state)
        tokens = tokens + self.time_embedding[None, :, None, :]
        tokens = tokens + self.entity_embedding[None, None, :, :]
        tokens = tokens.reshape(batch, self.time_steps * self.entities, -1)
        flat_mask = state_mask.reshape(batch, self.time_steps * self.entities).bool()
        encoded = self.geometry_encoder(tokens, src_key_padding_mask=~flat_mask)
        geometry = (encoded * flat_mask.unsqueeze(-1)).sum(dim=1) / flat_mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        scalar = self.scalar_projection(scalar_context)

        use_form = condition != "state"
        form_context = (
            self.team_form_projection(team_form)
            if use_form
            else self._zeros(geometry, 64)
        )
        effective = torch.log1p(profile_effective_sample_size).unsqueeze(-1) / 5.0
        profile_input = torch.cat([profiles, profile_uncertainty, effective], dim=-1)
        projected_profiles = self.profile_projection(profile_input)

        use_actor = condition in {"actor_profile", "additive_profiles", "full_matchup"}
        actor_context = (
            projected_profiles[:, 0]
            if use_actor
            else self._zeros(geometry, self.profile_projection_width)
        )
        if condition in {"additive_profiles", "full_matchup"}:
            additive_context = self.additive_projection(
                torch.cat(
                    [
                        projected_profiles[:, 0],
                        projected_profiles[:, 1:3].mean(dim=1),
                        projected_profiles[:, 3:6].mean(dim=1),
                    ],
                    dim=-1,
                )
            )
        else:
            additive_context = self._zeros(geometry, self.profile_projection_width)

        if condition == "full_matchup":
            actor = profiles[:, 0:1].expand(-1, 3, -1)
            opponents = profiles[:, 3:6]
            pair_input = torch.cat(
                [actor, opponents, actor - opponents, pair_geometry], dim=-1
            )
            opponent_tokens = self.opponent_pair_mlp(pair_input)
            weights = torch.softmax(self.opponent_attention(opponent_tokens), dim=1)
            opponent_sum = (weights * opponent_tokens).sum(dim=1)
            opponent_max = opponent_tokens.max(dim=1).values
            matchup_context = torch.cat([opponent_sum, opponent_max], dim=-1)

            actor_teammate = profiles[:, 0:1].expand(-1, 2, -1)
            teammates = profiles[:, 1:3]
            teammate_input = torch.cat(
                [
                    actor_teammate,
                    teammates,
                    actor_teammate - teammates,
                    teammate_geometry,
                ],
                dim=-1,
            )
            teammate_context = self.teammate_pair_mlp(teammate_input).mean(dim=1)
        else:
            matchup_context = self._zeros(geometry, 2 * self.pair_output_dimension)
            teammate_context = self._zeros(geometry, self.pair_output_dimension)

        logits = self.outcome_head(
            torch.cat(
                [
                    geometry,
                    scalar,
                    form_context,
                    actor_context,
                    additive_context,
                    matchup_context,
                    teammate_context,
                ],
                dim=-1,
            )
        )
        probabilities = torch.softmax(logits, dim=-1)
        return {
            "outcome_logits": logits,
            "outcome_probabilities": probabilities,
            "state_value": probabilities[:, 1] - probabilities[:, 2],
        }


def critical_value_loss(outputs: dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(outputs["outcome_logits"], labels.long())


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
