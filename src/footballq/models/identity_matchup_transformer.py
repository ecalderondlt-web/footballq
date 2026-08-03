"""Identity-conditioned transformer for RLCS next-touch decisions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as functional
from torch import nn

IdentityCondition = Literal["anonymous", "actor_only", "roster_only", "full"]
IDENTITY_CONDITIONS = ("anonymous", "actor_only", "roster_only", "full")


@dataclass(frozen=True)
class MatchupLossWeights:
    """Frozen weights for the four preregistered objectives."""

    next_touch_entity: float = 1.0
    next_touch_zone: float = 1.0
    retained_possession: float = 0.25
    goal_within_8s: float = 0.10
    focal_gamma: float = 2.0


def apply_identity_condition(
    identity_indices: torch.Tensor, condition: IdentityCondition
) -> torch.Tensor:
    """Apply one of the four conditions without changing roster geometry."""

    if identity_indices.ndim != 2 or identity_indices.shape[1] != 6:
        raise ValueError("identity_indices must have shape [batch, 6].")
    if condition not in IDENTITY_CONDITIONS:
        raise ValueError(f"Unknown identity condition {condition!r}.")
    conditioned = identity_indices.clone()
    if condition == "anonymous":
        conditioned.zero_()
    elif condition == "actor_only":
        conditioned[:, 1:] = 0
    elif condition == "roster_only":
        conditioned[:, 0] = 0
    return conditioned


def permute_within_roster_identities(
    identity_indices: torch.Tensor, *, generator: torch.Generator
) -> torch.Tensor:
    """Shuffle identities within actor-team and opponent-team slots."""

    if identity_indices.ndim != 2 or identity_indices.shape[1] != 6:
        raise ValueError("identity_indices must have shape [batch, 6].")
    output = identity_indices.clone()
    for row in range(output.shape[0]):
        team_permutation = torch.randperm(3, generator=generator, device="cpu").to(
            output.device
        )
        opponent_permutation = torch.randperm(3, generator=generator, device="cpu").to(
            output.device
        )
        source = identity_indices[row]
        output[row, :3] = source[:3][team_permutation]
        output[row, 3:] = source[3:][opponent_permutation]
    return output


class IdentityMatchupTransformer(nn.Module):
    """Matched geometry backbone with explicitly ablated identity inputs."""

    def __init__(
        self,
        *,
        num_player_identities: int,
        input_features: int = 27,
        time_steps: int = 20,
        entities: int = 7,
        width: int = 192,
        layers: int = 3,
        attention_heads: int = 6,
        feed_forward_width: int = 768,
        dropout: float = 0.10,
        identity_embedding_dim: int = 48,
    ) -> None:
        super().__init__()
        if num_player_identities < 1:
            raise ValueError("Identity vocabulary must include at least the UNK row.")
        if time_steps != 20 or entities != 7 or input_features != 27:
            raise ValueError("Version 1 requires a [20, 7, 27] state tensor.")
        self.time_steps = int(time_steps)
        self.entities = int(entities)
        self.width = int(width)
        self.input_projection = nn.Linear(input_features, width)
        self.temporal_embedding = nn.Parameter(torch.empty(time_steps, width))
        self.entity_role_embedding = nn.Parameter(torch.empty(entities, width))
        self.player_embedding = nn.Embedding(num_player_identities, identity_embedding_dim)
        self.identity_projection = nn.Linear(identity_embedding_dim, width, bias=False)
        self.matchup_projection = nn.Sequential(
            nn.Linear(identity_embedding_dim * 4, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.context_mlp = nn.Sequential(
            nn.Linear(3, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=attention_heads,
            dim_feedforward=feed_forward_width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=layers,
            norm=nn.LayerNorm(width),
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(width)
        self.next_touch_entity_head = nn.Linear(width, 6)
        self.next_touch_zone_head = nn.Linear(width, 18)
        self.retained_possession_head = nn.Linear(width, 1)
        self.goal_within_8s_head = nn.Linear(width, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.temporal_embedding, std=0.02)
        nn.init.normal_(self.entity_role_embedding, std=0.02)
        nn.init.normal_(self.player_embedding.weight, std=0.02)

    def _identity_tokens(
        self, identity_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.player_embedding(identity_indices)
        actor = embedding[:, 0]
        teammates = embedding[:, 1:3].mean(dim=1)
        opponents = embedding[:, 3:6].mean(dim=1)
        matchup = torch.cat([actor, teammates, opponents, actor * opponents], dim=-1)
        return self.identity_projection(embedding), self.matchup_projection(matchup)

    @staticmethod
    def _context_vector(
        seconds_remaining: torch.Tensor,
        score_diff_actor: torch.Tensor,
        overtime: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            [
                seconds_remaining.float().clamp(0, 300) / 120.0,
                score_diff_actor.float().clamp(-5, 5) / 5.0,
                overtime.float(),
            ],
            dim=-1,
        )

    def forward(
        self,
        state: torch.Tensor,
        state_mask: torch.Tensor,
        identity_indices: torch.Tensor,
        seconds_remaining: torch.Tensor,
        score_diff_actor: torch.Tensor,
        overtime: torch.Tensor,
        *,
        condition: IdentityCondition = "full",
    ) -> dict[str, torch.Tensor]:
        """Return four logits heads; state and ordering are identical across ablations."""

        expected = (self.time_steps, self.entities, self.input_projection.in_features)
        if tuple(state.shape[1:]) != expected:
            raise ValueError(f"state must have shape [batch, {expected}], got {state.shape}.")
        if tuple(state_mask.shape[1:]) != (self.time_steps, self.entities):
            raise ValueError("state_mask must have shape [batch, 20, 7].")
        conditioned_ids = apply_identity_condition(identity_indices.long(), condition)
        x = self.input_projection(torch.nan_to_num(state.float(), nan=0.0))
        x = (
            x
            + self.temporal_embedding[None, :, None, :]
            + self.entity_role_embedding[None, None, :, :]
        )
        identity_tokens, matchup_token = self._identity_tokens(conditioned_ids)
        car_identity = identity_tokens[:, None, :, :].expand(-1, self.time_steps, -1, -1)
        x[:, :, 1:, :] = x[:, :, 1:, :] + car_identity
        context = self.context_mlp(
            self._context_vector(seconds_remaining, score_diff_actor, overtime)
        )
        matchup_token = matchup_token + context
        sequence = x.reshape(x.shape[0], self.time_steps * self.entities, self.width)
        sequence = torch.cat([sequence, matchup_token[:, None, :]], dim=1)
        padding_mask = ~state_mask.bool().reshape(state.shape[0], -1)
        padding_mask = torch.cat(
            [padding_mask, torch.zeros((state.shape[0], 1), dtype=torch.bool, device=state.device)],
            dim=1,
        )
        encoded = self.encoder(sequence, src_key_padding_mask=padding_mask)
        pooled = self.output_norm(encoded[:, -1])
        return {
            "next_touch_entity_logits": self.next_touch_entity_head(pooled),
            "next_touch_zone_logits": self.next_touch_zone_head(pooled),
            "retained_possession_logit": self.retained_possession_head(pooled).squeeze(-1),
            "goal_within_8s_logit": self.goal_within_8s_head(pooled).squeeze(-1),
        }


def focal_binary_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, *, gamma: float = 2.0
) -> torch.Tensor:
    """Mean focal BCE using logits."""

    targets = targets.float()
    base = functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = torch.sigmoid(logits)
    probability_true = probability * targets + (1.0 - probability) * (1.0 - targets)
    return ((1.0 - probability_true).pow(float(gamma)) * base).mean()


def identity_matchup_loss(
    outputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    *,
    weights: MatchupLossWeights = MatchupLossWeights(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the frozen multi-head training objective."""

    entity = functional.cross_entropy(
        outputs["next_touch_entity_logits"], targets["next_touch_entity"].long()
    )
    zone = functional.cross_entropy(
        outputs["next_touch_zone_logits"], targets["next_touch_zone"].long()
    )
    retained = functional.binary_cross_entropy_with_logits(
        outputs["retained_possession_logit"], targets["retained_possession"].float()
    )
    goal = focal_binary_cross_entropy(
        outputs["goal_within_8s_logit"],
        targets["goal_for_within_8s"],
        gamma=weights.focal_gamma,
    )
    total = (
        weights.next_touch_entity * entity
        + weights.next_touch_zone * zone
        + weights.retained_possession * retained
        + weights.goal_within_8s * goal
    )
    return total, {
        "loss": total.detach(),
        "entity_ce": entity.detach(),
        "zone_ce": zone.detach(),
        "retained_bce": retained.detach(),
        "goal_focal_bce": goal.detach(),
    }


def factorized_joint_nll(
    outputs: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    """Per-example preregistered entity-plus-zone negative log likelihood."""

    entity = functional.cross_entropy(
        outputs["next_touch_entity_logits"],
        targets["next_touch_entity"].long(),
        reduction="none",
    )
    zone = functional.cross_entropy(
        outputs["next_touch_zone_logits"],
        targets["next_touch_zone"].long(),
        reduction="none",
    )
    return entity + zone
