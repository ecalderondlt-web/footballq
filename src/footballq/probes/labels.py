"""Label extraction and derived tactical targets for frozen probes."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from footballq.data.normalize import denormalize_velocity_to_mps, denormalize_xy_to_meters
from footballq.data.windows import (
    BALL_INDEX,
    ENTITY_PLAYER,
    TEAM_AWAY,
    TEAM_HOME,
    TrackingWindowTensorData,
)

POSSESSION_TEAM_LABELS = {"home": 0, "away": 1, "neutral": 2, "unknown": 3}
POSSESSION_AVAILABLE_LABELS = {"unavailable": 0, "available": 1}
PROGRESSION_LABELS = {"backward": 0, "neutral": 1, "forward": 2}
SHAPE_CHANGE_LABELS = {"compressing": 0, "stable": 1, "expanding": 2}

CLASSIFICATION_TARGETS = {
    "possession_team",
    "has_ball_or_possession_available",
    "phase",
    "future_ball_global_x_bucket",
    "future_ball_progression_bucket",
    "future_ball_progression_attacking_bucket",
    "team_shape_change_bucket",
}
REGRESSION_TARGETS = {
    "future_ball_dx_global_m",
    "future_ball_dx_m",
    "future_ball_dy_m",
    "future_ball_displacement_m",
    "future_ball_progression_attacking_m",
    "team_centroid_shift_m",
    "team_width_change_m",
    "team_length_change_m",
    "stretch_index_change_m",
}
SUPPORTED_TARGETS = CLASSIFICATION_TARGETS | REGRESSION_TARGETS


@dataclass
class DerivedTargets:
    """Container returned by target derivation."""

    targets: dict[str, torch.Tensor]
    masks: dict[str, torch.Tensor]
    target_types: dict[str, str]
    label_maps: dict[str, dict[str, int]]
    warnings: list[str]


def _feature_index(windows: TrackingWindowTensorData, name: str) -> int | None:
    try:
        return windows.feature_names.index(name)
    except ValueError:
        return None


def _state_xy_m(windows: TrackingWindowTensorData) -> torch.Tensor:
    xy_idx = [_feature_index(windows, "x_norm"), _feature_index(windows, "y_norm")]
    if any(idx is None for idx in xy_idx):
        raise ValueError("Window features must include x_norm and y_norm for probe labels.")
    return denormalize_xy_to_meters(windows.past[:, -1, :, [int(xy_idx[0]), int(xy_idx[1])]])


def _future_xy_m(windows: TrackingWindowTensorData) -> torch.Tensor:
    return denormalize_xy_to_meters(windows.future_xy[:, -1])


def _entity_visible_now_and_future(windows: TrackingWindowTensorData) -> torch.Tensor:
    return windows.past_mask[:, -1] & windows.future_mask[:, -1]


def _masked_centroid(xy: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    count = mask.sum(dim=1).clamp_min(1).float()
    centroid = (xy * mask.unsqueeze(-1).float()).sum(dim=1) / count.unsqueeze(-1)
    return centroid, mask.any(dim=1)


def _masked_spread(
    xy: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    large = torch.full_like(xy, 1e9)
    small = torch.full_like(xy, -1e9)
    min_xy = torch.where(mask.unsqueeze(-1), xy, large).amin(dim=1)
    max_xy = torch.where(mask.unsqueeze(-1), xy, small).amax(dim=1)
    valid = mask.sum(dim=1) >= 2
    width = torch.where(valid, max_xy[:, 1] - min_xy[:, 1], torch.zeros_like(max_xy[:, 1]))
    length = torch.where(valid, max_xy[:, 0] - min_xy[:, 0], torch.zeros_like(max_xy[:, 0]))
    return width, length, valid


def _masked_stretch(xy: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    centroid, centroid_valid = _masked_centroid(xy, mask)
    distances = torch.linalg.norm(xy - centroid.unsqueeze(1), dim=-1)
    count = mask.sum(dim=1).clamp_min(1).float()
    stretch = (distances * mask.float()).sum(dim=1) / count
    valid = centroid_valid & (mask.sum(dim=1) >= 2)
    return torch.where(valid, stretch, torch.zeros_like(stretch)), valid


def _all_player_mask(windows: TrackingWindowTensorData) -> torch.Tensor:
    return windows.entity_type == ENTITY_PLAYER


def _all_player_shape_change(
    current_xy_m: torch.Tensor,
    future_xy_m: torch.Tensor,
    visible: torch.Tensor,
    windows: TrackingWindowTensorData,
) -> dict[str, torch.Tensor]:
    players = _all_player_mask(windows) & visible
    current_centroid, current_centroid_valid = _masked_centroid(current_xy_m, players)
    future_centroid, future_centroid_valid = _masked_centroid(future_xy_m, players)
    current_width, current_length, current_span_valid = _masked_spread(current_xy_m, players)
    future_width, future_length, future_span_valid = _masked_spread(future_xy_m, players)
    current_stretch, current_stretch_valid = _masked_stretch(current_xy_m, players)
    future_stretch, future_stretch_valid = _masked_stretch(future_xy_m, players)
    valid = (
        current_centroid_valid
        & future_centroid_valid
        & current_span_valid
        & future_span_valid
        & current_stretch_valid
        & future_stretch_valid
    )
    return {
        "valid": valid,
        "centroid_shift_m": torch.linalg.norm(future_centroid - current_centroid, dim=-1),
        "width_change_m": future_width - current_width,
        "length_change_m": future_length - current_length,
        "stretch_index_change_m": future_stretch - current_stretch,
    }


def _bucket(
    values: torch.Tensor, threshold: float, low: int, neutral: int, high: int
) -> torch.Tensor:
    out = torch.full(values.shape, neutral, dtype=torch.long)
    out = torch.where(values < -threshold, torch.full_like(out, low), out)
    out = torch.where(values > threshold, torch.full_like(out, high), out)
    return out


def _window_metadata_values(windows: TrackingWindowTensorData, name: str) -> list[str]:
    values = getattr(windows, name, None)
    if values is None:
        return ["unknown"] * len(windows.match_id)
    out: list[str] = []
    for value in values:
        text = str(value).strip().lower()
        if not text or text in {"none", "nan", "<na>", "null"}:
            text = "unknown"
        out.append(text)
    return out


def _encode_dynamic_labels(values: list[str]) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    valid_values = sorted({value for value in values if value != "unknown"})
    if not valid_values:
        return (
            torch.zeros(len(values), dtype=torch.long),
            torch.zeros(len(values), dtype=torch.bool),
            {},
        )
    label_map = {value: idx for idx, value in enumerate(valid_values)}
    encoded = torch.zeros(len(values), dtype=torch.long)
    mask = torch.zeros(len(values), dtype=torch.bool)
    for idx, value in enumerate(values):
        if value in label_map:
            encoded[idx] = label_map[value]
            mask[idx] = True
    return encoded, mask, label_map


def derive_probe_targets(
    windows: TrackingWindowTensorData,
    target_names: list[str],
    progression_neutral_m: float = 1.0,
    shape_neutral_m: float = 1.0,
) -> DerivedTargets:
    """Derive supported probe targets from aligned tracking windows.

    The current implementation only uses fields preserved in the fixed window
    tensor payload. ``phase`` is intentionally not invented here because phase
    strings are not stored in that artifact.
    """

    requested = list(dict.fromkeys(target_names))
    targets: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}
    target_types: dict[str, str] = {}
    label_maps: dict[str, dict[str, int]] = {}
    warnings: list[str] = []

    unsupported = [name for name in requested if name not in SUPPORTED_TARGETS]
    for name in unsupported:
        warnings.append(f"target {name!r} is unavailable from window tensors and was skipped")
    requested = [name for name in requested if name in SUPPORTED_TARGETS]
    if not requested:
        return DerivedTargets(targets, masks, target_types, label_maps, warnings)

    current_xy_m = _state_xy_m(windows)
    future_xy_m = _future_xy_m(windows)
    visible = _entity_visible_now_and_future(windows)
    ball_visible = visible[:, BALL_INDEX]
    ball_delta = future_xy_m[:, BALL_INDEX, :] - current_xy_m[:, BALL_INDEX, :]
    ball_dx = ball_delta[:, 0]
    ball_dy = ball_delta[:, 1]
    ball_displacement = torch.linalg.norm(ball_delta, dim=-1)
    shape = _all_player_shape_change(current_xy_m, future_xy_m, visible, windows)

    if "phase" in requested:
        phase_values = _window_metadata_values(windows, "phase")
        phase_target, phase_mask, phase_map = _encode_dynamic_labels(phase_values)
        if phase_mask.any():
            targets["phase"] = phase_target
            masks["phase"] = phase_mask
            target_types["phase"] = "classification"
            label_maps["phase"] = phase_map
        else:
            warnings.append("phase is unavailable in the aligned window metadata and was skipped")

    if "possession_team" in requested or "has_ball_or_possession_available" in requested:
        possession_idx = _feature_index(windows, "is_possession_team")
        has_possession_idx = _feature_index(windows, "has_possession")
        n = len(windows.match_id)
        metadata_possession = _window_metadata_values(windows, "possession_team_id")
        metadata_available = torch.tensor(
            [
                bool(value) or team in {"home", "away"}
                for value, team in zip(
                    getattr(windows, "possession_available", [False] * n),
                    metadata_possession,
                    strict=True,
                )
            ],
            dtype=torch.bool,
        )
        if any(team in {"home", "away", "neutral"} for team in metadata_possession):
            available = metadata_available
            possession = torch.full((n,), POSSESSION_TEAM_LABELS["unknown"], dtype=torch.long)
            for idx, team in enumerate(metadata_possession):
                if team in POSSESSION_TEAM_LABELS:
                    possession[idx] = POSSESSION_TEAM_LABELS[team]
        elif possession_idx is None or has_possession_idx is None:
            available = torch.zeros(n, dtype=torch.bool)
            possession = torch.full((n,), POSSESSION_TEAM_LABELS["unknown"], dtype=torch.long)
            warnings.append("possession features are missing; possession labels are unknown")
        else:
            current = windows.past[:, -1]
            is_possession_team = current[:, :, possession_idx] > 0.5
            has_possession = current[:, :, has_possession_idx] > 0.5
            available = has_possession.any(dim=1) | is_possession_team.any(dim=1)
            home = (is_possession_team & (windows.team_id == TEAM_HOME)).any(dim=1)
            away = (is_possession_team & (windows.team_id == TEAM_AWAY)).any(dim=1)
            possession = torch.full((n,), POSSESSION_TEAM_LABELS["unknown"], dtype=torch.long)
            possession = torch.where(
                home & ~away,
                torch.full_like(possession, POSSESSION_TEAM_LABELS["home"]),
                possession,
            )
            possession = torch.where(
                away & ~home,
                torch.full_like(possession, POSSESSION_TEAM_LABELS["away"]),
                possession,
            )
            possession = torch.where(
                available & ~(home | away),
                torch.full_like(possession, POSSESSION_TEAM_LABELS["neutral"]),
                possession,
            )
            if not available.any():
                warnings.append(
                    "possession fields are present but empty; possession_team is all unknown"
                )
        if "possession_team" in requested:
            targets["possession_team"] = possession
            masks["possession_team"] = torch.ones_like(possession, dtype=torch.bool)
            target_types["possession_team"] = "classification"
            label_maps["possession_team"] = dict(POSSESSION_TEAM_LABELS)
        if "has_ball_or_possession_available" in requested:
            targets["has_ball_or_possession_available"] = available.long()
            masks["has_ball_or_possession_available"] = torch.ones_like(available, dtype=torch.bool)
            target_types["has_ball_or_possession_available"] = "classification"
            label_maps["has_ball_or_possession_available"] = dict(POSSESSION_AVAILABLE_LABELS)

    if "future_ball_global_x_bucket" in requested or "future_ball_progression_bucket" in requested:
        bucket = _bucket(
            ball_dx,
            threshold=progression_neutral_m,
            low=PROGRESSION_LABELS["backward"],
            neutral=PROGRESSION_LABELS["neutral"],
            high=PROGRESSION_LABELS["forward"],
        )
        if "future_ball_global_x_bucket" in requested:
            targets["future_ball_global_x_bucket"] = bucket
            masks["future_ball_global_x_bucket"] = ball_visible
            target_types["future_ball_global_x_bucket"] = "classification"
            label_maps["future_ball_global_x_bucket"] = dict(PROGRESSION_LABELS)
        if "future_ball_progression_bucket" in requested:
            targets["future_ball_progression_bucket"] = bucket
            masks["future_ball_progression_bucket"] = ball_visible
            target_types["future_ball_progression_bucket"] = "classification"
            label_maps["future_ball_progression_bucket"] = dict(PROGRESSION_LABELS)
            warnings.append(
                "future_ball_progression_bucket is a deprecated alias for "
                "future_ball_global_x_bucket because attacking direction is unknown"
            )

    if "future_ball_progression_attacking_bucket" in requested:
        warnings.append(
            "future_ball_progression_attacking_bucket is unavailable because reliable "
            "causal attacking-direction metadata is not stored in window tensors"
        )

    if "team_shape_change_bucket" in requested:
        targets["team_shape_change_bucket"] = _bucket(
            shape["stretch_index_change_m"],
            threshold=shape_neutral_m,
            low=SHAPE_CHANGE_LABELS["compressing"],
            neutral=SHAPE_CHANGE_LABELS["stable"],
            high=SHAPE_CHANGE_LABELS["expanding"],
        )
        masks["team_shape_change_bucket"] = shape["valid"]
        target_types["team_shape_change_bucket"] = "classification"
        label_maps["team_shape_change_bucket"] = dict(SHAPE_CHANGE_LABELS)
        warnings.append(
            "team_shape_change_bucket uses all visible players rather than one team "
            "so it remains comparable when possession changes during the horizon"
        )

    regression_values = {
        "future_ball_dx_global_m": (ball_dx, ball_visible),
        "future_ball_dx_m": (ball_dx, ball_visible),
        "future_ball_dy_m": (ball_dy, ball_visible),
        "future_ball_displacement_m": (ball_displacement, ball_visible),
        "team_centroid_shift_m": (shape["centroid_shift_m"], shape["valid"]),
        "team_width_change_m": (shape["width_change_m"], shape["valid"]),
        "team_length_change_m": (shape["length_change_m"], shape["valid"]),
        "stretch_index_change_m": (shape["stretch_index_change_m"], shape["valid"]),
    }
    for name, (values, valid) in regression_values.items():
        if name not in requested:
            continue
        targets[name] = values.float()
        masks[name] = valid.bool()
        target_types[name] = "regression"
        if name == "future_ball_dx_m":
            warnings.append("future_ball_dx_m is a deprecated alias for future_ball_dx_global_m")

    if "future_ball_progression_attacking_m" in requested:
        warnings.append(
            "future_ball_progression_attacking_m is unavailable because reliable causal "
            "attacking-direction metadata is not stored in window tensors"
        )

    return DerivedTargets(targets, masks, target_types, label_maps, warnings)


def raw_state_summary_features(
    windows: TrackingWindowTensorData,
) -> tuple[torch.Tensor, list[str]]:
    """Build non-future summary features from the current state only."""

    current_xy_m = _state_xy_m(windows)
    vx_idx = _feature_index(windows, "vx_norm")
    vy_idx = _feature_index(windows, "vy_norm")
    if vx_idx is None or vy_idx is None:
        velocity_mps = torch.zeros_like(current_xy_m)
    else:
        velocity_mps = denormalize_velocity_to_mps(
            windows.past[:, -1, :, [int(vx_idx), int(vy_idx)]]
        )
    current_visible = windows.past_mask[:, -1]
    ball_xy = torch.where(
        current_visible[:, BALL_INDEX].unsqueeze(-1),
        current_xy_m[:, BALL_INDEX],
        torch.zeros_like(current_xy_m[:, BALL_INDEX]),
    )
    ball_velocity = torch.where(
        current_visible[:, BALL_INDEX].unsqueeze(-1),
        velocity_mps[:, BALL_INDEX],
        torch.zeros_like(velocity_mps[:, BALL_INDEX]),
    )

    parts = [ball_xy, ball_velocity]
    names = ["ball_x_m", "ball_y_m", "ball_vx_mps", "ball_vy_mps"]
    team_summaries: dict[str, torch.Tensor] = {}
    for team_name, team_code in [("home", TEAM_HOME), ("away", TEAM_AWAY)]:
        mask = (windows.team_id == team_code) & current_visible
        centroid, centroid_valid = _masked_centroid(current_xy_m, mask)
        width, length, span_valid = _masked_spread(current_xy_m, mask)
        stretch, stretch_valid = _masked_stretch(current_xy_m, mask)
        valid = centroid_valid & span_valid & stretch_valid
        centroid = torch.where(valid.unsqueeze(-1), centroid, torch.zeros_like(centroid))
        width = torch.where(valid, width, torch.zeros_like(width))
        length = torch.where(valid, length, torch.zeros_like(length))
        stretch = torch.where(valid, stretch, torch.zeros_like(stretch))
        team_summaries[f"{team_name}_centroid"] = centroid
        team_summaries[f"{team_name}_valid"] = valid.float().unsqueeze(-1)
        parts.extend([centroid, width.unsqueeze(-1), length.unsqueeze(-1), stretch.unsqueeze(-1)])
        names.extend(
            [
                f"{team_name}_centroid_x_m",
                f"{team_name}_centroid_y_m",
                f"{team_name}_width_m",
                f"{team_name}_length_m",
                f"{team_name}_stretch_index_m",
            ]
        )

    home_centroid = team_summaries["home_centroid"]
    away_centroid = team_summaries["away_centroid"]
    home_valid = team_summaries["home_valid"].bool().squeeze(-1)
    away_valid = team_summaries["away_valid"].bool().squeeze(-1)
    both_valid = home_valid & away_valid
    centroid_distance = torch.linalg.norm(home_centroid - away_centroid, dim=-1)
    centroid_distance = torch.where(
        both_valid,
        centroid_distance,
        torch.zeros_like(centroid_distance),
    )
    ball_home_distance = torch.linalg.norm(ball_xy - home_centroid, dim=-1)
    ball_away_distance = torch.linalg.norm(ball_xy - away_centroid, dim=-1)
    ball_home_distance = torch.where(
        home_valid, ball_home_distance, torch.zeros_like(ball_home_distance)
    )
    ball_away_distance = torch.where(
        away_valid, ball_away_distance, torch.zeros_like(ball_away_distance)
    )
    parts.extend(
        [
            centroid_distance.unsqueeze(-1),
            ball_home_distance.unsqueeze(-1),
            ball_away_distance.unsqueeze(-1),
        ]
    )
    names.extend(
        [
            "team_centroid_distance_m",
            "ball_distance_to_home_centroid_m",
            "ball_distance_to_away_centroid_m",
        ]
    )
    features = torch.cat(parts, dim=1).float()
    return torch.nan_to_num(features), names
