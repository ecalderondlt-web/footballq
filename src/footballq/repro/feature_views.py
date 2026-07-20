"""Leakage-aware feature views for representation learning and controls."""

from __future__ import annotations

from dataclasses import dataclass

import torch

FULL_STATE_LEGACY = "full_state_legacy"
GEOMETRY_ONLY = "geometry_only"
POSITION_ONLY = "position_only"
MISSINGNESS_ONLY_CONTROL = "missingness_only_control"
RAW_KINEMATICS_CONTROL = "raw_kinematics_control"


@dataclass(frozen=True)
class FeatureView:
    name: str
    feature_names: list[str]
    validity_note: str


FEATURE_VIEW_NAMES = {
    FULL_STATE_LEGACY,
    GEOMETRY_ONLY,
    POSITION_ONLY,
    MISSINGNESS_ONLY_CONTROL,
    RAW_KINEMATICS_CONTROL,
}


def feature_view_names(source_feature_names: list[str], view: str) -> list[str]:
    """Return ordered feature names for a named view."""

    names = list(source_feature_names)
    if view == FULL_STATE_LEGACY:
        return names
    if view == GEOMETRY_ONLY:
        wanted = ["x_norm", "y_norm", "vx_norm", "vy_norm", "is_ball", "is_home", "is_away"]
    elif view == POSITION_ONLY:
        wanted = ["x_norm", "y_norm", "is_ball", "is_home", "is_away"]
    elif view == MISSINGNESS_ONLY_CONTROL:
        wanted = ["is_ball", "is_home", "is_away", "visible_mask"]
    elif view == RAW_KINEMATICS_CONTROL:
        wanted = ["x_norm", "y_norm", "vx_norm", "vy_norm", "visible_mask"]
    else:
        raise ValueError(f"Unknown feature view {view!r}. Expected {sorted(FEATURE_VIEW_NAMES)}")
    missing = [name for name in wanted if name not in names]
    if missing:
        raise ValueError(f"Feature view {view!r} is missing source features: {', '.join(missing)}")
    return wanted


def feature_view_indices(source_feature_names: list[str], view: str) -> list[int]:
    selected = feature_view_names(source_feature_names, view)
    return [source_feature_names.index(name) for name in selected]


def apply_feature_view(
    state: torch.Tensor,
    source_feature_names: list[str],
    view: str,
) -> tuple[torch.Tensor, list[str]]:
    """Select a leakage-controlled feature view from a state tensor."""

    indices = feature_view_indices(source_feature_names, view)
    return state[..., indices].contiguous(), feature_view_names(source_feature_names, view)


def probe_validity_class(target_name: str, feature_view: str) -> str:
    """Classify probe evidence under a feature view."""

    if target_name in {"possession_team", "has_ball_or_possession_available"}:
        return "leakage_sanity_check" if feature_view == FULL_STATE_LEGACY else "unavailable"
    if target_name in {
        "future_ball_dx_global_m",
        "future_ball_global_x_bucket",
        "future_ball_displacement_m",
    }:
        return "geometry_control"
    if target_name in {
        "team_shape_change_bucket",
        "team_centroid_shift_m",
        "team_width_change_m",
        "team_length_change_m",
        "stretch_index_change_m",
    }:
        return "semantic_candidate"
    return "unavailable"
