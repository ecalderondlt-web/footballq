import torch

from footballq.data.windows import FEATURE_NAMES
from footballq.repro.feature_views import apply_feature_view, feature_view_names


def test_geometry_only_excludes_possession_channels():
    names = feature_view_names(list(FEATURE_NAMES), "geometry_only")
    assert "is_possession_team" not in names
    assert "has_possession" not in names
    assert "visible_mask" not in names


def test_position_only_excludes_velocity_and_possession_channels():
    names = feature_view_names(list(FEATURE_NAMES), "position_only")

    assert names == ["x_norm", "y_norm", "is_ball", "is_home", "is_away"]


def test_missingness_only_control_runs():
    state = torch.randn(2, 3, 23, len(FEATURE_NAMES))
    selected, names = apply_feature_view(state, list(FEATURE_NAMES), "missingness_only_control")
    assert selected.shape[-1] == len(names)
    assert "visible_mask" in names
