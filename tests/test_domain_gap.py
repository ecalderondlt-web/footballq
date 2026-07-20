import numpy as np
import pytest
import torch

from footballq.analysis.domain_gap import (
    METRIC_UNITS,
    compare_metric_samples,
    deterministic_indices,
    extract_geometry_metrics,
    select_train_shards_by_match,
)
from footballq.data.normalize import XY_SCALE_M

FEATURE_NAMES = [
    "x_norm",
    "y_norm",
    "vx_norm",
    "vy_norm",
    "is_ball",
    "is_home",
    "is_away",
]


def test_geometry_metrics_restore_physical_units_and_masks():
    state = torch.zeros((1, 2, 23, len(FEATURE_NAMES)), dtype=torch.float32)
    mask = torch.zeros((1, 2, 23), dtype=torch.bool)
    mask[:, :, :3] = True
    x_scale, y_scale = (float(value) for value in XY_SCALE_M)
    state[0, :, 1, 0] = torch.tensor([0.0, 5.0 / x_scale])
    state[0, :, 2, 0] = torch.tensor([10.0 / x_scale, 10.0 / x_scale])
    state[0, 0, 1, 2] = 1.0 / x_scale
    state[0, 1, 1, 2] = 2.0 / x_scale
    state[0, :, 2, 3] = 1.0 / y_scale

    metrics = extract_geometry_metrics(state, mask, fps=10.0, feature_names=FEATURE_NAMES)

    assert sorted(metrics["player_speed_mps"].tolist()) == pytest.approx([1.0, 2.0])
    assert metrics["player_acceleration_mps2"].tolist() == pytest.approx([10.0, 0.0])
    assert metrics["visible_player_count"].tolist() == [2.0]
    assert metrics["nearest_player_distance_m"].tolist() == pytest.approx([5.0, 5.0])


def test_gap_comparison_is_zero_for_identical_samples_and_ranks_shift():
    base = np.linspace(0.0, 1.0, 100)
    real = {name: base.copy() for name in METRIC_UNITS}
    identical = {name: values.copy() for name, values in real.items()}
    shifted = {name: values.copy() for name, values in real.items()}
    shifted["player_speed_mps"] = shifted["player_speed_mps"] + 2.0

    assert all(row["gap_score"] == 0.0 for row in compare_metric_samples(real, identical))
    assert compare_metric_samples(real, shifted)[0]["metric"] == "player_speed_mps"


def test_robust_gap_scale_is_not_reduced_by_extreme_outlier_variance():
    base = np.linspace(0.0, 1.0, 1000)
    shifted = base + 1.0
    shifted_with_outlier = np.concatenate([shifted, [10000.0]])
    real = {name: base.copy() for name in METRIC_UNITS}
    candidate = {name: base.copy() for name in METRIC_UNITS}
    outlier_candidate = {name: base.copy() for name in METRIC_UNITS}
    candidate["player_acceleration_mps2"] = shifted
    outlier_candidate["player_acceleration_mps2"] = shifted_with_outlier

    score = next(
        row["gap_score"]
        for row in compare_metric_samples(real, candidate)
        if row["metric"] == "player_acceleration_mps2"
    )
    outlier_score = next(
        row["gap_score"]
        for row in compare_metric_samples(real, outlier_candidate)
        if row["metric"] == "player_acceleration_mps2"
    )

    assert outlier_score >= score * 0.95


def test_train_shard_selection_ignores_validation_and_spans_each_match():
    shards = [
        {"path": f"a{idx}", "match_id": "a", "split": "train"} for idx in range(5)
    ] + [
        {"path": "b0", "match_id": "b", "split": "train"},
        {"path": "heldout", "match_id": "c", "split": "val"},
    ]

    selected = select_train_shards_by_match(shards, max_shards_per_match=3)

    assert [row["path"] for row in selected] == ["a0", "a2", "a4", "b0"]
    assert all(row["split"] == "train" for row in selected)


def test_deterministic_indices_are_seeded_unique_and_bounded():
    first = deterministic_indices(100, 20, key="shard", seed=7)
    second = deterministic_indices(100, 20, key="shard", seed=7)

    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 20
    assert first.min() >= 0 and first.max() < 100
