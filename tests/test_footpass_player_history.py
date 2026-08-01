from __future__ import annotations

from datetime import date

import numpy as np

from footballq.analysis.footpass_player_history import (
    EventAccumulator,
    ExtractedFootpassData,
    FootpassAppearance,
    PlayerMatchStats,
    TrackingAccumulator,
    _history_stats_for_player,
    broad_role,
    compute_appearance_outcomes,
    event_profile_feature_names,
    fit_logistic_probe,
    load_extracted_footpass_data,
    load_logistic_probes,
    save_extracted_footpass_data,
    save_logistic_probes,
)
from footballq.io.footpass import FOOTPASS_PLAYER_IDS


def _appearance(
    match_id: str,
    match_date: date,
    *,
    partition: str = "development_train",
) -> FootpassAppearance:
    return FootpassAppearance(
        team_id="team_a",
        team_name="Team A",
        match_id=match_id,
        match_date=match_date,
        focal_team_index=0,
        partition=partition,
        player_by_shirt={10: "team_a:player_10"},
        player_name_by_id={"team_a:player_10": "Player 10"},
    )


def _synthetic_extracted_data() -> ExtractedFootpassData:
    slot_count = len(FOOTPASS_PLAYER_IDS)
    snapshot_player_id = np.tile(
        np.asarray(FOOTPASS_PLAYER_IDS, dtype=np.int16),
        (4, 1),
    )
    snapshot_team_index = np.tile(
        np.asarray(
            [0 if player_id < 200 else 1 for player_id in FOOTPASS_PLAYER_IDS],
            dtype=np.int8,
        ),
        (4, 1),
    )
    snapshot_shirt_number = np.full((4, slot_count), -1, dtype=np.int16)
    snapshot_shirt_number[:, 0] = 10
    snapshot_role_id = np.ones((4, slot_count), dtype=np.int8)
    snapshot_left_to_right = np.ones((4, slot_count), dtype=np.int8)
    snapshot_geometry = np.zeros((4, slot_count, 4), dtype=np.float32)
    return ExtractedFootpassData(
        metadata={
            "selected_match_ids": ["1"],
            "selected_appearance_ids": ["team_a:1"],
            "half_bounds": {"1:1": {"first_frame": 0, "last_frame": 1000}},
            "tracking_stride_frames": 125,
            "event_count": 4,
            "snapshot_count": 4,
            "confirmation_match_ids_included": [],
        },
        event_match_id=np.asarray(["1"] * 4),
        event_period=np.asarray([1, 1, 1, 1], dtype=np.int8),
        event_frame=np.asarray([100, 200, 400, 450], dtype=np.int64),
        event_team_index=np.asarray([0, 0, 0, 1], dtype=np.int8),
        event_player_id=np.asarray([100, 100, 100, 200], dtype=np.int16),
        event_shirt_number=np.asarray([10, 10, 10, 20], dtype=np.int16),
        event_role_id=np.asarray([10, 10, 10, 2], dtype=np.int8),
        event_left_to_right=np.asarray([1, 1, 1, 0], dtype=np.int8),
        event_action_class=np.asarray([2, 2, 2, 2], dtype=np.int8),
        event_geometry=np.asarray(
            [
                [0.50, 0.50, 0.01, 0.00],
                [0.90, 0.50, 0.01, 0.00],
                [0.50, 0.50, 0.01, 0.00],
                [0.50, 0.50, 0.01, 0.00],
            ],
            dtype=np.float32,
        ),
        event_snapshot_index=np.arange(4, dtype=np.int32),
        snapshot_player_id=snapshot_player_id,
        snapshot_team_index=snapshot_team_index,
        snapshot_shirt_number=snapshot_shirt_number,
        snapshot_role_id=snapshot_role_id,
        snapshot_left_to_right=snapshot_left_to_right,
        snapshot_geometry=snapshot_geometry,
        snapshot_active_count=np.asarray([22, 22, 22, 22], dtype=np.int8),
        tracking_stats={
            "team_a:1": {
                "team_a:player_10": TrackingAccumulator(),
            }
        },
    )


def test_broad_role_groups_cover_footpass_roles() -> None:
    assert broad_role(1) == 0
    assert {broad_role(value) for value in (2, 3, 4, 5, 13)} == {1}
    assert {broad_role(value) for value in (6, 7, 8, 9)} == {2}
    assert {broad_role(value) for value in (10, 11, 12)} == {3}
    assert broad_role(-1) == 4


def test_event_profile_is_finite_and_matches_declared_schema() -> None:
    accumulator = EventAccumulator()
    accumulator.update_event(
        action_class=2,
        x_attack=0.6,
        y=0.4,
        vx_attack=0.02,
        vy=-0.01,
        x_edges=[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0001],
        y_edges=[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0001],
    )
    accumulator.update_outcome(
        action_class=2,
        x_attack=0.6,
        turnover=1,
        penalty_entry=0,
        x_edges=[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0001],
    )
    vector = accumulator.vector()
    assert vector.shape == (len(event_profile_feature_names()),)
    assert np.isfinite(vector).all()


def test_future_labels_stop_at_opponent_possession_and_respect_horizons() -> None:
    outcomes = compute_appearance_outcomes(
        _synthetic_extracted_data(),
        _appearance("1", date(2023, 1, 1)),
        query_classes={1, 2, 3, 4, 6},
        possession_classes={1, 2, 3, 4, 5, 6},
        primary_horizon_frames=250,
        turnover_horizon_frames=125,
        penalty_area={
            "attacking_x_min": 0.84,
            "y_min": 0.20,
            "y_max": 0.80,
        },
        minimum_active_players=20,
    )
    assert set(outcomes) == {0, 2}
    assert outcomes[0].penalty_area_action_10s == 1
    assert outcomes[0].turnover_5s == 0
    assert outcomes[2].penalty_area_action_10s == 0
    assert outcomes[2].turnover_5s == 1


def test_history_uses_strictly_earlier_matches_and_caps_recent_support() -> None:
    appearances = [
        _appearance("1", date(2023, 1, 1)),
        _appearance("2", date(2023, 1, 2)),
        _appearance("3", date(2023, 1, 3)),
        _appearance("4", date(2023, 1, 4)),
    ]
    stats = {
        item.appearance_id: {
            "team_a:player_10": PlayerMatchStats(),
        }
        for item in appearances
    }
    histories = _history_stats_for_player(
        appearances[3],
        "team_a:player_10",
        {"team_a": appearances},
        stats,
        support_cap=2,
    )
    assert histories == [
        stats["team_a:3"]["team_a:player_10"],
        stats["team_a:2"]["team_a:player_10"],
    ]
    reverse = _history_stats_for_player(
        appearances[0],
        "team_a:player_10",
        {"team_a": appearances},
        stats,
        support_cap=2,
        reverse=True,
    )
    assert reverse == [
        stats["team_a:2"]["team_a:player_10"],
        stats["team_a:3"]["team_a:player_10"],
    ]


def test_extracted_cache_round_trip_does_not_require_pickle(tmp_path) -> None:
    path = tmp_path / "cache.npz"
    source = _synthetic_extracted_data()
    save_extracted_footpass_data(path, source)
    loaded = load_extracted_footpass_data(path)
    assert loaded.metadata == source.metadata
    assert np.array_equal(loaded.event_frame, source.event_frame)
    assert np.array_equal(loaded.snapshot_player_id, source.snapshot_player_id)
    assert set(loaded.tracking_stats["team_a:1"]) == {"team_a:player_10"}


def test_logistic_probe_learns_signal_and_round_trips(tmp_path) -> None:
    features = np.asarray(
        [[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]],
        dtype=np.float64,
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int8)
    config = {
        "standardize_epsilon": 1.0e-8,
        "fit_seed": 7,
        "max_iterations": 100,
        "tolerance_grad": 1.0e-9,
        "tolerance_change": 1.0e-11,
        "l2_coefficient": 0.01,
    }
    probe = fit_logistic_probe(features, labels, ["signal"], config)
    probabilities = probe.predict(features)
    assert np.isfinite(probabilities).all()
    assert probabilities[-1] > probabilities[0]

    path = tmp_path / "probes.npz"
    save_logistic_probes(path, {"target::view": probe})
    loaded = load_logistic_probes(path)["target::view"]
    assert loaded.feature_names == ["signal"]
    assert np.allclose(loaded.predict(features), probabilities)
