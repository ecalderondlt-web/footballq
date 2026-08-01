from __future__ import annotations

import numpy as np
import pandas as pd

from footballq.analysis.wyscout_cross_team_action import (
    build_destination_aggregates,
    build_penalty_aggregates,
    destination_metrics,
    predict_destination_probabilities,
    predict_penalty_probabilities,
    prepare_destination_cache,
    prepare_penalty_cache,
)


def _passes(
    player_id: int,
    team_id: int,
    role: int,
    destination_zone: int,
) -> list[dict[str, float | int]]:
    rows = []
    for match_id in range(1, 7):
        for index in range(4):
            rows.append(
                {
                    "match_id": match_id,
                    "dateutc": f"2018-01-{match_id:02d}",
                    "player_id": player_id,
                    "team_id": team_id,
                    "role": role,
                    "start_zone": 10,
                    "start_x": 55.0,
                    "start_y": 50.0,
                    "destination_x": 90.0 if destination_zone >= 25 else 70.0,
                    "destination_y": 50.0,
                    "destination_zone": destination_zone,
                }
            )
    return rows


def test_conditional_profile_recovers_player_destination_tendency() -> None:
    support = pd.DataFrame(
        [
            *_passes(1, 10, 2, 26),
            *_passes(2, 10, 2, 5),
        ]
    )
    query = pd.DataFrame(
        [
            *_passes(1, 100, 2, 26),
            *_passes(2, 100, 2, 5),
        ]
    )
    aggregates = build_destination_aggregates(support, match_cap=5)
    cache = prepare_destination_cache(
        query,
        aggregates,
        minimum_prior_matches=5,
    )

    probabilities = predict_destination_probabilities(
        cache,
        context_prior_strength=10.0,
        team_prior_strength=10.0,
        player_prior_strength=1.0,
        residual_ratio_limit=4.0,
    )

    rolling = destination_metrics(
        cache.destination_zone,
        probabilities["rolling_player"],
    )
    conditional = destination_metrics(
        cache.destination_zone,
        probabilities["conditional_player"],
    )
    assert conditional["nll"] <= rolling["nll"]
    assert conditional["top1"] == 1.0


def test_destination_metrics_reports_multiclass_brier_and_topk() -> None:
    labels = np.asarray([0, 2])
    probability = np.full((2, 30), 0.01)
    probability[0, 0] = 0.71
    probability[1, 2] = 0.71
    probability /= probability.sum(axis=1, keepdims=True)

    metrics = destination_metrics(labels, probability)

    assert metrics["examples"] == 2
    assert metrics["top1"] == 1.0
    assert metrics["top3"] == 1.0
    assert float(metrics["brier"]) >= 0.0


def _conditional_penalty_rows(
    player_id: int,
    *,
    enter_from_zone_10: bool,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for match_id in range(1, 7):
        for start_zone, start_y in ((10, 40.0), (11, 60.0)):
            enters = enter_from_zone_10 == (start_zone == 10)
            rows.append(
                {
                    "match_id": match_id,
                    "dateutc": f"2018-01-{match_id:02d}",
                    "player_id": player_id,
                    "team_id": 10,
                    "role": 2,
                    "start_zone": start_zone,
                    "start_x": 60.0,
                    "start_y": start_y,
                    "destination_x": 90.0 if enters else 75.0,
                    "destination_y": 50.0,
                    "destination_zone": 26 if enters else 20,
                }
            )
    return rows


def test_exact_penalty_profile_adds_context_beyond_player_total() -> None:
    support = pd.DataFrame(
        [
            *_conditional_penalty_rows(1, enter_from_zone_10=True),
            *_conditional_penalty_rows(2, enter_from_zone_10=False),
        ]
    )
    query = support.copy()
    query["team_id"] = 100
    aggregates = build_penalty_aggregates(
        support,
        match_cap=5,
        query_start_x_min=50.0,
    )
    cache = prepare_penalty_cache(
        query,
        aggregates,
        minimum_prior_matches=5,
        query_start_x_min=50.0,
    )

    probabilities = predict_penalty_probabilities(
        cache,
        context_prior_strength=10.0,
        team_prior_strength=10.0,
        player_prior_strength=1.0,
        residual_ratio_limit=4.0,
    )

    rolling_error = np.mean(
        (probabilities["rolling_player"] - cache.outcome) ** 2
    )
    conditional_error = np.mean(
        (probabilities["conditional_player"] - cache.outcome) ** 2
    )
    assert conditional_error < rolling_error


def test_penalty_target_uses_exact_coordinates_and_shuffle_keeps_totals() -> None:
    support = pd.DataFrame(
        [
            *_conditional_penalty_rows(1, enter_from_zone_10=True),
            *_conditional_penalty_rows(2, enter_from_zone_10=False),
        ]
    )
    query = support.iloc[[0, 1]].copy()
    query.loc[query.index[0], "destination_zone"] = 26
    query.loc[query.index[0], "destination_x"] = 90.0
    query.loc[query.index[0], "destination_y"] = 10.0
    aggregates = build_penalty_aggregates(
        support,
        match_cap=5,
        query_start_x_min=50.0,
    )
    genuine = prepare_penalty_cache(
        query,
        aggregates,
        minimum_prior_matches=5,
        query_start_x_min=50.0,
    )
    shuffled = prepare_penalty_cache(
        query,
        aggregates,
        minimum_prior_matches=5,
        query_start_x_min=50.0,
        profile_mapping={1: 2, 2: 1},
    )

    assert genuine.outcome[0] == 0.0
    np.testing.assert_array_equal(
        genuine.player_total_successes,
        shuffled.player_total_successes,
    )
    assert not np.array_equal(
        genuine.player_context_successes,
        shuffled.player_context_successes,
    )
