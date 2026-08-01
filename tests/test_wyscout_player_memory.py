from __future__ import annotations

import numpy as np
import pandas as pd

from footballq.analysis.wyscout_player_memory import (
    binary_nll,
    build_aggregate_tables,
    match_bootstrap_nll_gain,
    predict_probabilities,
    prepare_prediction_cache,
    same_team_role_shuffle,
    select_last_player_matches,
)


def _support_fixture() -> pd.DataFrame:
    rows = []
    for player_id, successes in [(1, 4), (2, 0), (3, 2)]:
        for match_id in range(1, 7):
            rows.append(
                {
                    "match_id": player_id * 100 + match_id,
                    "dateutc": f"2017-01-{match_id:02d}",
                    "player_id": player_id,
                    "team_id": 10 if player_id < 3 else 20,
                    "role": 2,
                    "start_zone": 10,
                    "subevent_id": 85,
                    "shot_within_horizon": int(match_id <= successes),
                }
            )
    return pd.DataFrame(rows)


def test_last_k_support_is_distinct_match_capped() -> None:
    support = _support_fixture()

    selected = select_last_player_matches(support, 3)

    counts = selected.groupby("player_id")["match_id"].nunique().to_dict()
    assert counts == {1: 3, 2: 3, 3: 3}
    assert selected.loc[selected["player_id"] == 1, "match_id"].min() == 104


def test_player_probability_falls_back_when_query_has_no_history() -> None:
    support = _support_fixture()
    aggregates = build_aggregate_tables(
        support,
        outcome="shot_within_horizon",
        match_cap=5,
    )
    query = pd.DataFrame(
        [
            {
                "match_id": 999,
                "player_id": 999,
                "role": 2,
                "start_zone": 10,
                "subevent_id": 85,
                "shot_within_horizon": 0,
            }
        ]
    )
    cache = prepare_prediction_cache(
        query,
        aggregates,
        outcome="shot_within_horizon",
        minimum_prior_matches=5,
    )

    probabilities = predict_probabilities(
        cache,
        context_prior_strength=10.0,
        team_prior_strength=10.0,
        player_prior_strength=10.0,
    )

    assert probabilities["player"][0] == probabilities["team"][0]


def test_same_team_role_shuffle_is_a_derangement_where_possible() -> None:
    aggregates = build_aggregate_tables(
        _support_fixture(),
        outcome="shot_within_horizon",
        match_cap=5,
    )

    mapping = same_team_role_shuffle(aggregates.player_catalog, seed=7)

    assert mapping[1] == 2
    assert mapping[2] == 1
    assert mapping[3] == 3


def test_match_bootstrap_reports_positive_gain_for_better_predictions() -> None:
    outcome = np.array([0.0, 1.0, 0.0, 1.0])
    baseline = np.array([0.5, 0.5, 0.5, 0.5])
    challenger = np.array([0.1, 0.9, 0.1, 0.9])
    match_ids = np.array([1, 1, 2, 2])

    result = match_bootstrap_nll_gain(
        outcome,
        baseline,
        challenger,
        match_ids,
        replicates=100,
        seed=11,
        confidence_level=0.95,
    )

    assert binary_nll(outcome, challenger) < binary_nll(outcome, baseline)
    assert result["point_gain"] > 0
    assert result["ci_lower"] > 0
