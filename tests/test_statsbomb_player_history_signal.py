from __future__ import annotations

import numpy as np

from footballq.analysis.statsbomb_player_history_signal import (
    CURRENT_FEATURE_NAMES,
    Cohort,
    MatchRecord,
    PassSummary,
    active_lineup_player_ids,
    aggregate_history,
    cohort_for_match,
    current_features,
    is_open_play_pass,
    pass_labels,
    profile_vectors,
    query_conditioned_profile_vectors,
    summarize_player_passes,
)


def _pass_event(
    *,
    event_id: str = "pass-1",
    minute: int = 1,
    second: int = 0,
    start: tuple[float, float] = (40.0, 40.0),
    end: tuple[float, float] = (60.0, 40.0),
    outcome: str | None = None,
    possession: int = 1,
) -> dict:
    payload = {"end_location": list(end)}
    if outcome is not None:
        payload["outcome"] = {"name": outcome}
    return {
        "id": event_id,
        "index": 1,
        "minute": minute,
        "second": second,
        "period": 1,
        "possession": possession,
        "type": {"name": "Pass"},
        "play_pattern": {"name": "Regular Play"},
        "team": {"id": 10, "name": "Example FC"},
        "player": {"id": 100, "name": "Example Player"},
        "position": {"name": "Center Midfield"},
        "location": list(start),
        "pass": payload,
    }


def test_cohort_selection_uses_competition_season_and_focal_team() -> None:
    record = MatchRecord(
        match_id="1",
        match_date="2024-01-01",
        competition_name="League",
        season_name="2023/2024",
        home_team_name="Example FC",
        away_team_name="Other FC",
        has_360=True,
    )
    cohort = Cohort(
        name="example",
        split="train",
        competition_name="League",
        season_name="2023/2024",
        focal_team_name="Example FC",
    )
    assert cohort_for_match(record, [cohort]) == cohort


def test_open_play_pass_rejects_dead_ball_passes() -> None:
    event = _pass_event()
    assert is_open_play_pass(event)
    event["pass"]["type"] = {"name": "Free Kick"}
    assert not is_open_play_pass(event)


def test_pass_labels_use_future_events_without_exposing_them_as_features() -> None:
    current = _pass_event()
    future = {
        "id": "future",
        "minute": 1,
        "second": 4,
        "period": 1,
        "possession": 2,
        "type": {"name": "Ball Recovery"},
        "team": {"id": 20, "name": "Other FC"},
    }
    labels = pass_labels([current, future], 0)
    assert labels["progressive"] is True
    assert labels["turnover_5s"] is True
    assert labels["complete"] is True


def test_history_aggregation_is_bounded_by_support_size() -> None:
    history = []
    for match_index in range(3):
        summary = PassSummary.empty()
        summary.match_count = 1
        summary.pass_count = match_index + 1
        history.append((f"2024-01-0{match_index + 1}", str(match_index), summary))
    aggregate = aggregate_history(history, support_size=2)
    assert aggregate.match_count == 2
    assert aggregate.pass_count == 5


def test_profile_vectors_are_finite_and_richer_than_rolling_stats() -> None:
    summary = summarize_player_passes([_pass_event()])["100"]
    rolling, rich = profile_vectors(summary)
    conditioned_rolling, conditioned_rich = query_conditioned_profile_vectors(
        summary,
        (40.0, 40.0),
        False,
    )
    assert np.isfinite(rolling).all()
    assert np.isfinite(rich).all()
    assert np.isfinite(conditioned_rich).all()
    assert rich.shape[0] > rolling.shape[0]
    assert conditioned_rich.shape[0] > rich.shape[0]
    assert np.array_equal(conditioned_rolling, rolling)
    assert rolling[0] == 1.0


def test_current_features_have_stable_shape() -> None:
    event = _pass_event()
    freeze_frame = [
        {"actor": True, "teammate": True, "keeper": False, "location": [40.0, 40.0]},
        {"actor": False, "teammate": True, "keeper": False, "location": [50.0, 35.0]},
        {"actor": False, "teammate": False, "keeper": False, "location": [45.0, 40.0]},
    ]
    features = current_features(event, freeze_frame)
    assert features.shape == (len(CURRENT_FEATURE_NAMES),)
    assert np.isfinite(features).all()


def test_active_lineup_uses_only_players_active_at_event_time() -> None:
    event = _pass_event(minute=60)
    event["period"] = 2
    lineup = {
        "lineup": [
            {
                "player_id": 1,
                "positions": [
                    {
                        "from": "00:00",
                        "to": "55:00",
                        "from_period": 1,
                        "to_period": 2,
                    }
                ],
            },
            {
                "player_id": 2,
                "positions": [
                    {
                        "from": "55:00",
                        "to": None,
                        "from_period": 2,
                        "to_period": None,
                    }
                ],
            },
        ]
    }
    assert active_lineup_player_ids(lineup, event) == ["2"]
