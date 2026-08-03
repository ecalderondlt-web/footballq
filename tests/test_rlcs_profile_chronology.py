from __future__ import annotations

import numpy as np
import pandas as pd

from footballq.data.rlcs_player_profiles import (
    PROFILE_DIMENSION,
    build_profile_snapshots,
    build_v2_split_frame,
    fit_profile_priors,
)


def _profile(value: float) -> list[float]:
    return np.full(PROFILE_DIMENSION, value, dtype=np.float32).tolist()


def test_profile_snapshot_uses_strictly_earlier_games_only():
    games = pd.DataFrame(
        [
            {
                "replay_id": "support-a",
                "series_id": "s1",
                "region": "EU",
                "event_time_utc": "2025-01-01T00:00:00Z",
                "v2_stage": "profile_support",
                "player_id": "p1",
                "profile": _profile(1.0),
                "team_win": 1.0,
                "team_goal_diff": 2,
            },
            {
                "replay_id": "same-time",
                "series_id": "s2",
                "region": "EU",
                "event_time_utc": "2025-01-01T00:00:00Z",
                "v2_stage": "profile_support",
                "player_id": "p1",
                "profile": _profile(99.0),
                "team_win": 0.0,
                "team_goal_diff": -2,
            },
            {
                "replay_id": "query",
                "series_id": "s3",
                "region": "EU",
                "event_time_utc": "2025-01-02T00:00:00Z",
                "v2_stage": "train",
                "player_id": "p1",
                "profile": _profile(500.0),
                "team_win": 0.0,
                "team_goal_diff": -5,
            },
        ]
    )
    priors = fit_profile_priors(games)
    snapshots = build_profile_snapshots(games, priors)
    first = snapshots.loc[snapshots["replay_id"] == "support-a"].iloc[0]
    same_time = snapshots.loc[snapshots["replay_id"] == "same-time"].iloc[0]
    query = snapshots.loc[snapshots["replay_id"] == "query"].iloc[0]
    assert first["n_prior_games"] == 0
    assert same_time["n_prior_games"] == 0
    assert query["n_prior_games"] == 2
    assert pd.Timestamp(query["latest_prior_time_utc"]) < pd.Timestamp(
        "2025-01-02T00:00:00Z"
    )
    assert max(query["profile"]) < 500.0


def test_v2_split_allocates_complete_series_by_region():
    rows = []
    for region in ("EU", "NA"):
        for index in range(99):
            rows.append(
                {
                    "replay_id": f"{region}-{index}",
                    "series_id": f"{region}-series-{index:03d}",
                    "region": region,
                    "event_time_utc": pd.Timestamp("2025-01-01", tz="UTC")
                    + pd.Timedelta(days=index),
                    "split_number": 1,
                    "regional_number": 1 + index // 33,
                }
            )
    split = build_v2_split_frame(pd.DataFrame(rows))
    for region in ("EU", "NA"):
        counts = split.loc[split["region"] == region, "v2_stage"].value_counts()
        assert counts.to_dict() == {
            "train": 44,
            "profile_support": 35,
            "internal_development": 20,
        }
    assert split.groupby("series_id")["v2_stage"].nunique().max() == 1
