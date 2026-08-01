from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import torch

from footballq.analysis.player_profile_proof import (
    TARGET_PENALTY_ENTRY,
    TARGET_TURNOVER,
    MatchInfo,
    _standardize,
    anchor_player_ids,
    build_history_features,
    build_possession_opportunities,
    timestamp_seconds,
)


def _event(
    *,
    index: int,
    possession: int,
    seconds: float,
    team: int,
    x: float,
    y: float = 40.0,
) -> dict:
    return {
        "index": index,
        "period": 1,
        "possession": possession,
        "timestamp": f"00:00:{seconds:06.3f}",
        "type": {"name": "Pass"},
        "possession_team": {"id": team},
        "location": [x, y],
    }


def test_timestamp_seconds_is_period_local() -> None:
    assert timestamp_seconds("00:01:02.500") == 62.5


def test_possession_labels_use_future_without_using_it_to_select_rows() -> None:
    events = [
        _event(index=1, possession=1, seconds=1.0, team=10, x=60.0),
        _event(index=2, possession=1, seconds=3.0, team=10, x=104.0),
        _event(index=3, possession=2, seconds=5.0, team=20, x=40.0),
        _event(index=4, possession=3, seconds=15.0, team=10, x=60.0),
    ]
    opportunities = build_possession_opportunities(events)
    assert len(opportunities) == 3
    assert opportunities[0].turnover_within_5s == 1
    assert opportunities[0].penalty_area_entry_within_5s == 1
    assert opportunities[0].penalty_area_entry_valid
    assert opportunities[1].turnover_within_5s == 0


def test_history_support_is_strictly_earlier_and_capped_by_k() -> None:
    query_date = datetime(2022, 11, 30, 20, 0)
    matches = {
        "past_a": MatchInfo(
            "past_a", "1", datetime(2022, 11, 20, 20, 0), "Group Stage", 1,
            "support", "A", "B"
        ),
        "past_b": MatchInfo(
            "past_b", "2", datetime(2022, 11, 25, 20, 0), "Group Stage", 2,
            "train", "A", "C"
        ),
        "query": MatchInfo(
            "query", "3", query_date, "Group Stage", 3, "val", "A", "D"
        ),
        "future": MatchInfo(
            "future", "4", datetime(2022, 12, 1, 20, 0), "Group Stage", 3,
            "val", "A", "E"
        ),
    }
    profile = {
        "past_a": {"p1": {"mean": torch.ones(2), "variance": 0.1, "clips": 3}},
        "past_b": {"p1": {"mean": torch.full((2,), 3.0), "variance": 0.2, "clips": 4}},
        "query": {"p1": {"mean": torch.full((2,), 100.0), "variance": 0.3, "clips": 5}},
        "future": {"p1": {"mean": torch.full((2,), 200.0), "variance": 0.4, "clips": 6}},
    }
    stats = {
        match_id: {"p1": np.array([value], dtype=np.float32)}
        for match_id, value in (
            ("past_a", 1.0),
            ("past_b", 3.0),
            ("query", 100.0),
            ("future", 200.0),
        )
    }
    player_ids = [[None, "p1", *([None] * 21)]]
    event, latent, audit = build_history_features(
        player_ids,
        [query_date],
        k=1,
        match_profiles=profile,
        match_event_stats=stats,
        matches_by_id=matches,
        embedding_dim=2,
    )
    event = event.view(1, 23, -1)
    latent = latent.view(1, 23, -1)
    assert event[0, 1, 0] == 3.0
    assert torch.equal(latent[0, 1, :2], torch.full((2,), 3.0))
    assert audit["support_count_distribution"] == {"1": 1}


def test_penalty_start_inside_is_masked() -> None:
    events = [
        _event(index=1, possession=1, seconds=1.0, team=10, x=104.0),
        _event(index=2, possession=2, seconds=8.0, team=20, x=50.0),
    ]
    opportunities = build_possession_opportunities(events)
    assert not opportunities[0].penalty_area_entry_valid
    assert opportunities[0].penalty_area_entry_within_5s == 0
    assert TARGET_TURNOVER == "turnover_within_5s"
    assert TARGET_PENALTY_ENTRY == "penalty_area_entry_within_5s"


def test_train_constant_history_column_does_not_explode_later_support() -> None:
    features = torch.tensor([[1.0, 2.0], [1.0, 4.0], [3.0, 6.0]])
    standardized = _standardize(features, [0, 1])
    assert standardized[2, 0] == 2.0
    assert torch.isfinite(standardized).all()


def test_anchor_player_ids_joins_dynamic_slot_to_stable_player(tmp_path) -> None:
    shard = tmp_path / "tracking.parquet"
    pd.DataFrame(
        {
            "frame_id": [100, 100, 100],
            "agent_id": ["ball", "home_slot_00", "away_slot_00"],
            "team_id": ["neutral", "home", "away"],
            "jersey_number": [None, 9, 10],
        }
    ).to_parquet(shard, index=False)
    manifest = {
        "shards": [
            {
                "path": shard.name,
                "start_frame": 100,
                "end_frame": 100,
            }
        ]
    }
    result = anchor_player_ids(
        tmp_path / "manifest.json",
        manifest,
        [100],
        {("home", 9): "home-player", ("away", 10): "away-player"},
    )
    assert result[100][1] == "home-player"
    assert result[100][12] == "away-player"
