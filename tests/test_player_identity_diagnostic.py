from __future__ import annotations

from datetime import datetime, timedelta

import torch

from footballq.analysis.player_identity_diagnostic import (
    aggregate_match_profiles,
    raw_clip_features,
    retrieval_metrics,
    retrieval_rows,
)
from footballq.analysis.player_profile_proof import MatchInfo


def _match(match_id: str, day: int, split: str) -> MatchInfo:
    return MatchInfo(
        pff_match_id=match_id,
        statsbomb_match_id=match_id,
        match_datetime=datetime(2022, 11, 20) + timedelta(days=day),
        stage="Group Stage",
        match_week=day + 1,
        split=split,
        home_team_name="A",
        away_team_name="B",
    )


def test_raw_clip_features_do_not_use_identity_channels() -> None:
    state = torch.zeros(1, 3, 23, 5)
    mask = torch.zeros(1, 3, 23, dtype=torch.bool)
    state[0, :, 1, 0] = torch.tensor([0.0, 0.1, 0.2])
    state[0, :, 1, 1] = 0.5
    mask[0, :, 1] = True
    features = raw_clip_features(state, mask)
    assert features.shape == (1, 23, 11)
    assert torch.isclose(features[0, 1, 0], torch.tensor(0.1))
    assert torch.isclose(features[0, 1, -1], torch.tensor(0.2))


def test_aggregate_profiles_keeps_stable_player_assignment() -> None:
    tokens = torch.zeros(2, 23, 2)
    raw = torch.zeros(2, 23, 11)
    masks = torch.zeros(2, 3, 23, dtype=torch.bool)
    tokens[:, 1] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    raw[:, 1, 0] = torch.tensor([2.0, 4.0])
    masks[:, :, 1] = True
    ids = [[None, "p1", *([None] * 21)] for _ in range(2)]
    profiles = aggregate_match_profiles(
        tokens,
        raw,
        masks,
        ids,
        roles={"p1": "forward"},
        teams={"p1": "A"},
    )
    assert torch.equal(profiles["p1"]["latent"], torch.tensor([2.0, 3.0]))
    assert profiles["p1"]["raw"][0] == 3.0
    assert profiles["p1"]["clips"] == 2


def test_retrieval_support_is_strictly_earlier_and_same_role_team() -> None:
    matches = {
        "m1": _match("m1", 0, "support"),
        "m2": _match("m2", 1, "train"),
        "m3": _match("m3", 2, "val"),
    }

    def profile(value: list[float], player: str) -> dict:
        return {
            "latent": torch.tensor(value),
            "raw": torch.tensor(value),
            "role": "forward",
            "team": "A",
            "clips": 2,
            "player": player,
        }

    profiles = {
        "m1": {"p1": profile([1.0, 0.0], "p1"), "p2": profile([0.0, 1.0], "p2")},
        "m2": {"p1": profile([1.0, 0.0], "p1"), "p2": profile([0.0, 1.0], "p2")},
        "m3": {"p1": profile([1.0, 0.0], "p1"), "p2": profile([0.0, 1.0], "p2")},
    }
    rows = retrieval_rows(
        profiles,
        matches,
        key="latent",
        query_splits={"val"},
        support_size=2,
        normalizer=None,
    )
    assert len(rows) == 2
    assert all(row["support_count"] == 2 for row in rows)
    assert all(row["rank"] == 1 for row in rows)
    metrics = retrieval_metrics(rows)
    assert metrics["top1_accuracy"] == 1.0
    assert metrics["chance_top1"] == 0.5


def test_retrieval_never_uses_test_as_query_when_not_requested() -> None:
    matches = {
        "m1": _match("m1", 0, "support"),
        "m2": _match("m2", 1, "test"),
    }
    profiles = {
        match_id: {
            player_id: {
                "latent": torch.tensor(value),
                "raw": torch.tensor(value),
                "role": "forward",
                "team": "A",
                "clips": 2,
            }
            for player_id, value in (
                ("p1", [1.0, 0.0]),
                ("p2", [0.0, 1.0]),
            )
        }
        for match_id in matches
    }
    rows = retrieval_rows(
        profiles,
        matches,
        key="latent",
        query_splits={"train", "val"},
        support_size=2,
        normalizer=None,
    )
    assert rows == []
