import json

import pandas as pd

from footballq.data.synthetic_visibility import (
    apply_pff_like_visibility,
    build_pff_visibility_profile,
)
from footballq.synthetic.generate import generate_synthetic_tracking


def _tracking_frame() -> pd.DataFrame:
    rows = []
    for frame_id in range(2):
        rows.append(
            {
                "match_id": "grf",
                "period": 1,
                "frame_id": frame_id,
                "agent_id": "ball",
                "agent_type": "ball",
                "x_m": 50.0,
                "y_m": 34.0,
                "visible": True,
                "is_visible": True,
                "is_observed": True,
                "provider_visibility": "visible",
            }
        )
        for player_id in range(4):
            rows.append(
                {
                    "match_id": "grf",
                    "period": 1,
                    "frame_id": frame_id,
                    "agent_id": f"home_{player_id}",
                    "agent_type": "player",
                    "x_m": 48.0 + player_id,
                    "y_m": 34.0,
                    "visible": True,
                    "is_visible": True,
                    "is_observed": True,
                    "provider_visibility": "visible",
                }
            )
    return pd.DataFrame(rows)


def test_pff_like_visibility_is_deterministic_and_matches_count():
    profile = {
        "observed_player_count_probabilities": {"2": 1.0},
        "player_observed_rate": 0.5,
        "ball_observed_rate": 0.0,
        "distance_bin_edges_m": [0.0, 10.0, 100.0],
        "distance_bin_observed_probabilities": [0.9, 0.1],
    }
    first = apply_pff_like_visibility(_tracking_frame(), profile, seed=7)
    second = apply_pff_like_visibility(_tracking_frame(), profile, seed=7)

    assert first["visible"].tolist() == second["visible"].tolist()
    players = first[first["agent_type"] == "player"]
    assert players.groupby("frame_id")["visible"].sum().tolist() == [2, 2]
    assert not first[first["agent_type"] == "ball"]["visible"].any()


def test_pff_like_visibility_preserves_frozen_agent_selection_sequence():
    profile = {
        "observed_player_count_probabilities": {"5": 0.4, "8": 0.6},
        "player_observed_rate": 0.5,
        "ball_observed_rate": 0.5,
        "distance_bin_edges_m": [0.0, 20.0, 110.0],
        "distance_bin_observed_probabilities": [0.8, 0.2],
    }
    tracking = generate_synthetic_tracking(
        match_id="visibility-regression",
        duration_s=0.5,
        fps=10.0,
        seed=3,
    )
    masked = apply_pff_like_visibility(tracking, profile, seed=17)
    visible_agents = [
        group.loc[group["visible"], "agent_id"].astype(str).tolist()
        for _, group in masked.groupby("frame_id", sort=False)
    ]

    assert visible_agents == [
        ["home_01", "home_04", "home_05", "home_10", "away_11"],
        ["home_09", "home_11", "away_02", "away_03", "away_07"],
        [
            "home_01",
            "home_09",
            "home_11",
            "away_03",
            "away_04",
            "away_07",
            "away_08",
            "away_11",
            "ball",
        ],
        ["home_01", "home_09", "home_10", "away_05", "away_10"],
        [
            "home_07",
            "home_08",
            "home_10",
            "away_03",
            "away_04",
            "away_08",
            "away_10",
            "away_11",
            "ball",
        ],
        ["home_06", "home_08", "away_09", "away_10", "away_11", "ball"],
    ]


def test_visibility_profile_uses_only_requested_canonical_split(tmp_path):
    root = tmp_path / "canonical"
    match_root = root / "train" / "1"
    match_root.mkdir(parents=True)
    rows = []
    for frame_id, observed_players in ((0, 1), (1, 2)):
        rows.append(
            {
                "match_id": "1",
                "period": 1,
                "frame_id": frame_id,
                "agent_type": "ball",
                "x_m": 50.0,
                "y_m": 34.0,
                "is_observed": frame_id == 0,
            }
        )
        for player_id in range(2):
            rows.append(
                {
                    "match_id": "1",
                    "period": 1,
                    "frame_id": frame_id,
                    "agent_type": "player",
                    "x_m": 49.0 + player_id,
                    "y_m": 34.0,
                    "is_observed": player_id < observed_players,
                }
            )
    pd.DataFrame(rows).to_parquet(match_root / "tracking.parquet", index=False)
    (match_root / "manifest.json").write_text(
        json.dumps({"shards": [{"path": "tracking.parquet"}]}), encoding="utf-8"
    )
    (root / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "matches": [{"match_id": "1", "split": "train"}],
                "split_manifest_path": "split.json",
                "split_manifest_sha256": "split-hash",
                "manifest_payload_sha256": "canonical-hash",
            }
        ),
        encoding="utf-8",
    )

    profile = build_pff_visibility_profile(root, frame_stride=1)

    assert profile["sampled_frame_count"] == 2
    assert profile["observed_player_count_probabilities"] == {"1": 0.5, "2": 0.5}
    assert profile["player_observed_rate"] == 0.75
    assert profile["ball_observed_rate"] == 0.5
    assert profile["split_manifest_sha256"] == "split-hash"
