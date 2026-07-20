import json

import pytest

from footballq.data.windows import build_tracking_windows
from footballq.io.gfootball import (
    GFootballAdapter,
    gfootball_xy_to_meters,
    observations_to_tracking,
)


def _obs(frame: int, players: int = 11) -> dict:
    left = [[-0.6 + idx * 0.02 + frame * 0.001, -0.2 + idx * 0.01] for idx in range(players)]
    right = [[0.6 - idx * 0.02 - frame * 0.001, 0.2 - idx * 0.01] for idx in range(players)]
    zeros = [[0.001, 0.0] for _ in range(players)]
    return {
        "episode_id": 0,
        "frame_id": frame,
        "fps": 10.0,
        "observation": {
            "ball": [0.0 + frame * 0.001, 0.0, 0.0],
            "ball_direction": [0.001, 0.0, 0.0],
            "ball_owned_team": 0,
            "ball_owned_player": 1,
            "left_team": left,
            "left_team_direction": zeros,
            "left_team_active": [True] * players,
            "left_team_roles": list(range(players)),
            "right_team": right,
            "right_team_direction": zeros,
            "right_team_active": [True] * players,
            "right_team_roles": list(range(players)),
            "active": 1,
            "game_mode": 0,
            "score": [1, 0],
            "steps_left": 100,
        },
    }


def test_gfootball_coordinate_conversion_matches_pitch_corners():
    assert gfootball_xy_to_meters(-1.0, -0.42) == pytest.approx((0.0, 0.0))
    assert gfootball_xy_to_meters(1.0, 0.42) == pytest.approx((105.0, 68.0))
    assert gfootball_xy_to_meters(0.0, 0.0) == pytest.approx((52.5, 34.0))


def test_gfootball_observations_convert_to_canonical_tracking_rows():
    tracking = observations_to_tracking([_obs(0), _obs(1)], match_id="grf")

    assert set(tracking["dataset"]) == {"gfootball"}
    assert set(tracking["match_id"]) == {"grf_episode_0"}
    assert len(tracking) == 46
    ball = tracking[(tracking["agent_id"] == "ball") & (tracking["frame_id"] == 0)].iloc[0]
    assert ball["x_m"] == pytest.approx(52.5)
    assert ball["y_m"] == pytest.approx(34.0)
    owner = tracking[(tracking["agent_id"] == "home_01") & (tracking["frame_id"] == 0)].iloc[0]
    assert bool(owner["has_possession"])
    assert owner["team_id"] == "home"
    assert owner["score_home"] == 1
    assert owner["score_away"] == 0
    assert owner["steps_left"] == 100


def test_gfootball_tracking_builds_fixed_footballq_windows():
    tracking = observations_to_tracking([_obs(frame) for frame in range(4)], match_id="grf")

    windows = build_tracking_windows(
        tracking,
        fps_out=10.0,
        context_seconds=0.2,
        horizon_seconds=0.2,
        stride_seconds=0.2,
    )

    assert windows.past.shape == (1, 2, 23, 10)
    assert windows.future_xy.shape == (1, 2, 23, 2)
    assert windows.match_id == ["grf_episode_0"]


def test_gfootball_adapter_namespaces_episode_ids_across_collection_shards(tmp_path):
    for shard in ("seed_1", "seed_2"):
        (tmp_path / f"{shard}.jsonl").write_text(json.dumps(_obs(0)) + "\n", encoding="utf-8")

    tracking = GFootballAdapter(tmp_path, match_id="grf").load_tracking()

    assert set(tracking["match_id"]) == {
        "grf_seed_1_episode_0",
        "grf_seed_2_episode_0",
    }
