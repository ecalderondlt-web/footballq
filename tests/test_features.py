import numpy as np
import pandas as pd

from footballq.processing.features import compute_features
from footballq.schema import canonical_tracking_frame


def _simple_tracking() -> pd.DataFrame:
    rows = []
    for frame, time_s in enumerate([0.0, 1.0, 2.0]):
        rows.append(
            {
                "match_id": "m1",
                "dataset": "synthetic",
                "period": 1,
                "frame_id": frame,
                "time_s": time_s,
                "agent_id": "home_01",
                "agent_type": "player",
                "team_id": "home",
                "player_id": "home_01",
                "jersey_number": 1,
                "role": "player",
                "x_m": float(frame),
                "y_m": 0.0,
            }
        )
        rows.append(
            {
                "match_id": "m1",
                "dataset": "synthetic",
                "period": 1,
                "frame_id": frame,
                "time_s": time_s,
                "agent_id": "away_01",
                "agent_type": "player",
                "team_id": "away",
                "player_id": "away_01",
                "jersey_number": 1,
                "role": "player",
                "x_m": 10.0,
                "y_m": 0.0,
            }
        )
        rows.append(
            {
                "match_id": "m1",
                "dataset": "synthetic",
                "period": 1,
                "frame_id": frame,
                "time_s": time_s,
                "agent_id": "ball",
                "agent_type": "ball",
                "team_id": "ball",
                "role": "ball",
                "x_m": 0.0,
                "y_m": 0.0,
            }
        )
    return canonical_tracking_frame(pd.DataFrame(rows))


def test_velocity_calculation_correctness():
    features = compute_features(_simple_tracking())
    player = features[features["agent_id"] == "home_01"].sort_values("time_s")
    assert np.allclose(player["vx_mps"], [1.0, 1.0, 1.0])
    assert np.allclose(player["vy_mps"], [0.0, 0.0, 0.0])
    assert np.allclose(player["speed_mps"], [1.0, 1.0, 1.0])
    assert np.allclose(player["ax_mps2"], [0.0, 0.0, 0.0])


def test_distance_and_nearest_opponent_features():
    features = compute_features(_simple_tracking())
    row = features[(features["agent_id"] == "home_01") & (features["frame_id"] == 0)].iloc[0]
    assert row["distance_to_ball_m"] == 0.0
    assert row["nearest_opponent_distance_m"] == 10.0

