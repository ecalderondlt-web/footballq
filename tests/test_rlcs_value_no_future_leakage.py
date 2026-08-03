from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from rlcs_test_utils import synthetic_replay

from footballq.data.rlcs_player_profiles import PROFILE_DIMENSION
from footballq.data.rlcs_value_windows import build_replay_value_rows, write_value_parquet


def _extend_events(parsed, roster, future_poison: bool = False):
    first = parsed.events.iloc[0].to_dict()
    prefixes = list(roster)
    extra = []
    for number, frame in enumerate(range(24, 61, 4), start=2):
        prefix = prefixes[number % len(prefixes)]
        team = "blue" if prefix.startswith("blue") else "orange"
        player_id = roster[prefix].split(":", 1)[1]
        row = dict(first)
        row.update(
            {
                "event_number": number,
                "event_type": "touch",
                "frame_number": frame,
                "observed_frame_number": frame,
                "seconds_elapsed": int(frame / 10),
                "game_time_s_precise": float(frame / 10),
                "event_team": team,
                "event_player_1_id": player_id,
                "event_player_1_name": f"Player {player_id}",
                "event_player_1_team": team,
                "event_ball_pos_x": float(parsed.frames.loc[frame, "ball_pos_x"]),
                "event_ball_pos_y": float(parsed.frames.loc[frame, "ball_pos_y"]),
                "event_ball_pos_z": float(parsed.frames.loc[frame, "ball_pos_z"]),
                "ball_pos_x": float(parsed.frames.loc[frame, "ball_pos_x"]),
                "ball_pos_y": float(parsed.frames.loc[frame, "ball_pos_y"]),
                "ball_pos_z": float(parsed.frames.loc[frame, "ball_pos_z"]),
            }
        )
        extra.append(row)
    parsed.events = pd.DataFrame([parsed.events.iloc[0].to_dict(), *extra])
    if future_poison:
        parsed.frames.loc[parsed.frames["observed_frame_number"] > 20, "ball_pos_x"] = 1e9


def _snapshots(roster):
    return {
        player_id: {
            "profile": [0.0] * PROFILE_DIMENSION,
            "uncertainty": [1.0] * PROFILE_DIMENSION,
            "effective_sample_size": 2.0,
            "n_prior_games": 1,
            "prior_win_rate": 0.5,
            "prior_goal_diff": 0.0,
        }
        for player_id in roster.values()
    }


def test_future_frame_poison_cannot_change_current_state(tmp_path: Path):
    clean, observations, roster, inventory = synthetic_replay()
    poisoned, poisoned_observations, poisoned_roster, poisoned_inventory = synthetic_replay()
    _extend_events(clean, roster)
    _extend_events(poisoned, poisoned_roster, future_poison=True)
    clean_rows = build_replay_value_rows(
        clean.frames,
        clean.events,
        replay_id=clean.replay_id,
        inventory=inventory,
        stage="train",
        observations=observations,
        roster_ids=roster,
        snapshots=_snapshots(roster),
        exclude_goal_reset_seconds=0.0,
    )
    poisoned_rows = build_replay_value_rows(
        poisoned.frames,
        poisoned.events,
        replay_id=poisoned.replay_id,
        inventory=poisoned_inventory,
        stage="train",
        observations=poisoned_observations,
        roster_ids=poisoned_roster,
        snapshots=_snapshots(poisoned_roster),
        exclude_goal_reset_seconds=0.0,
    )
    clean_current = next(row for row in clean_rows if row["frame_idx"] == 20)
    poisoned_current = next(row for row in poisoned_rows if row["frame_idx"] == 20)
    assert clean_current["state_flat"] == poisoned_current["state_flat"]
    assert clean_current["horizon_end_frame"] == poisoned_current["horizon_end_frame"]
    parquet_path = write_value_parquet(clean_rows, tmp_path / "train.parquet")
    assert len(pq.read_table(parquet_path)) == len(clean_rows)
