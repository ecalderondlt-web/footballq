from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from footballq.data.rlcs_replay import ParsedReplay, ReplayQC, roster_observations


def synthetic_replay(
    *,
    boundary_event: str | None = None,
    future_poison: bool = False,
    next_player_prefix: str = "orange_player_1",
) -> tuple[ParsedReplay, list[Any], dict[str, str], dict[str, Any]]:
    frame_count = 61
    times = np.arange(frame_count, dtype=np.float32) / 10.0
    frames: dict[str, Any] = {
        "game_id": ["synthetic"] * frame_count,
        "team_size": [3] * frame_count,
        "frame_number": np.arange(frame_count),
        "observed_frame_number": np.arange(frame_count),
        "stint_number": np.ones(frame_count, dtype=np.float32),
        "seconds_elapsed": np.floor(times),
        "game_time_s_precise": times,
        "ball_pos_x": np.linspace(0, 300, frame_count),
        "ball_pos_y": np.linspace(-500, 500, frame_count),
        "ball_pos_z": np.full(frame_count, 100.0),
        "ball_vel_x": np.full(frame_count, 50.0),
        "ball_vel_y": np.full(frame_count, 100.0),
        "ball_vel_z": np.zeros(frame_count),
        "ball_ang_vel_x": np.zeros(frame_count),
        "ball_ang_vel_y": np.zeros(frame_count),
        "ball_ang_vel_z": np.ones(frame_count),
    }
    positions = {
        "blue_player_1": (0.0, -1000.0, 17.0),
        "blue_player_2": (100.0, -900.0, 17.0),
        "blue_player_3": (1000.0, -1500.0, 17.0),
        "orange_player_1": (200.0, 800.0, 17.0),
        "orange_player_2": (-700.0, 1200.0, 17.0),
        "orange_player_3": (900.0, 1500.0, 17.0),
    }
    roster_ids: dict[str, str] = {}
    for index, (prefix, position) in enumerate(positions.items(), start=100):
        platform_id = str(index)
        roster_ids[prefix] = f"steam:{platform_id}"
        frames[f"{prefix}_id"] = [platform_id] * frame_count
        frames[f"{prefix}_network_id"] = [f"steam:{platform_id}"] * frame_count
        frames[f"{prefix}_name"] = [f"Player {index}"] * frame_count
        frames[f"{prefix}_platform"] = ["OnlinePlatform_Steam"] * frame_count
        frames[f"{prefix}_is_bot"] = [False] * frame_count
        for axis, value in zip("xyz", position, strict=True):
            frames[f"{prefix}_pos_{axis}"] = np.full(frame_count, value)
        frames[f"{prefix}_vel_x"] = np.full(frame_count, index - 100.0)
        frames[f"{prefix}_vel_y"] = np.full(frame_count, 20.0)
        frames[f"{prefix}_vel_z"] = np.zeros(frame_count)
        frames[f"{prefix}_ang_vel_x"] = np.zeros(frame_count)
        frames[f"{prefix}_ang_vel_y"] = np.zeros(frame_count)
        frames[f"{prefix}_ang_vel_z"] = np.full(frame_count, 0.1)
        frames[f"{prefix}_rot_x"] = np.zeros(frame_count)
        frames[f"{prefix}_rot_y"] = np.zeros(frame_count)
        frames[f"{prefix}_rot_z"] = np.full(frame_count, 0.5)
        frames[f"{prefix}_boost"] = np.full(frame_count, 50)
        frames[f"{prefix}_jumped"] = [False] * frame_count
        frames[f"{prefix}_flipped"] = [False] * frame_count
        frames[f"{prefix}_double_jump_active"] = [False] * frame_count
    frame_table = pd.DataFrame(frames)
    if future_poison:
        frame_table.loc[frame_table["observed_frame_number"] > 20, "ball_pos_x"] = 999_999.0

    def event(number: int, frame: int, prefix: str, event_type: str = "touch") -> dict[str, Any]:
        team = "blue" if prefix.startswith("blue") else "orange"
        player_id = roster_ids[prefix].split(":", 1)[1]
        index = int(player_id)
        return {
            "event_number": number,
            "event_type": event_type,
            "frame_number": frame,
            "observed_frame_number": frame,
            "seconds_elapsed": int(frame / 10),
            "game_time_s_precise": float(frame / 10),
            "event_team": team,
            "event_player_1_id": player_id,
            "event_player_1_name": f"Player {index}",
            "event_player_1_team": team,
            "event_ball_pos_x": float(frame_table.loc[frame, "ball_pos_x"]),
            "event_ball_pos_y": float(frame_table.loc[frame, "ball_pos_y"]),
            "event_ball_pos_z": float(frame_table.loc[frame, "ball_pos_z"]),
            "ball_pos_x": float(frame_table.loc[frame, "ball_pos_x"]),
            "ball_pos_y": float(frame_table.loc[frame, "ball_pos_y"]),
            "ball_pos_z": float(frame_table.loc[frame, "ball_pos_z"]),
            "blue_score": 0,
            "orange_score": 0,
        }

    events = [event(1, 20, "blue_player_1"), event(2, 30, next_player_prefix)]
    if boundary_event:
        boundary = event(3, 25, "blue_player_1", boundary_event)
        if boundary_event == "goal":
            boundary["blue_score"] = 1
            events[1]["blue_score"] = 1
        events.insert(1, boundary)
    event_table = pd.DataFrame(events).sort_values("observed_frame_number").reset_index(drop=True)
    parsed = ParsedReplay(
        replay_id="synthetic",
        frames=frame_table,
        events=event_table,
        qc=ReplayQC(
            accepted=True,
            reasons=(),
            team_size=3,
            player_slots=6,
            duration_seconds=300.0,
            frame_count=frame_count,
            event_count=len(event_table),
        ),
    )
    observations = roster_observations(
        frame_table,
        replay_id="synthetic",
        split="train",
        event_time_utc="2025-01-01T12:00:00Z",
        group_id="series-1",
    )
    inventory = {
        "series_id": "series-1",
        "leaf_group_id": "series-1",
        "group_path": "EU/Split 1/Regional 1/Swiss",
        "region": "EU",
        "event_time_utc": "2025-01-01T12:00:00Z",
    }
    return parsed, observations, roster_ids, inventory
