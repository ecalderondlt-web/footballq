"""Generate small synthetic football-like tracking samples."""

from __future__ import annotations

import numpy as np
import pandas as pd

from footballq.constants import DATASET_SYNTHETIC, PITCH_LENGTH_M, PITCH_WIDTH_M
from footballq.schema import canonical_tracking_frame


def _formation(team: str) -> np.ndarray:
    if team == "home":
        xs = [8, 24, 24, 24, 24, 42, 42, 42, 62, 62, 72]
    else:
        xs = [97, 81, 81, 81, 81, 63, 63, 63, 43, 43, 33]
    ys = [34, 12, 26, 42, 56, 18, 34, 50, 22, 46, 34]
    return np.column_stack([xs, ys]).astype(float)


def _smooth_player_positions(
    base: np.ndarray,
    times: np.ndarray,
    rng: np.random.Generator,
    team_sign: float,
) -> np.ndarray:
    n_frames = len(times)
    n_players = base.shape[0]
    positions = np.zeros((n_frames, n_players, 2), dtype=float)
    phase = rng.uniform(0, 2 * np.pi, size=(n_players, 2))
    amplitude = rng.uniform(0.4, 1.2, size=(n_players, 2))
    drift = np.column_stack(
        [
            team_sign * 0.05 * times,
            0.2 * np.sin(times / 4.0),
        ]
    )
    for idx, t in enumerate(times):
        wobble = amplitude * np.column_stack(
            [
                np.sin(0.55 * t + phase[:, 0]),
                np.cos(0.45 * t + phase[:, 1]),
            ]
        )
        positions[idx] = base + wobble + drift[idx]
    positions[..., 0] = np.clip(positions[..., 0], 0.0, PITCH_LENGTH_M)
    positions[..., 1] = np.clip(positions[..., 1], 0.0, PITCH_WIDTH_M)
    return positions


def generate_synthetic_tracking(
    match_id: str = "synthetic_match",
    duration_s: float = 12.0,
    fps: float = 10.0,
    seed: int = 7,
) -> pd.DataFrame:
    """Create a deterministic, small football-like tracking dataframe."""

    rng = np.random.default_rng(seed)
    times = np.round(np.arange(0.0, duration_s + 1.0 / fps / 2.0, 1.0 / fps), 10)
    frame_ids = np.arange(len(times))

    home_positions = _smooth_player_positions(_formation("home"), times, rng, team_sign=1.0)
    away_positions = _smooth_player_positions(_formation("away"), times, rng, team_sign=-1.0)
    all_positions = np.concatenate([home_positions, away_positions], axis=1)
    agent_ids = [f"home_{i:02d}" for i in range(1, 12)] + [f"away_{i:02d}" for i in range(1, 12)]
    team_ids = ["home"] * 11 + ["away"] * 11
    jersey_numbers = list(range(1, 12)) + list(range(1, 12))

    passers = [7, 8, 10, 9, 6]
    segment_edges = np.linspace(0, len(times) - 1, len(passers)).astype(int)
    ball_positions = np.zeros((len(times), 2), dtype=float)
    for segment, start_idx in enumerate(segment_edges[:-1]):
        end_idx = segment_edges[segment + 1]
        passer = passers[segment]
        receiver = passers[segment + 1]
        span = max(end_idx - start_idx, 1)
        for frame_idx in range(start_idx, end_idx + 1):
            alpha = (frame_idx - start_idx) / span
            start_pos = all_positions[frame_idx, passer]
            end_pos = all_positions[frame_idx, receiver]
            arc = np.array([0.0, np.sin(alpha * np.pi) * 1.5])
            ball_positions[frame_idx] = (1 - alpha) * start_pos + alpha * end_pos + arc
    ball_positions[:, 0] = np.clip(ball_positions[:, 0], 0.0, PITCH_LENGTH_M)
    ball_positions[:, 1] = np.clip(ball_positions[:, 1], 0.0, PITCH_WIDTH_M)

    rows = []
    for frame_idx, time_s in zip(frame_ids, times, strict=True):
        for agent_idx, agent_id in enumerate(agent_ids):
            x_m, y_m = all_positions[frame_idx, agent_idx]
            jersey_number = jersey_numbers[agent_idx]
            rows.append(
                {
                    "match_id": match_id,
                    "dataset": DATASET_SYNTHETIC,
                    "period": 1,
                    "frame_id": int(frame_idx),
                    "time_s": float(time_s),
                    "agent_id": agent_id,
                    "agent_type": "player",
                    "team_id": team_ids[agent_idx],
                    "player_id": agent_id,
                    "jersey_number": jersey_number,
                    "role": "goalkeeper" if jersey_number == 1 else "player",
                    "x_m": float(x_m),
                    "y_m": float(y_m),
                    "z_m": np.nan,
                    "raw_x": float(x_m / PITCH_LENGTH_M),
                    "raw_y": float(y_m / PITCH_WIDTH_M),
                    "is_visible": True,
                    "source_file": "synthetic",
                }
            )
        ball_x, ball_y = ball_positions[frame_idx]
        rows.append(
            {
                "match_id": match_id,
                "dataset": DATASET_SYNTHETIC,
                "period": 1,
                "frame_id": int(frame_idx),
                "time_s": float(time_s),
                "agent_id": "ball",
                "agent_type": "ball",
                "team_id": "ball",
                "player_id": pd.NA,
                "jersey_number": pd.NA,
                "role": "ball",
                "x_m": float(ball_x),
                "y_m": float(ball_y),
                "z_m": np.nan,
                "raw_x": float(ball_x / PITCH_LENGTH_M),
                "raw_y": float(ball_y / PITCH_WIDTH_M),
                "is_visible": True,
                "source_file": "synthetic",
            }
        )

    return canonical_tracking_frame(pd.DataFrame(rows))

