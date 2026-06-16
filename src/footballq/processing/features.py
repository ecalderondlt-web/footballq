"""Movement and tactical feature computation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from footballq.constants import AGENT_BALL, AGENT_PLAYER
from footballq.schema import FEATURE_VALUE_COLUMNS, canonical_tracking_frame

FRAME_KEYS = ["match_id", "period", "frame_id"]


def _safe_gradient(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros(len(values), dtype=float)
    values = values.astype(float)
    times = times.astype(float)
    valid_times = np.isfinite(times)
    if not valid_times.all() or len(np.unique(times)) < 2:
        return np.zeros(len(values), dtype=float)
    if np.any(np.diff(times) <= 0):
        out = np.full(len(values), np.nan, dtype=float)
        dt = np.diff(times)
        dv = np.diff(values)
        slopes = np.divide(dv, dt, out=np.full_like(dv, np.nan), where=dt > 0)
        if len(slopes):
            out[:-1] = slopes
            out[-1] = slopes[-1]
        return out
    return np.gradient(values, times, edge_order=1)


def add_kinematics(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Compute velocity, speed, acceleration, and acceleration magnitude."""

    df = canonical_tracking_frame(tracking_df).sort_values(
        ["match_id", "period", "agent_id", "time_s", "frame_id"],
        kind="mergesort",
    )
    for column in ["vx_mps", "vy_mps", "ax_mps2", "ay_mps2"]:
        df[column] = np.nan

    group_columns = ["match_id", "period", "agent_id"]
    for _, group in df.groupby(group_columns, dropna=False, sort=False):
        idx = group.index
        times = group["time_s"].to_numpy(dtype=float)
        x = group["x_m"].to_numpy(dtype=float)
        y = group["y_m"].to_numpy(dtype=float)
        vx = _safe_gradient(x, times)
        vy = _safe_gradient(y, times)
        ax = _safe_gradient(vx, times)
        ay = _safe_gradient(vy, times)
        df.loc[idx, "vx_mps"] = vx
        df.loc[idx, "vy_mps"] = vy
        df.loc[idx, "ax_mps2"] = ax
        df.loc[idx, "ay_mps2"] = ay

    df["speed_mps"] = np.hypot(df["vx_mps"], df["vy_mps"])
    df["accel_mps2"] = np.hypot(df["ax_mps2"], df["ay_mps2"])
    return df


def add_distance_to_ball(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-frame distance from each agent to the ball."""

    out = df.copy()
    ball = (
        out[out["agent_type"] == AGENT_BALL][FRAME_KEYS + ["x_m", "y_m"]]
        .drop_duplicates(FRAME_KEYS)
        .rename(columns={"x_m": "ball_x_m", "y_m": "ball_y_m"})
    )
    out = out.merge(ball, on=FRAME_KEYS, how="left")
    out["distance_to_ball_m"] = np.hypot(out["x_m"] - out["ball_x_m"], out["y_m"] - out["ball_y_m"])
    out = out.drop(columns=["ball_x_m", "ball_y_m"])
    return out


def add_team_shape_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute nearest-player and team-shape features for player rows."""

    out = df.copy()
    for column in [
        "nearest_teammate_distance_m",
        "nearest_opponent_distance_m",
        "team_centroid_x_m",
        "team_centroid_y_m",
        "team_width_m",
        "team_length_m",
    ]:
        out[column] = np.nan

    players = out[out["agent_type"] == AGENT_PLAYER].copy()
    if players.empty:
        return out

    team_stats = (
        players.groupby(FRAME_KEYS + ["team_id"], dropna=False)
        .agg(
            team_centroid_x_m=("x_m", "mean"),
            team_centroid_y_m=("y_m", "mean"),
            team_width_m=(
                "y_m",
                lambda s: float(s.max() - s.min()) if s.notna().any() else np.nan,
            ),
            team_length_m=(
                "x_m",
                lambda s: float(s.max() - s.min()) if s.notna().any() else np.nan,
            ),
        )
        .reset_index()
    )
    out = out.merge(team_stats, on=FRAME_KEYS + ["team_id"], how="left", suffixes=("", "_new"))
    for column in ["team_centroid_x_m", "team_centroid_y_m", "team_width_m", "team_length_m"]:
        new_column = f"{column}_new"
        if new_column in out.columns:
            out[column] = out[new_column].combine_first(out[column])
            out = out.drop(columns=[new_column])

    for _, frame in players.groupby(FRAME_KEYS, dropna=False, sort=False):
        coords = frame[["x_m", "y_m"]].to_numpy(dtype=float)
        teams = frame["team_id"].astype(str).to_numpy()
        indices = frame.index.to_numpy()
        for row_number, row_index in enumerate(indices):
            if not np.isfinite(coords[row_number]).all():
                continue
            deltas = coords - coords[row_number]
            distances = np.hypot(deltas[:, 0], deltas[:, 1])
            finite = np.isfinite(distances) & (np.arange(len(distances)) != row_number)
            teammate = finite & (teams == teams[row_number])
            opponent = finite & (teams != teams[row_number])
            if teammate.any():
                out.loc[row_index, "nearest_teammate_distance_m"] = float(distances[teammate].min())
            if opponent.any():
                out.loc[row_index, "nearest_opponent_distance_m"] = float(distances[opponent].min())

    return out


def compute_features(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Phase 1 movement and tactical features."""

    out = add_kinematics(tracking_df)
    out = add_distance_to_ball(out)
    out = add_team_shape_features(out)
    for column in FEATURE_VALUE_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    return out
