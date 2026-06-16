"""Fixed-length trajectory window export."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from footballq.config import WindowConfig
from footballq.constants import AGENT_BALL, AGENT_PLAYER

DEFAULT_FEATURE_NAMES = ["x_m", "y_m", "vx_mps", "vy_mps", "speed_mps", "distance_to_ball_m"]
DEFAULT_TARGET_NAMES = ["x_m", "y_m"]


@dataclass
class WindowBatch:
    X_history: np.ndarray
    Y_future: np.ndarray
    agent_mask_history: np.ndarray
    agent_mask_future: np.ndarray
    agent_ids: np.ndarray
    feature_names: list[str]
    target_names: list[str]
    window_meta: pd.DataFrame


def _natural_key(value: object) -> tuple[str, int]:
    text = str(value)
    match = re.search(r"(\d+)$", text)
    number = int(match.group(1)) if match else -1
    prefix = re.sub(r"\d+$", "", text)
    return prefix, number


def stable_agent_order(df: pd.DataFrame, max_agents: int = 23) -> list[str]:
    """Order agents as home players, away players, other players, then ball."""

    players = df[df["agent_type"] == AGENT_PLAYER]
    home = sorted(
        players[players["team_id"] == "home"]["agent_id"].dropna().unique(),
        key=_natural_key,
    )
    away = sorted(
        players[players["team_id"] == "away"]["agent_id"].dropna().unique(),
        key=_natural_key,
    )
    others = sorted(
        players[~players["team_id"].isin(["home", "away"])]["agent_id"].dropna().unique(),
        key=_natural_key,
    )
    balls = sorted(
        df[df["agent_type"] == AGENT_BALL]["agent_id"].dropna().unique(),
        key=_natural_key,
    )
    ordered = list(home) + list(away) + list(others)
    if balls:
        return (ordered[: max_agents - 1] + [balls[0]])[:max_agents]
    return ordered[:max_agents]


def build_windows(
    tracking_df: pd.DataFrame,
    features_df: pd.DataFrame | None = None,
    history_s: float = 5.0,
    future_s: float = 5.0,
    fps: float = 10.0,
    max_agents: int = 23,
    feature_names: list[str] | None = None,
    target_names: list[str] | None = None,
    stride_steps: int = 1,
) -> WindowBatch:
    """Build fixed-length history/future windows for future model training."""

    config = WindowConfig(history_s=history_s, future_s=future_s, fps=fps, max_agents=max_agents)
    feature_source = features_df if features_df is not None else tracking_df
    feature_names = feature_names or [
        column for column in DEFAULT_FEATURE_NAMES if column in feature_source.columns
    ]
    if not feature_names:
        feature_names = ["x_m", "y_m"]
    target_names = target_names or DEFAULT_TARGET_NAMES
    source = features_df.copy() if features_df is not None else tracking_df.copy()
    source = source.sort_values(
        ["match_id", "period", "time_s", "frame_id", "agent_id"],
        kind="mergesort",
    )

    history_arrays: list[np.ndarray] = []
    future_arrays: list[np.ndarray] = []
    history_masks: list[np.ndarray] = []
    future_masks: list[np.ndarray] = []
    agent_id_rows: list[list[str]] = []
    meta_rows: list[dict[str, object]] = []

    total_steps = config.history_steps + config.future_steps
    for (match_id, period), match_period in source.groupby(
        ["match_id", "period"],
        dropna=False,
        sort=False,
    ):
        times = np.array(sorted(match_period["time_s"].dropna().unique()), dtype=float)
        if len(times) < total_steps:
            continue
        agents = stable_agent_order(match_period, max_agents=max_agents)
        padded_agents = agents + [""] * (max_agents - len(agents))
        indexed = {
            (float(row.time_s), str(row.agent_id)): row
            for row in match_period.itertuples(index=False)
        }

        for start in range(0, len(times) - total_steps + 1, stride_steps):
            history_times = times[start : start + config.history_steps]
            future_times = times[start + config.history_steps : start + total_steps]
            x_hist = np.full((config.history_steps, max_agents, len(feature_names)), np.nan)
            y_future = np.full((config.future_steps, max_agents, len(target_names)), np.nan)
            mask_hist = np.zeros((config.history_steps, max_agents), dtype=bool)
            mask_future = np.zeros((config.future_steps, max_agents), dtype=bool)

            for t_idx, time_s in enumerate(history_times):
                for a_idx, agent_id in enumerate(agents):
                    row = indexed.get((float(time_s), str(agent_id)))
                    if row is None:
                        continue
                    values = [getattr(row, name, np.nan) for name in feature_names]
                    x_hist[t_idx, a_idx, :] = values
                    mask_hist[t_idx, a_idx] = bool(pd.notna(getattr(row, "x_m", np.nan)))

            for t_idx, time_s in enumerate(future_times):
                for a_idx, agent_id in enumerate(agents):
                    row = indexed.get((float(time_s), str(agent_id)))
                    if row is None:
                        continue
                    values = [getattr(row, name, np.nan) for name in target_names]
                    y_future[t_idx, a_idx, :] = values
                    mask_future[t_idx, a_idx] = bool(pd.notna(getattr(row, "x_m", np.nan)))

            history_arrays.append(x_hist)
            future_arrays.append(y_future)
            history_masks.append(mask_hist)
            future_masks.append(mask_future)
            agent_id_rows.append(padded_agents)
            meta_rows.append(
                {
                    "match_id": match_id,
                    "period": period,
                    "start_time_s": float(history_times[0]),
                    "history_end_time_s": float(history_times[-1]),
                    "future_start_time_s": float(future_times[0]),
                    "end_time_s": float(future_times[-1]),
                    "history_steps": config.history_steps,
                    "future_steps": config.future_steps,
                    "fps": fps,
                }
            )

    if not history_arrays:
        empty_history = np.empty((0, config.history_steps, max_agents, len(feature_names)))
        empty_future = np.empty((0, config.future_steps, max_agents, len(target_names)))
        empty_history_mask = np.empty((0, config.history_steps, max_agents), dtype=bool)
        empty_future_mask = np.empty((0, config.future_steps, max_agents), dtype=bool)
        empty_agents = np.empty((0, max_agents), dtype=object)
        return WindowBatch(
            empty_history,
            empty_future,
            empty_history_mask,
            empty_future_mask,
            empty_agents,
            feature_names,
            target_names,
            pd.DataFrame(meta_rows),
        )

    return WindowBatch(
        np.stack(history_arrays),
        np.stack(future_arrays),
        np.stack(history_masks),
        np.stack(future_masks),
        np.asarray(agent_id_rows, dtype=object),
        feature_names,
        target_names,
        pd.DataFrame(meta_rows),
    )


def export_windows_npz(batch: WindowBatch, out: str | Path) -> tuple[Path, Path]:
    """Write windows to NPZ plus parquet metadata."""

    out_path = Path(out)
    if out_path.suffix.lower() == ".npz":
        out_path.parent.mkdir(parents=True, exist_ok=True)
        npz_path = out_path
        meta_path = out_path.with_name(f"{out_path.stem}_meta.parquet")
    else:
        out_path.mkdir(parents=True, exist_ok=True)
        npz_path = out_path / "windows.npz"
        meta_path = out_path / "window_meta.parquet"

    np.savez_compressed(
        npz_path,
        X_history=batch.X_history,
        Y_future=batch.Y_future,
        agent_mask_history=batch.agent_mask_history,
        agent_mask_future=batch.agent_mask_future,
        agent_ids=batch.agent_ids,
        feature_names=np.asarray(batch.feature_names, dtype=object),
        target_names=np.asarray(batch.target_names, dtype=object),
    )
    batch.window_meta.to_parquet(meta_path, index=False)
    return npz_path, meta_path
