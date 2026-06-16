"""Tracking resampling utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from footballq.schema import canonical_tracking_frame


def resample_tracking(
    tracking_df: pd.DataFrame,
    target_fps: float = 10.0,
    max_gap_s: float = 0.5,
) -> pd.DataFrame:
    """Resample tracking to a target FPS without crossing period boundaries."""

    df = canonical_tracking_frame(tracking_df)
    rows: list[pd.DataFrame] = []
    dt = 1.0 / target_fps
    coordinate_columns = ["x_m", "y_m", "z_m", "raw_x", "raw_y"]
    group_columns = ["match_id", "period", "agent_id"]

    for _, group in df.groupby(group_columns, dropna=False, sort=False):
        group = group.sort_values("time_s", kind="mergesort")
        if group.empty:
            continue
        start = float(group["time_s"].min())
        end = float(group["time_s"].max())
        grid = np.round(np.arange(start, end + dt / 2.0, dt), 10)
        base = pd.DataFrame({"time_s": grid})
        merged = base.merge(group, on="time_s", how="left", suffixes=("", "_orig"))

        for column in df.columns:
            if column == "time_s":
                continue
            if column not in merged.columns:
                merged[column] = pd.NA

        for column in coordinate_columns:
            if column in merged.columns:
                merged[f"{column}_was_missing"] = merged[column].isna()
                series = pd.to_numeric(merged[column], errors="coerce")
                merged[column] = series.interpolate(
                    method="linear",
                    limit=int(round(max_gap_s * target_fps)),
                    limit_area="inside",
                )

        for column in df.columns:
            if column in coordinate_columns or column == "time_s":
                continue
            merged[column] = merged[column].ffill().bfill()

        merged["frame_id"] = np.round(merged["time_s"] * target_fps).astype(int)
        mask_columns = [c for c in merged.columns if c.endswith("_was_missing")]
        rows.append(merged[df.columns.tolist() + mask_columns])

    if not rows:
        return canonical_tracking_frame(df.iloc[0:0])
    return pd.concat(rows, ignore_index=True)
