"""Structured data-quality checks."""

from __future__ import annotations

import pandas as pd

from footballq.constants import AGENT_BALL, BALL_SPEED_WARNING_MPS, PLAYER_SPEED_WARNING_MPS
from footballq.processing.coordinates import validate_pitch_bounds
from footballq.processing.features import compute_features


def check_quality(tracking_df: pd.DataFrame) -> dict[str, object]:
    """Return a structured report for canonical tracking data."""

    df = tracking_df.copy()
    report: dict[str, object] = {
        "ok": True,
        "frame_monotonicity": [],
        "coordinate_bounds": validate_pitch_bounds(df),
        "impossible_speeds": [],
        "missingness_by_agent": {},
    }

    monotonicity_issues: list[dict[str, object]] = []
    for key, group in df.groupby(["match_id", "period", "agent_id"], dropna=False):
        ordered = group.sort_values("time_s", kind="mergesort")
        if ordered["time_s"].diff().dropna().lt(0).any():
            monotonicity_issues.append({"group": key, "issue": "time_s decreases"})
        if ordered["frame_id"].diff().dropna().lt(0).any():
            monotonicity_issues.append({"group": key, "issue": "frame_id decreases"})
    report["frame_monotonicity"] = monotonicity_issues

    features = compute_features(df) if "speed_mps" not in df.columns else df
    speed_issues: list[dict[str, object]] = []
    player_fast = features[
        (features["agent_type"] != AGENT_BALL) & (features["speed_mps"] > PLAYER_SPEED_WARNING_MPS)
    ]
    ball_fast = features[
        (features["agent_type"] == AGENT_BALL) & (features["speed_mps"] > BALL_SPEED_WARNING_MPS)
    ]
    for _, row in player_fast.iterrows():
        speed_issues.append(
            {
                "match_id": row["match_id"],
                "frame_id": int(row["frame_id"]),
                "agent_id": row["agent_id"],
                "speed_mps": float(row["speed_mps"]),
                "threshold_mps": PLAYER_SPEED_WARNING_MPS,
            }
        )
    for _, row in ball_fast.iterrows():
        speed_issues.append(
            {
                "match_id": row["match_id"],
                "frame_id": int(row["frame_id"]),
                "agent_id": row["agent_id"],
                "speed_mps": float(row["speed_mps"]),
                "threshold_mps": BALL_SPEED_WARNING_MPS,
            }
        )
    report["impossible_speeds"] = speed_issues

    missingness = {}
    for agent_id, group in df.groupby("agent_id", dropna=False):
        missingness[str(agent_id)] = {
            "rows": int(len(group)),
            "missing_x_m": int(group["x_m"].isna().sum()),
            "missing_y_m": int(group["y_m"].isna().sum()),
        }
    report["missingness_by_agent"] = missingness

    report["ok"] = not (
        monotonicity_issues
        or speed_issues
        or not bool(report["coordinate_bounds"]["ok"])  # type: ignore[index]
    )
    return report

