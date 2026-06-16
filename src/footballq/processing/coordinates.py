"""Coordinate conversion and validation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from footballq.constants import PITCH_LENGTH_M, PITCH_WIDTH_M


def to_meters_from_normalized(
    x: Any,
    y: Any,
    pitch_length: float = PITCH_LENGTH_M,
    pitch_width: float = PITCH_WIDTH_M,
) -> tuple[Any, Any]:
    """Convert normalized provider coordinates to canonical pitch meters."""

    return x * pitch_length, y * pitch_width


def clip_to_pitch(
    df: pd.DataFrame,
    pitch_length: float = PITCH_LENGTH_M,
    pitch_width: float = PITCH_WIDTH_M,
) -> pd.DataFrame:
    """Clip `x_m` and `y_m` coordinates to canonical pitch bounds."""

    out = df.copy()
    if "x_m" in out.columns:
        out["x_m"] = pd.to_numeric(out["x_m"], errors="coerce").clip(0.0, pitch_length)
    if "y_m" in out.columns:
        out["y_m"] = pd.to_numeric(out["y_m"], errors="coerce").clip(0.0, pitch_width)
    return out


def validate_pitch_bounds(
    df: pd.DataFrame,
    pitch_length: float = PITCH_LENGTH_M,
    pitch_width: float = PITCH_WIDTH_M,
) -> dict[str, object]:
    """Return a structured report for out-of-bounds coordinates."""

    if "x_m" not in df.columns or "y_m" not in df.columns:
        return {
            "ok": False,
            "out_of_bounds_rows": 0,
            "bad_indices": [],
            "message": "Dataframe must contain x_m and y_m columns.",
        }

    x = pd.to_numeric(df["x_m"], errors="coerce")
    y = pd.to_numeric(df["y_m"], errors="coerce")
    present = x.notna() & y.notna()
    in_bounds = x.between(0.0, pitch_length) & y.between(0.0, pitch_width)
    bad = present & ~in_bounds
    bad_indices = list(df.index[bad])
    return {
        "ok": len(bad_indices) == 0,
        "out_of_bounds_rows": len(bad_indices),
        "bad_indices": bad_indices,
        "message": "ok" if len(bad_indices) == 0 else "Coordinates outside canonical pitch bounds.",
    }


def flip_coordinates_left_to_right(
    df: pd.DataFrame,
    pitch_length: float = PITCH_LENGTH_M,
    pitch_width: float = PITCH_WIDTH_M,
) -> pd.DataFrame:
    """Flip coordinates for future attacking-direction normalization."""

    out = df.copy()
    if "x_m" in out.columns:
        out["x_m"] = pitch_length - pd.to_numeric(out["x_m"], errors="coerce")
    if "y_m" in out.columns:
        out["y_m"] = pitch_width - pd.to_numeric(out["y_m"], errors="coerce")
    if "vx_mps" in out.columns:
        out["vx_mps"] = -pd.to_numeric(out["vx_mps"], errors="coerce")
    if "vy_mps" in out.columns:
        out["vy_mps"] = -pd.to_numeric(out["vy_mps"], errors="coerce")
    return out

