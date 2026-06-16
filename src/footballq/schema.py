"""Canonical schema helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from footballq.constants import AGENT_TYPES

TRACKING_COLUMNS = [
    "match_id",
    "dataset",
    "period",
    "frame_id",
    "time_s",
    "agent_id",
    "agent_type",
    "team_id",
    "player_id",
    "jersey_number",
    "role",
    "x_m",
    "y_m",
    "z_m",
    "raw_x",
    "raw_y",
    "is_visible",
    "source_file",
]

EVENT_COLUMNS = [
    "match_id",
    "dataset",
    "period",
    "time_s",
    "frame_id",
    "team_id",
    "player_id",
    "event_type",
    "event_subtype",
    "x_m",
    "y_m",
    "end_x_m",
    "end_y_m",
    "outcome",
    "raw_event",
]

FEATURE_VALUE_COLUMNS = [
    "vx_mps",
    "vy_mps",
    "speed_mps",
    "ax_mps2",
    "ay_mps2",
    "accel_mps2",
    "distance_to_ball_m",
    "nearest_teammate_distance_m",
    "nearest_opponent_distance_m",
    "team_centroid_x_m",
    "team_centroid_y_m",
    "team_width_m",
    "team_length_m",
]

FEATURE_COLUMNS = TRACKING_COLUMNS + FEATURE_VALUE_COLUMNS

TRACKING_IDENTIFIER_COLUMNS = [
    "match_id",
    "dataset",
    "period",
    "frame_id",
    "time_s",
    "agent_id",
    "agent_type",
    "team_id",
    "player_id",
]


@dataclass(frozen=True)
class SchemaValidationResult:
    """Result returned by lightweight schema validation."""

    ok: bool
    missing_columns: list[str]
    invalid_agent_types: list[str]


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return a copy with all requested columns present."""

    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
    return out.loc[:, list(columns) + [c for c in out.columns if c not in columns]]


def order_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Order known columns first while preserving provider-specific extras."""

    ordered = [c for c in columns if c in df.columns]
    extras = [c for c in df.columns if c not in ordered]
    return df.loc[:, ordered + extras]


def canonical_tracking_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tracking dataframe with canonical columns first."""

    out = ensure_columns(df, TRACKING_COLUMNS)
    numeric_columns = [
        "period",
        "frame_id",
        "time_s",
        "jersey_number",
        "x_m",
        "y_m",
        "z_m",
        "raw_x",
        "raw_y",
    ]
    for column in numeric_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return order_columns(out, TRACKING_COLUMNS)


def canonical_event_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return an events dataframe with canonical columns first."""

    out = ensure_columns(df, EVENT_COLUMNS)
    for column in ["period", "time_s", "frame_id", "x_m", "y_m", "end_x_m", "end_y_m"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return order_columns(out, EVENT_COLUMNS)


def validate_tracking_schema(df: pd.DataFrame) -> SchemaValidationResult:
    """Validate required columns and agent types."""

    missing = [column for column in TRACKING_COLUMNS if column not in df.columns]
    invalid: list[str] = []
    if "agent_type" in df.columns:
        values = set(df["agent_type"].dropna().astype(str).unique())
        invalid = sorted(values - AGENT_TYPES)
    return SchemaValidationResult(
        ok=not missing and not invalid,
        missing_columns=missing,
        invalid_agent_types=invalid,
    )
