"""Metrica Sports sample-data adapter."""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from footballq.constants import AGENT_BALL, AGENT_PLAYER, DATASET_METRICA
from footballq.io.base import TrackingDataAdapter
from footballq.processing.coordinates import to_meters_from_normalized
from footballq.schema import canonical_event_frame, canonical_tracking_frame


def _is_unnamed(value: object) -> bool:
    text = str(value).strip()
    return not text or text.lower().startswith("unnamed")


def flatten_metrica_columns(columns: pd.Index | pd.MultiIndex) -> list[str]:
    """Flatten Metrica multi-row headers and forward-fill player labels."""

    if not isinstance(columns, pd.MultiIndex):
        return [str(column).strip() for column in columns]

    flattened: list[str] = []
    current_entity: str | None = None
    for column in columns:
        levels = [str(level).strip() for level in column]
        first_named = next((level for level in levels if not _is_unnamed(level)), "")
        coord = next(
            (level.lower() for level in reversed(levels) if level.lower() in {"x", "y"}),
            None,
        )
        if coord is None:
            current_entity = None
            flattened.append(first_named)
            continue
        entity = next((level for level in levels[:-1] if not _is_unnamed(level)), None)
        if entity is not None and entity.lower() not in {"x", "y"}:
            current_entity = entity
        if current_entity is None:
            flattened.append(coord)
        else:
            flattened.append(f"{current_entity}_{coord}")
    return flattened


def _read_metrica_csv(path: Path, require_xy_pairs: bool = False) -> pd.DataFrame:
    for header in ([0, 1], [0, 1, 2], 0):
        try:
            df = pd.read_csv(path, header=header)
        except (pd.errors.ParserError, ValueError):
            continue
        df.columns = flatten_metrica_columns(df.columns)
        has_frame_columns = {"Period", "Frame"}.issubset(set(df.columns)) or any(
            column.lower().startswith("period") for column in df.columns
        )
        has_xy_pairs = bool(infer_xy_pairs(list(df.columns)))
        if has_frame_columns and (has_xy_pairs or not require_xy_pairs):
            return df
    raise ValueError(f"Could not parse Metrica CSV headers: {path}")


def _sanitize_agent_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
    return cleaned.lower() or "unknown"


def infer_xy_pairs(columns: list[str]) -> list[tuple[str, str, str]]:
    """Infer wide x/y coordinate pairs from flattened Metrica columns."""

    by_prefix: dict[str, dict[str, str]] = {}
    for column in columns:
        match = re.match(r"(.+?)[_\s.-]+([xy])$", str(column).strip(), flags=re.IGNORECASE)
        if not match:
            continue
        prefix = match.group(1).strip()
        axis = match.group(2).lower()
        by_prefix.setdefault(prefix, {})[axis] = column
    pairs = []
    for prefix, axes in by_prefix.items():
        if "x" in axes and "y" in axes:
            pairs.append((prefix, axes["x"], axes["y"]))
    return pairs


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {re.sub(r"[^a-z0-9]", "", column.lower()): column for column in columns}
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in normalized:
            return normalized[key]
    return None


def metrica_tracking_wide_to_long(
    wide_df: pd.DataFrame,
    team: str,
    match_id: str,
    source_file: str | None = None,
) -> pd.DataFrame:
    """Convert a Metrica wide tracking table to canonical long rows."""

    wide = wide_df.copy()
    if isinstance(wide.columns, pd.MultiIndex):
        wide.columns = flatten_metrica_columns(wide.columns)
    wide.columns = [str(column).strip() for column in wide.columns]
    columns = list(wide.columns)
    period_col = _find_column(columns, ["Period"])
    frame_col = _find_column(columns, ["Frame"])
    time_col = _find_column(columns, ["Time [s]", "Time", "Time_s"])
    if period_col is None or frame_col is None or time_col is None:
        raise ValueError("Metrica tracking data must include Period, Frame, and Time columns.")

    rows: list[pd.DataFrame] = []
    for prefix, x_col, y_col in infer_xy_pairs(columns):
        is_ball = "ball" in prefix.lower()
        raw_x = pd.to_numeric(wide[x_col], errors="coerce")
        raw_y = pd.to_numeric(wide[y_col], errors="coerce")
        x_m, y_m = to_meters_from_normalized(raw_x, raw_y)
        sanitized = _sanitize_agent_label(prefix)
        jersey_match = re.search(r"(\d+)$", prefix)
        jersey_number = int(jersey_match.group(1)) if jersey_match else pd.NA
        if is_ball:
            agent_id = "ball"
            team_id = "ball"
            player_id = pd.NA
            role = "ball"
            agent_type = AGENT_BALL
            jersey_number = pd.NA
        else:
            agent_id = f"{team}_{sanitized}"
            team_id = team
            player_id = agent_id
            role = "player"
            agent_type = AGENT_PLAYER
        rows.append(
            pd.DataFrame(
                {
                    "match_id": match_id,
                    "dataset": DATASET_METRICA,
                    "period": pd.to_numeric(wide[period_col], errors="coerce"),
                    "frame_id": pd.to_numeric(wide[frame_col], errors="coerce"),
                    "time_s": pd.to_numeric(wide[time_col], errors="coerce"),
                    "agent_id": agent_id,
                    "agent_type": agent_type,
                    "team_id": team_id,
                    "player_id": player_id,
                    "jersey_number": jersey_number,
                    "role": role,
                    "x_m": x_m,
                    "y_m": y_m,
                    "z_m": np.nan,
                    "raw_x": raw_x,
                    "raw_y": raw_y,
                    "is_visible": raw_x.notna() & raw_y.notna(),
                    "source_file": source_file,
                }
            )
        )

    if not rows:
        warnings.warn("No Metrica x/y coordinate pairs were inferred.", stacklevel=2)
        return canonical_tracking_frame(pd.DataFrame())
    return canonical_tracking_frame(pd.concat(rows, ignore_index=True))


def _scale_metrica_coordinate(value: object, axis: str) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return float("nan")
    scale = 105.0 if axis == "x" else 68.0
    return float(numeric) * scale


class MetricaAdapter(TrackingDataAdapter):
    """Adapter for Metrica Sports sample CSV tracking and event files."""

    dataset = DATASET_METRICA

    def discover_files(self) -> dict[str, Path | None]:
        csv_files = list(self.raw_dir.rglob("*.csv")) if self.raw_dir.exists() else []
        files: dict[str, Path | None] = {"home": None, "away": None, "events": None}
        for path in csv_files:
            name = path.name.lower()
            if "tracking" in name and "home" in name and files["home"] is None:
                files["home"] = path
            elif "tracking" in name and "away" in name and files["away"] is None:
                files["away"] = path
            elif "event" in name and files["events"] is None:
                files["events"] = path
        return files

    def load_tracking(self) -> pd.DataFrame:
        files = self.discover_files()
        rows = []
        for team in ["home", "away"]:
            path = files[team]
            if path is None:
                warnings.warn(
                    f"No Metrica {team} tracking file found in {self.raw_dir}.",
                    stacklevel=2,
                )
                continue
            wide = _read_metrica_csv(path, require_xy_pairs=True)
            rows.append(
                metrica_tracking_wide_to_long(
                    wide,
                    team=team,
                    match_id=self.match_id,
                    source_file=str(path),
                )
            )
        if not rows:
            raise FileNotFoundError(
                "No Metrica tracking CSVs found. Place sample-data CSV files under "
                f"{self.raw_dir} or pass --raw-dir to the directory containing them."
            )
        out = pd.concat(rows, ignore_index=True)
        out = out.drop_duplicates(["match_id", "period", "frame_id", "time_s", "agent_id"])
        return canonical_tracking_frame(out)

    def load_events(self) -> pd.DataFrame:
        files = self.discover_files()
        path = files["events"]
        if path is None:
            warnings.warn(f"No Metrica event file found in {self.raw_dir}.", stacklevel=2)
            return canonical_event_frame(pd.DataFrame())

        events = pd.read_csv(path)
        events.columns = [str(column).strip() for column in events.columns]
        columns = list(events.columns)
        period_col = _find_column(columns, ["Period"])
        time_col = _find_column(columns, ["Start Time [s]", "Start Time", "Time [s]", "Time"])
        frame_col = _find_column(columns, ["Start Frame", "Frame"])
        team_col = _find_column(columns, ["Team"])
        player_col = _find_column(columns, ["From", "Player"])
        type_col = _find_column(columns, ["Type"])
        subtype_col = _find_column(columns, ["Subtype", "Sub Type"])
        outcome_col = _find_column(columns, ["Outcome"])
        start_x = _find_column(columns, ["Start X", "Start_x", "x"])
        start_y = _find_column(columns, ["Start Y", "Start_y", "y"])
        end_x = _find_column(columns, ["End X", "End_x"])
        end_y = _find_column(columns, ["End Y", "End_y"])

        records = []
        for _, row in events.iterrows():
            records.append(
                {
                    "match_id": self.match_id,
                    "dataset": DATASET_METRICA,
                    "period": row.get(period_col) if period_col else pd.NA,
                    "time_s": row.get(time_col) if time_col else pd.NA,
                    "frame_id": row.get(frame_col) if frame_col else pd.NA,
                    "team_id": str(row.get(team_col, "")).lower() if team_col else pd.NA,
                    "player_id": row.get(player_col) if player_col else pd.NA,
                    "event_type": row.get(type_col) if type_col else pd.NA,
                    "event_subtype": row.get(subtype_col) if subtype_col else pd.NA,
                    "x_m": _scale_metrica_coordinate(row.get(start_x), "x") if start_x else np.nan,
                    "y_m": _scale_metrica_coordinate(row.get(start_y), "y") if start_y else np.nan,
                    "end_x_m": _scale_metrica_coordinate(row.get(end_x), "x") if end_x else np.nan,
                    "end_y_m": _scale_metrica_coordinate(row.get(end_y), "y") if end_y else np.nan,
                    "outcome": row.get(outcome_col) if outcome_col else pd.NA,
                    "raw_event": json.dumps(row.to_dict(), default=str),
                }
            )
        return canonical_event_frame(pd.DataFrame(records))
