"""IDSSE / Sportec tracking adapter.

The public IDSSE Bundesliga files have appeared in a few tabular exports. This
adapter intentionally accepts a conservative long-form subset and normalizes the
column names into footballq's canonical table. It raises a clear error when a
local file uses an unsupported layout instead of guessing silently.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from footballq.constants import AGENT_BALL, AGENT_PLAYER
from footballq.io.base import TrackingDataAdapter
from footballq.schema import canonical_tracking_frame


class IDSSEAdapter(TrackingDataAdapter):
    """Load locally downloaded IDSSE/Sportec long-form tracking files."""

    dataset = "idsse"

    def _candidate_files(self) -> list[Path]:
        if self.raw_dir.is_file():
            return [self.raw_dir]
        if not self.raw_dir.exists():
            return []
        return sorted(
            [
                *self.raw_dir.rglob("*.parquet"),
                *self.raw_dir.rglob("*.csv"),
            ]
        )

    @staticmethod
    def _find(columns: list[str], candidates: list[str]) -> str | None:
        normalized = {column.lower().replace(" ", "_"): column for column in columns}
        for candidate in candidates:
            key = candidate.lower().replace(" ", "_")
            if key in normalized:
                return normalized[key]
        return None

    def _read_file(self, path: Path) -> pd.DataFrame:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)

    def load_tracking(self) -> pd.DataFrame:
        files = self._candidate_files()
        if not files:
            raise FileNotFoundError(
                "No IDSSE/Sportec CSV or parquet files found. Pass --raw to a long-form "
                "tracking file or a directory containing one."
            )

        frames: list[pd.DataFrame] = []
        for path in files:
            raw = self._read_file(path)
            raw.columns = [str(column).strip() for column in raw.columns]
            columns = list(raw.columns)
            frame_col = self._find(columns, ["frame_id", "frame", "frame_number"])
            time_col = self._find(columns, ["time_s", "time", "timestamp", "seconds"])
            entity_col = self._find(
                columns,
                ["entity_id", "agent_id", "object_id", "track_id", "player_id"],
            )
            type_col = self._find(columns, ["entity_type", "agent_type", "object_type"])
            team_col = self._find(columns, ["team_id", "team", "side"])
            x_col = self._find(columns, ["x_m", "x", "x_position"])
            y_col = self._find(columns, ["y_m", "y", "y_position"])
            if not all([frame_col, time_col, entity_col, x_col, y_col]):
                continue

            entity_type = (
                raw[type_col].astype(str).str.lower()
                if type_col
                else raw[entity_col].astype(str).str.contains("ball", case=False).map(
                    {True: AGENT_BALL, False: AGENT_PLAYER}
                )
            )
            team = raw[team_col] if team_col else pd.NA
            match = raw.get("match_id", self.match_id)
            period = raw.get("period", 1)
            x = pd.to_numeric(raw[x_col], errors="coerce")
            y = pd.to_numeric(raw[y_col], errors="coerce")
            out = pd.DataFrame(
                {
                    "match_id": match,
                    "dataset": self.dataset,
                    "period": period,
                    "frame_id": pd.to_numeric(raw[frame_col], errors="coerce"),
                    "time_s": pd.to_numeric(raw[time_col], errors="coerce"),
                    "agent_id": raw[entity_col].astype(str),
                    "agent_type": entity_type.where(entity_type.eq(AGENT_BALL), AGENT_PLAYER),
                    "team_id": team,
                    "player_id": raw[entity_col].where(~entity_type.eq(AGENT_BALL), pd.NA),
                    "jersey_number": raw.get("jersey_number", pd.NA),
                    "role": raw.get("role", pd.NA),
                    "x_m": x,
                    "y_m": y,
                    "z_m": raw.get("z_m", np.nan),
                    "raw_x": x,
                    "raw_y": y,
                    "is_visible": x.notna() & y.notna(),
                    "source_file": str(path),
                }
            )
            frames.append(out)

        if not frames:
            raise ValueError(
                "IDSSE/Sportec files were found, but none matched the supported long-form "
                "columns: frame/time/entity/x/y."
            )
        return canonical_tracking_frame(pd.concat(frames, ignore_index=True))
