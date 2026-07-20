"""StatsBomb Open Data event adapter."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from footballq.constants import DATASET_STATSBOMB
from footballq.io.base import TrackingDataAdapter
from footballq.schema import canonical_event_frame


class StatsBombAdapter(TrackingDataAdapter):
    """Load StatsBomb events as context, not continuous tracking."""

    dataset = DATASET_STATSBOMB

    def load_tracking(self) -> pd.DataFrame:
        raise NotImplementedError(
            "StatsBomb Open Data does not provide continuous full tracking. Use load_events() "
            "for event context and future freeze-frame joins."
        )

    def _data_dir(self) -> Path | None:
        direct = self.raw_dir
        nested = self.raw_dir / "data"
        for candidate in (direct, nested):
            if (candidate / "events").is_dir():
                return candidate
        return None

    def _match_file(self, directory: str) -> Path | None:
        data_dir = self._data_dir()
        if data_dir is None:
            return None
        path = data_dir / directory / f"{self.match_id}.json"
        return path if path.is_file() else None

    def _event_files(self) -> list[Path]:
        path = self._match_file("events")
        return [path] if path is not None else []

    @staticmethod
    def _sb_location_to_meters(location: object) -> tuple[float, float]:
        if not isinstance(location, list) or len(location) < 2:
            return float("nan"), float("nan")
        return float(location[0]) / 120.0 * 105.0, float(location[1]) / 80.0 * 68.0

    @staticmethod
    def _event_payload(event: dict[str, Any]) -> dict[str, Any]:
        event_name = str((event.get("type") or {}).get("name") or "")
        key = re.sub(r"[^a-z0-9]+", "_", event_name.lower()).strip("_")
        payload = event.get(key)
        if isinstance(payload, dict):
            return payload
        return {}

    @staticmethod
    def _nested_value(payload: dict[str, Any], key: str, field: str) -> object:
        value = payload.get(key)
        return value.get(field) if isinstance(value, dict) else None

    def load_events(self) -> pd.DataFrame:
        files = self._event_files()
        if not files:
            return canonical_event_frame(pd.DataFrame())

        records = []
        for path in files:
            with path.open("r", encoding="utf-8") as handle:
                events = json.load(handle)
            match_id = path.stem
            for event in events:
                x_m, y_m = self._sb_location_to_meters(event.get("location"))
                payload = self._event_payload(event)
                end_location = payload.get("end_location")
                end_x_m, end_y_m = self._sb_location_to_meters(end_location)
                minute = event.get("minute") or 0
                second = event.get("second") or 0
                team = event.get("team") or {}
                player = event.get("player") or {}
                event_type = event.get("type") or {}
                play_pattern = event.get("play_pattern") or {}
                position = event.get("position") or {}
                records.append(
                    {
                        "match_id": match_id or self.match_id,
                        "dataset": DATASET_STATSBOMB,
                        "period": event.get("period"),
                        "time_s": float(minute) * 60.0 + float(second),
                        "frame_id": np.nan,
                        "team_id": team.get("id"),
                        "player_id": player.get("id"),
                        "event_type": event_type.get("name"),
                        "event_subtype": self._nested_value(payload, "type", "name"),
                        "x_m": x_m,
                        "y_m": y_m,
                        "end_x_m": end_x_m,
                        "end_y_m": end_y_m,
                        "outcome": self._nested_value(payload, "outcome", "name"),
                        "raw_event": json.dumps(event, default=str),
                        "event_id": event.get("id"),
                        "event_index": event.get("index"),
                        "timestamp": event.get("timestamp"),
                        "duration_s": event.get("duration"),
                        "event_type_id": event_type.get("id"),
                        "event_subtype_id": self._nested_value(payload, "type", "id"),
                        "outcome_id": self._nested_value(payload, "outcome", "id"),
                        "team_name": team.get("name"),
                        "player_name": player.get("name"),
                        "possession": event.get("possession"),
                        "possession_team_id": (event.get("possession_team") or {}).get("id"),
                        "play_pattern_id": play_pattern.get("id"),
                        "play_pattern": play_pattern.get("name"),
                        "position_id": position.get("id"),
                        "position": position.get("name"),
                        "under_pressure": bool(event.get("under_pressure", False)),
                        "counterpress": bool(event.get("counterpress", False)),
                    }
                )
        return canonical_event_frame(pd.DataFrame(records))

    def load_360(self) -> pd.DataFrame:
        """Load sparse StatsBomb 360 rows for this match without implying continuous tracking."""

        path = self._match_file("three-sixty")
        columns = [
            "match_id",
            "dataset",
            "event_id",
            "freeze_frame_count",
            "visible_area",
            "freeze_frame",
        ]
        if path is None:
            return pd.DataFrame(columns=columns)
        with path.open("r", encoding="utf-8-sig") as handle:
            rows = json.load(handle)
        records = []
        for row in rows:
            freeze_frame = row.get("freeze_frame") or []
            records.append(
                {
                    "match_id": str(self.match_id),
                    "dataset": DATASET_STATSBOMB,
                    "event_id": row.get("event_uuid"),
                    "freeze_frame_count": len(freeze_frame),
                    "visible_area": json.dumps(row.get("visible_area") or []),
                    "freeze_frame": json.dumps(freeze_frame),
                }
            )
        return pd.DataFrame.from_records(records, columns=columns)

