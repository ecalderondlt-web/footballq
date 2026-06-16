"""StatsBomb Open Data event adapter."""

from __future__ import annotations

import json
from pathlib import Path

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

    def _event_files(self) -> list[Path]:
        if not self.raw_dir.exists():
            return []
        event_dir = self.raw_dir / "data" / "events"
        if event_dir.exists():
            return sorted(event_dir.glob("*.json"))
        return sorted(self.raw_dir.rglob("events/*.json"))

    @staticmethod
    def _sb_location_to_meters(location: object) -> tuple[float, float]:
        if not isinstance(location, list) or len(location) < 2:
            return float("nan"), float("nan")
        return float(location[0]) / 120.0 * 105.0, float(location[1]) / 80.0 * 68.0

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
                end_location = None
                for payload in event.values():
                    if isinstance(payload, dict) and "end_location" in payload:
                        end_location = payload.get("end_location")
                        break
                end_x_m, end_y_m = self._sb_location_to_meters(end_location)
                minute = event.get("minute") or 0
                second = event.get("second") or 0
                records.append(
                    {
                        "match_id": match_id or self.match_id,
                        "dataset": DATASET_STATSBOMB,
                        "period": event.get("period"),
                        "time_s": float(minute) * 60.0 + float(second),
                        "frame_id": np.nan,
                        "team_id": (event.get("team") or {}).get("name"),
                        "player_id": (event.get("player") or {}).get("id"),
                        "event_type": (event.get("type") or {}).get("name"),
                        "event_subtype": pd.NA,
                        "x_m": x_m,
                        "y_m": y_m,
                        "end_x_m": end_x_m,
                        "end_y_m": end_y_m,
                        "outcome": pd.NA,
                        "raw_event": json.dumps(event, default=str),
                    }
                )
        return canonical_event_frame(pd.DataFrame(records))

