"""Base dataset adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class TrackingDataAdapter(ABC):
    """Abstract interface for providers that can expose canonical tables."""

    dataset: str

    def __init__(self, raw_dir: str | Path, match_id: str) -> None:
        self.raw_dir = Path(raw_dir)
        self.match_id = match_id

    @abstractmethod
    def load_tracking(self) -> pd.DataFrame:
        """Load canonical tracking rows."""

    def load_events(self) -> pd.DataFrame:
        """Load canonical event rows if available."""

        return pd.DataFrame()

    def write_outputs(self, out_dir: str | Path) -> dict[str, Path]:
        """Write canonical parquet outputs."""

        output = Path(out_dir)
        output.mkdir(parents=True, exist_ok=True)
        tracking = self.load_tracking()
        events = self.load_events()
        tracking_path = output / "tracking.parquet"
        events_path = output / "events.parquet"
        tracking.to_parquet(tracking_path, index=False)
        paths = {"tracking": tracking_path}
        if not events.empty:
            events.to_parquet(events_path, index=False)
            paths["events"] = events_path
        return paths

