"""SkillCorner Open Data adapter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from footballq.constants import (
    AGENT_BALL,
    AGENT_PLAYER,
    DATASET_SKILLCORNER,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
)
from footballq.io.base import TrackingDataAdapter
from footballq.schema import canonical_tracking_frame


class SkillCornerAdapter(TrackingDataAdapter):
    """Minimal adapter for locally present SkillCorner Open Data JSONL files.

    SkillCorner tracking is documented as frame-level JSONL at 10 fps with `ball_data`,
    `player_data`, `frame`, `timestamp`, and `period`. Coordinates are in meters around the
    pitch center. This Phase 1 loader converts a small obvious subset into `footballq`'s
    top-left-origin canonical schema. Dynamic-event ingestion is left for later phases.
    """

    dataset = DATASET_SKILLCORNER

    def _tracking_files(self) -> list[Path]:
        if not self.raw_dir.exists():
            return []
        return sorted(self.raw_dir.rglob("*tracking*.jsonl"))

    @staticmethod
    def _center_to_canonical(x: object, y: object) -> tuple[float, float]:
        x_num = pd.to_numeric(x, errors="coerce")
        y_num = pd.to_numeric(y, errors="coerce")
        if pd.isna(x_num) or pd.isna(y_num):
            return float("nan"), float("nan")
        return float(x_num) + PITCH_LENGTH_M / 2.0, PITCH_WIDTH_M / 2.0 - float(y_num)

    def load_tracking(self) -> pd.DataFrame:
        files = self._tracking_files()
        if not files:
            raise NotImplementedError(
                "No SkillCorner tracking JSONL files found. Place opendata match folders under "
                f"{self.raw_dir}; expected files like *_tracking_extrapolated.jsonl."
            )

        rows = []
        path = files[0]
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                frame = json.loads(line)
                frame_id = frame.get("frame")
                time_s = frame.get("timestamp")
                period = frame.get("period")
                ball = frame.get("ball_data") or {}
                if ball:
                    x_m, y_m = self._center_to_canonical(ball.get("x"), ball.get("y"))
                    rows.append(
                        {
                            "match_id": self.match_id,
                            "dataset": DATASET_SKILLCORNER,
                            "period": period,
                            "frame_id": frame_id,
                            "time_s": time_s,
                            "agent_id": "ball",
                            "agent_type": AGENT_BALL,
                            "team_id": "ball",
                            "player_id": pd.NA,
                            "jersey_number": pd.NA,
                            "role": "ball",
                            "x_m": x_m,
                            "y_m": y_m,
                            "z_m": np.nan,
                            "raw_x": ball.get("x"),
                            "raw_y": ball.get("y"),
                            "is_visible": ball.get("is_detected", True),
                            "source_file": str(path),
                        }
                    )
                for player in frame.get("player_data") or []:
                    x_m, y_m = self._center_to_canonical(player.get("x"), player.get("y"))
                    player_id = str(player.get("player_id"))
                    team_id = player.get("group") or player.get("team") or pd.NA
                    rows.append(
                        {
                            "match_id": self.match_id,
                            "dataset": DATASET_SKILLCORNER,
                            "period": period,
                            "frame_id": frame_id,
                            "time_s": time_s,
                            "agent_id": player_id,
                            "agent_type": AGENT_PLAYER,
                            "team_id": team_id,
                            "player_id": player_id,
                            "jersey_number": player.get("number"),
                            "role": "player",
                            "x_m": x_m,
                            "y_m": y_m,
                            "z_m": np.nan,
                            "raw_x": player.get("x"),
                            "raw_y": player.get("y"),
                            "is_visible": player.get("is_detected", True),
                            "source_file": str(path),
                        }
                    )
        return canonical_tracking_frame(pd.DataFrame(rows))

