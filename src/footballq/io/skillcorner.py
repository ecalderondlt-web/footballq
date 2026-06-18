"""SkillCorner Open Data adapter."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

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
    """Adapter for locally present SkillCorner Open Data tracking files.

    The public files commonly expose frame-level JSONL with ball and player
    entries. This loader accepts the common ``ball_data``/``player_data`` shape,
    plus small variants seen in examples, and converts centered meter coordinates
    into footballq's top-left-origin canonical pitch coordinates.
    """

    dataset = DATASET_SKILLCORNER

    def _tracking_files(self) -> list[Path]:
        if not self.raw_dir.exists():
            return []
        if self.raw_dir.is_file():
            return [self.raw_dir]
        candidates = [
            path
            for path in [*self.raw_dir.rglob("*.jsonl"), *self.raw_dir.rglob("*.json")]
            if "tracking" in path.name.lower()
        ]
        return sorted(candidates)

    @staticmethod
    def _first(mapping: dict[str, Any], keys: list[str], default: Any = None) -> Any:
        for key in keys:
            if key in mapping and mapping[key] is not None:
                return mapping[key]
        return default

    @staticmethod
    def _normalize_team(value: object) -> object:
        text = str(value).strip().lower()
        if not text or text in {"nan", "<na>", "none", "null"}:
            return pd.NA
        text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
        if text in {"home", "h", "home_team", "team_home", "1", "team_1"}:
            return "home"
        if text in {"away", "a", "away_team", "team_away", "2", "team_2"}:
            return "away"
        return text

    @staticmethod
    def _center_to_canonical(x: object, y: object) -> tuple[float, float]:
        x_num = pd.to_numeric(x, errors="coerce")
        y_num = pd.to_numeric(y, errors="coerce")
        if pd.isna(x_num) or pd.isna(y_num):
            return float("nan"), float("nan")
        x_float = float(x_num)
        y_float = float(y_num)
        return x_float + PITCH_LENGTH_M / 2.0, PITCH_WIDTH_M / 2.0 - y_float

    @staticmethod
    def _timestamp_to_seconds(value: object) -> float | None:
        if value is None or pd.isna(value):
            return None
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.notna(numeric):
            return float(numeric)
        text = str(value).strip()
        match = re.match(r"^(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)$", text)
        if not match:
            return None
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2))
        seconds = float(match.group(3))
        return hours * 3600.0 + minutes * 60.0 + seconds

    @staticmethod
    def _iter_json_records(path: Path) -> Iterable[dict[str, Any]]:
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8-sig") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)
            return

        text = path.read_text(encoding="utf-8-sig").strip()
        if not text:
            return
        payload = json.loads(text)
        if isinstance(payload, list):
            yield from payload
            return
        if isinstance(payload, dict):
            for key in ["frames", "tracking", "data"]:
                value = payload.get(key)
                if isinstance(value, list):
                    yield from value
                    return
            yield payload
            return
        raise ValueError(f"Unsupported SkillCorner JSON payload in {path}")

    @staticmethod
    def _players_from_frame(frame: dict[str, Any]) -> list[dict[str, Any]]:
        for key in ["player_data", "players"]:
            value = frame.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        data = frame.get("data")
        if isinstance(data, list):
            players = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                label = str(
                    item.get("object_type")
                    or item.get("type")
                    or item.get("trackable_object_type")
                    or ""
                ).lower()
                if "ball" not in label:
                    players.append(item)
            return players
        return []

    @staticmethod
    def _ball_from_frame(frame: dict[str, Any]) -> dict[str, Any]:
        for key in ["ball_data", "ball"]:
            value = frame.get(key)
            if isinstance(value, dict):
                return value
        data = frame.get("data")
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                label = str(
                    item.get("object_type")
                    or item.get("type")
                    or item.get("trackable_object_type")
                    or ""
                ).lower()
                if "ball" in label:
                    return item
        return {}

    def _match_id_for_path(self, path: Path, multiple_files: bool) -> str:
        if not multiple_files:
            return self.match_id
        parent = path.parent.name
        if parent and parent != self.raw_dir.name:
            return parent
        return path.stem

    def _load_match_metadata(self, tracking_path: Path) -> dict[str, dict[str, Any]]:
        candidates = sorted(tracking_path.parent.glob("*match*.json"))
        if not candidates:
            return {}
        try:
            payload = json.loads(candidates[0].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        home_team_id = payload.get("home_team", {}).get("id")
        away_team_id = payload.get("away_team", {}).get("id")
        players: dict[str, dict[str, Any]] = {}
        for player in payload.get("players") or []:
            team = pd.NA
            if player.get("team_id") == home_team_id:
                team = "home"
            elif player.get("team_id") == away_team_id:
                team = "away"
            role = player.get("player_role") or {}
            record = {
                "team_id": team,
                "jersey_number": player.get("number"),
                "role": role.get("name") or role.get("acronym") or "player",
            }
            for key in ["id", "trackable_object", "team_player_id"]:
                value = player.get(key)
                if value is not None:
                    players[str(value)] = record
        return players

    def _frame_time_s(self, frame: dict[str, Any], fps: float) -> float | Any:
        time_s = self._timestamp_to_seconds(
            self._first(frame, ["time_s", "timestamp", "time", "game_clock"])
        )
        if time_s is not None:
            return time_s
        frame_id = pd.to_numeric(
            self._first(frame, ["frame_id", "frame", "index"]),
            errors="coerce",
        )
        if pd.notna(frame_id):
            return float(frame_id) / fps
        return pd.NA

    def load_tracking(self) -> pd.DataFrame:
        files = self._tracking_files()
        if not files:
            raise FileNotFoundError(
                "No SkillCorner tracking JSON/JSONL files found. Place Open Data match folders "
                f"under {self.raw_dir}; expected files with 'tracking' in the filename, such as "
                "*_tracking_extrapolated.jsonl."
            )

        rows = []
        multiple_files = len(files) > 1
        for path in files:
            match_id = self._match_id_for_path(path, multiple_files)
            player_metadata = self._load_match_metadata(path)
            records = self._iter_json_records(path)
            for frame in records:
                fps_value = pd.to_numeric(frame.get("fps", 10.0), errors="coerce")
                fps = float(fps_value) if pd.notna(fps_value) else 10.0
                frame_id = self._first(frame, ["frame_id", "frame", "index"])
                time_s = self._frame_time_s(frame, fps=fps)
                period = self._first(frame, ["period", "period_id"], 1)
                possession_team = self._normalize_team(
                    self._first(frame, ["possession_team_id", "possession_team", "possession"])
                )
                event_type = self._first(frame, ["event_type", "event"])
                phase = self._first(frame, ["phase"])
                ball = self._ball_from_frame(frame)
                if ball:
                    x_m, y_m = self._center_to_canonical(
                        self._first(ball, ["x", "x_m"]),
                        self._first(ball, ["y", "y_m"]),
                    )
                    rows.append(
                        {
                            "match_id": match_id,
                            "dataset": DATASET_SKILLCORNER,
                            "period": period,
                            "frame_id": frame_id,
                            "time_s": time_s,
                            "fps": fps,
                            "entity_id": "ball",
                            "entity_type": AGENT_BALL,
                            "agent_id": "ball",
                            "agent_type": AGENT_BALL,
                            "team_id": "neutral",
                            "player_id": pd.NA,
                            "jersey_number": pd.NA,
                            "role": "ball",
                            "x_m": x_m,
                            "y_m": y_m,
                            "vx_mps": np.nan,
                            "vy_mps": np.nan,
                            "z_m": np.nan,
                            "raw_x": self._first(ball, ["x", "x_m"]),
                            "raw_y": self._first(ball, ["y", "y_m"]),
                            "is_visible": self._first(
                                ball,
                                ["is_detected", "is_visible", "visible"],
                                True,
                            ),
                            "visible": self._first(
                                ball,
                                ["is_detected", "is_visible", "visible"],
                                True,
                            ),
                            "has_possession": False,
                            "possession_team_id": possession_team,
                            "phase": phase,
                            "event_type": event_type,
                            "source_file": str(path),
                        }
                    )

                for player in self._players_from_frame(frame):
                    player_id = str(
                        self._first(
                            player,
                            ["player_id", "trackable_object", "trackable_object_id", "id"],
                        )
                    )
                    metadata = player_metadata.get(player_id, {})
                    team_id = self._normalize_team(
                        self._first(
                            player,
                            ["group", "team", "team_id", "side", "group_name"],
                            metadata.get("team_id"),
                        )
                    )
                    x_m, y_m = self._center_to_canonical(
                        self._first(player, ["x", "x_m"]),
                        self._first(player, ["y", "y_m"]),
                    )
                    visible = self._first(player, ["is_detected", "is_visible", "visible"], True)
                    rows.append(
                        {
                            "match_id": match_id,
                            "dataset": DATASET_SKILLCORNER,
                            "period": period,
                            "frame_id": frame_id,
                            "time_s": time_s,
                            "fps": fps,
                            "entity_id": player_id,
                            "entity_type": AGENT_PLAYER,
                            "agent_id": player_id,
                            "agent_type": AGENT_PLAYER,
                            "team_id": team_id,
                            "player_id": player_id,
                            "jersey_number": self._first(
                                player,
                                ["number", "jersey_number"],
                                metadata.get("jersey_number"),
                            ),
                            "role": self._first(
                                player,
                                ["role", "position"],
                                metadata.get("role", "player"),
                            ),
                            "x_m": x_m,
                            "y_m": y_m,
                            "vx_mps": np.nan,
                            "vy_mps": np.nan,
                            "z_m": np.nan,
                            "raw_x": self._first(player, ["x", "x_m"]),
                            "raw_y": self._first(player, ["y", "y_m"]),
                            "is_visible": visible,
                            "visible": visible,
                            "has_possession": bool(
                                self._first(player, ["has_possession", "in_possession"], False)
                            ),
                            "possession_team_id": possession_team,
                            "phase": phase,
                            "event_type": event_type,
                            "source_file": str(path),
                        }
                    )
        if not rows:
            raise ValueError(
                f"SkillCorner tracking files were found under {self.raw_dir}, but no ball/player "
                "rows could be parsed. Expected frame records with ball_data/player_data or data."
            )
        return canonical_tracking_frame(pd.DataFrame(rows))

