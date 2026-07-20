"""Google Research Football raw-observation adapter."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from footballq.constants import (
    AGENT_BALL,
    AGENT_PLAYER,
    DATASET_GFOOTBALL,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
)
from footballq.io.base import TrackingDataAdapter
from footballq.schema import canonical_tracking_frame

GRF_X_MIN = -1.0
GRF_X_MAX = 1.0
GRF_Y_MIN = -0.42
GRF_Y_MAX = 0.42

ROLE_NAMES = {
    0: "goalkeeper",
    1: "center_back",
    2: "left_back",
    3: "right_back",
    4: "defensive_midfield",
    5: "central_midfield",
    6: "left_midfield",
    7: "right_midfield",
    8: "attacking_midfield",
    9: "center_forward",
}


def gfootball_xy_to_meters(x: object, y: object) -> tuple[float, float]:
    """Convert GRF normalized pitch coordinates to footballq canonical meters."""

    x_num = pd.to_numeric(x, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    if pd.isna(x_num) or pd.isna(y_num):
        return float("nan"), float("nan")
    x_m = (float(x_num) - GRF_X_MIN) / (GRF_X_MAX - GRF_X_MIN) * PITCH_LENGTH_M
    y_m = (float(y_num) - GRF_Y_MIN) / (GRF_Y_MAX - GRF_Y_MIN) * PITCH_WIDTH_M
    return x_m, y_m


def gfootball_direction_to_mps(dx: object, dy: object, fps: float) -> tuple[float, float]:
    """Convert GRF per-step movement vectors to meters per second."""

    dx_num = pd.to_numeric(dx, errors="coerce")
    dy_num = pd.to_numeric(dy, errors="coerce")
    if pd.isna(dx_num) or pd.isna(dy_num):
        return float("nan"), float("nan")
    vx_mps = float(dx_num) / (GRF_X_MAX - GRF_X_MIN) * PITCH_LENGTH_M * fps
    vy_mps = float(dy_num) / (GRF_Y_MAX - GRF_Y_MIN) * PITCH_WIDTH_M * fps
    return vx_mps, vy_mps


def _records_from_path(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            yield from payload
            return
        if isinstance(payload, dict):
            for key in ("frames", "observations", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    yield from value
                    return
            yield payload
            return
    raise ValueError(f"Unsupported GFootball observation file: {path}")


def _observation_files(raw_dir: Path) -> list[Path]:
    if raw_dir.is_file():
        return [raw_dir]
    if not raw_dir.exists():
        return []
    return sorted([*raw_dir.rglob("*.jsonl"), *raw_dir.rglob("*.json")])


def _extract_observation(record: Any) -> dict[str, Any]:
    value = record
    if isinstance(value, list):
        if not value:
            raise ValueError("GFootball observation list is empty.")
        value = value[0]
    if not isinstance(value, dict):
        raise ValueError("GFootball observation records must be dictionaries.")
    for key in ("observation", "obs", "state"):
        nested = value.get(key)
        if nested is not None:
            return _extract_observation(nested)
    return value


def _as_xy_array(value: object, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    try:
        return arr.reshape(-1, 2)
    except ValueError as exc:
        raise ValueError(f"GFootball field {name!r} must contain x/y pairs.") from exc


def _as_vector(value: object, *, length: int | None = None, default: float = 0.0) -> np.ndarray:
    if value is None:
        if length is None:
            return np.asarray([], dtype=np.float32)
        return np.full(length, default, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if length is not None and len(arr) < length:
        arr = np.pad(arr, (0, length - len(arr)), constant_values=default)
    return arr


def _record_match_id(base_match_id: str, record: dict[str, Any]) -> str:
    explicit = record.get("match_id")
    if explicit not in (None, ""):
        return str(explicit)
    episode_id = record.get("episode_id", record.get("episode"))
    if episode_id in (None, ""):
        return base_match_id
    return f"{base_match_id}_episode_{episode_id}"


def _record_period(record: dict[str, Any]) -> int:
    value = record.get("period", 1)
    numeric = pd.to_numeric(value, errors="coerce")
    return int(numeric) if pd.notna(numeric) else 1


def _record_frame_id(
    record: dict[str, Any],
    match_id: str,
    period: int,
    counters: defaultdict[tuple[str, int], int],
) -> int:
    for key in ("frame_id", "frame", "step"):
        value = record.get(key)
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.notna(numeric):
            frame_id = int(numeric)
            counters[(match_id, period)] = max(counters[(match_id, period)], frame_id + 1)
            return frame_id
    frame_id = counters[(match_id, period)]
    counters[(match_id, period)] += 1
    return frame_id


def _record_time_s(record: dict[str, Any], frame_id: int, fps: float) -> float:
    for key in ("time_s", "timestamp"):
        value = record.get(key)
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.notna(numeric):
            return float(numeric)
    return float(frame_id) / float(fps)


def _int_value(value: object, default: int) -> int:
    numeric = pd.to_numeric(value, errors="coerce")
    return int(numeric) if pd.notna(numeric) else int(default)


def _float_value(value: object, default: float) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    return float(numeric) if pd.notna(numeric) else float(default)


def _role_name(roles: np.ndarray, idx: int) -> str:
    if idx >= len(roles) or not np.isfinite(roles[idx]):
        return "player"
    return ROLE_NAMES.get(int(roles[idx]), f"role_{int(roles[idx])}")


def _team_rows(
    *,
    obs: dict[str, Any],
    record: dict[str, Any],
    team_key: str,
    direction_key: str,
    active_key: str,
    roles_key: str,
    team_id: str,
    match_id: str,
    period: int,
    frame_id: int,
    time_s: float,
    fps: float,
    source_file: str | None,
) -> list[dict[str, Any]]:
    positions = _as_xy_array(obs.get(team_key, []), name=team_key)
    directions = _as_xy_array(obs.get(direction_key, np.zeros_like(positions)), name=direction_key)
    if len(directions) < len(positions):
        directions = np.vstack(
            [directions, np.zeros((len(positions) - len(directions), 2), dtype=np.float32)]
        )
    active_flags = _as_vector(obs.get(active_key), length=len(positions), default=1.0).astype(bool)
    roles = _as_vector(obs.get(roles_key), length=len(positions), default=np.nan)
    ball_owned_team = _int_value(obs.get("ball_owned_team", -1), -1)
    ball_owned_player = _int_value(obs.get("ball_owned_player", -1), -1)
    owner_team_id = {"home": 0, "away": 1}[team_id]
    possession_team_id = (
        "home" if ball_owned_team == 0 else "away" if ball_owned_team == 1 else "neutral"
    )
    score = _as_vector(obs.get("score"), length=2, default=0.0)
    steps_left = _int_value(obs.get("steps_left", -1), -1)

    rows = []
    for idx, xy in enumerate(positions):
        x_m, y_m = gfootball_xy_to_meters(xy[0], xy[1])
        vx_mps, vy_mps = gfootball_direction_to_mps(directions[idx][0], directions[idx][1], fps)
        visible = bool(active_flags[idx]) and np.isfinite(x_m) and np.isfinite(y_m)
        has_possession = ball_owned_team == owner_team_id and ball_owned_player == idx
        player_id = f"{team_id}_{idx:02d}"
        rows.append(
            {
                "match_id": match_id,
                "dataset": DATASET_GFOOTBALL,
                "period": period,
                "frame_id": frame_id,
                "time_s": time_s,
                "fps": fps,
                "agent_id": player_id,
                "agent_type": AGENT_PLAYER,
                "team_id": team_id,
                "player_id": player_id,
                "jersey_number": idx + 1,
                "role": _role_name(roles, idx),
                "x_m": x_m,
                "y_m": y_m,
                "vx_mps": vx_mps,
                "vy_mps": vy_mps,
                "z_m": np.nan,
                "raw_x": float(xy[0]),
                "raw_y": float(xy[1]),
                "is_visible": visible,
                "visible": visible,
                "has_possession": has_possession,
                "possession_team_id": possession_team_id,
                "game_mode": obs.get("game_mode"),
                "score_home": int(score[0]),
                "score_away": int(score[1]),
                "steps_left": steps_left,
                "source_file": source_file,
                "provider": record.get("provider", "google_research_football"),
            }
        )
    return rows


def observations_to_tracking(
    records: Iterable[dict[str, Any]],
    *,
    match_id: str = "gfootball",
    fps: float = 10.0,
    source_file: str | None = None,
) -> pd.DataFrame:
    """Convert serializable GRF raw observations to canonical tracking rows.

    Expected input is one observation per frame, either as a raw observation dict
    or as ``{"episode_id": ..., "frame_id": ..., "observation": raw_obs}``.
    """

    rows: list[dict[str, Any]] = []
    counters: defaultdict[tuple[str, int], int] = defaultdict(int)
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("GFootball records must be dictionaries.")
        obs = _extract_observation(record)
        frame_fps = _float_value(record.get("fps", fps), fps)
        frame_match_id = _record_match_id(match_id, record)
        period = _record_period(record)
        frame_id = _record_frame_id(record, frame_match_id, period, counters)
        time_s = _record_time_s(record, frame_id, frame_fps)
        ball = _as_vector(obs.get("ball"), length=3, default=np.nan)
        ball_direction = _as_vector(obs.get("ball_direction"), length=3, default=0.0)
        ball_x_m, ball_y_m = gfootball_xy_to_meters(ball[0], ball[1])
        ball_vx_mps, ball_vy_mps = gfootball_direction_to_mps(
            ball_direction[0], ball_direction[1], frame_fps
        )
        ball_owned_team = _int_value(obs.get("ball_owned_team", -1), -1)
        possession_team_id = (
            "home" if ball_owned_team == 0 else "away" if ball_owned_team == 1 else "neutral"
        )
        score = _as_vector(obs.get("score"), length=2, default=0.0)
        steps_left = _int_value(obs.get("steps_left", -1), -1)
        rows.append(
            {
                "match_id": frame_match_id,
                "dataset": DATASET_GFOOTBALL,
                "period": period,
                "frame_id": frame_id,
                "time_s": time_s,
                "fps": frame_fps,
                "agent_id": "ball",
                "agent_type": AGENT_BALL,
                "team_id": "neutral",
                "player_id": pd.NA,
                "jersey_number": pd.NA,
                "role": "ball",
                "x_m": ball_x_m,
                "y_m": ball_y_m,
                "vx_mps": ball_vx_mps,
                "vy_mps": ball_vy_mps,
                "z_m": np.nan,
                "raw_x": float(ball[0]) if np.isfinite(ball[0]) else np.nan,
                "raw_y": float(ball[1]) if np.isfinite(ball[1]) else np.nan,
                "is_visible": np.isfinite(ball_x_m) and np.isfinite(ball_y_m),
                "visible": np.isfinite(ball_x_m) and np.isfinite(ball_y_m),
                "has_possession": False,
                "possession_team_id": possession_team_id,
                "game_mode": obs.get("game_mode"),
                "score_home": int(score[0]),
                "score_away": int(score[1]),
                "steps_left": steps_left,
                "source_file": source_file,
                "provider": record.get("provider", "google_research_football"),
            }
        )
        rows.extend(
            _team_rows(
                obs=obs,
                record=record,
                team_key="left_team",
                direction_key="left_team_direction",
                active_key="left_team_active",
                roles_key="left_team_roles",
                team_id="home",
                match_id=frame_match_id,
                period=period,
                frame_id=frame_id,
                time_s=time_s,
                fps=frame_fps,
                source_file=source_file,
            )
        )
        rows.extend(
            _team_rows(
                obs=obs,
                record=record,
                team_key="right_team",
                direction_key="right_team_direction",
                active_key="right_team_active",
                roles_key="right_team_roles",
                team_id="away",
                match_id=frame_match_id,
                period=period,
                frame_id=frame_id,
                time_s=time_s,
                fps=frame_fps,
                source_file=source_file,
            )
        )
    if not rows:
        raise ValueError("No GFootball observations were found.")
    return canonical_tracking_frame(pd.DataFrame(rows))


class GFootballAdapter(TrackingDataAdapter):
    """Adapter for saved Google Research Football raw observation JSON/JSONL files."""

    dataset = DATASET_GFOOTBALL

    def __init__(self, raw_dir: str | Path, match_id: str = "gfootball", fps: float = 10.0) -> None:
        super().__init__(raw_dir=raw_dir, match_id=match_id)
        self.fps = float(fps)

    def load_tracking(self) -> pd.DataFrame:
        files = _observation_files(self.raw_dir)
        if not files:
            raise FileNotFoundError(
                "No GFootball observation JSON/JSONL files found. Run "
                "scripts/collect_gfootball_tracking.py first or pass a saved observation file."
            )
        frames = []
        for path in files:
            file_match_id = self.match_id
            if len(files) > 1:
                relative = path.relative_to(self.raw_dir).with_suffix("")
                shard_name = "__".join(relative.parts)
                file_match_id = f"{self.match_id}_{shard_name}"
            frames.append(
                observations_to_tracking(
                    _records_from_path(path),
                    match_id=file_match_id,
                    fps=self.fps,
                    source_file=str(path),
                )
            )
        return canonical_tracking_frame(pd.concat(frames, ignore_index=True))
