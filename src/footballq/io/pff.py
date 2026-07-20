"""PFF FC broadcast-tracking adapter for the World Cup 2022 delivery."""

from __future__ import annotations

import bz2
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import pandas as pd

from footballq.constants import (
    AGENT_BALL,
    AGENT_PLAYER,
    DATASET_PFF,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
)
from footballq.io.base import TrackingDataAdapter
from footballq.schema import canonical_event_frame, canonical_tracking_frame

PFF_NTSC_FPS = 30_000 / 1_001
_ALL_MATCHES = {"all", "*"}


def pff_match_id(path: str | Path) -> str:
    """Return the game identifier encoded in a PFF tracking filename."""

    name = Path(path).name
    for suffix in (".jsonl.bz2", ".jsonl"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def discover_pff_tracking_files(raw_dir: str | Path) -> dict[str, Path]:
    """Discover one physical tracking file per game, preferring extracted JSONL."""

    root = Path(raw_dir)
    if not root.exists():
        return {}
    if root.is_file():
        return {pff_match_id(root): root}

    candidates = [
        path
        for path in [*root.rglob("*.jsonl"), *root.rglob("*.jsonl.bz2")]
        if path.is_file()
    ]
    by_match: dict[str, list[Path]] = {}
    for path in candidates:
        by_match.setdefault(pff_match_id(path), []).append(path)

    selected: dict[str, Path] = {}
    for match_id, paths in by_match.items():
        selected[match_id] = min(
            paths,
            key=lambda path: (
                path.name.lower().endswith(".bz2"),
                len(path.parts),
                str(path).lower(),
            ),
        )
    return dict(sorted(selected.items(), key=lambda item: int(item[0])))


def _open_text(path: Path) -> TextIO:
    if path.name.lower().endswith(".bz2"):
        return bz2.open(path, mode="rt", encoding="utf-8")
    return path.open(mode="r", encoding="utf-8")


def iter_pff_records(path: str | Path) -> Iterable[dict[str, Any]]:
    """Stream JSON objects from an extracted or bzip2-compressed PFF file."""

    source = Path(path)
    with _open_text(source) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid PFF JSON at {source}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"PFF record at {source}:{line_number} is not an object")
            yield payload


def pff_xy_to_meters(
    x: object,
    y: object,
    *,
    pitch_length_m: float = PITCH_LENGTH_M,
    pitch_width_m: float = PITCH_WIDTH_M,
) -> tuple[float, float]:
    """Convert PFF center-origin camera coordinates to canonical pitch meters."""

    x_num = pd.to_numeric(x, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    if pd.isna(x_num) or pd.isna(y_num):
        return float("nan"), float("nan")
    return float(x_num) + pitch_length_m / 2.0, pitch_width_m / 2.0 - float(y_num)


def _objects(value: object) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _players_for_frame(value: object) -> list[dict[str, Any]]:
    """Keep the first coordinate for each jersey in provider-duplicated arrays."""

    players: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, player in enumerate(_objects(value)):
        jersey = str(player.get("jerseyNum", "")).strip()
        key = jersey or f"missing:{index}"
        if key in seen:
            continue
        seen.add(key)
        players.append(player)
    return players


def _team_for_event(game_event: dict[str, Any]) -> object:
    home_team = game_event.get("home_team")
    if home_team is True or home_team == 1:
        return "home"
    if home_team is False or home_team == 0:
        return "away"
    return pd.NA


def pff_record_to_tracking_rows(
    record: dict[str, Any],
    *,
    source_file: str | Path,
    fallback_match_id: str,
    use_smoothed: bool = True,
    pitch_length_m: float = PITCH_LENGTH_M,
    pitch_width_m: float = PITCH_WIDTH_M,
) -> list[dict[str, Any]]:
    """Convert one deduplicated PFF frame record to canonical row dictionaries."""

    match_id = str(record.get("gameRefId") or fallback_match_id)
    frame_id = int(record.get("frameNum", -1))
    period = int(record.get("period", 1))
    time_s = pd.to_numeric(record.get("periodElapsedTime"), errors="coerce")
    if pd.isna(time_s):
        video_ms = pd.to_numeric(record.get("videoTimeMs"), errors="coerce")
        time_s = float(video_ms) / 1000.0 if pd.notna(video_ms) else pd.NA
    game_event = record.get("game_event") or {}
    possession_event = record.get("possession_event") or {}
    event_type = possession_event.get("possession_event_type") or game_event.get(
        "game_event_type"
    )
    possession_team = _team_for_event(game_event)
    coordinate_variant = "smoothed" if use_smoothed else "raw"
    source_text = str(source_file)

    def xy(item: dict[str, Any]) -> tuple[float, float]:
        return pff_xy_to_meters(
            item.get("x"),
            item.get("y"),
            pitch_length_m=pitch_length_m,
            pitch_width_m=pitch_width_m,
        )

    common = {
        "match_id": match_id,
        "dataset": DATASET_PFF,
        "period": period,
        "frame_id": frame_id,
        "time_s": time_s,
        "fps": PFF_NTSC_FPS,
        "fps_source": "inferred_from_tracking_timestamps",
        "coordinate_variant": coordinate_variant,
        "pitch_dimensions_source": "pff_spec_105x68_example",
        "possession_team_id": possession_team,
        "event_type": event_type,
        "game_event_id": record.get("game_event_id"),
        "possession_event_id": record.get("possession_event_id"),
        "source_file": source_text,
    }
    rows: list[dict[str, Any]] = []
    ball_key = "ballsSmoothed" if use_smoothed else "balls"
    balls = _objects(record.get(ball_key))
    if balls:
        ball = balls[0]
        x_m, y_m = xy(ball)
        visibility = str(ball.get("visibility") or "UNKNOWN").upper()
        available = bool(np.isfinite(x_m) and np.isfinite(y_m))
        rows.append(
            {
                **common,
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
                "z_m": pd.to_numeric(ball.get("z"), errors="coerce"),
                "raw_x": ball.get("x"),
                "raw_y": ball.get("y"),
                "is_visible": available,
                "visible": available,
                "is_observed": visibility == "VISIBLE",
                "provider_visibility": visibility,
                "has_possession": False,
            }
        )

    player_keys = (
        ("home", "homePlayersSmoothed" if use_smoothed else "homePlayers"),
        ("away", "awayPlayersSmoothed" if use_smoothed else "awayPlayers"),
    )
    for team_id, key in player_keys:
        for player in _players_for_frame(record.get(key)):
            jersey = str(player.get("jerseyNum") or "unknown").strip()
            agent_id = f"{team_id}_{jersey}"
            x_m, y_m = xy(player)
            visibility = str(player.get("visibility") or "UNKNOWN").upper()
            available = bool(np.isfinite(x_m) and np.isfinite(y_m))
            rows.append(
                {
                    **common,
                    "entity_id": agent_id,
                    "entity_type": AGENT_PLAYER,
                    "agent_id": agent_id,
                    "agent_type": AGENT_PLAYER,
                    "team_id": team_id,
                    "player_id": pd.NA,
                    "jersey_number": player.get("jerseyNum"),
                    "role": "player",
                    "x_m": x_m,
                    "y_m": y_m,
                    "z_m": np.nan,
                    "raw_x": player.get("x"),
                    "raw_y": player.get("y"),
                    "speed_mps": pd.to_numeric(player.get("speed"), errors="coerce"),
                    "is_visible": available,
                    "visible": available,
                    "is_observed": visibility == "VISIBLE",
                    "provider_visibility": visibility,
                    "provider_confidence": player.get("confidence"),
                    "has_possession": False,
                }
            )
    return rows


class PFFAdapter(TrackingDataAdapter):
    """Convert PFF FC World Cup tracking records to footballq's canonical rows.

    The delivery can contain repeated records for one video frame and, on those
    records, repeated player coordinates. Only the first geometry per frame and
    jersey is retained. Provider visibility is preserved separately because
    ``ESTIMATED`` off-camera locations are usable coordinates but are not direct
    observations.
    """

    dataset = DATASET_PFF

    def __init__(
        self,
        raw_dir: str | Path,
        match_id: str,
        *,
        use_smoothed: bool = True,
        max_frames: int | None = None,
        pitch_length_m: float = PITCH_LENGTH_M,
        pitch_width_m: float = PITCH_WIDTH_M,
    ) -> None:
        super().__init__(raw_dir=raw_dir, match_id=match_id)
        self.use_smoothed = use_smoothed
        self.max_frames = max_frames
        self.pitch_length_m = float(pitch_length_m)
        self.pitch_width_m = float(pitch_width_m)

    def _tracking_files(self) -> list[Path]:
        discovered = discover_pff_tracking_files(self.raw_dir)
        if not discovered:
            raise FileNotFoundError(
                f"No PFF .jsonl or .jsonl.bz2 tracking files found under {self.raw_dir}."
            )
        if self.match_id.lower() in _ALL_MATCHES:
            return list(discovered.values())
        selected = discovered.get(str(self.match_id))
        if selected is not None:
            return [selected]
        if self.raw_dir.is_file() or len(discovered) == 1:
            return [next(iter(discovered.values()))]
        available = ", ".join(discovered)
        raise ValueError(
            f"PFF match {self.match_id!r} was not found. Available match IDs: {available}"
        )

    def load_tracking(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for path in self._tracking_files():
            fallback_match_id = pff_match_id(path)
            seen_frames: set[tuple[str, int, int]] = set()
            for record in iter_pff_records(path):
                match_id = str(record.get("gameRefId") or fallback_match_id)
                frame_id = int(record.get("frameNum", -1))
                period = int(record.get("period", 1))
                frame_key = (match_id, period, frame_id)
                if frame_key in seen_frames:
                    continue
                seen_frames.add(frame_key)
                if self.max_frames is not None and len(seen_frames) > self.max_frames:
                    break
                rows.extend(
                    pff_record_to_tracking_rows(
                        record,
                        source_file=path,
                        fallback_match_id=fallback_match_id,
                        use_smoothed=self.use_smoothed,
                        pitch_length_m=self.pitch_length_m,
                        pitch_width_m=self.pitch_width_m,
                    )
                )

        if not rows:
            raise ValueError(f"PFF files under {self.raw_dir} contained no usable tracking rows.")
        return canonical_tracking_frame(pd.DataFrame(rows))

    def load_events(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for path in self._tracking_files():
            fallback_match_id = pff_match_id(path)
            for record in iter_pff_records(path):
                match_id = str(record.get("gameRefId") or fallback_match_id)
                period = record.get("period", 1)
                frame_id = record.get("frameNum")
                time_s = record.get("periodElapsedTime")
                game_event = record.get("game_event") or {}
                for kind, id_key, payload, type_key in (
                    ("game", "game_event_id", game_event, "game_event_type"),
                    (
                        "possession",
                        "possession_event_id",
                        record.get("possession_event") or {},
                        "possession_event_type",
                    ),
                ):
                    event_id = record.get(id_key)
                    if not payload or event_id is None:
                        continue
                    unique_key = (match_id, kind, str(event_id))
                    if unique_key in seen:
                        continue
                    seen.add(unique_key)
                    rows.append(
                        {
                            "match_id": match_id,
                            "dataset": DATASET_PFF,
                            "period": period,
                            "time_s": time_s,
                            "frame_id": frame_id,
                            "team_id": _team_for_event(game_event),
                            "player_id": game_event.get("player_id"),
                            "event_type": payload.get(type_key),
                            "event_subtype": kind,
                            "outcome": pd.NA,
                            "raw_event": json.dumps(payload, sort_keys=True),
                            "source_event_id": event_id,
                        }
                    )
        if not rows:
            return pd.DataFrame()
        return canonical_event_frame(pd.DataFrame(rows))


def audit_pff_match(path: str | Path, *, max_records: int = 10_000) -> dict[str, Any]:
    """Return a bounded, JSON-serializable structural audit of one PFF file."""

    source = Path(path)
    records = 0
    frames: set[tuple[int, int]] = set()
    periods: set[int] = set()
    duplicate_records = 0
    duplicated_player_arrays = 0
    observed_players = 0
    estimated_players = 0
    time_deltas: list[float] = []
    previous_by_period: dict[int, tuple[int, float]] = {}

    for record in iter_pff_records(source):
        records += 1
        period = int(record.get("period", 1))
        frame_id = int(record.get("frameNum", -1))
        frame_key = (period, frame_id)
        if frame_key in frames:
            duplicate_records += 1
        else:
            frames.add(frame_key)
            video_ms = pd.to_numeric(record.get("videoTimeMs"), errors="coerce")
            previous = previous_by_period.get(period)
            if previous and pd.notna(video_ms) and frame_id > previous[0]:
                delta_frames = frame_id - previous[0]
                delta_seconds = (float(video_ms) - previous[1]) / 1000.0
                if delta_seconds > 0:
                    time_deltas.append(delta_seconds / delta_frames)
            if pd.notna(video_ms):
                previous_by_period[period] = (frame_id, float(video_ms))
        periods.add(period)

        for key in ("homePlayersSmoothed", "awayPlayersSmoothed"):
            players = _objects(record.get(key))
            if len(players) != len(_players_for_frame(players)):
                duplicated_player_arrays += 1
            for player in _players_for_frame(players):
                if str(player.get("visibility", "")).upper() == "VISIBLE":
                    observed_players += 1
                else:
                    estimated_players += 1
        if records >= max_records:
            break

    median_delta = float(np.median(time_deltas)) if time_deltas else None
    return {
        "match_id": pff_match_id(source),
        "source_file": str(source),
        "records_sampled": records,
        "unique_frames_sampled": len(frames),
        "duplicate_records": duplicate_records,
        "duplicated_player_array_records": duplicated_player_arrays,
        "periods_sampled": sorted(periods),
        "inferred_fps": (1.0 / median_delta) if median_delta else None,
        "observed_player_entries": observed_players,
        "estimated_player_entries": estimated_players,
    }
