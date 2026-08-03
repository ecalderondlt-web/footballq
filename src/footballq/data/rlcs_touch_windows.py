"""Leakage-controlled touch-decision windows for the RLCS identity experiment."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from footballq.data.rlcs_replay import (
    IdentityObservation,
    ParsedReplay,
    normalize_handle,
    repair_score_columns,
)
from footballq.repro.manifest import file_sha256

TIME_STEPS = 20
N_ENTITIES = 7
N_PLAYERS = 6
N_FEATURES = 27
STATE_SIZE = TIME_STEPS * N_ENTITIES * N_FEATURES
STATE_MASK_SIZE = TIME_STEPS * N_ENTITIES

FIELD_SCALE = np.asarray([4096.0, 5120.0, 2044.0], dtype=np.float32)
ANGULAR_VELOCITY_SCALE = 5.5
UNK_IDENTITY_INDEX = 0

FEATURE_NAMES = (
    "position_x",
    "position_y",
    "position_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "rotation_6d_1",
    "rotation_6d_2",
    "rotation_6d_3",
    "rotation_6d_4",
    "rotation_6d_5",
    "rotation_6d_6",
    "angular_velocity_x",
    "angular_velocity_y",
    "angular_velocity_z",
    "boost_fraction",
    "on_ground",
    "jump_available",
    "dodge_available",
    "demolished",
    "team_sign",
    "relative_position_x",
    "relative_position_y",
    "relative_position_z",
    "relative_velocity_x",
    "relative_velocity_y",
    "relative_velocity_z",
)

CONTACT_EVENT_TYPES = frozenset(
    {
        "air-dribble",
        "double-tap",
        "flick",
        "flip-reset",
        "goal",
        "ground-dribble",
        "kickoff",
        "pass",
        "rebound",
        "retrieval",
        "save",
        "shot",
        "touch",
        "turnover",
    }
)
BOUNDARY_EVENT_TYPES = frozenset({"goal", "kickoff"})


class TouchWindowError(ValueError):
    """Raised when a decision window violates the frozen protocol."""


@dataclass(frozen=True)
class Touch:
    """One deduplicated physical ball contact."""

    frame_idx: int
    game_time_s: float
    player_prefix: str
    player_id: str
    team: str
    ball_position: tuple[float, float, float]
    blue_score: int
    orange_score: int


@dataclass(frozen=True)
class ContextSelection:
    """Frame row indices selected for a past-only context."""

    row_indices: tuple[int, ...]
    requested_times: tuple[float, ...]
    observed_times: tuple[float, ...]


@dataclass(frozen=True)
class _FrameTimeline:
    times: np.ndarray
    frame_ids: np.ndarray
    stints: np.ndarray | None

    @classmethod
    def from_frames(cls, frames: pd.DataFrame) -> _FrameTimeline:
        times = np.round(frames["game_time_s_precise"].to_numpy(dtype=np.float64), 6)
        frame_ids = frames["observed_frame_number"].to_numpy(dtype=np.int64)
        if np.any(np.diff(times) < 0) or np.any(np.diff(frame_ids) <= 0):
            raise TouchWindowError("Frame timeline is not strictly ordered.")
        stints = None
        if "stint_number" in frames:
            stints = pd.to_numeric(frames["stint_number"], errors="coerce").to_numpy(
                dtype=np.float64
            )
        return cls(times=times, frame_ids=frame_ids, stints=stints)

    def select(
        self,
        *,
        touch_frame_idx: int,
        touch_time_s: float,
        fps: float,
        context_seconds: float,
        max_frame_lag_seconds: float,
    ) -> ContextSelection:
        steps = int(round(float(fps) * float(context_seconds)))
        if steps != TIME_STEPS:
            raise TouchWindowError(
                f"Frozen schema requires {TIME_STEPS} context steps, got {steps}."
            )
        requested = np.round(
            touch_time_s - np.arange(steps - 1, -1, -1, dtype=np.float64) / fps,
            6,
        )
        positions = np.searchsorted(self.times, requested, side="right") - 1
        if np.any(positions < 0):
            raise TouchWindowError("Insufficient past context before current touch.")
        observed = self.times[positions]
        selected_frames = self.frame_ids[positions]
        if np.any(observed > requested + 1e-7) or np.any(
            selected_frames > int(touch_frame_idx)
        ):
            raise TouchWindowError("Future frame entered the context window.")
        if np.any(requested - observed > float(max_frame_lag_seconds)):
            raise TouchWindowError("Context contains a parser gap larger than the allowed lag.")
        if len(set(int(value) for value in positions)) != steps:
            raise TouchWindowError("Context grid reused a frame; replay sampling is too sparse.")
        return ContextSelection(
            row_indices=tuple(int(value) for value in positions),
            requested_times=tuple(float(value) for value in requested),
            observed_times=tuple(float(value) for value in observed),
        )

    def last_row_at_or_before(self, frame_idx: int) -> int | None:
        position = int(np.searchsorted(self.frame_ids, int(frame_idx), side="right") - 1)
        return None if position < 0 else position

    def crosses_parser_segment(
        self, *, context_rows: Sequence[int], next_touch_frame: int
    ) -> bool:
        if self.stints is None:
            return False
        target_position = self.last_row_at_or_before(next_touch_frame)
        if target_position is None:
            return True
        target_stint = self.stints[target_position]
        context_stints = self.stints[np.asarray(context_rows, dtype=np.int64)]
        values = set(float(value) for value in context_stints[~np.isnan(context_stints)])
        if not math.isfinite(float(target_stint)):
            return len(values) > 1
        return len(values | {float(target_stint)}) > 1


@dataclass(frozen=True)
class _EventIndex:
    boundary_frames: np.ndarray
    boundary_times: np.ndarray
    blue_goal_times: np.ndarray
    orange_goal_times: np.ndarray

    @classmethod
    def from_events(cls, events: pd.DataFrame) -> _EventIndex:
        boundary = _boundary_event_mask(events)
        frame_values = pd.to_numeric(
            events["observed_frame_number"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        event_times = pd.to_numeric(events["game_time_s_precise"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        boundary_values = boundary.to_numpy(dtype=np.bool_)
        score_change = events[["blue_score", "orange_score"]].diff().fillna(0)
        scoring_team = np.where(
            score_change["blue_score"] > 0,
            "blue",
            np.where(score_change["orange_score"] > 0, "orange", ""),
        )
        score_changed = (score_change.sum(axis=1) > 0).to_numpy(dtype=np.bool_)
        return cls(
            boundary_frames=frame_values[boundary_values],
            boundary_times=event_times[boundary_values],
            blue_goal_times=event_times[score_changed & (scoring_team == "blue")],
            orange_goal_times=event_times[score_changed & (scoring_team == "orange")],
        )

    def crosses_boundary(self, *, context_start_frame: int, next_touch_frame: int) -> bool:
        return bool(
            (
                (self.boundary_frames > int(context_start_frame))
                & (self.boundary_frames <= int(next_touch_frame))
            ).any()
        )

    def near_reset_boundary(self, *, touch_time: float, exclusion_seconds: float) -> bool:
        return bool(
            (
                (self.boundary_times <= float(touch_time))
                & (float(touch_time) - self.boundary_times <= float(exclusion_seconds))
            ).any()
        )

    def goal_for_within(
        self,
        *,
        current_time: float,
        actor_team: str,
        horizon_seconds: float,
    ) -> bool:
        goal_times = (
            self.blue_goal_times
            if str(actor_team).casefold() == "blue"
            else self.orange_goal_times
        )
        return bool(
            (
                (goal_times > float(current_time))
                & (goal_times <= float(current_time) + float(horizon_seconds))
            ).any()
        )


@dataclass(frozen=True)
class IdentityVocabulary:
    """Training-only canonical identity vocabulary."""

    player_to_index: Mapping[str, int]
    counts: Mapping[str, int]

    def encode(self, player_id: str) -> tuple[int, bool]:
        index = self.player_to_index.get(str(player_id))
        return (UNK_IDENTITY_INDEX, False) if index is None else (int(index), True)

    @property
    def size(self) -> int:
        return 1 + len(self.player_to_index)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "unk_index": UNK_IDENTITY_INDEX,
            "fit_split": "train",
            "player_to_index": dict(self.player_to_index),
            "counts": dict(self.counts),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> IdentityVocabulary:
        if payload.get("fit_split") != "train" or int(payload.get("unk_index", -1)) != 0:
            raise TouchWindowError("Identity vocabulary is not a frozen train-only vocabulary.")
        return cls(
            player_to_index={
                str(key): int(value) for key, value in payload["player_to_index"].items()
            },
            counts={str(key): int(value) for key, value in payload.get("counts", {}).items()},
        )


def fit_identity_vocabulary(
    train_player_rosters: Iterable[Sequence[str]], *, minimum_count: int = 1
) -> IdentityVocabulary:
    """Fit indices from training rosters and nothing else."""

    counts = Counter(str(player) for roster in train_player_rosters for player in roster)
    retained = [player for player, count in counts.items() if count >= int(minimum_count)]
    retained.sort(key=lambda player: (-counts[player], player))
    return IdentityVocabulary(
        player_to_index={player: index + 1 for index, player in enumerate(retained)},
        counts={player: counts[player] for player in sorted(counts)},
    )


def save_identity_vocabulary(vocabulary: IdentityVocabulary, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(vocabulary.to_dict(), indent=2) + "\n", encoding="utf-8")
    return destination


def load_identity_vocabulary(path: str | Path) -> IdentityVocabulary:
    return IdentityVocabulary.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _event_player_prefix(
    row: Mapping[str, Any], observations: Sequence[IdentityObservation]
) -> str | None:
    raw_id = str(row.get("event_player_1_id") or "").strip()
    handle = normalize_handle(row.get("event_player_1_name"))
    candidates: set[str] = set()
    for item in observations:
        if raw_id and raw_id in {item.platform_id, item.platform_key}:
            candidates.add(item.prefix)
        if handle and handle == item.normalized_handle:
            candidates.add(item.prefix)
    return next(iter(candidates)) if len(candidates) == 1 else None


def extract_touches(
    events: pd.DataFrame,
    observations: Sequence[IdentityObservation],
    roster_ids: Mapping[str, str],
    *,
    scores_repaired: bool = False,
) -> list[Touch]:
    """Extract physical contacts and merge same-player contacts under 0.20 seconds."""

    contacts: list[Touch] = []
    ordered = (
        events.sort_values(["observed_frame_number", "event_number"], kind="stable")
        if scores_repaired
        else repair_score_columns(events)
    )
    for row in ordered.to_dict(orient="records"):
        event_type = normalize_handle(row.get("event_type"))
        if event_type not in CONTACT_EVENT_TYPES:
            continue
        prefix = _event_player_prefix(row, observations)
        if prefix is None or prefix not in roster_ids:
            continue
        team = str(row.get("event_player_1_team") or row.get("event_team") or "").casefold()
        if team not in {"blue", "orange"}:
            team = "blue" if prefix.startswith("blue_") else "orange"
        position = []
        for axis in "xyz":
            value = row.get(f"event_ball_pos_{axis}")
            if value is None or not math.isfinite(float(value)):
                value = row.get(f"ball_pos_{axis}")
            position.append(float(value))
        contact = Touch(
            frame_idx=int(row["observed_frame_number"]),
            game_time_s=float(row["game_time_s_precise"]),
            player_prefix=prefix,
            player_id=str(roster_ids[prefix]),
            team=team,
            ball_position=(position[0], position[1], position[2]),
            blue_score=int(row.get("blue_score") or 0),
            orange_score=int(row.get("orange_score") or 0),
        )
        if contacts and contact.frame_idx == contacts[-1].frame_idx:
            if contact.player_id == contacts[-1].player_id:
                contacts[-1] = contact
            continue
        if (
            contacts
            and contact.player_id == contacts[-1].player_id
            and contact.game_time_s - contacts[-1].game_time_s < 0.20
        ):
            contacts[-1] = contact
        else:
            contacts.append(contact)
    return contacts


def select_past_context(
    frames: pd.DataFrame,
    *,
    touch_frame_idx: int,
    touch_time_s: float,
    fps: float = 10.0,
    context_seconds: float = 2.0,
    max_frame_lag_seconds: float = 0.15,
) -> ContextSelection:
    """Select an evenly spaced context using only frames at or before each grid time."""

    return _FrameTimeline.from_frames(frames).select(
        touch_frame_idx=touch_frame_idx,
        touch_time_s=touch_time_s,
        fps=fps,
        context_seconds=context_seconds,
        max_frame_lag_seconds=max_frame_lag_seconds,
    )


def _position_at(row: Mapping[str, Any], prefix: str) -> np.ndarray:
    return np.asarray([row[f"{prefix}_pos_{axis}"] for axis in "xyz"], dtype=np.float32)


def relative_player_order(
    frame: Mapping[str, Any], *, actor_prefix: str, observations: Sequence[IdentityObservation]
) -> list[str]:
    """Order cars by current role and distance, never by identity or name."""

    by_prefix = {item.prefix: item for item in observations}
    if actor_prefix not in by_prefix:
        raise TouchWindowError(f"Actor prefix {actor_prefix!r} is absent from roster.")
    actor_team = by_prefix[actor_prefix].team
    ball = np.asarray([frame[f"ball_pos_{axis}"] for axis in "xyz"], dtype=np.float32)
    teammates = [
        item.prefix
        for item in observations
        if item.team == actor_team and item.prefix != actor_prefix
    ]
    opponents = [item.prefix for item in observations if item.team != actor_team]
    if len(teammates) != 2 or len(opponents) != 3:
        raise TouchWindowError("Relative role ordering requires an exact 3v3 roster.")

    def key(prefix: str) -> tuple[float, str]:
        position = _position_at(frame, prefix)
        return (float(np.linalg.norm(position - ball)), prefix)

    return [actor_prefix, *sorted(teammates, key=key), *sorted(opponents, key=key)]


def _rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
    return rz @ ry @ rx


def _rotation_6d(row: Mapping[str, Any], prefix: str, transform: np.ndarray) -> np.ndarray:
    values = [float(row.get(f"{prefix}_rot_{axis}") or 0.0) for axis in "xyz"]
    rotation = _rotation_matrix(*values)
    rotation = transform @ rotation @ transform
    return np.concatenate([rotation[:, 0], rotation[:, 1]]).astype(np.float32)


def _bool_value(row: Mapping[str, Any], column: str, default: bool = False) -> float:
    value = row.get(column, default)
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return float(default)
    return float(bool(value))


def build_state_tensor(
    frames: pd.DataFrame,
    selection: ContextSelection,
    *,
    car_order: Sequence[str],
    actor_team: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the frozen ``20 x 7 x 27`` actor-oriented state tensor."""

    if len(car_order) != N_PLAYERS:
        raise TouchWindowError("State construction requires six relative car slots.")
    side = np.eye(3, dtype=np.float32)
    if str(actor_team).casefold() == "orange":
        side[1, 1] = -1.0
    state = np.zeros((TIME_STEPS, N_ENTITIES, N_FEATURES), dtype=np.float32)
    mask = np.zeros((TIME_STEPS, N_ENTITIES), dtype=np.bool_)
    for time_index, row_index in enumerate(selection.row_indices):
        row = frames.iloc[row_index].to_dict()
        ball_pos_raw = np.asarray([row[f"ball_pos_{axis}"] for axis in "xyz"], dtype=np.float32)
        ball_vel_raw = np.asarray([row[f"ball_vel_{axis}"] for axis in "xyz"], dtype=np.float32)
        ball_pos = (side @ ball_pos_raw) / FIELD_SCALE
        ball_vel = (side @ ball_vel_raw) / FIELD_SCALE
        ball_ang = np.asarray(
            [row.get(f"ball_ang_vel_{axis}", 0.0) for axis in "xyz"], dtype=np.float32
        )
        ball_valid = bool(np.isfinite(ball_pos).all() and np.isfinite(ball_vel).all())
        if ball_valid:
            state[time_index, 0, 0:3] = ball_pos
            state[time_index, 0, 3:6] = ball_vel
            state[time_index, 0, 12:15] = (side @ ball_ang) / ANGULAR_VELOCITY_SCALE
            mask[time_index, 0] = True
        for entity_index, prefix in enumerate(car_order, start=1):
            pos_raw = np.asarray([row.get(f"{prefix}_pos_{axis}") for axis in "xyz"], dtype=float)
            vel_raw = np.asarray([row.get(f"{prefix}_vel_{axis}") for axis in "xyz"], dtype=float)
            valid = bool(np.isfinite(pos_raw).all() and np.isfinite(vel_raw).all())
            demolished = not valid
            if not valid:
                state[time_index, entity_index, 19] = 1.0
                continue
            pos = (side @ pos_raw.astype(np.float32)) / FIELD_SCALE
            vel = (side @ vel_raw.astype(np.float32)) / FIELD_SCALE
            angular = np.asarray(
                [row.get(f"{prefix}_ang_vel_{axis}", 0.0) for axis in "xyz"],
                dtype=np.float32,
            )
            on_ground = float(pos_raw[2] <= 40.0)
            jumped = _bool_value(row, f"{prefix}_jumped")
            flipped = max(
                _bool_value(row, f"{prefix}_flipped"),
                _bool_value(row, f"{prefix}_double_jump_active"),
            )
            team_sign = 1.0 if entity_index <= 3 else -1.0
            features = state[time_index, entity_index]
            features[0:3] = pos
            features[3:6] = vel
            features[6:12] = _rotation_6d(row, prefix, side)
            features[12:15] = (side @ angular) / ANGULAR_VELOCITY_SCALE
            features[15] = float(row.get(f"{prefix}_boost", 0.0) or 0.0) / 100.0
            features[16] = on_ground
            features[17] = max(on_ground, 1.0 - jumped)
            features[18] = max(on_ground, 1.0 - flipped)
            features[19] = float(demolished)
            features[20] = team_sign
            features[21:24] = pos - ball_pos
            features[24:27] = vel - ball_vel
            mask[time_index, entity_index] = True
    if not bool(mask.all()):
        raise TouchWindowError("Context contains a missing ball or car rigid body state.")
    return state, mask


def reflect_state_x(state: np.ndarray) -> np.ndarray:
    """Apply the train-only left-right reflection to geometry and orientation."""

    reflected = np.asarray(state, dtype=np.float32).copy()
    reflected[..., 0] *= -1.0
    reflected[..., 3] *= -1.0
    reflected[..., 12] *= -1.0
    reflected[..., 21] *= -1.0
    reflected[..., 24] *= -1.0
    # For F = diag(-1, 1, 1), reflecting a rotation matrix R is F R F.
    # Only R's first two columns are stored. Their exact transformed signs are
    # [x, -y, -z] and [-x, y, z], so no per-entity matrix reconstruction is
    # needed. This is algebraically identical and keeps augmentation off the
    # Python hot path during training.
    reflected[..., 7:9] *= -1.0
    reflected[..., 9] *= -1.0
    return reflected


def reflect_next_touch_zone(zone: int) -> int:
    """Mirror one longitudinal-major 3 x 6 zone across the field's x-axis."""

    value = int(zone)
    if value < 0 or value >= 18:
        raise ValueError(f"next-touch zone must be in [0, 17], got {value}.")
    longitudinal, lateral = divmod(value, 3)
    return longitudinal * 3 + (2 - lateral)


def next_touch_zone(ball_position: Sequence[float], *, actor_team: str) -> int:
    """Return a longitudinal-major 3 x 6 field-zone class."""

    x, y = float(ball_position[0]), float(ball_position[1])
    if str(actor_team).casefold() == "orange":
        y = -y
    lateral = int(np.clip(np.floor((x + 4096.0) / (8192.0 / 3.0)), 0, 2))
    longitudinal = int(np.clip(np.floor((y + 5120.0) / (10240.0 / 6.0)), 0, 5))
    return longitudinal * 3 + lateral


def _roster_hash(player_ids: Sequence[str]) -> bytes:
    payload = "\n".join(sorted(str(value) for value in player_ids)).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).digest()


def _crosses_boundary(
    events: pd.DataFrame, *, context_start_frame: int, next_touch_frame: int
) -> bool:
    return _EventIndex.from_events(events).crosses_boundary(
        context_start_frame=context_start_frame,
        next_touch_frame=next_touch_frame,
    )


def _near_reset_boundary(
    events: pd.DataFrame, *, touch_time: float, exclusion_seconds: float
) -> bool:
    return _EventIndex.from_events(events).near_reset_boundary(
        touch_time=touch_time,
        exclusion_seconds=exclusion_seconds,
    )


def _boundary_event_mask(events: pd.DataFrame) -> pd.Series:
    event_types = events["event_type"].astype(str).str.casefold()
    kickoff = event_types.eq("kickoff")
    if "official_goal" in events:
        official_goal = events["official_goal"].fillna(False).astype(bool) | event_types.eq(
            "goal"
        )
    else:
        official_goal = event_types.eq("goal")
    return kickoff | official_goal


def _crosses_parser_segment(
    frames: pd.DataFrame, *, context_rows: Sequence[int], next_touch_frame: int
) -> bool:
    return _FrameTimeline.from_frames(frames).crosses_parser_segment(
        context_rows=context_rows,
        next_touch_frame=next_touch_frame,
    )


def _goal_for_within(
    events: pd.DataFrame,
    *,
    current_time: float,
    actor_team: str,
    horizon_seconds: float = 8.0,
    scores_repaired: bool = False,
) -> bool:
    repaired = events if scores_repaired else repair_score_columns(events)
    return _EventIndex.from_events(repaired).goal_for_within(
        current_time=current_time,
        actor_team=actor_team,
        horizon_seconds=horizon_seconds,
    )


def build_replay_decisions(
    parsed: ParsedReplay,
    *,
    inventory: Mapping[str, Any],
    split: str,
    observations: Sequence[IdentityObservation],
    roster_ids: Mapping[str, str],
    vocabulary: IdentityVocabulary | None = None,
    fps: float = 10.0,
    context_seconds: float = 2.0,
    min_next_touch_dt: float = 0.20,
    max_next_touch_dt: float = 4.00,
    exclude_goal_reset_seconds: float = 2.0,
) -> list[dict[str, Any]]:
    """Build all clean decision rows for one accepted replay."""

    if not parsed.qc.accepted:
        return []
    if len(roster_ids) != N_PLAYERS:
        raise TouchWindowError("Decision construction requires six resolved identities.")
    repaired_events = repair_score_columns(
        parsed.events,
        expected_blue_score=inventory.get("blue_score"),
        expected_orange_score=inventory.get("orange_score"),
    )
    frame_timeline = _FrameTimeline.from_frames(parsed.frames)
    event_index = _EventIndex.from_events(repaired_events)
    touches = extract_touches(
        repaired_events,
        observations,
        roster_ids,
        scores_repaired=True,
    )
    rows: list[dict[str, Any]] = []
    for current, following in zip(touches, touches[1:], strict=False):
        delta = following.game_time_s - current.game_time_s
        if current.player_id == following.player_id:
            continue
        if delta < float(min_next_touch_dt) or delta > float(max_next_touch_dt):
            continue
        if event_index.near_reset_boundary(
            touch_time=current.game_time_s,
            exclusion_seconds=exclude_goal_reset_seconds,
        ):
            continue
        current_row_index = frame_timeline.last_row_at_or_before(current.frame_idx)
        if current_row_index is None:
            continue
        current_row = parsed.frames.iloc[current_row_index]
        try:
            selection = frame_timeline.select(
                touch_frame_idx=current.frame_idx,
                touch_time_s=current.game_time_s,
                fps=fps,
                context_seconds=context_seconds,
                max_frame_lag_seconds=0.15,
            )
            order = relative_player_order(
                current_row.to_dict(),
                actor_prefix=current.player_prefix,
                observations=observations,
            )
        except TouchWindowError:
            continue
        context_start_frame = int(
            parsed.frames.iloc[selection.row_indices[0]]["observed_frame_number"]
        )
        if event_index.crosses_boundary(
            context_start_frame=context_start_frame,
            next_touch_frame=following.frame_idx,
        ) or frame_timeline.crosses_parser_segment(
            context_rows=selection.row_indices,
            next_touch_frame=following.frame_idx,
        ):
            continue
        try:
            state, state_mask = build_state_tensor(
                parsed.frames,
                selection,
                car_order=order,
                actor_team=current.team,
            )
        except TouchWindowError:
            continue
        try:
            next_entity = order.index(following.player_prefix)
        except ValueError:
            continue
        player_ids = [str(roster_ids[prefix]) for prefix in order]
        if vocabulary is None:
            identity_indices = [UNK_IDENTITY_INDEX] * N_PLAYERS
            known_mask = [False] * N_PLAYERS
        else:
            encoded = [vocabulary.encode(player_id) for player_id in player_ids]
            identity_indices = [value[0] for value in encoded]
            known_mask = [value[1] for value in encoded]
        team_ids = player_ids[:3]
        opponent_ids = player_ids[3:]
        score_diff = (
            current.blue_score - current.orange_score
            if current.team == "blue"
            else current.orange_score - current.blue_score
        )
        overtime = current.game_time_s > 300.0
        series_id = str(inventory.get("series_id") or inventory.get("leaf_group_id") or "")
        raw_stint = current_row.get("stint_number")
        stint = int(raw_stint) if raw_stint is not None and math.isfinite(float(raw_stint)) else 0
        sample_id = f"{parsed.replay_id}:stint_{stint}:touch_{current.frame_idx}"
        rows.append(
            {
                "sample_id": sample_id,
                "replay_id": parsed.replay_id,
                "series_id": series_id,
                "group_path": str(inventory.get("group_path") or ""),
                "region": str(inventory.get("region") or ""),
                "event_time_utc": inventory.get("event_time_utc"),
                "split": str(split),
                "frame_idx": int(current.frame_idx),
                "game_time_s": np.float32(current.game_time_s),
                "seconds_remaining": np.float32(max(300.0 - current.game_time_s, 0.0)),
                "score_diff_actor": int(np.clip(score_diff, -127, 127)),
                "overtime": bool(overtime),
                "actor_player_id": current.player_id,
                "player_ids": player_ids,
                "player_identity_idx": identity_indices,
                "player_known_mask": known_mask,
                "team_roster_hash": _roster_hash(team_ids),
                "opponent_roster_hash": _roster_hash(opponent_ids),
                "state_flat": state.reshape(-1).tolist(),
                "state_mask": state_mask.reshape(-1).tolist(),
                "next_touch_entity": int(next_entity),
                "next_touch_zone": next_touch_zone(
                    following.ball_position, actor_team=current.team
                ),
                "next_touch_dt_s": np.float32(delta),
                "retained_possession": bool(following.team == current.team),
                "goal_for_within_8s": event_index.goal_for_within(
                    current_time=current.game_time_s,
                    actor_team=current.team,
                    horizon_seconds=8.0,
                ),
            }
        )
    return rows


def encode_decision_identities(
    rows: Sequence[Mapping[str, Any]], vocabulary: IdentityVocabulary
) -> list[dict[str, Any]]:
    """Apply the frozen train vocabulary after all split assignments are fixed."""

    encoded_rows: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        encoded = [vocabulary.encode(str(player)) for player in row["player_ids"]]
        row["player_identity_idx"] = [value[0] for value in encoded]
        row["player_known_mask"] = [value[1] for value in encoded]
        encoded_rows.append(row)
    return encoded_rows


def decision_arrow_schema() -> Any:
    """Return the exact version-1 Arrow schema."""

    try:
        import pyarrow as pa
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("RLCS dataset writing requires pyarrow.") from exc
    return pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("replay_id", pa.string(), nullable=False),
            pa.field("series_id", pa.string(), nullable=False),
            pa.field("group_path", pa.string(), nullable=False),
            pa.field("region", pa.dictionary(pa.int8(), pa.string()), nullable=False),
            pa.field("event_time_utc", pa.timestamp("ms", tz="UTC"), nullable=True),
            pa.field("frame_idx", pa.int32(), nullable=False),
            pa.field("game_time_s", pa.float32(), nullable=False),
            pa.field("seconds_remaining", pa.float32(), nullable=False),
            pa.field("score_diff_actor", pa.int8(), nullable=False),
            pa.field("overtime", pa.bool_(), nullable=False),
            pa.field("actor_player_id", pa.string(), nullable=False),
            pa.field("player_identity_idx", pa.list_(pa.int32(), N_PLAYERS), nullable=False),
            pa.field("player_known_mask", pa.list_(pa.bool_(), N_PLAYERS), nullable=False),
            pa.field("team_roster_hash", pa.binary(16), nullable=False),
            pa.field("opponent_roster_hash", pa.binary(16), nullable=False),
            pa.field("state_flat", pa.list_(pa.float32(), STATE_SIZE), nullable=False),
            pa.field("state_mask", pa.list_(pa.bool_(), STATE_MASK_SIZE), nullable=False),
            pa.field("next_touch_entity", pa.uint8(), nullable=False),
            pa.field("next_touch_zone", pa.uint8(), nullable=False),
            pa.field("next_touch_dt_s", pa.float32(), nullable=False),
            pa.field("retained_possession", pa.bool_(), nullable=False),
            pa.field("goal_for_within_8s", pa.bool_(), nullable=False),
        ],
        metadata={
            b"footballq_schema": b"rlcs_touch_decisions_v1",
            b"state_shape": b"20,7,27",
            b"feature_names": json.dumps(FEATURE_NAMES).encode("utf-8"),
        },
    )


def _parse_utc(value: Any) -> datetime | None:
    if value is None or not str(value).strip():
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(UTC)
    else:
        timestamp = timestamp.tz_convert(UTC)
    return timestamp.to_pydatetime()


def _arrow_row(source: Mapping[str, Any]) -> dict[str, Any]:
    row = {key: value for key, value in source.items() if key != "player_ids" and key != "split"}
    row["event_time_utc"] = _parse_utc(row.get("event_time_utc"))
    row["state_flat"] = np.asarray(row["state_flat"], dtype=np.float32).tolist()
    row["state_mask"] = np.asarray(row["state_mask"], dtype=np.bool_).tolist()
    row["player_identity_idx"] = np.asarray(row["player_identity_idx"], dtype=np.int32).tolist()
    row["player_known_mask"] = np.asarray(row["player_known_mask"], dtype=np.bool_).tolist()
    return row


def write_decision_parquet_batches(
    row_batches: Iterable[Sequence[Mapping[str, Any]]],
    path: str | Path,
    *,
    max_rows_per_group: int = 256,
) -> Path:
    """Stream bounded row batches into one atomically replaced Parquet split."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("RLCS dataset writing requires pyarrow.") from exc
    if int(max_rows_per_group) <= 0:
        raise ValueError("max_rows_per_group must be positive.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    schema = decision_arrow_schema()
    writer = None
    buffer: list[Mapping[str, Any]] = []
    sample_ids: set[str] = set()
    row_count = 0

    def write_rows(rows: Sequence[Mapping[str, Any]]) -> None:
        nonlocal writer, row_count
        if not rows:
            return
        incoming_ids = [str(row["sample_id"]) for row in rows]
        if len(incoming_ids) != len(set(incoming_ids)) or any(
            sample_id in sample_ids for sample_id in incoming_ids
        ):
            raise TouchWindowError("Duplicate sample_id values detected.")
        sample_ids.update(incoming_ids)
        table = pa.Table.from_pylist([_arrow_row(row) for row in rows], schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(
                temporary,
                schema,
                compression="zstd",
                use_dictionary=["region"],
            )
        writer.write_table(table)
        row_count += len(rows)

    try:
        for batch in row_batches:
            buffer.extend(batch)
            while len(buffer) >= int(max_rows_per_group):
                write_rows(buffer[: int(max_rows_per_group)])
                del buffer[: int(max_rows_per_group)]
        write_rows(buffer)
        if row_count == 0:
            raise TouchWindowError("Refusing to write an empty scientific split.")
        if writer is not None:
            writer.close()
            writer = None
        temporary.replace(destination)
    except BaseException:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise
    return destination


def write_decision_parquet(rows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write one in-memory split through the bounded streaming implementation."""

    return write_decision_parquet_batches([rows], path)


def write_dataset_manifest(
    *,
    output_dir: str | Path,
    split_paths: Mapping[str, str | Path],
    vocabulary_path: str | Path,
    split_manifest_path: str | Path,
    parser_version: str,
    quality_report_path: str | Path,
) -> Path:
    """Record immutable source and output hashes for training."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("RLCS dataset manifests require pyarrow.") from exc
    split_payload: dict[str, Any] = {}
    for split, raw_path in split_paths.items():
        path = Path(raw_path)
        split_payload[split] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "rows": int(pq.read_metadata(path).num_rows),
        }
    payload = {
        "version": 1,
        "schema": "rlcs_touch_decisions_v1",
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "parser_version": str(parser_version),
        "split_manifest": {
            "path": str(split_manifest_path),
            "sha256": file_sha256(split_manifest_path),
        },
        "identity_vocabulary": {
            "path": str(vocabulary_path),
            "sha256": file_sha256(vocabulary_path),
        },
        "quality_report": {
            "path": str(quality_report_path),
            "sha256": file_sha256(quality_report_path),
        },
        "splits": split_payload,
    }
    destination = Path(output_dir) / "dataset_manifest.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def is_critical_state(row: Mapping[str, Any]) -> bool:
    return abs(int(row["score_diff_actor"])) <= 1 and (
        float(row["seconds_remaining"]) <= 120.0 or bool(row["overtime"])
    )
