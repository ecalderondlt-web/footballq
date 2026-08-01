"""Causal FOOTPASS player-history prediction experiment."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import torch

from footballq.io.footpass import FOOTPASS_PLAYER_IDS

try:
    import h5py
except ImportError:  # pragma: no cover - exercised only in incomplete environments
    h5py = None


FRAME = 0
PLAYER_ID = 1
LEFT_TO_RIGHT = 2
SHIRT_NUMBER = 3
ROLE_ID = 4
X = 5
Y = 6
SPEED_X = 7
SPEED_Y = 8
ACTION_CLASS = 13

ROLE_COUNT = 13
ACTION_COUNT = 8
TRACKING_FEATURE_NAMES = (
    "x_attack",
    "y",
    "vx_attack",
    "vy",
    "speed",
    "team_relative_x",
    "team_relative_y",
    "nearest_teammate_distance",
    "nearest_opponent_distance",
)
EVENT_CONTINUOUS_NAMES = (
    "x_attack",
    "y",
    "vx_attack",
    "vy",
    "speed",
)
PLAYER_SLOT = {player_id: index for index, player_id in enumerate(FOOTPASS_PLAYER_IDS)}


def broad_role(role_id: int) -> int:
    """Map FOOTPASS's 13 tactical roles into four broad role groups."""

    role = int(role_id)
    if role == 1:
        return 0
    if role in {2, 3, 4, 5, 13}:
        return 1
    if role in {6, 7, 8, 9}:
        return 2
    if role in {10, 11, 12}:
        return 3
    return 4


def _stable_choice_index(seed: int, key: str, count: int) -> int:
    if count < 1:
        raise ValueError("Stable choice requires at least one candidate.")
    payload = f"{int(seed)}:{key}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % count


def _x_attack(x: float, left_to_right: int) -> float:
    return float(x) if int(left_to_right) == 1 else 1.0 - float(x)


def _vx_attack(vx: float, left_to_right: int) -> float:
    return float(vx) if int(left_to_right) == 1 else -float(vx)


def _bin_index(value: float, edges: list[float]) -> int:
    index = int(np.searchsorted(np.asarray(edges), float(value), side="right") - 1)
    return max(0, min(index, len(edges) - 2))


def _safe_mean(vectors: list[np.ndarray], size: int) -> np.ndarray:
    if not vectors:
        return np.zeros(size, dtype=np.float64)
    return np.mean(np.stack(vectors, axis=0), axis=0)


@dataclass(frozen=True)
class FootpassAppearance:
    """Externally resolved focal-team appearance."""

    team_id: str
    team_name: str
    match_id: str
    match_date: date
    focal_team_index: int
    partition: str
    player_by_shirt: dict[int, str]
    player_name_by_id: dict[str, str]

    @property
    def appearance_id(self) -> str:
        return f"{self.team_id}:{self.match_id}"


def load_footpass_appearances(
    identity_manifest_paths: Iterable[str | Path],
) -> list[FootpassAppearance]:
    """Load and validate season-scoped identities and research partitions."""

    appearances: list[FootpassAppearance] = []
    seen: set[tuple[str, str]] = set()
    partition_names = (
        "profile_support_only",
        "development_train",
        "development_validation",
        "confirmatory_reserve_do_not_read_until_frozen",
    )
    for manifest_path in identity_manifest_paths:
        path = Path(manifest_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        team_id = str(payload["team"]["team_id"])
        team_name = str(payload["team"]["name"])
        player_by_shirt = {
            int(item["shirt_number"]): str(item["player_id"])
            for item in payload["players"]
        }
        player_name_by_id = {
            str(item["player_id"]): str(item["player_name"])
            for item in payload["players"]
        }
        partition_by_match: dict[str, str] = {}
        for partition_name in partition_names:
            for match_id in payload["research_partition"][partition_name]:
                match_key = str(match_id)
                if match_key in partition_by_match:
                    raise ValueError(
                        f"Identity partition overlap for {team_id}/{match_key}."
                    )
                partition_by_match[match_key] = partition_name
        for match in payload["matches"]:
            match_id = str(match["footpass_match_id"])
            key = (team_id, match_id)
            if key in seen:
                raise ValueError(f"Duplicate FOOTPASS appearance: {key}.")
            seen.add(key)
            if match_id not in partition_by_match:
                raise ValueError(f"Missing partition for {team_id}/{match_id}.")
            missing = {
                int(shirt) for shirt in match["focal_starting_shirts"]
            } - set(player_by_shirt)
            if missing:
                raise ValueError(
                    f"Starting shirts lack identity mappings for {team_id}/{match_id}: "
                    f"{sorted(missing)}."
                )
            appearances.append(
                FootpassAppearance(
                    team_id=team_id,
                    team_name=team_name,
                    match_id=match_id,
                    match_date=date.fromisoformat(str(match["match_date"])),
                    focal_team_index=int(match["focal_team_index"]),
                    partition=partition_by_match[match_id],
                    player_by_shirt=dict(player_by_shirt),
                    player_name_by_id=dict(player_name_by_id),
                )
            )
    return sorted(
        appearances,
        key=lambda item: (item.match_date, int(item.match_id), item.team_id),
    )


@dataclass
class EventAccumulator:
    """Sufficient statistics for a player's action history."""

    event_count: int = 0
    action_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(ACTION_COUNT, dtype=np.float64)
    )
    x_bin_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    y_bin_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    continuous_sum: np.ndarray = field(
        default_factory=lambda: np.zeros(len(EVENT_CONTINUOUS_NAMES), dtype=np.float64)
    )
    continuous_sumsq: np.ndarray = field(
        default_factory=lambda: np.zeros(len(EVENT_CONTINUOUS_NAMES), dtype=np.float64)
    )
    outcome_count: int = 0
    turnover_positive: int = 0
    entry_positive: int = 0
    outcome_x_count: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    turnover_x_positive: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    entry_x_positive: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    outcome_class_count: np.ndarray = field(
        default_factory=lambda: np.zeros(ACTION_COUNT, dtype=np.float64)
    )
    turnover_class_positive: np.ndarray = field(
        default_factory=lambda: np.zeros(ACTION_COUNT, dtype=np.float64)
    )
    entry_class_positive: np.ndarray = field(
        default_factory=lambda: np.zeros(ACTION_COUNT, dtype=np.float64)
    )

    def update_event(
        self,
        *,
        action_class: int,
        x_attack: float,
        y: float,
        vx_attack: float,
        vy: float,
        x_edges: list[float],
        y_edges: list[float],
    ) -> None:
        action_index = int(action_class) - 1
        if not 0 <= action_index < ACTION_COUNT:
            return
        x_bin = _bin_index(x_attack, x_edges)
        y_bin = _bin_index(y, y_edges)
        values = np.asarray(
            [
                x_attack,
                y,
                vx_attack,
                vy,
                math.hypot(vx_attack, vy),
            ],
            dtype=np.float64,
        )
        self.event_count += 1
        self.action_counts[action_index] += 1.0
        self.x_bin_counts[x_bin] += 1.0
        self.y_bin_counts[y_bin] += 1.0
        self.continuous_sum += values
        self.continuous_sumsq += values * values

    def update_outcome(
        self,
        *,
        action_class: int,
        x_attack: float,
        turnover: int,
        penalty_entry: int,
        x_edges: list[float],
    ) -> None:
        action_index = int(action_class) - 1
        if not 0 <= action_index < ACTION_COUNT:
            return
        x_bin = _bin_index(x_attack, x_edges)
        self.outcome_count += 1
        self.turnover_positive += int(turnover)
        self.entry_positive += int(penalty_entry)
        self.outcome_x_count[x_bin] += 1.0
        self.turnover_x_positive[x_bin] += int(turnover)
        self.entry_x_positive[x_bin] += int(penalty_entry)
        self.outcome_class_count[action_index] += 1.0
        self.turnover_class_positive[action_index] += int(turnover)
        self.entry_class_positive[action_index] += int(penalty_entry)

    def merge(self, other: EventAccumulator) -> None:
        self.event_count += other.event_count
        self.action_counts += other.action_counts
        self.x_bin_counts += other.x_bin_counts
        self.y_bin_counts += other.y_bin_counts
        self.continuous_sum += other.continuous_sum
        self.continuous_sumsq += other.continuous_sumsq
        self.outcome_count += other.outcome_count
        self.turnover_positive += other.turnover_positive
        self.entry_positive += other.entry_positive
        self.outcome_x_count += other.outcome_x_count
        self.turnover_x_positive += other.turnover_x_positive
        self.entry_x_positive += other.entry_x_positive
        self.outcome_class_count += other.outcome_class_count
        self.turnover_class_positive += other.turnover_class_positive
        self.entry_class_positive += other.entry_class_positive

    def copy(self) -> EventAccumulator:
        copied = EventAccumulator()
        copied.merge(self)
        return copied

    def vector(self) -> np.ndarray:
        total = float(self.event_count)
        action_prob = (self.action_counts + 1.0) / (total + ACTION_COUNT)
        x_prob = (self.x_bin_counts + 1.0) / (total + 3.0)
        y_prob = (self.y_bin_counts + 1.0) / (total + 3.0)
        denominator = max(total, 1.0)
        mean = self.continuous_sum / denominator
        variance = np.maximum(
            self.continuous_sumsq / denominator - mean * mean,
            0.0,
        )
        overall_outcome = np.asarray(
            [
                (self.turnover_positive + 1.0) / (self.outcome_count + 2.0),
                (self.entry_positive + 1.0) / (self.outcome_count + 2.0),
            ]
        )
        turnover_x = (self.turnover_x_positive + 1.0) / (
            self.outcome_x_count + 2.0
        )
        entry_x = (self.entry_x_positive + 1.0) / (
            self.outcome_x_count + 2.0
        )
        turnover_class = (self.turnover_class_positive + 1.0) / (
            self.outcome_class_count + 2.0
        )
        entry_class = (self.entry_class_positive + 1.0) / (
            self.outcome_class_count + 2.0
        )
        return np.concatenate(
            [
                np.asarray([math.log1p(total)]),
                action_prob,
                x_prob,
                y_prob,
                mean,
                np.sqrt(variance),
                overall_outcome,
                turnover_x,
                entry_x,
                turnover_class,
                entry_class,
            ]
        ).astype(np.float64)


def event_profile_feature_names() -> list[str]:
    names = ["log_event_count"]
    names.extend(f"action_probability_{index}" for index in range(1, ACTION_COUNT + 1))
    names.extend(f"x_bin_probability_{index}" for index in range(3))
    names.extend(f"y_bin_probability_{index}" for index in range(3))
    names.extend(f"mean_{name}" for name in EVENT_CONTINUOUS_NAMES)
    names.extend(f"std_{name}" for name in EVENT_CONTINUOUS_NAMES)
    names.extend(["turnover_rate", "penalty_entry_rate"])
    names.extend(f"turnover_rate_x_bin_{index}" for index in range(3))
    names.extend(f"penalty_entry_rate_x_bin_{index}" for index in range(3))
    names.extend(f"turnover_rate_action_{index}" for index in range(1, ACTION_COUNT + 1))
    names.extend(
        f"penalty_entry_rate_action_{index}" for index in range(1, ACTION_COUNT + 1)
    )
    return names


@dataclass
class TrackingAccumulator:
    """Sufficient statistics for sampled off-ball tracking history."""

    sample_count: int = 0
    feature_sum: np.ndarray = field(
        default_factory=lambda: np.zeros(len(TRACKING_FEATURE_NAMES), dtype=np.float64)
    )
    feature_sumsq: np.ndarray = field(
        default_factory=lambda: np.zeros(len(TRACKING_FEATURE_NAMES), dtype=np.float64)
    )
    x_bin_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    y_bin_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    role_counts: Counter[int] = field(default_factory=Counter)

    def update(
        self,
        values: np.ndarray,
        *,
        x_bin: int,
        y_bin: int,
        role_id: int,
    ) -> None:
        vector = np.asarray(values, dtype=np.float64)
        self.sample_count += 1
        self.feature_sum += vector
        self.feature_sumsq += vector * vector
        self.x_bin_counts[int(x_bin)] += 1.0
        self.y_bin_counts[int(y_bin)] += 1.0
        self.role_counts[int(role_id)] += 1

    def merge(self, other: TrackingAccumulator) -> None:
        self.sample_count += other.sample_count
        self.feature_sum += other.feature_sum
        self.feature_sumsq += other.feature_sumsq
        self.x_bin_counts += other.x_bin_counts
        self.y_bin_counts += other.y_bin_counts
        self.role_counts.update(other.role_counts)

    def vector(self) -> np.ndarray:
        total = float(self.sample_count)
        denominator = max(total, 1.0)
        mean = self.feature_sum / denominator
        variance = np.maximum(
            self.feature_sumsq / denominator - mean * mean,
            0.0,
        )
        x_prob = (self.x_bin_counts + 1.0) / (total + 3.0)
        y_prob = (self.y_bin_counts + 1.0) / (total + 3.0)
        return np.concatenate(
            [
                np.asarray([math.log1p(total)]),
                mean,
                np.sqrt(variance),
                x_prob,
                y_prob,
            ]
        ).astype(np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "feature_sum": self.feature_sum.tolist(),
            "feature_sumsq": self.feature_sumsq.tolist(),
            "x_bin_counts": self.x_bin_counts.tolist(),
            "y_bin_counts": self.y_bin_counts.tolist(),
            "role_counts": {str(key): value for key, value in self.role_counts.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrackingAccumulator:
        result = cls()
        result.sample_count = int(payload["sample_count"])
        result.feature_sum = np.asarray(payload["feature_sum"], dtype=np.float64)
        result.feature_sumsq = np.asarray(payload["feature_sumsq"], dtype=np.float64)
        result.x_bin_counts = np.asarray(payload["x_bin_counts"], dtype=np.float64)
        result.y_bin_counts = np.asarray(payload["y_bin_counts"], dtype=np.float64)
        result.role_counts = Counter(
            {int(key): int(value) for key, value in payload["role_counts"].items()}
        )
        return result


def tracking_profile_feature_names() -> list[str]:
    names = ["log_tracking_sample_count"]
    names.extend(f"mean_{name}" for name in TRACKING_FEATURE_NAMES)
    names.extend(f"std_{name}" for name in TRACKING_FEATURE_NAMES)
    names.extend(f"tracking_x_bin_probability_{index}" for index in range(3))
    names.extend(f"tracking_y_bin_probability_{index}" for index in range(3))
    return names


@dataclass
class ExtractedFootpassData:
    """Compact event snapshots plus sampled tracking sufficient statistics."""

    metadata: dict[str, Any]
    event_match_id: np.ndarray
    event_period: np.ndarray
    event_frame: np.ndarray
    event_team_index: np.ndarray
    event_player_id: np.ndarray
    event_shirt_number: np.ndarray
    event_role_id: np.ndarray
    event_left_to_right: np.ndarray
    event_action_class: np.ndarray
    event_geometry: np.ndarray
    event_snapshot_index: np.ndarray
    snapshot_player_id: np.ndarray
    snapshot_team_index: np.ndarray
    snapshot_shirt_number: np.ndarray
    snapshot_role_id: np.ndarray
    snapshot_left_to_right: np.ndarray
    snapshot_geometry: np.ndarray
    snapshot_active_count: np.ndarray
    tracking_stats: dict[str, dict[str, TrackingAccumulator]]


def _iter_complete_frame_batches(
    dataset: Any,
    *,
    chunk_rows: int = 500_000,
) -> Iterable[np.ndarray]:
    carry = np.empty((0, int(dataset.shape[1])), dtype=np.float32)
    for start in range(0, int(dataset.shape[0]), chunk_rows):
        rows = np.asarray(dataset[start : start + chunk_rows], dtype=np.float32)
        if carry.size:
            rows = np.concatenate([carry, rows], axis=0)
        if rows.size == 0:
            continue
        last_frame = rows[-1, FRAME]
        split = int(np.searchsorted(rows[:, FRAME], last_frame, side="left"))
        complete = rows[:split]
        carry = rows[split:]
        if complete.size:
            yield complete
    if carry.size:
        yield carry


def _tracking_updates_for_frame(
    rows: np.ndarray,
    appearance: FootpassAppearance,
    *,
    x_edges: list[float],
    y_edges: list[float],
    accumulator_by_player: dict[str, TrackingAccumulator],
) -> None:
    focal_mask = (
        rows[:, PLAYER_ID] < 200
        if appearance.focal_team_index == 0
        else rows[:, PLAYER_ID] >= 200
    )
    focal_rows = rows[focal_mask]
    opponent_rows = rows[~focal_mask]
    focal_finite = focal_rows[
        np.isfinite(focal_rows[:, [X, Y, SPEED_X, SPEED_Y]]).all(axis=1)
    ]
    opponent_finite = opponent_rows[
        np.isfinite(opponent_rows[:, [X, Y, SPEED_X, SPEED_Y]]).all(axis=1)
    ]
    if focal_finite.size == 0:
        return
    direction_values = focal_finite[:, LEFT_TO_RIGHT].astype(np.int64)
    direction = int(np.median(direction_values))
    focal_xy = np.column_stack(
        [
            np.asarray(
                [_x_attack(value, direction) for value in focal_finite[:, X]]
            ),
            focal_finite[:, Y],
        ]
    )
    opponent_xy = np.column_stack(
        [
            np.asarray(
                [_x_attack(value, direction) for value in opponent_finite[:, X]]
            ),
            opponent_finite[:, Y],
        ]
    )
    team_centroid = focal_xy.mean(axis=0)
    for index, row in enumerate(focal_finite):
        shirt = int(row[SHIRT_NUMBER])
        persistent_id = appearance.player_by_shirt.get(shirt)
        if persistent_id is None:
            continue
        xy = focal_xy[index]
        teammate_distance = np.linalg.norm(focal_xy - xy, axis=1)
        teammate_distance = teammate_distance[teammate_distance > 1e-12]
        nearest_teammate = (
            float(teammate_distance.min()) if teammate_distance.size else 1.5
        )
        opponent_distance = np.linalg.norm(opponent_xy - xy, axis=1)
        nearest_opponent = (
            float(opponent_distance.min()) if opponent_distance.size else 1.5
        )
        vx = _vx_attack(float(row[SPEED_X]), direction)
        vy = float(row[SPEED_Y])
        values = np.asarray(
            [
                float(xy[0]),
                float(xy[1]),
                vx,
                vy,
                math.hypot(vx, vy),
                float(xy[0] - team_centroid[0]),
                float(xy[1] - team_centroid[1]),
                nearest_teammate,
                nearest_opponent,
            ]
        )
        accumulator_by_player.setdefault(
            persistent_id, TrackingAccumulator()
        ).update(
            values,
            x_bin=_bin_index(float(xy[0]), x_edges),
            y_bin=_bin_index(float(xy[1]), y_edges),
            role_id=int(row[ROLE_ID]),
        )


def extract_footpass_experiment_data(
    hdf5_path: str | Path,
    appearances: list[FootpassAppearance],
    *,
    selected_match_ids: set[str],
    tracking_stride_frames: int,
    x_edges: list[float],
    y_edges: list[float],
    progress: Callable[[str], None] | None = None,
) -> ExtractedFootpassData:
    """Scan selected matches while preserving event-frame geometry."""

    if h5py is None:
        raise ImportError("FOOTPASS experiment extraction requires h5py.")
    selected_appearances = [
        item for item in appearances if item.match_id in selected_match_ids
    ]
    by_match: dict[str, list[FootpassAppearance]] = defaultdict(list)
    for appearance in selected_appearances:
        by_match[appearance.match_id].append(appearance)
    if set(by_match) != set(selected_match_ids):
        missing = sorted(set(selected_match_ids) - set(by_match), key=int)
        raise ValueError(f"Selected matches lack focal appearances: {missing}.")

    event_rows: list[tuple[Any, ...]] = []
    snapshot_player_id: list[np.ndarray] = []
    snapshot_team_index: list[np.ndarray] = []
    snapshot_shirt_number: list[np.ndarray] = []
    snapshot_role_id: list[np.ndarray] = []
    snapshot_left_to_right: list[np.ndarray] = []
    snapshot_geometry: list[np.ndarray] = []
    snapshot_active_count: list[int] = []
    tracking: dict[str, dict[str, TrackingAccumulator]] = {
        appearance.appearance_id: {} for appearance in selected_appearances
    }
    half_bounds: dict[str, dict[str, int]] = {}

    with h5py.File(Path(hdf5_path), "r") as h5_file:
        for match_id in sorted(selected_match_ids, key=int):
            for period in (1, 2):
                key = f"game_{int(match_id)}_H{period}"
                if progress is not None:
                    progress(f"extracting {key}")
                dataset = h5_file[key]
                if int(dataset.shape[1]) != 14:
                    raise ValueError(f"{key} has no action-label column.")
                first_frame = int(dataset[0, FRAME])
                last_frame = int(dataset[-1, FRAME])
                half_bounds[f"{match_id}:{period}"] = {
                    "first_frame": first_frame,
                    "last_frame": last_frame,
                }
                next_tracking_frame = first_frame
                for batch in _iter_complete_frame_batches(dataset):
                    frames, starts, counts = np.unique(
                        batch[:, FRAME].astype(np.int64),
                        return_index=True,
                        return_counts=True,
                    )
                    tracking_positions: list[int] = []
                    tracking_target = next_tracking_frame
                    while True:
                        tracking_position = int(
                            np.searchsorted(frames, tracking_target, side="left")
                        )
                        if tracking_position >= len(frames):
                            break
                        tracking_positions.append(tracking_position)
                        tracking_target = (
                            int(frames[tracking_position]) + tracking_stride_frames
                        )
                    if tracking_positions:
                        next_tracking_frame = tracking_target
                    action_frame_values = np.unique(
                        batch[batch[:, ACTION_CLASS] > 0, FRAME].astype(np.int64)
                    )
                    action_positions = np.searchsorted(
                        frames,
                        action_frame_values,
                    ).astype(np.int64)
                    tracking_position_set = set(tracking_positions)
                    selected_positions = sorted(
                        tracking_position_set | set(action_positions.tolist())
                    )
                    for position in selected_positions:
                        frame = int(frames[position])
                        start = int(starts[position])
                        count = int(counts[position])
                        rows = batch[start : start + count]
                        if position in tracking_position_set:
                            for appearance in by_match[match_id]:
                                _tracking_updates_for_frame(
                                    rows,
                                    appearance,
                                    x_edges=x_edges,
                                    y_edges=y_edges,
                                    accumulator_by_player=tracking[
                                        appearance.appearance_id
                                    ],
                                )
                        action_rows = rows[rows[:, ACTION_CLASS] > 0]
                        if action_rows.size == 0:
                            continue

                        player_ids = np.full(len(FOOTPASS_PLAYER_IDS), -1, dtype=np.int16)
                        teams = np.full(len(FOOTPASS_PLAYER_IDS), -1, dtype=np.int8)
                        shirts = np.full(len(FOOTPASS_PLAYER_IDS), -1, dtype=np.int16)
                        roles = np.full(len(FOOTPASS_PLAYER_IDS), -1, dtype=np.int8)
                        directions = np.full(len(FOOTPASS_PLAYER_IDS), -1, dtype=np.int8)
                        geometry = np.full(
                            (len(FOOTPASS_PLAYER_IDS), 4),
                            np.nan,
                            dtype=np.float32,
                        )
                        active = 0
                        for row in rows:
                            raw_player_id = int(row[PLAYER_ID])
                            slot = PLAYER_SLOT.get(raw_player_id)
                            if slot is None:
                                raise ValueError(
                                    f"Unexpected FOOTPASS player_id {raw_player_id}."
                                )
                            player_ids[slot] = raw_player_id
                            teams[slot] = 0 if raw_player_id < 200 else 1
                            shirts[slot] = int(row[SHIRT_NUMBER])
                            roles[slot] = int(row[ROLE_ID])
                            directions[slot] = int(row[LEFT_TO_RIGHT])
                            geometry[slot] = row[[X, Y, SPEED_X, SPEED_Y]]
                            active += 1
                        snapshot_index = len(snapshot_player_id)
                        snapshot_player_id.append(player_ids)
                        snapshot_team_index.append(teams)
                        snapshot_shirt_number.append(shirts)
                        snapshot_role_id.append(roles)
                        snapshot_left_to_right.append(directions)
                        snapshot_geometry.append(geometry)
                        snapshot_active_count.append(active)
                        for row in action_rows:
                            raw_player_id = int(row[PLAYER_ID])
                            event_rows.append(
                                (
                                    str(match_id),
                                    int(period),
                                    int(frame),
                                    0 if raw_player_id < 200 else 1,
                                    raw_player_id,
                                    int(row[SHIRT_NUMBER]),
                                    int(row[ROLE_ID]),
                                    int(row[LEFT_TO_RIGHT]),
                                    int(row[ACTION_CLASS]),
                                    float(row[X]),
                                    float(row[Y]),
                                    float(row[SPEED_X]),
                                    float(row[SPEED_Y]),
                                    snapshot_index,
                                )
                            )
                if progress is not None:
                    progress(
                        f"completed {key}: {len(event_rows)} cumulative action rows"
                    )

    event_rows.sort(key=lambda item: (int(item[0]), item[1], item[2], item[4]))
    metadata = {
        "version": 1,
        "selected_match_ids": sorted(selected_match_ids, key=int),
        "selected_appearance_ids": [
            item.appearance_id for item in selected_appearances
        ],
        "half_bounds": half_bounds,
        "tracking_stride_frames": int(tracking_stride_frames),
        "event_count": len(event_rows),
        "snapshot_count": len(snapshot_player_id),
        "confirmation_match_ids_included": sorted(
            {
                item.match_id
                for item in selected_appearances
                if item.partition
                == "confirmatory_reserve_do_not_read_until_frozen"
            },
            key=int,
        ),
    }
    return ExtractedFootpassData(
        metadata=metadata,
        event_match_id=np.asarray([row[0] for row in event_rows], dtype="U16"),
        event_period=np.asarray([row[1] for row in event_rows], dtype=np.int8),
        event_frame=np.asarray([row[2] for row in event_rows], dtype=np.int64),
        event_team_index=np.asarray([row[3] for row in event_rows], dtype=np.int8),
        event_player_id=np.asarray([row[4] for row in event_rows], dtype=np.int16),
        event_shirt_number=np.asarray([row[5] for row in event_rows], dtype=np.int16),
        event_role_id=np.asarray([row[6] for row in event_rows], dtype=np.int8),
        event_left_to_right=np.asarray([row[7] for row in event_rows], dtype=np.int8),
        event_action_class=np.asarray([row[8] for row in event_rows], dtype=np.int8),
        event_geometry=np.asarray([row[9:13] for row in event_rows], dtype=np.float32),
        event_snapshot_index=np.asarray([row[13] for row in event_rows], dtype=np.int32),
        snapshot_player_id=np.stack(snapshot_player_id),
        snapshot_team_index=np.stack(snapshot_team_index),
        snapshot_shirt_number=np.stack(snapshot_shirt_number),
        snapshot_role_id=np.stack(snapshot_role_id),
        snapshot_left_to_right=np.stack(snapshot_left_to_right),
        snapshot_geometry=np.stack(snapshot_geometry),
        snapshot_active_count=np.asarray(snapshot_active_count, dtype=np.int8),
        tracking_stats=tracking,
    )


def save_extracted_footpass_data(
    path: str | Path,
    data: ExtractedFootpassData,
) -> Path:
    """Save extracted arrays without pickle-backed object values."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tracking_payload = {
        appearance_id: {
            player_id: accumulator.to_dict()
            for player_id, accumulator in player_stats.items()
        }
        for appearance_id, player_stats in data.tracking_stats.items()
    }
    np.savez_compressed(
        output,
        metadata_json=np.asarray(json.dumps(data.metadata, sort_keys=True)),
        tracking_json=np.asarray(json.dumps(tracking_payload, sort_keys=True)),
        event_match_id=data.event_match_id,
        event_period=data.event_period,
        event_frame=data.event_frame,
        event_team_index=data.event_team_index,
        event_player_id=data.event_player_id,
        event_shirt_number=data.event_shirt_number,
        event_role_id=data.event_role_id,
        event_left_to_right=data.event_left_to_right,
        event_action_class=data.event_action_class,
        event_geometry=data.event_geometry,
        event_snapshot_index=data.event_snapshot_index,
        snapshot_player_id=data.snapshot_player_id,
        snapshot_team_index=data.snapshot_team_index,
        snapshot_shirt_number=data.snapshot_shirt_number,
        snapshot_role_id=data.snapshot_role_id,
        snapshot_left_to_right=data.snapshot_left_to_right,
        snapshot_geometry=data.snapshot_geometry,
        snapshot_active_count=data.snapshot_active_count,
    )
    return output


def load_extracted_footpass_data(path: str | Path) -> ExtractedFootpassData:
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        tracking_raw = json.loads(str(payload["tracking_json"].item()))
        tracking = {
            appearance_id: {
                player_id: TrackingAccumulator.from_dict(stats)
                for player_id, stats in player_stats.items()
            }
            for appearance_id, player_stats in tracking_raw.items()
        }
        return ExtractedFootpassData(
            metadata=metadata,
            tracking_stats=tracking,
            **{
                name: np.asarray(payload[name])
                for name in (
                    "event_match_id",
                    "event_period",
                    "event_frame",
                    "event_team_index",
                    "event_player_id",
                    "event_shirt_number",
                    "event_role_id",
                    "event_left_to_right",
                    "event_action_class",
                    "event_geometry",
                    "event_snapshot_index",
                    "snapshot_player_id",
                    "snapshot_team_index",
                    "snapshot_shirt_number",
                    "snapshot_role_id",
                    "snapshot_left_to_right",
                    "snapshot_geometry",
                    "snapshot_active_count",
                )
            },
        )


def combine_extracted_footpass_data(
    left: ExtractedFootpassData,
    right: ExtractedFootpassData,
) -> ExtractedFootpassData:
    """Combine disjoint development and confirmatory extraction caches."""

    overlap = set(left.metadata["selected_match_ids"]) & set(
        right.metadata["selected_match_ids"]
    )
    if overlap:
        raise ValueError(f"Cannot combine overlapping FOOTPASS caches: {sorted(overlap)}.")
    snapshot_offset = int(left.snapshot_player_id.shape[0])
    metadata = {
        "version": 1,
        "selected_match_ids": sorted(
            set(left.metadata["selected_match_ids"])
            | set(right.metadata["selected_match_ids"]),
            key=int,
        ),
        "selected_appearance_ids": sorted(
            set(left.metadata["selected_appearance_ids"])
            | set(right.metadata["selected_appearance_ids"])
        ),
        "half_bounds": {
            **left.metadata["half_bounds"],
            **right.metadata["half_bounds"],
        },
        "tracking_stride_frames": left.metadata["tracking_stride_frames"],
        "event_count": int(len(left.event_frame) + len(right.event_frame)),
        "snapshot_count": int(
            len(left.snapshot_player_id) + len(right.snapshot_player_id)
        ),
        "confirmation_match_ids_included": sorted(
            set(left.metadata["confirmation_match_ids_included"])
            | set(right.metadata["confirmation_match_ids_included"]),
            key=int,
        ),
    }
    tracking = {
        **left.tracking_stats,
        **right.tracking_stats,
    }
    event_order = np.lexsort(
        (
            np.concatenate([left.event_player_id, right.event_player_id]),
            np.concatenate([left.event_frame, right.event_frame]),
            np.concatenate([left.event_period, right.event_period]),
            np.asarray(
                [
                    int(value)
                    for value in np.concatenate(
                        [left.event_match_id, right.event_match_id]
                    )
                ]
            ),
        )
    )

    def event_concat(name: str) -> np.ndarray:
        left_value = getattr(left, name)
        right_value = getattr(right, name)
        if name == "event_snapshot_index":
            right_value = right_value + snapshot_offset
        return np.concatenate([left_value, right_value], axis=0)[event_order]

    def snapshot_concat(name: str) -> np.ndarray:
        return np.concatenate([getattr(left, name), getattr(right, name)], axis=0)

    return ExtractedFootpassData(
        metadata=metadata,
        tracking_stats=tracking,
        **{
            name: event_concat(name)
            for name in (
                "event_match_id",
                "event_period",
                "event_frame",
                "event_team_index",
                "event_player_id",
                "event_shirt_number",
                "event_role_id",
                "event_left_to_right",
                "event_action_class",
                "event_geometry",
                "event_snapshot_index",
            )
        },
        **{
            name: snapshot_concat(name)
            for name in (
                "snapshot_player_id",
                "snapshot_team_index",
                "snapshot_shirt_number",
                "snapshot_role_id",
                "snapshot_left_to_right",
                "snapshot_geometry",
                "snapshot_active_count",
            )
        },
    )


@dataclass(frozen=True)
class OpportunityOutcome:
    penalty_area_action_10s: int
    turnover_5s: int


def event_is_in_penalty_area(
    data: ExtractedFootpassData,
    event_index: int,
    penalty_area: dict[str, float],
) -> bool:
    x_value = _x_attack(
        float(data.event_geometry[event_index, 0]),
        int(data.event_left_to_right[event_index]),
    )
    y_value = float(data.event_geometry[event_index, 1])
    return (
        x_value >= float(penalty_area["attacking_x_min"])
        and float(penalty_area["y_min"]) <= y_value <= float(penalty_area["y_max"])
    )


def compute_appearance_outcomes(
    data: ExtractedFootpassData,
    appearance: FootpassAppearance,
    *,
    query_classes: set[int],
    possession_classes: set[int],
    primary_horizon_frames: int,
    turnover_horizon_frames: int,
    penalty_area: dict[str, float],
    minimum_active_players: int,
) -> dict[int, OpportunityOutcome]:
    """Compute future labels without exposing future frames as input features."""

    match_indices = np.flatnonzero(data.event_match_id == appearance.match_id)
    ordered = sorted(
        match_indices.tolist(),
        key=lambda index: (
            int(data.event_period[index]),
            int(data.event_frame[index]),
            int(data.event_player_id[index]),
        ),
    )
    outcomes: dict[int, OpportunityOutcome] = {}
    for offset, event_index in enumerate(ordered):
        if int(data.event_team_index[event_index]) != appearance.focal_team_index:
            continue
        if int(data.event_action_class[event_index]) not in query_classes:
            continue
        if int(data.snapshot_active_count[data.event_snapshot_index[event_index]]) < int(
            minimum_active_players
        ):
            continue
        if event_is_in_penalty_area(data, event_index, penalty_area):
            continue
        current_period = int(data.event_period[event_index])
        current_frame = int(data.event_frame[event_index])
        turnover = 0
        entry = 0
        first_possession_seen = False
        for future_index in ordered[offset + 1 :]:
            if int(data.event_period[future_index]) != current_period:
                break
            future_frame = int(data.event_frame[future_index])
            delta = future_frame - current_frame
            if delta <= 0:
                continue
            if delta > primary_horizon_frames:
                break
            if int(data.event_action_class[future_index]) not in possession_classes:
                continue
            future_is_focal = (
                int(data.event_team_index[future_index])
                == appearance.focal_team_index
            )
            if not first_possession_seen and delta <= turnover_horizon_frames:
                turnover = 0 if future_is_focal else 1
                first_possession_seen = True
            if not future_is_focal:
                break
            if event_is_in_penalty_area(data, future_index, penalty_area):
                entry = 1
                break
        outcomes[event_index] = OpportunityOutcome(
            penalty_area_action_10s=entry,
            turnover_5s=turnover,
        )
    return outcomes


@dataclass
class PlayerMatchStats:
    event: EventAccumulator = field(default_factory=EventAccumulator)
    tracking: TrackingAccumulator = field(default_factory=TrackingAccumulator)
    role_counts: Counter[int] = field(default_factory=Counter)

    @property
    def role_id(self) -> int:
        if not self.role_counts:
            return -1
        return int(self.role_counts.most_common(1)[0][0])

    def merge(self, other: PlayerMatchStats) -> None:
        self.event.merge(other.event)
        self.tracking.merge(other.tracking)
        self.role_counts.update(other.role_counts)


def build_player_match_stats(
    data: ExtractedFootpassData,
    appearances: list[FootpassAppearance],
    config: dict[str, Any],
) -> tuple[
    dict[str, dict[str, PlayerMatchStats]],
    dict[str, dict[int, OpportunityOutcome]],
]:
    opportunity_config = config["opportunities"]
    feature_config = config["features"]
    fps = float(opportunity_config["fps"])
    outcomes_by_appearance: dict[str, dict[int, OpportunityOutcome]] = {}
    stats_by_appearance: dict[str, dict[str, PlayerMatchStats]] = {}
    for appearance in appearances:
        if appearance.match_id not in set(data.metadata["selected_match_ids"]):
            continue
        outcomes = compute_appearance_outcomes(
            data,
            appearance,
            query_classes={
                int(value) for value in opportunity_config["query_action_classes"]
            },
            possession_classes={
                int(value)
                for value in opportunity_config["possession_action_classes"]
            },
            primary_horizon_frames=round(
                fps * float(opportunity_config["primary_horizon_seconds"])
            ),
            turnover_horizon_frames=round(
                fps * float(opportunity_config["turnover_horizon_seconds"])
            ),
            penalty_area=opportunity_config["penalty_area"],
            minimum_active_players=int(
                opportunity_config["minimum_active_players"]
            ),
        )
        outcomes_by_appearance[appearance.appearance_id] = outcomes
        player_stats: dict[str, PlayerMatchStats] = {}
        for player_id, tracking_stats in data.tracking_stats.get(
            appearance.appearance_id, {}
        ).items():
            stats = player_stats.setdefault(player_id, PlayerMatchStats())
            stats.tracking.merge(tracking_stats)
            stats.role_counts.update(tracking_stats.role_counts)
        focal_event_indices = np.flatnonzero(
            (data.event_match_id == appearance.match_id)
            & (data.event_team_index == appearance.focal_team_index)
        )
        for event_index in focal_event_indices.tolist():
            shirt = int(data.event_shirt_number[event_index])
            persistent_id = appearance.player_by_shirt.get(shirt)
            if persistent_id is None:
                continue
            stats = player_stats.setdefault(persistent_id, PlayerMatchStats())
            role_id = int(data.event_role_id[event_index])
            stats.role_counts[role_id] += 1
            x_value = _x_attack(
                float(data.event_geometry[event_index, 0]),
                int(data.event_left_to_right[event_index]),
            )
            y_value = float(data.event_geometry[event_index, 1])
            vx_value = _vx_attack(
                float(data.event_geometry[event_index, 2]),
                int(data.event_left_to_right[event_index]),
            )
            vy_value = float(data.event_geometry[event_index, 3])
            stats.event.update_event(
                action_class=int(data.event_action_class[event_index]),
                x_attack=x_value,
                y=y_value,
                vx_attack=vx_value,
                vy=vy_value,
                x_edges=[float(value) for value in feature_config["spatial_x_bins"]],
                y_edges=[float(value) for value in feature_config["spatial_y_bins"]],
            )
            outcome = outcomes.get(event_index)
            if outcome is not None:
                stats.event.update_outcome(
                    action_class=int(data.event_action_class[event_index]),
                    x_attack=x_value,
                    turnover=outcome.turnover_5s,
                    penalty_entry=outcome.penalty_area_action_10s,
                    x_edges=[
                        float(value) for value in feature_config["spatial_x_bins"]
                    ],
                )
        stats_by_appearance[appearance.appearance_id] = player_stats
    return stats_by_appearance, outcomes_by_appearance


def full_player_profile_feature_names() -> list[str]:
    return [
        *event_profile_feature_names(),
        *tracking_profile_feature_names(),
        "history_match_count_log",
        "history_available",
    ]


def _profile_vector_from_stats(
    stats: list[PlayerMatchStats],
    *,
    event_only: bool = False,
) -> np.ndarray:
    merged = PlayerMatchStats()
    for item in stats:
        merged.merge(item)
    event_vector = merged.event.vector()
    tracking_vector = (
        np.zeros(len(tracking_profile_feature_names()), dtype=np.float64)
        if event_only
        else merged.tracking.vector()
    )
    return np.concatenate(
        [
            event_vector,
            tracking_vector,
            np.asarray([math.log1p(len(stats)), float(bool(stats))]),
        ]
    )


def _history_stats_for_player(
    query: FootpassAppearance,
    player_id: str,
    appearances_by_team: dict[str, list[FootpassAppearance]],
    stats_by_appearance: dict[str, dict[str, PlayerMatchStats]],
    *,
    support_cap: int,
    reverse: bool = False,
) -> list[PlayerMatchStats]:
    candidates = []
    for appearance in appearances_by_team[query.team_id]:
        if appearance.appearance_id not in stats_by_appearance:
            continue
        allowed = (
            appearance.match_date > query.match_date
            if reverse
            else appearance.match_date < query.match_date
        )
        if not allowed:
            continue
        stats = stats_by_appearance[appearance.appearance_id].get(player_id)
        if stats is not None:
            candidates.append((appearance.match_date, int(appearance.match_id), stats))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=not reverse)
    return [item[2] for item in candidates[: int(support_cap)]]


def geometry_feature_names(nearest_teammates: int, nearest_opponents: int) -> list[str]:
    names = [
        "time_in_half",
        "actor_x_attack",
        "actor_y",
        "actor_vx_attack",
        "actor_vy",
        "actor_speed",
    ]
    names.extend(f"current_action_{index}" for index in range(1, ACTION_COUNT + 1))
    aggregate_names = (
        "active_fraction",
        "mean_x",
        "mean_y",
        "std_x",
        "std_y",
        "mean_vx",
        "mean_vy",
        "std_vx",
        "std_vy",
        "min_x",
        "max_x",
        "min_y",
        "max_y",
        "fraction_ahead_actor",
        "fraction_final_third",
        "fraction_penalty_area",
    )
    names.extend(f"focal_{name}" for name in aggregate_names)
    names.extend(f"opponent_{name}" for name in aggregate_names)
    relative_names = ("dx", "dy", "dvx", "dvy", "distance", "available")
    for index in range(nearest_teammates):
        names.extend(f"teammate_{index}_{name}" for name in relative_names)
    for index in range(nearest_opponents):
        names.extend(f"opponent_{index}_{name}" for name in relative_names)
    return names


def _side_aggregate(
    values: np.ndarray,
    *,
    actor_x: float,
    penalty_area: dict[str, float],
) -> np.ndarray:
    if values.size == 0:
        return np.zeros(16, dtype=np.float64)
    x_value = values[:, 0]
    y_value = values[:, 1]
    vx_value = values[:, 2]
    vy_value = values[:, 3]
    in_penalty = (
        (x_value >= float(penalty_area["attacking_x_min"]))
        & (y_value >= float(penalty_area["y_min"]))
        & (y_value <= float(penalty_area["y_max"]))
    )
    return np.asarray(
        [
            len(values) / 11.0,
            x_value.mean(),
            y_value.mean(),
            x_value.std(),
            y_value.std(),
            vx_value.mean(),
            vy_value.mean(),
            vx_value.std(),
            vy_value.std(),
            x_value.min(),
            x_value.max(),
            y_value.min(),
            y_value.max(),
            np.mean(x_value > actor_x),
            np.mean(x_value >= 2.0 / 3.0),
            np.mean(in_penalty),
        ],
        dtype=np.float64,
    )


def _nearest_relative(
    values: np.ndarray,
    actor: np.ndarray,
    count: int,
    *,
    exclude_actor: bool,
) -> np.ndarray:
    if values.size == 0:
        return np.zeros(count * 6, dtype=np.float64)
    delta = values[:, :2] - actor[:2]
    distance = np.linalg.norm(delta, axis=1)
    if exclude_actor:
        keep = distance > 1e-12
        values = values[keep]
        delta = delta[keep]
        distance = distance[keep]
    order = np.argsort(distance, kind="stable")
    features: list[float] = []
    for index in range(count):
        if index >= len(order):
            features.extend([0.0] * 6)
            continue
        selected = int(order[index])
        features.extend(
            [
                float(delta[selected, 0]),
                float(delta[selected, 1]),
                float(values[selected, 2] - actor[2]),
                float(values[selected, 3] - actor[3]),
                float(distance[selected]),
                1.0,
            ]
        )
    return np.asarray(features, dtype=np.float64)


def current_geometry_features(
    data: ExtractedFootpassData,
    event_index: int,
    appearance: FootpassAppearance,
    config: dict[str, Any],
) -> np.ndarray:
    feature_config = config["features"]
    opportunity_config = config["opportunities"]
    snapshot_index = int(data.event_snapshot_index[event_index])
    player_ids = data.snapshot_player_id[snapshot_index]
    teams = data.snapshot_team_index[snapshot_index]
    geometry = data.snapshot_geometry[snapshot_index].astype(np.float64)
    finite = (player_ids >= 0) & np.isfinite(geometry).all(axis=1)
    actor_raw_id = int(data.event_player_id[event_index])
    actor_slots = np.flatnonzero((player_ids == actor_raw_id) & finite)
    if len(actor_slots) != 1:
        raise ValueError(
            f"Cannot locate actor {actor_raw_id} in snapshot {snapshot_index}."
        )
    direction = int(data.event_left_to_right[event_index])
    oriented = geometry.copy()
    oriented[:, 0] = np.where(direction == 1, geometry[:, 0], 1.0 - geometry[:, 0])
    oriented[:, 2] = np.where(direction == 1, geometry[:, 2], -geometry[:, 2])
    oriented[:, :2] = np.clip(
        oriented[:, :2],
        float(feature_config["coordinate_clip_min"]),
        float(feature_config["coordinate_clip_max"]),
    )
    oriented[:, 2:] = np.clip(
        oriented[:, 2:],
        -float(feature_config["velocity_clip_abs"]),
        float(feature_config["velocity_clip_abs"]),
    )
    actor = oriented[int(actor_slots[0])]
    focal = oriented[
        finite & (teams == int(appearance.focal_team_index))
    ]
    opponent = oriented[
        finite & (teams != int(appearance.focal_team_index))
    ]
    half_key = f"{appearance.match_id}:{int(data.event_period[event_index])}"
    half_bounds = data.metadata["half_bounds"][half_key]
    duration = max(
        int(half_bounds["last_frame"]) - int(half_bounds["first_frame"]),
        1,
    )
    time_in_half = (
        int(data.event_frame[event_index]) - int(half_bounds["first_frame"])
    ) / duration
    action_one_hot = np.zeros(ACTION_COUNT, dtype=np.float64)
    action_class = int(data.event_action_class[event_index])
    if 1 <= action_class <= ACTION_COUNT:
        action_one_hot[action_class - 1] = 1.0
    return np.concatenate(
        [
            np.asarray(
                [
                    time_in_half,
                    actor[0],
                    actor[1],
                    actor[2],
                    actor[3],
                    math.hypot(actor[2], actor[3]),
                ]
            ),
            action_one_hot,
            _side_aggregate(
                focal,
                actor_x=float(actor[0]),
                penalty_area=opportunity_config["penalty_area"],
            ),
            _side_aggregate(
                opponent,
                actor_x=float(actor[0]),
                penalty_area=opportunity_config["penalty_area"],
            ),
            _nearest_relative(
                focal,
                actor,
                int(feature_config["nearest_teammates"]),
                exclude_actor=True,
            ),
            _nearest_relative(
                opponent,
                actor,
                int(feature_config["nearest_opponents"]),
                exclude_actor=False,
            ),
        ]
    ).astype(np.float64)


def _active_focal_players(
    data: ExtractedFootpassData,
    event_index: int,
    appearance: FootpassAppearance,
) -> list[tuple[str, int]]:
    snapshot_index = int(data.event_snapshot_index[event_index])
    teams = data.snapshot_team_index[snapshot_index]
    shirts = data.snapshot_shirt_number[snapshot_index]
    roles = data.snapshot_role_id[snapshot_index]
    player_ids = data.snapshot_player_id[snapshot_index]
    active: list[tuple[str, int]] = []
    for slot in np.flatnonzero(
        (player_ids >= 0) & (teams == appearance.focal_team_index)
    ).tolist():
        persistent_id = appearance.player_by_shirt.get(int(shirts[slot]))
        if persistent_id is not None:
            active.append((persistent_id, int(roles[slot])))
    return active


def role_features(
    data: ExtractedFootpassData,
    event_index: int,
    appearance: FootpassAppearance,
) -> np.ndarray:
    snapshot_index = int(data.event_snapshot_index[event_index])
    teams = data.snapshot_team_index[snapshot_index]
    roles = data.snapshot_role_id[snapshot_index]
    player_ids = data.snapshot_player_id[snapshot_index]
    actor_role = int(data.event_role_id[event_index])
    actor_one_hot = np.zeros(ROLE_COUNT, dtype=np.float64)
    if 1 <= actor_role <= ROLE_COUNT:
        actor_one_hot[actor_role - 1] = 1.0
    side_vectors: list[np.ndarray] = []
    for side in (appearance.focal_team_index, 1 - appearance.focal_team_index):
        counts = np.zeros(ROLE_COUNT, dtype=np.float64)
        selected = np.flatnonzero((player_ids >= 0) & (teams == side))
        for role in roles[selected]:
            if 1 <= int(role) <= ROLE_COUNT:
                counts[int(role) - 1] += 1.0
        if len(selected):
            counts /= len(selected)
        side_vectors.append(counts)
    return np.concatenate([actor_one_hot, *side_vectors])


def identity_features(
    data: ExtractedFootpassData,
    event_index: int,
    appearance: FootpassAppearance,
    player_index: dict[str, int],
) -> np.ndarray:
    actor = appearance.player_by_shirt[int(data.event_shirt_number[event_index])]
    actor_one_hot = np.zeros(len(player_index), dtype=np.float64)
    actor_one_hot[player_index[actor]] = 1.0
    lineup = np.zeros(len(player_index), dtype=np.float64)
    active = _active_focal_players(data, event_index, appearance)
    for player_id, _role in active:
        lineup[player_index[player_id]] = 1.0
    if active:
        lineup /= len(active)
    return np.concatenate([actor_one_hot, lineup])


@dataclass
class FootpassFeatureDataset:
    """Common rows and all predeclared feature components."""

    sample_ids: list[str]
    team_ids: list[str]
    match_ids: list[str]
    periods: list[int]
    frames: list[int]
    actor_ids: list[str]
    labels: dict[str, np.ndarray]
    components: dict[str, np.ndarray]
    component_feature_names: dict[str, list[str]]
    audit: dict[str, Any]

    def feature_views(self, main_support_cap: int) -> dict[str, tuple[np.ndarray, list[str]]]:
        prefix_components = ("geometry", "role", "identity", "rolling")

        def join(*component_names: str) -> tuple[np.ndarray, list[str]]:
            arrays = [self.components[name] for name in component_names]
            names = [
                f"{component}:{feature}"
                for component in component_names
                for feature in self.component_feature_names[component]
            ]
            return np.concatenate(arrays, axis=1), names

        views: dict[str, tuple[np.ndarray, list[str]]] = {
            "geometry": join("geometry"),
            "geometry_role": join("geometry", "role"),
            "geometry_role_identity": join("geometry", "role", "identity"),
            "rolling": join(*prefix_components),
        }
        for support_cap in (1, 3, 5):
            component = f"history_k{support_cap}"
            views[f"history_k{support_cap}"] = join(*prefix_components, component)
        views["history"] = views[f"history_k{main_support_cap}"]
        for control in (
            "role_mean_history",
            "event_only_history",
            "reverse_history",
        ):
            views[control] = join(*prefix_components, control)
        for name in sorted(self.components):
            if name.startswith("shuffled_history_seed_"):
                views[name] = join(*prefix_components, name)
        return views


def _history_profile_cache(
    query_appearances: list[FootpassAppearance],
    appearances_by_team: dict[str, list[FootpassAppearance]],
    stats_by_appearance: dict[str, dict[str, PlayerMatchStats]],
    all_player_ids: list[str],
    *,
    support_caps: list[int],
) -> dict[tuple[str, str, int, bool, bool], np.ndarray]:
    cache: dict[tuple[str, str, int, bool, bool], np.ndarray] = {}
    for query in query_appearances:
        for player_id in all_player_ids:
            if not player_id.startswith(f"{query.team_id}:"):
                continue
            for cap in support_caps:
                for reverse in (False, True):
                    stats = _history_stats_for_player(
                        query,
                        player_id,
                        appearances_by_team,
                        stats_by_appearance,
                        support_cap=cap,
                        reverse=reverse,
                    )
                    cache[(query.appearance_id, player_id, cap, reverse, False)] = (
                        _profile_vector_from_stats(stats)
                    )
                    cache[(query.appearance_id, player_id, cap, reverse, True)] = (
                        _profile_vector_from_stats(stats, event_only=True)
                    )
    return cache


def _available_profile(vector: np.ndarray) -> bool:
    return bool(vector[-1] > 0.5)


def _role_mean_profile(
    active: list[tuple[str, int]],
    profile_by_player: dict[str, np.ndarray],
    role_id: int,
    profile_size: int,
    *,
    exclude_player_id: str | None = None,
) -> np.ndarray:
    target_role = broad_role(role_id)
    candidates = [
        profile_by_player[player_id]
        for player_id, candidate_role in active
        if player_id != exclude_player_id
        and broad_role(candidate_role) == target_role
        and _available_profile(profile_by_player[player_id])
    ]
    if not candidates:
        candidates = [
            profile_by_player[player_id]
            for player_id, _candidate_role in active
            if player_id != exclude_player_id
            and _available_profile(profile_by_player[player_id])
        ]
    return _safe_mean(candidates, profile_size)


def _profile_components_for_event(
    *,
    sample_id: str,
    actor_id: str,
    actor_role: int,
    active: list[tuple[str, int]],
    profiles: dict[str, np.ndarray],
    event_only_profiles: dict[str, np.ndarray],
    shuffle_seeds: list[int],
) -> dict[str, np.ndarray]:
    profile_size = len(next(iter(profiles.values())))

    def profile_or_fallback(player_id: str, role_id: int) -> np.ndarray:
        profile = profiles[player_id]
        if _available_profile(profile):
            return profile
        fallback = _role_mean_profile(
            active,
            profiles,
            role_id,
            profile_size,
            exclude_player_id=player_id,
        ).copy()
        fallback[-2:] = 0.0
        return fallback

    actor_profile = profile_or_fallback(actor_id, actor_role)
    lineup_profiles = [
        profile_or_fallback(player_id, role_id)
        for player_id, role_id in active
    ]
    main = np.concatenate(
        [actor_profile, _safe_mean(lineup_profiles, profile_size)]
    )

    actor_role_mean = _role_mean_profile(
        active,
        profiles,
        actor_role,
        profile_size,
        exclude_player_id=actor_id,
    )
    lineup_role_means = [
        _role_mean_profile(
            active,
            profiles,
            role_id,
            profile_size,
            exclude_player_id=player_id,
        )
        for player_id, role_id in active
    ]
    role_mean = np.concatenate(
        [actor_role_mean, _safe_mean(lineup_role_means, profile_size)]
    )

    def event_only_or_fallback(player_id: str, role_id: int) -> np.ndarray:
        profile = event_only_profiles[player_id]
        if _available_profile(profile):
            return profile
        fallback = _role_mean_profile(
            active,
            event_only_profiles,
            role_id,
            profile_size,
            exclude_player_id=player_id,
        ).copy()
        fallback[-2:] = 0.0
        return fallback

    event_only = np.concatenate(
        [
            event_only_or_fallback(actor_id, actor_role),
            _safe_mean(
                [
                    event_only_or_fallback(player_id, role_id)
                    for player_id, role_id in active
                ],
                profile_size,
            ),
        ]
    )
    result = {
        "main": main,
        "role_mean": role_mean,
        "event_only": event_only,
    }
    for seed in shuffle_seeds:
        shuffled_by_player: dict[str, np.ndarray] = {}
        for player_id, role_id in active:
            candidates = [
                donor_id
                for donor_id, donor_role in active
                if donor_id != player_id
                and broad_role(donor_role) == broad_role(role_id)
                and _available_profile(profiles[donor_id])
            ]
            if not candidates:
                candidates = [
                    donor_id
                    for donor_id, vector in profiles.items()
                    if donor_id != player_id and _available_profile(vector)
                ]
            if candidates:
                candidates.sort()
                donor = candidates[
                    _stable_choice_index(seed, f"{sample_id}:{player_id}", len(candidates))
                ]
                shuffled_by_player[player_id] = profiles[donor]
            else:
                shuffled_by_player[player_id] = profile_or_fallback(player_id, role_id)
        shuffled_actor = shuffled_by_player.get(
            actor_id,
            profile_or_fallback(actor_id, actor_role),
        )
        result[f"shuffle_{seed}"] = np.concatenate(
            [
                shuffled_actor,
                _safe_mean(list(shuffled_by_player.values()), profile_size),
            ]
        )
    return result


def build_footpass_feature_dataset(
    data: ExtractedFootpassData,
    appearances: list[FootpassAppearance],
    config: dict[str, Any],
    *,
    query_partitions: set[str],
) -> FootpassFeatureDataset:
    """Build causal current features and strictly prior player histories."""

    available_match_ids = set(data.metadata["selected_match_ids"])
    selected_appearances = [
        item for item in appearances if item.match_id in available_match_ids
    ]
    query_appearances = [
        item for item in selected_appearances if item.partition in query_partitions
    ]
    stats_by_appearance, outcomes_by_appearance = build_player_match_stats(
        data,
        selected_appearances,
        config,
    )
    appearances_by_team: dict[str, list[FootpassAppearance]] = defaultdict(list)
    for appearance in selected_appearances:
        appearances_by_team[appearance.team_id].append(appearance)
    for team_appearances in appearances_by_team.values():
        team_appearances.sort(key=lambda item: (item.match_date, int(item.match_id)))
    all_player_ids = sorted(
        {
            player_id
            for appearance in appearances
            for player_id in appearance.player_by_shirt.values()
        }
    )
    player_index = {player_id: index for index, player_id in enumerate(all_player_ids)}
    support_caps = [
        int(value) for value in config["features"]["history_support_caps"]
    ]
    main_cap = int(config["features"]["main_history_support_cap"])
    profile_cache = _history_profile_cache(
        query_appearances,
        appearances_by_team,
        stats_by_appearance,
        all_player_ids,
        support_caps=support_caps,
    )
    shuffle_seeds = [
        int(value) for value in config["features"]["shuffled_history_seeds"]
    ]
    event_names = event_profile_feature_names()
    full_profile_names = full_player_profile_feature_names()
    history_names = [
        *[f"actor_{name}" for name in full_profile_names],
        *[f"lineup_mean_{name}" for name in full_profile_names],
    ]

    sample_ids: list[str] = []
    team_ids: list[str] = []
    match_ids: list[str] = []
    periods: list[int] = []
    frames: list[int] = []
    actor_ids: list[str] = []
    primary_labels: list[int] = []
    turnover_labels: list[int] = []
    components: dict[str, list[np.ndarray]] = defaultdict(list)
    missing_history_rows = 0
    support_counts: list[float] = []

    x_edges = [float(value) for value in config["features"]["spatial_x_bins"]]
    y_edges = [float(value) for value in config["features"]["spatial_y_bins"]]
    fps = float(config["opportunities"]["fps"])
    maturity_frames = round(
        fps * float(config["opportunities"]["primary_horizon_seconds"])
    )

    for appearance in query_appearances:
        outcomes = outcomes_by_appearance[appearance.appearance_id]
        focal_indices = np.flatnonzero(
            (data.event_match_id == appearance.match_id)
            & (data.event_team_index == appearance.focal_team_index)
        ).tolist()
        focal_indices.sort(
            key=lambda index: (
                int(data.event_period[index]),
                int(data.event_frame[index]),
                int(data.event_player_id[index]),
            )
        )
        rolling: dict[str, EventAccumulator] = defaultdict(EventAccumulator)
        pending: list[tuple[int, int, str, int]] = []
        grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
        for event_index in focal_indices:
            grouped[
                (
                    int(data.event_period[event_index]),
                    int(data.event_frame[event_index]),
                )
            ].append(event_index)
        for (period, frame), group_indices in sorted(grouped.items()):
            still_pending: list[tuple[int, int, str, int]] = []
            for pending_period, mature_frame, player_id, origin_index in pending:
                if pending_period < period or (
                    pending_period == period and mature_frame < frame
                ):
                    outcome = outcomes[origin_index]
                    rolling[player_id].update_outcome(
                        action_class=int(data.event_action_class[origin_index]),
                        x_attack=_x_attack(
                            float(data.event_geometry[origin_index, 0]),
                            int(data.event_left_to_right[origin_index]),
                        ),
                        turnover=outcome.turnover_5s,
                        penalty_entry=outcome.penalty_area_action_10s,
                        x_edges=x_edges,
                    )
                else:
                    still_pending.append(
                        (pending_period, mature_frame, player_id, origin_index)
                    )
            pending = still_pending

            for event_index in group_indices:
                outcome = outcomes.get(event_index)
                if outcome is None:
                    continue
                shirt = int(data.event_shirt_number[event_index])
                actor_id = appearance.player_by_shirt.get(shirt)
                if actor_id is None:
                    continue
                actor_role = int(data.event_role_id[event_index])
                active = _active_focal_players(data, event_index, appearance)
                active_ids = {player_id for player_id, _role in active}
                if actor_id not in active_ids:
                    active.append((actor_id, actor_role))
                sample_id = (
                    f"{appearance.team_id}:{appearance.match_id}:p{period}:"
                    f"f{frame}:slot{int(data.event_player_id[event_index])}"
                )

                components["geometry"].append(
                    current_geometry_features(data, event_index, appearance, config)
                )
                components["role"].append(
                    role_features(data, event_index, appearance)
                )
                components["identity"].append(
                    identity_features(data, event_index, appearance, player_index)
                )
                rolling_actor = rolling[actor_id].vector()
                rolling_lineup = _safe_mean(
                    [rolling[player_id].vector() for player_id, _role in active],
                    len(event_names),
                )
                components["rolling"].append(
                    np.concatenate([rolling_actor, rolling_lineup])
                )

                for cap in support_caps:
                    profile_by_player = {
                        player_id: profile_cache[
                            (
                                appearance.appearance_id,
                                player_id,
                                cap,
                                False,
                                False,
                            )
                        ]
                        for player_id, _role in active
                    }
                    event_only_by_player = {
                        player_id: profile_cache[
                            (
                                appearance.appearance_id,
                                player_id,
                                cap,
                                False,
                                True,
                            )
                        ]
                        for player_id, _role in active
                    }
                    if actor_id not in profile_by_player:
                        profile_by_player[actor_id] = profile_cache[
                            (
                                appearance.appearance_id,
                                actor_id,
                                cap,
                                False,
                                False,
                            )
                        ]
                        event_only_by_player[actor_id] = profile_cache[
                            (
                                appearance.appearance_id,
                                actor_id,
                                cap,
                                False,
                                True,
                            )
                        ]
                    profile_parts = _profile_components_for_event(
                        sample_id=sample_id,
                        actor_id=actor_id,
                        actor_role=actor_role,
                        active=active,
                        profiles=profile_by_player,
                        event_only_profiles=event_only_by_player,
                        shuffle_seeds=shuffle_seeds,
                    )
                    components[f"history_k{cap}"].append(profile_parts["main"])
                    if cap == main_cap:
                        components["role_mean_history"].append(
                            profile_parts["role_mean"]
                        )
                        components["event_only_history"].append(
                            profile_parts["event_only"]
                        )
                        for seed in shuffle_seeds:
                            components[f"shuffled_history_seed_{seed}"].append(
                                profile_parts[f"shuffle_{seed}"]
                            )
                        actor_main = profile_by_player[actor_id]
                        support_counts.append(float(math.expm1(actor_main[-2])))
                        if not _available_profile(actor_main):
                            missing_history_rows += 1

                reverse_profiles = {
                    player_id: profile_cache[
                        (
                            appearance.appearance_id,
                            player_id,
                            main_cap,
                            True,
                            False,
                        )
                    ]
                    for player_id, _role in active
                }
                reverse_event_only = {
                    player_id: profile_cache[
                        (
                            appearance.appearance_id,
                            player_id,
                            main_cap,
                            True,
                            True,
                        )
                    ]
                    for player_id, _role in active
                }
                if actor_id not in reverse_profiles:
                    reverse_profiles[actor_id] = profile_cache[
                        (
                            appearance.appearance_id,
                            actor_id,
                            main_cap,
                            True,
                            False,
                        )
                    ]
                    reverse_event_only[actor_id] = profile_cache[
                        (
                            appearance.appearance_id,
                            actor_id,
                            main_cap,
                            True,
                            True,
                        )
                    ]
                reverse_parts = _profile_components_for_event(
                    sample_id=f"reverse:{sample_id}",
                    actor_id=actor_id,
                    actor_role=actor_role,
                    active=active,
                    profiles=reverse_profiles,
                    event_only_profiles=reverse_event_only,
                    shuffle_seeds=[],
                )
                components["reverse_history"].append(reverse_parts["main"])

                sample_ids.append(sample_id)
                team_ids.append(appearance.team_id)
                match_ids.append(appearance.match_id)
                periods.append(period)
                frames.append(frame)
                actor_ids.append(actor_id)
                primary_labels.append(outcome.penalty_area_action_10s)
                turnover_labels.append(outcome.turnover_5s)

            for event_index in group_indices:
                shirt = int(data.event_shirt_number[event_index])
                player_id = appearance.player_by_shirt.get(shirt)
                if player_id is None:
                    continue
                x_value = _x_attack(
                    float(data.event_geometry[event_index, 0]),
                    int(data.event_left_to_right[event_index]),
                )
                y_value = float(data.event_geometry[event_index, 1])
                vx_value = _vx_attack(
                    float(data.event_geometry[event_index, 2]),
                    int(data.event_left_to_right[event_index]),
                )
                vy_value = float(data.event_geometry[event_index, 3])
                rolling[player_id].update_event(
                    action_class=int(data.event_action_class[event_index]),
                    x_attack=x_value,
                    y=y_value,
                    vx_attack=vx_value,
                    vy=vy_value,
                    x_edges=x_edges,
                    y_edges=y_edges,
                )
                if event_index in outcomes:
                    pending.append(
                        (
                            period,
                            frame + maturity_frames,
                            player_id,
                            event_index,
                        )
                    )

    arrays = {
        name: np.stack(rows).astype(np.float64)
        for name, rows in components.items()
    }
    if not sample_ids:
        raise ValueError("No FOOTPASS opportunities were built.")
    component_names = {
        "geometry": geometry_feature_names(
            int(config["features"]["nearest_teammates"]),
            int(config["features"]["nearest_opponents"]),
        ),
        "role": [
            *[f"actor_role_{index}" for index in range(1, ROLE_COUNT + 1)],
            *[f"focal_role_fraction_{index}" for index in range(1, ROLE_COUNT + 1)],
            *[
                f"opponent_role_fraction_{index}"
                for index in range(1, ROLE_COUNT + 1)
            ],
        ],
        "identity": [
            *[f"actor_identity_{player_id}" for player_id in all_player_ids],
            *[f"lineup_identity_{player_id}" for player_id in all_player_ids],
        ],
        "rolling": [
            *[f"actor_{name}" for name in event_names],
            *[f"lineup_mean_{name}" for name in event_names],
        ],
    }
    for name in arrays:
        if name.startswith("history_") or name.endswith("_history"):
            component_names[name] = history_names
        if name.startswith("shuffled_history_seed_"):
            component_names[name] = history_names
    finite_audit = {
        name: {
            "rows": int(value.shape[0]),
            "features": int(value.shape[1]),
            "finite": bool(np.isfinite(value).all()),
        }
        for name, value in arrays.items()
    }
    chronology_violations = 0
    for appearance in query_appearances:
        prior = [
            item
            for item in appearances_by_team[appearance.team_id]
            if item.match_date < appearance.match_date
        ]
        if any(item.match_date >= appearance.match_date for item in prior):
            chronology_violations += 1
    sample_id_duplicates = len(sample_ids) - len(set(sample_ids))
    audit = {
        "status": (
            "passed"
            if all(item["finite"] for item in finite_audit.values())
            and chronology_violations == 0
            and sample_id_duplicates == 0
            else "failed"
        ),
        "opportunity_count": len(sample_ids),
        "query_partitions": sorted(query_partitions),
        "query_match_ids": sorted(set(match_ids), key=int),
        "team_count": len(set(team_ids)),
        "player_count": len(set(actor_ids)),
        "primary_positive_count": int(sum(primary_labels)),
        "primary_prevalence": float(np.mean(primary_labels)),
        "turnover_positive_count": int(sum(turnover_labels)),
        "turnover_prevalence": float(np.mean(turnover_labels)),
        "missing_actor_history_rows": missing_history_rows,
        "missing_actor_history_fraction": missing_history_rows / len(sample_ids),
        "mean_actor_history_support_matches": float(np.mean(support_counts)),
        "sample_id_duplicates": sample_id_duplicates,
        "chronology_violations": chronology_violations,
        "component_audit": finite_audit,
        "confirmation_match_ids_loaded": data.metadata[
            "confirmation_match_ids_included"
        ],
    }
    return FootpassFeatureDataset(
        sample_ids=sample_ids,
        team_ids=team_ids,
        match_ids=match_ids,
        periods=periods,
        frames=frames,
        actor_ids=actor_ids,
        labels={
            "penalty_area_action_10s": np.asarray(primary_labels, dtype=np.int8),
            "turnover_5s": np.asarray(turnover_labels, dtype=np.int8),
        },
        components=arrays,
        component_feature_names=component_names,
        audit=audit,
    )


@dataclass
class LogisticProbe:
    feature_names: list[str]
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    bias: float
    l2_coefficient: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        if features.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Probe feature mismatch: expected {len(self.feature_names)}, "
                f"got {features.shape[1]}."
            )
        standardized = (features - self.mean) / self.scale
        logits = standardized @ self.weight + self.bias
        logits = np.clip(logits, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-logits))


def fit_logistic_probe(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    probe_config: dict[str, Any],
) -> LogisticProbe:
    """Fit one deterministic, unweighted, L2-regularized logistic probe."""

    x_value = np.asarray(features, dtype=np.float64)
    y_value = np.asarray(labels, dtype=np.float64)
    if len(np.unique(y_value)) != 2:
        raise ValueError("Logistic probe training requires both classes.")
    mean = x_value.mean(axis=0)
    scale = x_value.std(axis=0)
    epsilon = float(probe_config["standardize_epsilon"])
    scale = np.where(scale < epsilon, 1.0, scale)
    standardized = (x_value - mean) / scale

    torch.manual_seed(int(probe_config["fit_seed"]))
    x_tensor = torch.as_tensor(standardized, dtype=torch.float64)
    y_tensor = torch.as_tensor(y_value, dtype=torch.float64)
    weight = torch.zeros(x_value.shape[1], dtype=torch.float64, requires_grad=True)
    prevalence = float(np.clip(y_value.mean(), 1e-6, 1.0 - 1e-6))
    bias = torch.tensor(
        math.log(prevalence / (1.0 - prevalence)),
        dtype=torch.float64,
        requires_grad=True,
    )
    optimizer = torch.optim.LBFGS(
        [weight, bias],
        max_iter=int(probe_config["max_iterations"]),
        tolerance_grad=float(probe_config["tolerance_grad"]),
        tolerance_change=float(probe_config["tolerance_change"]),
        line_search_fn="strong_wolfe",
    )
    l2 = float(probe_config["l2_coefficient"])

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        logits = x_tensor @ weight + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            y_tensor,
        )
        loss = loss + 0.5 * l2 * weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return LogisticProbe(
        feature_names=list(feature_names),
        mean=mean,
        scale=scale,
        weight=weight.detach().numpy(),
        bias=float(bias.detach()),
        l2_coefficient=l2,
    )


def _average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    positive_count = int(labels.sum())
    if positive_count == 0:
        return None
    order = np.argsort(-probabilities, kind="stable")
    ordered_labels = labels[order]
    precision = np.cumsum(ordered_labels) / np.arange(1, len(labels) + 1)
    return float(np.sum(precision * ordered_labels) / positive_count)


def _roc_auc(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(probabilities, kind="stable")
    sorted_probabilities = probabilities[order]
    ranks = np.empty(len(labels), dtype=np.float64)
    start = 0
    while start < len(labels):
        end = start + 1
        while (
            end < len(labels)
            and sorted_probabilities[end] == sorted_probabilities[start]
        ):
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    positive_rank_sum = ranks[labels == 1].sum()
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    ece_bins: int,
) -> dict[str, Any]:
    labels_value = np.asarray(labels, dtype=np.int8)
    probabilities_value = np.clip(
        np.asarray(probabilities, dtype=np.float64),
        1e-9,
        1.0 - 1e-9,
    )
    losses = -(
        labels_value * np.log(probabilities_value)
        + (1 - labels_value) * np.log(1.0 - probabilities_value)
    )
    predictions = probabilities_value >= 0.5
    f1_values: list[float] = []
    for label_value in (0, 1):
        true_positive = int(
            np.sum((predictions == label_value) & (labels_value == label_value))
        )
        false_positive = int(
            np.sum((predictions == label_value) & (labels_value != label_value))
        )
        false_negative = int(
            np.sum((predictions != label_value) & (labels_value == label_value))
        )
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(
            0.0 if denominator == 0 else 2 * true_positive / denominator
        )
    ece = 0.0
    for index in range(int(ece_bins)):
        low = index / ece_bins
        high = (index + 1) / ece_bins
        selected = (probabilities_value >= low) & (
            probabilities_value < high
            if index < ece_bins - 1
            else probabilities_value <= high
        )
        if selected.any():
            ece += float(selected.mean()) * abs(
                float(probabilities_value[selected].mean())
                - float(labels_value[selected].mean())
            )
    return {
        "examples": len(labels_value),
        "positives": int(labels_value.sum()),
        "prevalence": float(labels_value.mean()),
        "nll": float(losses.mean()),
        "brier": float(np.mean((probabilities_value - labels_value) ** 2)),
        "average_precision": _average_precision(
            labels_value, probabilities_value
        ),
        "roc_auc": _roc_auc(labels_value, probabilities_value),
        "macro_f1_at_0p5": float(np.mean(f1_values)),
        "expected_calibration_error": ece,
    }


def blocked_bootstrap_nll_gain(
    labels: np.ndarray,
    baseline_probabilities: np.ndarray,
    history_probabilities: np.ndarray,
    block_ids: list[str],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    labels_value = np.asarray(labels, dtype=np.float64)
    baseline = np.clip(baseline_probabilities, 1e-9, 1.0 - 1e-9)
    history = np.clip(history_probabilities, 1e-9, 1.0 - 1e-9)
    baseline_loss = -(
        labels_value * np.log(baseline)
        + (1.0 - labels_value) * np.log(1.0 - baseline)
    )
    history_loss = -(
        labels_value * np.log(history)
        + (1.0 - labels_value) * np.log(1.0 - history)
    )
    unique_blocks = sorted(set(block_ids))
    block_count = np.asarray(
        [sum(value == block for value in block_ids) for block in unique_blocks],
        dtype=np.float64,
    )
    block_baseline_sum = np.asarray(
        [
            baseline_loss[
                np.asarray([value == block for value in block_ids], dtype=bool)
            ].sum()
            for block in unique_blocks
        ]
    )
    block_history_sum = np.asarray(
        [
            history_loss[
                np.asarray([value == block for value in block_ids], dtype=bool)
            ].sum()
            for block in unique_blocks
        ]
    )
    generator = np.random.default_rng(int(seed))
    gains = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        sampled = generator.integers(0, len(unique_blocks), len(unique_blocks))
        denominator = block_count[sampled].sum()
        gains[replicate] = (
            block_baseline_sum[sampled].sum()
            - block_history_sum[sampled].sum()
        ) / denominator
    return {
        "block_unit": "match_id:period",
        "block_count": len(unique_blocks),
        "replicates": int(replicates),
        "point_gain": float(baseline_loss.mean() - history_loss.mean()),
        "bootstrap_mean_gain": float(gains.mean()),
        "ci95": [
            float(np.quantile(gains, 0.025)),
            float(np.quantile(gains, 0.975)),
        ],
        "positive_fraction": float(np.mean(gains > 0.0)),
    }


def save_logistic_probes(
    path: str | Path,
    probes: dict[str, LogisticProbe],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {"keys": []}
    for index, (key, probe) in enumerate(sorted(probes.items())):
        prefix = f"probe_{index}"
        metadata["keys"].append(
            {
                "key": key,
                "prefix": prefix,
                "feature_names": probe.feature_names,
                "bias": probe.bias,
                "l2_coefficient": probe.l2_coefficient,
            }
        )
        arrays[f"{prefix}_mean"] = probe.mean
        arrays[f"{prefix}_scale"] = probe.scale
        arrays[f"{prefix}_weight"] = probe.weight
    np.savez_compressed(
        output,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        **arrays,
    )
    return output


def load_logistic_probes(path: str | Path) -> dict[str, LogisticProbe]:
    with np.load(Path(path), allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        probes: dict[str, LogisticProbe] = {}
        for item in metadata["keys"]:
            prefix = str(item["prefix"])
            probes[str(item["key"])] = LogisticProbe(
                feature_names=[str(value) for value in item["feature_names"]],
                mean=np.asarray(payload[f"{prefix}_mean"], dtype=np.float64),
                scale=np.asarray(payload[f"{prefix}_scale"], dtype=np.float64),
                weight=np.asarray(payload[f"{prefix}_weight"], dtype=np.float64),
                bias=float(item["bias"]),
                l2_coefficient=float(item["l2_coefficient"]),
            )
        return probes


def _subset_indices(match_ids: list[str], selected: set[str]) -> np.ndarray:
    return np.asarray(
        [index for index, match_id in enumerate(match_ids) if match_id in selected],
        dtype=np.int64,
    )


def run_development_probes(
    dataset: FootpassFeatureDataset,
    config: dict[str, Any],
    *,
    train_match_ids: set[str],
    validation_match_ids: set[str],
) -> tuple[dict[str, Any], dict[str, LogisticProbe], dict[str, np.ndarray]]:
    """Fit the frozen feature ladder and evaluate development validation."""

    if dataset.audit["status"] != "passed":
        raise ValueError("Cannot train probes from a failed feature audit.")
    train_indices = _subset_indices(dataset.match_ids, train_match_ids)
    validation_indices = _subset_indices(dataset.match_ids, validation_match_ids)
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("Development train and validation rows are required.")
    views = dataset.feature_views(
        int(config["features"]["main_history_support_cap"])
    )
    probes: dict[str, LogisticProbe] = {}
    validation_probabilities: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for target, labels in dataset.labels.items():
        metrics[target] = {}
        for view_name, (features, feature_names) in views.items():
            if view_name == "history":
                continue
            probe_key = f"{target}::{view_name}"
            probe = fit_logistic_probe(
                features[train_indices],
                labels[train_indices],
                feature_names,
                config["probe"],
            )
            probabilities = probe.predict(features[validation_indices])
            probes[probe_key] = probe
            validation_probabilities[probe_key] = probabilities
            metrics[target][view_name] = binary_metrics(
                labels[validation_indices],
                probabilities,
                ece_bins=int(config["evaluation"]["ece_bins"]),
            )
        main_history_view = (
            f"history_k{int(config['features']['main_history_support_cap'])}"
        )
        main_key = f"{target}::{main_history_view}"
        history_key = f"{target}::history"
        probes[history_key] = probes[main_key]
        validation_probabilities[history_key] = validation_probabilities[main_key]
        metrics[target]["history"] = dict(metrics[target][main_history_view])

    primary_target = str(config["evaluation"]["primary_target"])
    secondary_target = str(config["evaluation"]["secondary_target"])
    primary_labels = dataset.labels[primary_target][validation_indices]
    primary_baseline = validation_probabilities[f"{primary_target}::rolling"]
    primary_history = validation_probabilities[f"{primary_target}::history"]
    baseline_metrics = metrics[primary_target]["rolling"]
    history_metrics = metrics[primary_target]["history"]
    primary_gain = float(baseline_metrics["nll"] - history_metrics["nll"])
    relative_primary_gain = primary_gain / float(baseline_metrics["nll"])
    block_ids = [
        f"{dataset.match_ids[index]}:{dataset.periods[index]}"
        for index in validation_indices.tolist()
    ]
    bootstrap = blocked_bootstrap_nll_gain(
        primary_labels,
        primary_baseline,
        primary_history,
        block_ids,
        replicates=int(config["evaluation"]["bootstrap_replicates"]),
        seed=int(config["evaluation"]["bootstrap_seed"]),
    )
    per_match: dict[str, dict[str, Any]] = {}
    for match_id in sorted(validation_match_ids, key=int):
        local = np.asarray(
            [
                offset
                for offset, index in enumerate(validation_indices.tolist())
                if dataset.match_ids[index] == match_id
            ],
            dtype=np.int64,
        )
        per_match[match_id] = {
            "examples": len(local),
            "nll_gain": (
                binary_metrics(
                    primary_labels[local],
                    primary_baseline[local],
                    ece_bins=int(config["evaluation"]["ece_bins"]),
                )["nll"]
                - binary_metrics(
                    primary_labels[local],
                    primary_history[local],
                    ece_bins=int(config["evaluation"]["ece_bins"]),
                )["nll"]
            ),
        }
    secondary_baseline_metrics = metrics[secondary_target]["rolling"]
    secondary_history_metrics = metrics[secondary_target]["history"]
    secondary_gain = float(
        secondary_baseline_metrics["nll"] - secondary_history_metrics["nll"]
    )
    secondary_relative_gain = secondary_gain / float(
        secondary_baseline_metrics["nll"]
    )
    shuffle_names = sorted(
        name for name in views if name.startswith("shuffled_history_seed_")
    )
    shuffle_nll = {
        name: float(metrics[primary_target][name]["nll"])
        for name in shuffle_names
    }
    gate_config = config["development_gate"]
    checks = {
        "minimum_primary_relative_nll_improvement": (
            relative_primary_gain
            >= float(gate_config["minimum_primary_relative_nll_improvement"])
        ),
        "positive_primary_bootstrap_lower_bound": (
            float(bootstrap["ci95"][0]) > 0.0
            if bool(
                gate_config["require_positive_primary_bootstrap_lower_bound"]
            )
            else True
        ),
        "minimum_positive_validation_matches": (
            sum(float(value["nll_gain"]) > 0.0 for value in per_match.values())
            >= int(gate_config["minimum_positive_validation_matches"])
        ),
        "better_than_every_history_shuffle": (
            all(float(history_metrics["nll"]) < value for value in shuffle_nll.values())
            if bool(gate_config["require_better_than_every_history_shuffle"])
            else True
        ),
        "primary_brier_noninferiority": (
            float(history_metrics["brier"]) <= float(baseline_metrics["brier"])
            if bool(gate_config["require_primary_brier_noninferiority"])
            else True
        ),
        "secondary_nll_noninferiority": (
            secondary_relative_gain
            >= float(gate_config["minimum_secondary_relative_nll_change"])
        ),
        "integrity_audits": (
            dataset.audit["status"] == "passed"
            if bool(gate_config["require_integrity_audits"])
            else True
        ),
    }
    result = {
        "status": "development_metrics_opened",
        "claim_status": "development_only",
        "metrics": metrics,
        "primary_comparison": {
            "target": primary_target,
            "baseline": "rolling",
            "model": "history",
            "nll_gain": primary_gain,
            "relative_nll_improvement": relative_primary_gain,
            "brier_gain": float(
                baseline_metrics["brier"] - history_metrics["brier"]
            ),
            "blocked_bootstrap": bootstrap,
            "per_match": per_match,
            "shuffle_nll": shuffle_nll,
        },
        "secondary_comparison": {
            "target": secondary_target,
            "nll_gain": secondary_gain,
            "relative_nll_improvement": secondary_relative_gain,
        },
        "gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "thresholds": dict(gate_config),
        },
        "train_examples": len(train_indices),
        "validation_examples": len(validation_indices),
        "feature_audit": dataset.audit,
        "confirmatory_metrics_loaded": False,
    }
    return result, probes, validation_probabilities


def evaluate_frozen_probes(
    dataset: FootpassFeatureDataset,
    config: dict[str, Any],
    probes: dict[str, LogisticProbe],
) -> dict[str, Any]:
    """Apply development-frozen probes once to confirmation rows."""

    if dataset.audit["status"] != "passed":
        raise ValueError("Cannot evaluate a failed confirmation feature audit.")
    views = dataset.feature_views(
        int(config["features"]["main_history_support_cap"])
    )
    metrics: dict[str, dict[str, Any]] = {}
    probabilities: dict[str, np.ndarray] = {}
    for target, labels in dataset.labels.items():
        metrics[target] = {}
        for view_name, (features, feature_names) in views.items():
            key = f"{target}::{view_name}"
            probe = probes[key]
            if probe.feature_names != feature_names:
                raise ValueError(f"Frozen feature-name mismatch for {key}.")
            predicted = probe.predict(features)
            probabilities[key] = predicted
            metrics[target][view_name] = binary_metrics(
                labels,
                predicted,
                ece_bins=int(config["evaluation"]["ece_bins"]),
            )
    primary_target = str(config["evaluation"]["primary_target"])
    secondary_target = str(config["evaluation"]["secondary_target"])
    baseline_metrics = metrics[primary_target]["rolling"]
    history_metrics = metrics[primary_target]["history"]
    primary_gain = float(baseline_metrics["nll"] - history_metrics["nll"])
    relative_primary_gain = primary_gain / float(baseline_metrics["nll"])
    block_ids = [
        f"{match_id}:{period}"
        for match_id, period in zip(
            dataset.match_ids, dataset.periods, strict=True
        )
    ]
    bootstrap = blocked_bootstrap_nll_gain(
        dataset.labels[primary_target],
        probabilities[f"{primary_target}::rolling"],
        probabilities[f"{primary_target}::history"],
        block_ids,
        replicates=int(config["evaluation"]["bootstrap_replicates"]),
        seed=int(config["evaluation"]["bootstrap_seed"]) + 100_000,
    )
    per_match: dict[str, dict[str, Any]] = {}
    for match_id in sorted(set(dataset.match_ids), key=int):
        selected = np.flatnonzero(np.asarray(dataset.match_ids) == match_id)
        baseline_local = binary_metrics(
            dataset.labels[primary_target][selected],
            probabilities[f"{primary_target}::rolling"][selected],
            ece_bins=int(config["evaluation"]["ece_bins"]),
        )
        history_local = binary_metrics(
            dataset.labels[primary_target][selected],
            probabilities[f"{primary_target}::history"][selected],
            ece_bins=int(config["evaluation"]["ece_bins"]),
        )
        per_match[match_id] = {
            "examples": len(selected),
            "nll_gain": float(baseline_local["nll"] - history_local["nll"]),
        }
    secondary_baseline = metrics[secondary_target]["rolling"]
    secondary_history = metrics[secondary_target]["history"]
    secondary_relative_gain = (
        float(secondary_baseline["nll"]) - float(secondary_history["nll"])
    ) / float(secondary_baseline["nll"])
    shuffle_names = sorted(
        name for name in views if name.startswith("shuffled_history_seed_")
    )
    gate_config = config["development_gate"]
    checks = {
        "minimum_primary_relative_nll_improvement": (
            relative_primary_gain
            >= float(gate_config["minimum_primary_relative_nll_improvement"])
        ),
        "positive_primary_bootstrap_lower_bound": float(bootstrap["ci95"][0])
        > 0.0,
        "minimum_positive_confirmation_matches": (
            sum(float(value["nll_gain"]) > 0.0 for value in per_match.values())
            >= int(gate_config["minimum_positive_validation_matches"])
        ),
        "better_than_every_history_shuffle": all(
            float(history_metrics["nll"])
            < float(metrics[primary_target][name]["nll"])
            for name in shuffle_names
        ),
        "primary_brier_noninferiority": float(history_metrics["brier"])
        <= float(baseline_metrics["brier"]),
        "secondary_nll_noninferiority": secondary_relative_gain
        >= float(gate_config["minimum_secondary_relative_nll_change"]),
        "integrity_audits": dataset.audit["status"] == "passed",
    }
    return {
        "status": "confirmatory_metrics_opened_once",
        "metrics": metrics,
        "primary_comparison": {
            "target": primary_target,
            "nll_gain": primary_gain,
            "relative_nll_improvement": relative_primary_gain,
            "blocked_bootstrap": bootstrap,
            "per_match": per_match,
        },
        "secondary_comparison": {
            "target": secondary_target,
            "relative_nll_improvement": secondary_relative_gain,
        },
        "gate": {
            "passed": all(checks.values()),
            "checks": checks,
        },
        "feature_audit": dataset.audit,
        "confirmatory_metrics_loaded": True,
    }
