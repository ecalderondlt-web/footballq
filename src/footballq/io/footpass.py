"""Lazy reader and integrity audit for FOOTPASS tactical HDF5 data."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from footballq.repro.identity import make_sample_id
from footballq.repro.manifest import file_sha256
from footballq.repro.splits import load_split_manifest

try:
    import h5py
except ImportError:  # pragma: no cover - exercised only in incomplete environments
    h5py = None


FOOTPASS_FPS = 25.0
FOOTPASS_COLUMNS_WITH_CLASS = (
    "frame",
    "player_id",
    "left_to_right",
    "shirt_number",
    "role_id",
    "x",
    "y",
    "speed_x",
    "speed_y",
    "roi_x",
    "roi_y",
    "roi_width",
    "roi_height",
    "class",
)
FOOTPASS_COLUMNS_WITHOUT_CLASS = FOOTPASS_COLUMNS_WITH_CLASS[:-1]
FOOTPASS_GEOMETRY_FEATURE_NAMES = ("x_norm", "y_norm", "vx_norm", "vy_norm")
FOOTPASS_PLAYER_IDS = tuple(range(100, 116)) + tuple(range(200, 216))
FOOTPASS_ACTION_CLASSES = {
    0: "background",
    1: "drive",
    2: "pass",
    3: "cross",
    4: "throw_in",
    5: "shot",
    6: "header",
    7: "tackle",
    8: "block",
}

(
    FRAME,
    PLAYER_ID,
    LEFT_TO_RIGHT,
    SHIRT_NUMBER,
    ROLE_ID,
    X,
    Y,
    SPEED_X,
    SPEED_Y,
    ROI_X,
    ROI_Y,
    ROI_WIDTH,
    ROI_HEIGHT,
    ACTION_CLASS,
) = range(14)

_KEY_PATTERN = re.compile(r"^game_(?P<match_id>\d+)_H(?P<period>[12])$")
_PLAYER_SLOT = {player_id: index for index, player_id in enumerate(FOOTPASS_PLAYER_IDS)}


def _require_h5py() -> Any:
    if h5py is None:
        raise ImportError(
            "FOOTPASS HDF5 support requires h5py. Install the project dependencies "
            "with `python -m pip install -e .`."
        )
    return h5py


@dataclass(frozen=True)
class FootpassHalfKey:
    """Parsed match and period identity for one HDF5 dataset key."""

    dataset_key: str
    match_id: str
    period: int

    @classmethod
    def parse(cls, value: str) -> FootpassHalfKey:
        match = _KEY_PATTERN.fullmatch(str(value))
        if match is None:
            raise ValueError(f"Invalid FOOTPASS half key: {value!r}")
        return cls(
            dataset_key=str(value),
            match_id=match.group("match_id"),
            period=int(match.group("period")),
        )

    @classmethod
    def from_components(cls, match_id: str | int, period: int) -> FootpassHalfKey:
        if int(period) not in {1, 2}:
            raise ValueError("FOOTPASS period must be 1 or 2.")
        return cls.parse(f"game_{int(match_id)}_H{int(period)}")


@dataclass(frozen=True)
class FootpassWindow:
    """Fixed-roster view of a bounded FOOTPASS frame interval.

    Identity, tactical role, video ROI, and action labels remain separate from
    the four-channel geometry tensor so callers cannot include them by accident.
    """

    match_id: str
    period: int
    source_key: str
    frame_ids: np.ndarray
    sample_ids: tuple[str, ...]
    player_ids: np.ndarray
    geometry: np.ndarray
    active_mask: np.ndarray
    finite_geometry_mask: np.ndarray
    pitch_bounds_mask: np.ndarray
    team_index: np.ndarray
    left_to_right: np.ndarray
    shirt_number: np.ndarray
    role_id: np.ndarray
    roi_xywh: np.ndarray
    roi_valid_mask: np.ndarray
    action_class: np.ndarray | None

    @property
    def feature_names(self) -> tuple[str, ...]:
        return FOOTPASS_GEOMETRY_FEATURE_NAMES


@dataclass(frozen=True)
class FootpassLineupSignature:
    """Starting-lineup fingerprint for one anonymized match-side appearance."""

    match_id: str
    team_index: int
    period: int
    frame_id: int
    left_to_right: int
    shirt_numbers: tuple[int, ...]
    shirt_role_pairs: tuple[tuple[int, int], ...]

    @property
    def appearance_id(self) -> str:
        return f"{self.match_id}:{self.team_index}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "appearance_id": self.appearance_id,
            "match_id": self.match_id,
            "team_index": self.team_index,
            "period": self.period,
            "frame_id": self.frame_id,
            "left_to_right": self.left_to_right,
            "starter_count": len(self.shirt_numbers),
            "shirt_numbers": list(self.shirt_numbers),
            "shirt_role_pairs": [
                {"shirt_number": shirt, "role_id": role}
                for shirt, role in self.shirt_role_pairs
            ],
        }


class FootpassTacticalStore:
    """Read FOOTPASS tactical rows lazily without materializing the full file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"FOOTPASS HDF5 file not found: {self.path}")
        h5_module = _require_h5py()
        self._file = h5_module.File(self.path, "r")
        self._halves = self._parse_halves()

    def _parse_halves(self) -> tuple[FootpassHalfKey, ...]:
        halves = tuple(
            sorted(
                (FootpassHalfKey.parse(key) for key in self._file.keys()),
                key=lambda half: (int(half.match_id), half.period),
            )
        )
        if not halves:
            raise ValueError(f"FOOTPASS HDF5 file contains no half datasets: {self.path}")
        for half in halves:
            dataset = self._file[half.dataset_key]
            if len(dataset.shape) != 2 or dataset.shape[1] not in {13, 14}:
                raise ValueError(
                    f"FOOTPASS dataset {half.dataset_key!r} has unexpected shape "
                    f"{dataset.shape}; expected (N, 13) or (N, 14)."
                )
        return halves

    @property
    def halves(self) -> tuple[FootpassHalfKey, ...]:
        return self._halves

    @property
    def match_ids(self) -> tuple[str, ...]:
        return tuple(sorted({half.match_id for half in self.halves}, key=int))

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> FootpassTacticalStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _search_frame(dataset: Any, frame_id: int, *, right: bool) -> int:
        low = 0
        high = int(dataset.shape[0])
        while low < high:
            middle = (low + high) // 2
            value = int(dataset[middle, FRAME])
            move_right = value <= frame_id if right else value < frame_id
            if move_right:
                low = middle + 1
            else:
                high = middle
        return low

    def read_window(
        self,
        match_id: str | int,
        period: int,
        start_frame: int,
        end_frame: int,
    ) -> FootpassWindow:
        """Read an inclusive frame interval into fixed match-local roster slots."""

        if int(end_frame) < int(start_frame):
            raise ValueError("end_frame must be greater than or equal to start_frame.")
        half = FootpassHalfKey.from_components(match_id, period)
        if half.dataset_key not in self._file:
            raise KeyError(f"FOOTPASS half not found: {half.dataset_key}")
        dataset = self._file[half.dataset_key]
        start_index = self._search_frame(dataset, int(start_frame), right=False)
        end_index = self._search_frame(dataset, int(end_frame), right=True)
        rows = np.asarray(dataset[start_index:end_index], dtype=np.float32)
        if rows.size == 0:
            raise ValueError(
                f"No FOOTPASS rows found for {half.dataset_key} frames "
                f"{start_frame}..{end_frame}."
            )
        if np.any(np.diff(rows[:, FRAME]) < 0):
            raise ValueError(f"FOOTPASS rows are not frame-sorted in {half.dataset_key}.")

        frame_ids = np.unique(rows[:, FRAME].astype(np.int64))
        frame_lookup = {frame_id: index for index, frame_id in enumerate(frame_ids.tolist())}
        frame_count = len(frame_ids)
        slot_count = len(FOOTPASS_PLAYER_IDS)

        geometry = np.full((frame_count, slot_count, 4), np.nan, dtype=np.float32)
        active_mask = np.zeros((frame_count, slot_count), dtype=bool)
        team_index = np.full((frame_count, slot_count), -1, dtype=np.int8)
        left_to_right = np.full((frame_count, slot_count), -1, dtype=np.int8)
        shirt_number = np.full((frame_count, slot_count), -1, dtype=np.int16)
        role_id = np.full((frame_count, slot_count), -1, dtype=np.int8)
        roi_xywh = np.full((frame_count, slot_count, 4), np.nan, dtype=np.float32)
        action_class = (
            np.zeros((frame_count, slot_count), dtype=np.int8)
            if dataset.shape[1] == len(FOOTPASS_COLUMNS_WITH_CLASS)
            else None
        )

        for row in rows:
            player_id = int(row[PLAYER_ID])
            try:
                slot = _PLAYER_SLOT[player_id]
            except KeyError as exc:
                raise ValueError(f"Unexpected FOOTPASS player_id {player_id}.") from exc
            frame_index = frame_lookup[int(row[FRAME])]
            if active_mask[frame_index, slot]:
                raise ValueError(
                    f"Duplicate FOOTPASS player {player_id} at frame {int(row[FRAME])}."
                )
            active_mask[frame_index, slot] = True
            geometry[frame_index, slot] = row[[X, Y, SPEED_X, SPEED_Y]]
            team_index[frame_index, slot] = 0 if player_id < 200 else 1
            left_to_right[frame_index, slot] = int(row[LEFT_TO_RIGHT])
            shirt_number[frame_index, slot] = int(row[SHIRT_NUMBER])
            role_id[frame_index, slot] = int(row[ROLE_ID])
            roi_xywh[frame_index, slot] = row[[ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT]]
            if action_class is not None:
                action_class[frame_index, slot] = int(row[ACTION_CLASS])

        finite_geometry_mask = active_mask & np.isfinite(geometry).all(axis=-1)
        pitch_bounds_mask = (
            finite_geometry_mask
            & (geometry[..., 0] >= 0.0)
            & (geometry[..., 0] <= 1.0)
            & (geometry[..., 1] >= 0.0)
            & (geometry[..., 1] <= 1.0)
        )
        roi_valid_mask = (
            active_mask
            & np.isfinite(roi_xywh).all(axis=-1)
            & (roi_xywh[..., 2] > 0.0)
            & (roi_xywh[..., 3] > 0.0)
        )
        return FootpassWindow(
            match_id=half.match_id,
            period=half.period,
            source_key=half.dataset_key,
            frame_ids=frame_ids,
            sample_ids=tuple(
                make_sample_id(half.match_id, half.period, frame_id) for frame_id in frame_ids
            ),
            player_ids=np.asarray(FOOTPASS_PLAYER_IDS, dtype=np.int16),
            geometry=geometry,
            active_mask=active_mask,
            finite_geometry_mask=finite_geometry_mask,
            pitch_bounds_mask=pitch_bounds_mask,
            team_index=team_index,
            left_to_right=left_to_right,
            shirt_number=shirt_number,
            role_id=role_id,
            roi_xywh=roi_xywh,
            roi_valid_mask=roi_valid_mask,
            action_class=action_class,
        )


def extract_footpass_lineup_signatures(
    path: str | Path,
) -> tuple[FootpassLineupSignature, ...]:
    """Extract the first observed H1 lineup for both sides of every match."""

    signatures: list[FootpassLineupSignature] = []
    with FootpassTacticalStore(path) as store:
        for match_id in store.match_ids:
            half = FootpassHalfKey.from_components(match_id, 1)
            dataset = store._file[half.dataset_key]
            first_frame = int(dataset[0, FRAME])
            end_index = store._search_frame(dataset, first_frame, right=True)
            rows = np.asarray(dataset[:end_index], dtype=np.float32)
            for team_index in (0, 1):
                player_mask = (
                    rows[:, PLAYER_ID] < 200
                    if team_index == 0
                    else rows[:, PLAYER_ID] >= 200
                )
                team_rows = rows[player_mask]
                if team_rows.size == 0:
                    raise ValueError(
                        f"FOOTPASS {half.dataset_key} has no first-frame rows "
                        f"for team_index={team_index}."
                    )
                shirt_role_pairs = tuple(
                    sorted(
                        {
                            (int(row[SHIRT_NUMBER]), int(row[ROLE_ID]))
                            for row in team_rows
                        }
                    )
                )
                directions = {int(value) for value in team_rows[:, LEFT_TO_RIGHT]}
                if len(directions) != 1:
                    raise ValueError(
                        f"FOOTPASS {half.dataset_key} has inconsistent first-frame "
                        f"directions for team_index={team_index}: {sorted(directions)}."
                    )
                signatures.append(
                    FootpassLineupSignature(
                        match_id=match_id,
                        team_index=team_index,
                        period=1,
                        frame_id=first_frame,
                        left_to_right=directions.pop(),
                        shirt_numbers=tuple(
                            sorted(shirt for shirt, _ in shirt_role_pairs)
                        ),
                        shirt_role_pairs=shirt_role_pairs,
                    )
                )
    return tuple(
        sorted(signatures, key=lambda item: (int(item.match_id), item.team_index))
    )


def rank_footpass_lineup_matches(
    signatures: tuple[FootpassLineupSignature, ...] | list[FootpassLineupSignature],
    *,
    minimum_overlap: int = 5,
    exclude_same_match: bool = True,
) -> list[dict[str, Any]]:
    """Rank candidate repeated-team appearances by IDF-weighted shirt overlap."""

    if minimum_overlap < 1:
        raise ValueError("minimum_overlap must be positive.")
    if len(signatures) < 2:
        return []

    frequency = Counter(
        shirt for signature in signatures for shirt in signature.shirt_numbers
    )
    appearance_count = len(signatures)
    weights = {
        shirt: math.log((1.0 + appearance_count) / (1.0 + count)) + 1.0
        for shirt, count in frequency.items()
    }
    ranked: list[dict[str, Any]] = []
    ordered = sorted(
        signatures, key=lambda item: (int(item.match_id), item.team_index)
    )
    for left_index, left in enumerate(ordered):
        left_shirts = set(left.shirt_numbers)
        for right in ordered[left_index + 1 :]:
            if exclude_same_match and left.match_id == right.match_id:
                continue
            right_shirts = set(right.shirt_numbers)
            overlap = left_shirts & right_shirts
            if len(overlap) < minimum_overlap:
                continue
            union = left_shirts | right_shirts
            weighted_jaccard = sum(weights[shirt] for shirt in overlap) / sum(
                weights[shirt] for shirt in union
            )
            ranked.append(
                {
                    "left_appearance_id": left.appearance_id,
                    "right_appearance_id": right.appearance_id,
                    "weighted_jaccard": weighted_jaccard,
                    "plain_jaccard": len(overlap) / len(union),
                    "overlap_count": len(overlap),
                    "overlap_shirt_numbers": sorted(overlap),
                }
            )
    return sorted(
        ranked,
        key=lambda item: (
            -float(item["weighted_jaccard"]),
            -int(item["overlap_count"]),
            str(item["left_appearance_id"]),
            str(item["right_appearance_id"]),
        ),
    )


def _numeric_range(values: np.ndarray) -> tuple[float | None, float | None, int]:
    finite = np.isfinite(values)
    if not finite.any():
        return None, None, int((~finite).sum())
    return (
        float(values[finite].min()),
        float(values[finite].max()),
        int((~finite).sum()),
    )


def audit_footpass_tactical_data(
    path: str | Path,
    *,
    split_manifest_path: str | Path | None = None,
    full_scan: bool = True,
    hash_source: bool = False,
) -> dict[str, Any]:
    """Return a JSON-serializable source and quality audit."""

    source = Path(path)
    with FootpassTacticalStore(source) as store:
        halves = store.halves
        match_ids = list(store.match_ids)
        split_metadata: dict[str, Any] = {}
        if split_manifest_path is not None:
            split = load_split_manifest(split_manifest_path)
            if split.payload["dataset"] != "footpass":
                raise ValueError(
                    "FOOTPASS audit requires a split manifest with dataset='footpass'."
                )
            source_ids = set(match_ids)
            manifest_ids = set(split.all_match_ids)
            if source_ids != manifest_ids:
                missing = sorted(manifest_ids - source_ids, key=int)
                extra = sorted(source_ids - manifest_ids, key=int)
                raise ValueError(
                    "FOOTPASS source/split match mismatch: "
                    f"missing={missing}, extra={extra}."
                )
            split_metadata = split.metadata()

        period_counts = Counter(half.period for half in halves)
        shape_counts: Counter[str] = Counter()
        dtype_counts: Counter[str] = Counter()
        total_rows = 0
        label_columns_present = True
        for half in halves:
            dataset = store._file[half.dataset_key]
            total_rows += int(dataset.shape[0])
            shape_counts[str(dataset.shape[1])] += 1
            dtype_counts[str(dataset.dtype)] += 1
            label_columns_present &= dataset.shape[1] == 14

        report: dict[str, Any] = {
            "status": "complete" if full_scan else "inventory_only",
            "claim_status": "data_plumbing_only",
            "dataset": "footpass",
            "source_release_split": "train" if label_columns_present else "unknown_or_test",
            "source": {
                "path": str(source.resolve()),
                "size_bytes": source.stat().st_size,
                "mtime_ns": source.stat().st_mtime_ns,
                "sha256": file_sha256(source) if hash_source else None,
            },
            "match_count": len(match_ids),
            "match_ids": match_ids,
            "half_count": len(halves),
            "halves_by_period": {str(key): value for key, value in sorted(period_counts.items())},
            "total_rows": total_rows,
            "column_count_halves": dict(sorted(shape_counts.items())),
            "dtype_halves": dict(sorted(dtype_counts.items())),
            "columns": (
                list(FOOTPASS_COLUMNS_WITH_CLASS)
                if label_columns_present
                else list(FOOTPASS_COLUMNS_WITHOUT_CLASS)
            ),
            "geometry_feature_names": list(FOOTPASS_GEOMETRY_FEATURE_NAMES),
            "geometry_excluded_fields": [
                "player_id",
                "shirt_number",
                "role_id",
                "roi_x",
                "roi_y",
                "roi_width",
                "roi_height",
                "class",
            ],
            "ball_coordinates_present": False,
            "companion_video_count": len(list(source.parent.glob("*.mp4"))),
            "hdf5_root_attributes": {
                str(key): str(value) for key, value in store._file.attrs.items()
            },
            **split_metadata,
        }
        if not full_scan:
            return report

        frame_shape_counts: Counter[int] = Counter()
        class_counts: Counter[int] = Counter()
        total_frames = 0
        total_frame_gaps = 0
        roi_visible_rows = 0
        roi_visible_event_rows = 0
        event_rows = 0
        geometry_nan_rows = 0
        out_of_bounds = Counter()
        coordinate_ranges = {
            name: {"min": None, "max": None, "nan_rows": 0}
            for name in ("x", "y", "speed_x", "speed_y")
        }
        per_half: list[dict[str, Any]] = []
        for half in halves:
            dataset = store._file[half.dataset_key]
            rows = np.asarray(dataset[:], dtype=np.float32)
            frames = rows[:, FRAME].astype(np.int64)
            unique_frames, player_counts = np.unique(frames, return_counts=True)
            gaps = int(np.maximum(np.diff(unique_frames) - 1, 0).sum())
            total_frames += len(unique_frames)
            total_frame_gaps += gaps
            frame_shape_counts.update(player_counts.tolist())

            finite_geometry = np.isfinite(rows[:, [X, Y, SPEED_X, SPEED_Y]]).all(axis=1)
            geometry_nan_rows += int((~finite_geometry).sum())
            roi_visible = np.isfinite(rows[:, [ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT]]).all(axis=1)
            roi_visible_rows += int(roi_visible.sum())

            if dataset.shape[1] == 14:
                classes = rows[:, ACTION_CLASS].astype(np.int64)
                class_counts.update(classes.tolist())
                events = classes != 0
                event_rows += int(events.sum())
                roi_visible_event_rows += int((events & roi_visible).sum())

            out_of_bounds["x_lt_0"] += int((rows[:, X] < 0.0).sum())
            out_of_bounds["x_gt_1"] += int((rows[:, X] > 1.0).sum())
            out_of_bounds["y_lt_0"] += int((rows[:, Y] < 0.0).sum())
            out_of_bounds["y_gt_1"] += int((rows[:, Y] > 1.0).sum())
            out_of_bounds["abs_speed_x_gt_0p5"] += int((np.abs(rows[:, SPEED_X]) > 0.5).sum())
            out_of_bounds["abs_speed_y_gt_0p5"] += int((np.abs(rows[:, SPEED_Y]) > 0.5).sum())

            for name, index in (
                ("x", X),
                ("y", Y),
                ("speed_x", SPEED_X),
                ("speed_y", SPEED_Y),
            ):
                minimum, maximum, nan_rows = _numeric_range(rows[:, index])
                current = coordinate_ranges[name]
                current["nan_rows"] = int(current["nan_rows"]) + nan_rows
                if minimum is not None:
                    current["min"] = (
                        minimum if current["min"] is None else min(float(current["min"]), minimum)
                    )
                    current["max"] = (
                        maximum if current["max"] is None else max(float(current["max"]), maximum)
                    )

            per_half.append(
                {
                    "key": half.dataset_key,
                    "match_id": half.match_id,
                    "period": half.period,
                    "rows": int(rows.shape[0]),
                    "first_frame": int(unique_frames[0]),
                    "last_frame": int(unique_frames[-1]),
                    "unique_frames": len(unique_frames),
                    "frame_gap_count": gaps,
                    "min_players_per_frame": int(player_counts.min()),
                    "max_players_per_frame": int(player_counts.max()),
                    "frames_with_21_players": int((player_counts == 21).sum()),
                    "frames_with_22_players": int((player_counts == 22).sum()),
                }
            )

        report.update(
            {
                "total_unique_frames": total_frames,
                "duration_hours_at_25fps": total_frames / FOOTPASS_FPS / 3600.0,
                "frame_gap_count": total_frame_gaps,
                "frame_player_count_distribution": {
                    str(key): value for key, value in sorted(frame_shape_counts.items())
                },
                "geometry_nan_rows": geometry_nan_rows,
                "coordinate_ranges": coordinate_ranges,
                "coordinate_outlier_counts": dict(sorted(out_of_bounds.items())),
                "coordinate_outlier_fractions": {
                    key: value / total_rows for key, value in sorted(out_of_bounds.items())
                },
                "roi_visible_rows": roi_visible_rows,
                "roi_visible_fraction": roi_visible_rows / total_rows,
                "event_rows": event_rows,
                "event_class_counts": {
                    FOOTPASS_ACTION_CLASSES.get(key, f"unknown_{key}"): value
                    for key, value in sorted(class_counts.items())
                    if key != 0
                },
                "roi_visible_event_rows": roi_visible_event_rows,
                "roi_visible_event_fraction": (
                    roi_visible_event_rows / event_rows if event_rows else None
                ),
                "per_half": per_half,
            }
        )
        return report
