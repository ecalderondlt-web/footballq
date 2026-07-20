"""Leakage-controlled StatsBomb event shards and causal sequence windows."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections import Counter, OrderedDict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler

from footballq.data.statsbomb_events import (
    STATSBOMB_OPEN_DATA_COMMIT,
    file_sha256,
    resolve_statsbomb_data_dir,
    statsbomb_event_payload,
)
from footballq.repro.splits import load_split_manifest

CATEGORICAL_FEATURES = (
    "event_type",
    "play_pattern",
    "position",
    "subtype",
    "outcome",
)
CONTINUOUS_FEATURES = (
    "x_norm",
    "y_norm",
    "end_x_norm",
    "end_y_norm",
    "location_present",
    "location_in_bounds",
    "end_location_present",
    "end_location_in_bounds",
    "duration_log_scaled",
    "delta_time_log_scaled",
    "possession_change",
    "team_is_possession",
    "under_pressure",
    "counterpress",
    "period_scaled",
    "has_360",
    "visible_area_fraction",
)
FREEZE_FRAME_FEATURES = (
    "x_norm",
    "y_norm",
    "teammate",
    "actor",
    "keeper",
    "location_in_bounds",
)
MAX_FREEZE_FRAME_PLAYERS = 22


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _category_maps(schema_audit: dict[str, Any]) -> dict[str, dict[int, int]]:
    return {
        name: {
            int(entry["provider_id"]): int(entry["index"])
            for entry in schema_audit["vocabularies"][name]["entries"]
        }
        for name in CATEGORICAL_FEATURES
    }


def _category_index(value: object, mapping: dict[int, int]) -> int:
    if not isinstance(value, dict) or value.get("id") is None:
        return 0
    return mapping.get(int(value["id"]), 1)


def _normalized_location(value: object) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) < 2:
        return 0.0, 0.0, 0.0, 0.0
    x = float(value[0])
    y = float(value[1])
    finite = math.isfinite(x) and math.isfinite(y)
    in_bounds = finite and 0.0 <= x <= 120.0 and 0.0 <= y <= 80.0
    if not finite:
        return 0.0, 0.0, 1.0, 0.0
    return min(max(x / 120.0, 0.0), 1.0), min(max(y / 80.0, 0.0), 1.0), 1.0, float(
        in_bounds
    )


def _visible_area_fraction(value: object) -> float:
    if not isinstance(value, list) or len(value) < 6 or len(value) % 2:
        return 0.0
    points = [(float(value[index]), float(value[index + 1])) for index in range(0, len(value), 2)]
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return min(max(abs(area) * 0.5 / (120.0 * 80.0), 0.0), 1.0)


def _scaled_log_seconds(value: object, maximum: float = 60.0) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(seconds):
        return 0.0
    return math.log1p(min(max(seconds, 0.0), maximum)) / math.log1p(maximum)


def _event_time_seconds(event: dict[str, Any]) -> float:
    return float(event.get("minute") or 0) * 60.0 + float(event.get("second") or 0)


def _window_starts(periods: torch.Tensor, sequence_length: int, stride: int) -> torch.Tensor:
    starts = []
    begin = 0
    while begin < len(periods):
        end = begin + 1
        while end < len(periods) and int(periods[end]) == int(periods[begin]):
            end += 1
        last_start = end - sequence_length - 1
        if last_start >= begin:
            period_starts = list(range(begin, last_start + 1, stride))
            if period_starts[-1] != last_start:
                period_starts.append(last_start)
            starts.extend(period_starts)
        begin = end
    return torch.tensor(starts, dtype=torch.int64)


def _freeze_frame_tensor(
    players: list[dict[str, Any]],
    *,
    max_players: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(players) > max_players:
        raise ValueError(
            f"StatsBomb 360 frame has {len(players)} players; frozen maximum is {max_players}."
        )
    frame = torch.zeros((max_players, len(FREEZE_FRAME_FEATURES)), dtype=torch.float32)
    mask = torch.zeros(max_players, dtype=torch.bool)
    for index, player in enumerate(players):
        x, y, present, in_bounds = _normalized_location(player.get("location"))
        if not present:
            continue
        frame[index] = torch.tensor(
            [
                x,
                y,
                float(bool(player.get("teammate"))),
                float(bool(player.get("actor"))),
                float(bool(player.get("keeper"))),
                in_bounds,
            ],
            dtype=torch.float32,
        )
        mask[index] = True
    return frame, mask


def prepare_statsbomb_match_shard(
    data_dir: Path,
    match_id: str,
    split: str,
    category_maps: dict[str, dict[int, int]],
    *,
    sequence_length: int,
    stride: int,
    max_freeze_frame_players: int = MAX_FREEZE_FRAME_PLAYERS,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Convert one match into event tensors and period-bounded window starts."""

    events = _read_json(data_dir / "events" / f"{match_id}.json")
    events.sort(key=lambda event: (int(event.get("period") or 0), int(event.get("index") or 0)))
    event_id_to_index = {
        str(event["id"]): index for index, event in enumerate(events) if event.get("id") is not None
    }
    event_to_freeze = torch.full((len(events),), -1, dtype=torch.int64)
    visible_area = torch.zeros(len(events), dtype=torch.float32)
    freeze_frames = []
    freeze_masks = []
    quality: dict[str, int] = {
        "malformed_three_sixty_files": 0,
        "unmatched_three_sixty_rows": 0,
        "duplicate_three_sixty_event_rows": 0,
        "matched_three_sixty_rows": 0,
    }

    frame_path = data_dir / "three-sixty" / f"{match_id}.json"
    if frame_path.is_file():
        try:
            frame_rows = _read_json(frame_path)
        except json.JSONDecodeError:
            frame_rows = []
            quality["malformed_three_sixty_files"] = 1
        for row in frame_rows:
            event_index = event_id_to_index.get(str(row.get("event_uuid")))
            if event_index is None:
                quality["unmatched_three_sixty_rows"] += 1
                continue
            if int(event_to_freeze[event_index]) >= 0:
                quality["duplicate_three_sixty_event_rows"] += 1
                continue
            frame, mask = _freeze_frame_tensor(
                row.get("freeze_frame") or [],
                max_players=max_freeze_frame_players,
            )
            event_to_freeze[event_index] = len(freeze_frames)
            visible_area[event_index] = _visible_area_fraction(row.get("visible_area"))
            freeze_frames.append(frame)
            freeze_masks.append(mask)
            quality["matched_three_sixty_rows"] += 1

    categorical = torch.zeros((len(events), len(CATEGORICAL_FEATURES)), dtype=torch.int64)
    continuous = torch.zeros((len(events), len(CONTINUOUS_FEATURES)), dtype=torch.float32)
    periods = torch.zeros(len(events), dtype=torch.int64)
    previous_period = None
    previous_time = 0.0
    previous_possession = None
    for index, event in enumerate(events):
        payload = statsbomb_event_payload(event)
        categorical[index] = torch.tensor(
            [
                _category_index(event.get("type"), category_maps["event_type"]),
                _category_index(event.get("play_pattern"), category_maps["play_pattern"]),
                _category_index(event.get("position"), category_maps["position"]),
                _category_index(payload.get("type"), category_maps["subtype"]),
                _category_index(payload.get("outcome"), category_maps["outcome"]),
            ],
            dtype=torch.int64,
        )
        period = int(event.get("period") or 0)
        periods[index] = period
        current_time = _event_time_seconds(event)
        delta_time = current_time - previous_time if previous_period == period else 0.0
        possession = event.get("possession")
        possession_change = (
            previous_period == period
            and previous_possession is not None
            and possession is not None
            and possession != previous_possession
        )
        team_id = (event.get("team") or {}).get("id")
        possession_team_id = (event.get("possession_team") or {}).get("id")
        team_is_possession = (
            team_id is not None and possession_team_id is not None and team_id == possession_team_id
        )
        x, y, location_present, location_in_bounds = _normalized_location(event.get("location"))
        end_x, end_y, end_present, end_in_bounds = _normalized_location(
            payload.get("end_location")
        )
        has_360 = float(int(event_to_freeze[index]) >= 0)
        continuous[index] = torch.tensor(
            [
                x,
                y,
                end_x,
                end_y,
                location_present,
                location_in_bounds,
                end_present,
                end_in_bounds,
                _scaled_log_seconds(event.get("duration")),
                _scaled_log_seconds(delta_time),
                float(possession_change),
                float(team_is_possession),
                float(bool(event.get("under_pressure"))),
                float(bool(event.get("counterpress"))),
                min(max(float(period) / 5.0, 0.0), 1.0),
                has_360,
                float(visible_area[index]),
            ],
            dtype=torch.float32,
        )
        previous_period = period
        previous_time = current_time
        previous_possession = possession

    shard = {
        "version": 1,
        "dataset": "statsbomb_open_data",
        "source_commit": STATSBOMB_OPEN_DATA_COMMIT,
        "match_id": match_id,
        "split": split,
        "categorical_feature_names": list(CATEGORICAL_FEATURES),
        "continuous_feature_names": list(CONTINUOUS_FEATURES),
        "freeze_frame_feature_names": list(FREEZE_FRAME_FEATURES),
        "sequence_length": sequence_length,
        "stride": stride,
        "max_freeze_frame_players": max_freeze_frame_players,
        "categorical": categorical,
        "continuous": continuous,
        "period": periods,
        "event_to_freeze": event_to_freeze,
        "freeze_frame": (
            torch.stack(freeze_frames)
            if freeze_frames
            else torch.zeros(
                (0, max_freeze_frame_players, len(FREEZE_FRAME_FEATURES)),
                dtype=torch.float32,
            )
        ),
        "freeze_mask": (
            torch.stack(freeze_masks)
            if freeze_masks
            else torch.zeros((0, max_freeze_frame_players), dtype=torch.bool)
        ),
        "window_starts": _window_starts(periods, sequence_length, stride),
    }
    return shard, quality


def _load_source_hashes(source_manifest: dict[str, Any]) -> dict[str, str]:
    if not source_manifest.get("file_hashes_complete"):
        raise ValueError("StatsBomb source manifest does not contain complete file hashes.")
    return {str(row["path"]): str(row["sha256"]) for row in source_manifest["files"]}


def prepare_statsbomb_event_dataset(
    raw_root: str | Path,
    split_path: str | Path,
    source_manifest_path: str | Path,
    schema_audit_path: str | Path,
    out_dir: str | Path,
    *,
    sequence_length: int = 32,
    stride: int = 16,
    include_splits: tuple[str, ...] = ("train", "val"),
    match_limits: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Prepare resumable train/validation shards; test is deliberately unsupported."""

    if not include_splits or any(split not in {"train", "val"} for split in include_splits):
        raise ValueError("StatsBomb event preparation permits only train and val splits.")
    if sequence_length < 2 or stride < 1:
        raise ValueError("StatsBomb sequence_length must be >=2 and stride must be positive.")
    split = load_split_manifest(split_path)
    source_manifest_file = Path(source_manifest_path)
    source_manifest = json.loads(source_manifest_file.read_text(encoding="utf-8"))
    schema_audit_file = Path(schema_audit_path)
    schema_audit = json.loads(schema_audit_file.read_text(encoding="utf-8"))
    if schema_audit.get("scope") != "train_only" or schema_audit.get("loaded_splits") != [
        "train"
    ]:
        raise ValueError("StatsBomb vocabulary audit must be train-only.")
    if schema_audit.get("split_manifest_sha256") != split.sha256:
        raise ValueError("StatsBomb schema audit split hash does not match the requested split.")
    if source_manifest.get("split_manifest_sha256") != split.sha256:
        raise ValueError("StatsBomb source manifest split hash does not match the requested split.")

    data_dir = resolve_statsbomb_data_dir(raw_root)
    output = Path(out_dir)
    source_hashes = _load_source_hashes(source_manifest)
    category_maps = _category_maps(schema_audit)
    shards = []
    quality_totals: Counter[str] = Counter()
    split_counts: dict[str, dict[str, int]] = {}
    match_limits = match_limits or {}

    for split_name in include_splits:
        match_ids = list(getattr(split, f"{split_name}_match_ids"))
        if split_name in match_limits:
            match_ids = match_ids[: int(match_limits[split_name])]
        counters: Counter[str] = Counter()
        for match_id in match_ids:
            relative_path = Path("shards") / split_name / f"{match_id}.pt"
            path = output / relative_path
            event_source_path = f"events/{match_id}.json"
            frame_source_path = f"three-sixty/{match_id}.json"
            expected = {
                "match_id": match_id,
                "split": split_name,
                "sequence_length": sequence_length,
                "stride": stride,
                "source_event_sha256": source_hashes[event_source_path],
                "source_three_sixty_sha256": source_hashes.get(frame_source_path),
            }
            shard = None
            if path.is_file():
                existing = torch.load(path, map_location="cpu", weights_only=False)
                if all(existing.get(key) == value for key, value in expected.items()):
                    shard = existing
            if shard is None:
                shard, quality = prepare_statsbomb_match_shard(
                    data_dir,
                    match_id,
                    split_name,
                    category_maps,
                    sequence_length=sequence_length,
                    stride=stride,
                )
                shard.update(expected)
                shard["quality"] = quality
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(shard, path)
            quality_totals.update(shard["quality"])
            event_count = int(shard["categorical"].shape[0])
            window_count = int(shard["window_starts"].shape[0])
            counters["matches"] += 1
            counters["events"] += event_count
            counters["windows"] += window_count
            counters["matched_three_sixty_rows"] += int(shard["freeze_frame"].shape[0])
            shards.append(
                {
                    "match_id": match_id,
                    "split": split_name,
                    "path": relative_path.as_posix(),
                    "event_count": event_count,
                    "window_count": window_count,
                    "matched_three_sixty_rows": int(shard["freeze_frame"].shape[0]),
                    "source_event_sha256": expected["source_event_sha256"],
                    "source_three_sixty_sha256": expected["source_three_sixty_sha256"],
                    "tensor_sha256": file_sha256(path),
                }
            )
        split_counts[split_name] = dict(counters)

    manifest = {
        "version": 1,
        "dataset": "statsbomb_open_data",
        "source_commit": STATSBOMB_OPEN_DATA_COMMIT,
        "source_manifest_path": str(source_manifest_file),
        "source_manifest_sha256": file_sha256(source_manifest_file),
        "source_manifest_payload_sha256": source_manifest["manifest_payload_sha256"],
        "schema_audit_path": str(schema_audit_file),
        "schema_audit_sha256": file_sha256(schema_audit_file),
        "schema_audit_payload_sha256": schema_audit["audit_payload_sha256"],
        "vocabulary_payload_sha256": schema_audit["vocabulary_payload_sha256"],
        "split_manifest_path": str(split.path),
        "split_manifest_sha256": split.sha256,
        "loaded_splits": list(include_splits),
        "test_loaded": False,
        "sequence_length": sequence_length,
        "stride": stride,
        "categorical_feature_names": list(CATEGORICAL_FEATURES),
        "continuous_feature_names": list(CONTINUOUS_FEATURES),
        "freeze_frame_feature_names": list(FREEZE_FRAME_FEATURES),
        "max_freeze_frame_players": MAX_FREEZE_FRAME_PLAYERS,
        "categorical_vocabularies": schema_audit["vocabularies"],
        "match_limits": match_limits or None,
        "split_counts": split_counts,
        "quality_totals": dict(quality_totals),
        "tensor_hashes_complete": True,
        "shards": shards,
    }
    manifest["manifest_payload_sha256"] = _stable_hash(manifest)
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def audit_statsbomb_event_dataset(
    manifest_path: str | Path,
    *,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Validate every prepared train/validation tensor shard and causal window."""

    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "train_and_validation_only": manifest.get("loaded_splits") == ["train", "val"],
        "test_not_loaded": manifest.get("test_loaded") is False,
        "tensor_hashes_complete": manifest.get("tensor_hashes_complete") is True,
    }
    split = load_split_manifest(manifest["split_manifest_path"])
    expected_ids = {
        "train": set(split.train_match_ids),
        "val": set(split.val_match_ids),
    }
    actual_ids = {
        split_name: {
            str(row["match_id"])
            for row in manifest["shards"]
            if row["split"] == split_name
        }
        for split_name in ("train", "val")
    }
    if manifest.get("match_limits") is None:
        checks["complete_train_match_coverage"] = actual_ids["train"] == expected_ids["train"]
        checks["complete_val_match_coverage"] = actual_ids["val"] == expected_ids["val"]
    checks["no_test_match_shards"] = not any(
        str(row["match_id"]) in set(split.test_match_ids) for row in manifest["shards"]
    )
    counters = {
        split_name: Counter(
            {
                "matches": 0,
                "events": 0,
                "windows": 0,
                "matched_three_sixty_rows": 0,
            }
        )
        for split_name in ("train", "val")
    }
    failures = []
    sequence_length = int(manifest["sequence_length"])
    vocabularies = manifest["categorical_vocabularies"]
    categorical_names = manifest["categorical_feature_names"]
    continuous_names = manifest["continuous_feature_names"]
    has_360_index = continuous_names.index("has_360")
    for row in manifest["shards"]:
        split_name = str(row["split"])
        shard_path = path.parent / row["path"]
        if not shard_path.is_file():
            failures.append(f"missing:{row['path']}")
            continue
        if verify_hashes and file_sha256(shard_path) != row["tensor_sha256"]:
            failures.append(f"hash:{row['path']}")
            continue
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        categorical = shard["categorical"]
        continuous = shard["continuous"]
        periods = shard["period"]
        event_to_freeze = shard["event_to_freeze"]
        freeze_frame = shard["freeze_frame"]
        freeze_mask = shard["freeze_mask"]
        starts = shard["window_starts"]
        event_count = int(categorical.shape[0])
        frame_count = int(freeze_frame.shape[0])
        valid = True
        valid &= tuple(categorical.shape) == (event_count, len(categorical_names))
        valid &= tuple(continuous.shape) == (event_count, len(continuous_names))
        valid &= tuple(periods.shape) == (event_count,)
        valid &= tuple(event_to_freeze.shape) == (event_count,)
        valid &= tuple(freeze_frame.shape[1:]) == (
            int(manifest["max_freeze_frame_players"]),
            len(manifest["freeze_frame_feature_names"]),
        )
        valid &= tuple(freeze_mask.shape) == (
            frame_count,
            int(manifest["max_freeze_frame_players"]),
        )
        valid &= bool(torch.isfinite(continuous).all())
        valid &= bool(torch.isfinite(freeze_frame).all())
        valid &= int(event_to_freeze.min()) >= -1 if event_count else True
        valid &= int(event_to_freeze.max()) < frame_count if frame_count else bool(
            (event_to_freeze == -1).all()
        )
        valid &= bool(
            torch.equal(
                continuous[:, has_360_index] > 0.5,
                event_to_freeze >= 0,
            )
        )
        if len(starts):
            valid &= int(starts.min()) >= 0
            valid &= int(starts.max()) + sequence_length < event_count
            valid &= bool(
                torch.equal(
                    periods[starts],
                    periods[starts + sequence_length],
                )
            )
        for index, name in enumerate(categorical_names):
            values = categorical[:, index]
            valid &= int(values.min()) >= 0 if event_count else True
            valid &= int(values.max()) < int(vocabularies[name]["size"]) if event_count else True
            counters[split_name][f"{name}_missing"] += int((values == 0).sum())
            counters[split_name][f"{name}_unknown"] += int((values == 1).sum())
        if not valid:
            failures.append(f"shape_or_value:{row['path']}")
        counters[split_name]["matches"] += 1
        counters[split_name]["events"] += event_count
        counters[split_name]["windows"] += int(starts.shape[0])
        counters[split_name]["matched_three_sixty_rows"] += frame_count

    checks["all_shards_valid"] = not failures
    for split_name in ("train", "val"):
        expected = manifest["split_counts"][split_name]
        for field in ("matches", "events", "windows", "matched_three_sixty_rows"):
            checks[f"{split_name}_{field}_matches_manifest"] = (
                counters[split_name][field] == int(expected[field])
            )
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    audit = {
        "version": 1,
        "dataset": "statsbomb_open_data",
        "scope": "processed_train_and_validation_only",
        "loaded_splits": ["train", "val"],
        "test_loaded": False,
        "manifest_path": str(path),
        "manifest_sha256": file_sha256(path),
        "manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "verify_hashes": bool(verify_hashes),
        "status": "passed" if not failed_checks else "blocked",
        "checks": checks,
        "failed_checks": failed_checks,
        "failures": failures,
        "split_counts": {
            split_name: dict(counters[split_name]) for split_name in ("train", "val")
        },
    }
    audit["audit_payload_sha256"] = _stable_hash(audit)
    return audit


class ShardedStatsBombEventDataset(Dataset):
    """Map-style causal event windows backed by per-match tensor shards."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        *,
        cache_size: int = 2,
        verify_hashes_on_load: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if split not in {"train", "val"}:
            raise ValueError("StatsBomb event dataset exposes only train and val splits.")
        if split not in self.manifest["loaded_splits"]:
            raise ValueError(f"StatsBomb event manifest contains no prepared {split!r} split.")
        self.split = split
        self.cache_size = max(1, int(cache_size))
        self.verify_hashes_on_load = bool(verify_hashes_on_load)
        self.shards = [row for row in self.manifest["shards"] if row["split"] == split]
        self.ends = []
        offset = 0
        for shard in self.shards:
            offset += int(shard["window_count"])
            self.ends.append(offset)
        self.window_count = offset
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return self.window_count

    def _load_shard(self, index: int) -> dict[str, Any]:
        cached = self._cache.get(index)
        if cached is not None:
            self._cache.move_to_end(index)
            return cached
        row = self.shards[index]
        path = self.manifest_path.parent / row["path"]
        if self.verify_hashes_on_load and file_sha256(path) != row["tensor_sha256"]:
            raise ValueError(f"StatsBomb event tensor hash mismatch: {path}")
        shard = torch.load(path, map_location="cpu", weights_only=False)
        self._cache[index] = shard
        self._cache.move_to_end(index)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return shard

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.ends, index)
        previous_end = self.ends[shard_index - 1] if shard_index else 0
        shard = self._load_shard(shard_index)
        start = int(shard["window_starts"][index - previous_end])
        length = int(self.manifest["sequence_length"])
        input_slice = slice(start, start + length)
        target_slice = slice(start + 1, start + length + 1)
        event_to_freeze = shard["event_to_freeze"][input_slice]
        freeze_frame = torch.zeros(
            (
                length,
                int(self.manifest["max_freeze_frame_players"]),
                len(self.manifest["freeze_frame_feature_names"]),
            ),
            dtype=torch.float32,
        )
        freeze_mask = torch.zeros(
            (length, int(self.manifest["max_freeze_frame_players"])), dtype=torch.bool
        )
        present = event_to_freeze >= 0
        if bool(present.any()):
            selected = event_to_freeze[present]
            freeze_frame[present] = shard["freeze_frame"][selected]
            freeze_mask[present] = shard["freeze_mask"][selected]
        target_continuous = shard["continuous"][target_slice]
        return {
            "categorical": shard["categorical"][input_slice],
            "continuous": shard["continuous"][input_slice],
            "event_mask": torch.ones(length, dtype=torch.bool),
            "freeze_frame": freeze_frame,
            "freeze_mask": freeze_mask,
            "has_360": present,
            "target_event_type": shard["categorical"][target_slice, 0],
            "target_location": target_continuous[:, :2],
            "target_location_mask": (target_continuous[:, 4] > 0.5)
            & (target_continuous[:, 5] > 0.5),
            "match_id": str(shard["match_id"]),
            "period": int(shard["period"][start]),
            "start_event_index": start,
        }


class StatsBombShardGroupedSampler(Sampler[int]):
    """Shuffle matches and windows while retaining match-local disk reads."""

    def __init__(
        self,
        dataset: ShardedStatsBombEventDataset,
        *,
        shuffle: bool,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        shard_order = list(range(len(self.dataset.shards)))
        if self.shuffle:
            shard_order = torch.randperm(len(shard_order), generator=generator).tolist()
        for shard_index in shard_order:
            start = self.dataset.ends[shard_index - 1] if shard_index else 0
            end = self.dataset.ends[shard_index]
            if self.shuffle:
                local_order = torch.randperm(end - start, generator=generator).tolist()
                yield from (start + local_index for local_index in local_order)
            else:
                yield from range(start, end)
