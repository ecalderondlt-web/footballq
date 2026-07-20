"""Provider-explicit PFF event histories for frozen StatsBomb context studies."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from footballq.constants import PITCH_LENGTH_M, PITCH_WIDTH_M
from footballq.data.sharded_td_dataset import ShardedTDJEPADataset
from footballq.data.statsbomb_event_dataset import (
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
)
from footballq.io.pff import (
    discover_pff_tracking_files,
    iter_pff_records,
    pff_xy_to_meters,
)
from footballq.repro.manifest import file_sha256
from footballq.repro.splits import load_split_manifest

PFF_POSSESSION_EVENT_MAP = {
    "PA": "Pass",
    "CH": "Duel",
    "BC": "Carry",
    "CL": "Clearance",
    "CR": "Pass",
    "RE": "Ball Receipt*",
    "SH": "Shot",
}
PFF_GAME_EVENT_MAP = {
    "FIRSTKICKOFF": "Half Start",
    "SECONDKICKOFF": "Half Start",
    "END": "Half End",
    "SUB": "Substitution",
    "OFF": "Player Off",
    "ON": "Player On",
}
PFF_EXCLUDED_GAME_EVENTS = {"OTB"}
RAW_EVENT_CONTEXT_DIM = 128
EVENT_HISTORY_LENGTH = 32


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def pff_event_mapping_payload() -> dict[str, Any]:
    payload = {
        "version": 1,
        "possession_event_map": PFF_POSSESSION_EVENT_MAP,
        "game_event_map": PFF_GAME_EVENT_MAP,
        "excluded_game_events": sorted(PFF_EXCLUDED_GAME_EVENTS),
        "unknown_policy": "retain_as_statsbomb_unknown_event_type",
        "duplicate_policy": "first_unique_provider_event_id",
        "ordering": "period_then_event_frame_then_possession_before_game_then_provider_id",
    }
    payload["mapping_payload_sha256"] = _stable_hash(payload)
    return payload


def _statsbomb_event_indices(manifest: dict[str, Any]) -> dict[str, int]:
    vocabulary = manifest["categorical_vocabularies"]["event_type"]
    return {
        str(entry["name"]): int(entry["index"])
        for entry in vocabulary["entries"]
    }


def _scaled_log_seconds(value: object, maximum: float = 60.0) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(seconds):
        return 0.0
    return math.log1p(min(max(seconds, 0.0), maximum)) / math.log1p(maximum)


def _ball_location(record: dict[str, Any]) -> tuple[float, float, float, float]:
    value = record.get("ballsSmoothed")
    balls = [value] if isinstance(value, dict) else value if isinstance(value, list) else []
    ball = next((item for item in balls if isinstance(item, dict)), None)
    if ball is None:
        return 0.0, 0.0, 0.0, 0.0
    x_m, y_m = pff_xy_to_meters(ball.get("x"), ball.get("y"))
    finite = math.isfinite(x_m) and math.isfinite(y_m)
    if not finite:
        return 0.0, 0.0, 1.0, 0.0
    in_bounds = 0.0 <= x_m <= PITCH_LENGTH_M and 0.0 <= y_m <= PITCH_WIDTH_M
    return (
        min(max(x_m / PITCH_LENGTH_M, 0.0), 1.0),
        min(max(y_m / PITCH_WIDTH_M, 0.0), 1.0),
        1.0,
        float(in_bounds),
    )


def _event_candidates(record: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for priority, kind, id_key, payload_key, type_key in (
        (0, "possession", "possession_event_id", "possession_event", "possession_event_type"),
        (1, "game", "game_event_id", "game_event", "game_event_type"),
    ):
        payload = record.get(payload_key) or {}
        provider_id = record.get(id_key)
        provider_code = str(payload.get(type_key) or "").strip().upper()
        if not payload or provider_id is None or not provider_code:
            continue
        if kind == "game" and provider_code in PFF_EXCLUDED_GAME_EVENTS:
            continue
        event_frame = payload.get("start_frame")
        if event_frame is None:
            event_frame = record.get("frameNum")
        candidates.append(
            {
                "kind": kind,
                "priority": priority,
                "provider_id": str(provider_id),
                "provider_code": provider_code,
                "payload": payload,
                "frame_id": int(event_frame),
            }
        )
    return candidates


def build_pff_event_match_shard(
    source_path: str | Path,
    *,
    match_id: str,
    split: str,
    statsbomb_manifest: dict[str, Any],
    source_sha256: str,
) -> dict[str, Any]:
    """Build one period-aware PFF event shard using the frozen provider map."""

    event_indices = _statsbomb_event_indices(statsbomb_manifest)
    mapping = pff_event_mapping_payload()
    seen: set[tuple[str, str]] = set()
    rows = []
    excluded_counts: Counter[str] = Counter()
    raw_code_counts: Counter[str] = Counter()
    duplicate_counts: Counter[str] = Counter()
    for record in iter_pff_records(source_path):
        game_payload = record.get("game_event") or {}
        game_code = str(game_payload.get("game_event_type") or "").strip().upper()
        if game_code in PFF_EXCLUDED_GAME_EVENTS:
            game_id = record.get("game_event_id")
            key = ("game", str(game_id))
            if game_id is not None and key not in seen:
                seen.add(key)
                excluded_counts[game_code] += 1
        for candidate in _event_candidates(record):
            key = (candidate["kind"], candidate["provider_id"])
            if key in seen:
                duplicate_counts[candidate["kind"]] += 1
                continue
            seen.add(key)
            code = candidate["provider_code"]
            raw_code_counts[f"{candidate['kind']}:{code}"] += 1
            target_name = (
                PFF_POSSESSION_EVENT_MAP.get(code)
                if candidate["kind"] == "possession"
                else PFF_GAME_EVENT_MAP.get(code)
            )
            target_index = event_indices.get(target_name, 1) if target_name else 1
            x, y, location_present, location_in_bounds = _ball_location(record)
            period = int(record.get("period") or 0)
            event_time = record.get("periodElapsedTime")
            try:
                time_s = float(event_time)
            except (TypeError, ValueError):
                time_s = 0.0
            home_team = game_payload.get("home_team")
            rows.append(
                {
                    **candidate,
                    "period": period,
                    "time_s": time_s if math.isfinite(time_s) else 0.0,
                    "statsbomb_name": target_name,
                    "statsbomb_index": int(target_index),
                    "x": x,
                    "y": y,
                    "location_present": location_present,
                    "location_in_bounds": location_in_bounds,
                    "home_team": home_team if isinstance(home_team, bool) else None,
                }
            )

    rows.sort(
        key=lambda row: (
            row["period"],
            row["frame_id"],
            row["priority"],
            row["provider_id"],
        )
    )
    categorical = torch.zeros((len(rows), len(CATEGORICAL_FEATURES)), dtype=torch.int64)
    continuous = torch.zeros((len(rows), len(CONTINUOUS_FEATURES)), dtype=torch.float32)
    periods = torch.zeros(len(rows), dtype=torch.int64)
    frames = torch.zeros(len(rows), dtype=torch.int64)
    previous_period = None
    previous_time = 0.0
    previous_team = None
    mapped_count = 0
    for index, row in enumerate(rows):
        categorical[index, 0] = int(row["statsbomb_index"])
        periods[index] = int(row["period"])
        frames[index] = int(row["frame_id"])
        delta_time = row["time_s"] - previous_time if previous_period == row["period"] else 0.0
        team = row["home_team"]
        possession_change = (
            previous_period == row["period"]
            and previous_team is not None
            and team is not None
            and previous_team != team
        )
        continuous[index] = torch.tensor(
            [
                row["x"],
                row["y"],
                0.0,
                0.0,
                row["location_present"],
                row["location_in_bounds"],
                0.0,
                0.0,
                _scaled_log_seconds(row["payload"].get("duration")),
                _scaled_log_seconds(delta_time),
                float(possession_change),
                float(row["kind"] == "possession"),
                0.0,
                0.0,
                min(max(float(row["period"]) / 5.0, 0.0), 1.0),
                0.0,
                0.0,
            ],
            dtype=torch.float32,
        )
        if int(row["statsbomb_index"]) > 1:
            mapped_count += 1
        previous_period = row["period"]
        previous_time = row["time_s"]
        previous_team = team

    return {
        "version": 1,
        "dataset": "pff_fc",
        "match_id": str(match_id),
        "split": split,
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "statsbomb_manifest_payload_sha256": statsbomb_manifest["manifest_payload_sha256"],
        "statsbomb_vocabulary_payload_sha256": statsbomb_manifest[
            "vocabulary_payload_sha256"
        ],
        "mapping_payload_sha256": mapping["mapping_payload_sha256"],
        "categorical_feature_names": list(CATEGORICAL_FEATURES),
        "continuous_feature_names": list(CONTINUOUS_FEATURES),
        "period": periods,
        "frame_id": frames,
        "categorical": categorical,
        "continuous": continuous,
        "provider_kind": [row["kind"] for row in rows],
        "provider_code": [row["provider_code"] for row in rows],
        "provider_event_id": [row["provider_id"] for row in rows],
        "statsbomb_event_name": [row["statsbomb_name"] for row in rows],
        "quality": {
            "event_count": len(rows),
            "mapped_event_count": mapped_count,
            "unknown_event_count": len(rows) - mapped_count,
            "excluded_event_count": sum(excluded_counts.values()),
            "excluded_code_counts": dict(sorted(excluded_counts.items())),
            "raw_code_counts": dict(sorted(raw_code_counts.items())),
            "repeated_record_event_references": dict(sorted(duplicate_counts.items())),
        },
    }


def _prepare_match_task(task: dict[str, Any]) -> dict[str, Any]:
    destination = Path(task["destination"])
    expected = task["expected"]
    if destination.is_file():
        existing = torch.load(destination, map_location="cpu", weights_only=False)
        if all(existing.get(key) == value for key, value in expected.items()):
            return {
                "match_id": task["match_id"],
                "split": task["split"],
                "path": task["relative_path"],
                "tensor_sha256": file_sha256(destination),
                "quality": existing["quality"],
            }
    shard = build_pff_event_match_shard(
        task["source_path"],
        match_id=task["match_id"],
        split=task["split"],
        statsbomb_manifest=task["statsbomb_manifest"],
        source_sha256=task["source_sha256"],
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(shard, destination)
    return {
        "match_id": task["match_id"],
        "split": task["split"],
        "path": task["relative_path"],
        "tensor_sha256": file_sha256(destination),
        "quality": shard["quality"],
    }


def prepare_pff_event_context_dataset(
    raw_root: str | Path,
    canonical_root: str | Path,
    split_path: str | Path,
    statsbomb_manifest_path: str | Path,
    out_dir: str | Path,
    *,
    include_splits: tuple[str, ...] = ("train",),
    workers: int = 1,
) -> dict[str, Any]:
    """Prepare resumable PFF event shards; test preparation is forbidden."""

    if not include_splits or any(split not in {"train", "val"} for split in include_splits):
        raise ValueError("PFF event-context preparation permits only train and val splits.")
    split = load_split_manifest(split_path)
    statsbomb_path = Path(statsbomb_manifest_path)
    statsbomb_manifest = json.loads(statsbomb_path.read_text(encoding="utf-8"))
    mapping = pff_event_mapping_payload()
    output = Path(out_dir)
    discovered = discover_pff_tracking_files(raw_root)
    canonical = Path(canonical_root)
    tasks = []
    for split_name in include_splits:
        for match_id in getattr(split, f"{split_name}_match_ids"):
            if match_id not in discovered:
                raise FileNotFoundError(f"PFF raw event source missing match {match_id}.")
            match_manifest_path = canonical / split_name / match_id / "manifest.json"
            match_manifest = json.loads(match_manifest_path.read_text(encoding="utf-8"))
            source_sha256 = str(match_manifest["source"]["sha256"])
            relative = Path("shards") / split_name / f"{match_id}.pt"
            tasks.append(
                {
                    "match_id": match_id,
                    "split": split_name,
                    "source_path": str(discovered[match_id]),
                    "source_sha256": source_sha256,
                    "destination": str(output / relative),
                    "relative_path": relative.as_posix(),
                    "statsbomb_manifest": statsbomb_manifest,
                    "expected": {
                        "match_id": match_id,
                        "split": split_name,
                        "source_sha256": source_sha256,
                        "statsbomb_manifest_payload_sha256": statsbomb_manifest[
                            "manifest_payload_sha256"
                        ],
                        "mapping_payload_sha256": mapping["mapping_payload_sha256"],
                    },
                }
            )

    rows = []
    if int(workers) > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            futures = [executor.submit(_prepare_match_task, task) for task in tasks]
            for future in as_completed(futures):
                rows.append(future.result())
    else:
        rows = [_prepare_match_task(task) for task in tasks]
    rows.sort(key=lambda row: (row["split"], int(row["match_id"])))

    split_counts = {}
    quality_totals: Counter[str] = Counter()
    code_counts: Counter[str] = Counter()
    for split_name in include_splits:
        selected = [row for row in rows if row["split"] == split_name]
        counts: Counter[str] = Counter({"matches": len(selected)})
        for row in selected:
            quality = row["quality"]
            for name in (
                "event_count",
                "mapped_event_count",
                "unknown_event_count",
                "excluded_event_count",
            ):
                counts[name] += int(quality[name])
                quality_totals[name] += int(quality[name])
            code_counts.update(quality["raw_code_counts"])
        split_counts[split_name] = dict(counts)

    manifest = {
        "version": 1,
        "dataset": "pff_fc",
        "scope": "provider_event_histories_for_frozen_statsbomb_context",
        "raw_root": str(Path(raw_root)),
        "canonical_root": str(canonical),
        "split_manifest_path": str(split.path),
        "split_manifest_sha256": split.sha256,
        "statsbomb_manifest_path": str(statsbomb_path),
        "statsbomb_manifest_sha256": file_sha256(statsbomb_path),
        "statsbomb_manifest_payload_sha256": statsbomb_manifest["manifest_payload_sha256"],
        "statsbomb_vocabulary_payload_sha256": statsbomb_manifest[
            "vocabulary_payload_sha256"
        ],
        "mapping": mapping,
        "loaded_splits": list(include_splits),
        "test_loaded": False,
        "tensor_hashes_complete": True,
        "split_counts": split_counts,
        "quality_totals": dict(quality_totals),
        "raw_code_counts": dict(sorted(code_counts.items())),
        "shards": rows,
    }
    manifest["manifest_payload_sha256"] = _stable_hash(manifest)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def audit_pff_event_context_dataset(
    manifest_path: str | Path,
    *,
    require_train_only: bool,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Validate PFF event tensors and provider mapping coverage."""

    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected_splits = ["train"] if require_train_only else ["train", "val"]
    split = load_split_manifest(manifest["split_manifest_path"])
    checks: dict[str, bool] = {
        "expected_splits_only": manifest.get("loaded_splits") == expected_splits,
        "test_not_loaded": manifest.get("test_loaded") is False,
        "tensor_hashes_complete": manifest.get("tensor_hashes_complete") is True,
        "mapping_hash_matches": manifest["mapping"]["mapping_payload_sha256"]
        == pff_event_mapping_payload()["mapping_payload_sha256"],
    }
    failures = []
    counts: dict[str, Counter[str]] = {name: Counter() for name in expected_splits}
    actual_ids = {name: set() for name in expected_splits}
    for row in manifest["shards"]:
        split_name = str(row["split"])
        shard_path = path.parent / row["path"]
        if split_name not in counts:
            failures.append(f"unexpected_split:{row['path']}")
            continue
        actual_ids[split_name].add(str(row["match_id"]))
        if not shard_path.is_file():
            failures.append(f"missing:{row['path']}")
            continue
        if verify_hashes and file_sha256(shard_path) != row["tensor_sha256"]:
            failures.append(f"hash:{row['path']}")
            continue
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        event_count = int(shard["categorical"].shape[0])
        valid = tuple(shard["categorical"].shape) == (
            event_count,
            len(CATEGORICAL_FEATURES),
        )
        valid &= tuple(shard["continuous"].shape) == (
            event_count,
            len(CONTINUOUS_FEATURES),
        )
        valid &= tuple(shard["period"].shape) == (event_count,)
        valid &= tuple(shard["frame_id"].shape) == (event_count,)
        valid &= bool(torch.isfinite(shard["continuous"]).all())
        provider_keys = set(
            zip(shard["provider_kind"], shard["provider_event_id"], strict=True)
        )
        valid &= len(provider_keys) == event_count
        if event_count > 1:
            keys = torch.stack([shard["period"], shard["frame_id"]], dim=1)
            ordered = (keys[1:, 0] > keys[:-1, 0]) | (
                (keys[1:, 0] == keys[:-1, 0]) & (keys[1:, 1] >= keys[:-1, 1])
            )
            valid &= bool(
                ordered.all()
            )
        if not valid:
            failures.append(f"shape_or_value:{row['path']}")
        counts[split_name].update(
            {
                "matches": 1,
                "event_count": event_count,
                "mapped_event_count": int(shard["quality"]["mapped_event_count"]),
                "unknown_event_count": int(shard["quality"]["unknown_event_count"]),
                "excluded_event_count": int(shard["quality"]["excluded_event_count"]),
            }
        )

    for split_name in expected_splits:
        expected_ids = set(getattr(split, f"{split_name}_match_ids"))
        checks[f"complete_{split_name}_coverage"] = actual_ids[split_name] == expected_ids
        for field, value in manifest["split_counts"][split_name].items():
            checks[f"{split_name}_{field}_matches_manifest"] = counts[split_name][field] == int(
                value
            )
    checks["all_shards_valid"] = not failures
    failed = sorted(name for name, passed in checks.items() if not passed)
    report = {
        "version": 1,
        "dataset": "pff_fc",
        "scope": (
            "train_only_mapping_audit"
            if require_train_only
            else "train_validation_tensor_audit"
        ),
        "loaded_splits": expected_splits,
        "test_loaded": False,
        "manifest_path": str(path),
        "manifest_sha256": file_sha256(path),
        "manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "mapping_payload_sha256": manifest["mapping"]["mapping_payload_sha256"],
        "verify_hashes": bool(verify_hashes),
        "status": "passed" if not failed else "blocked",
        "checks": checks,
        "failed_checks": failed,
        "failures": failures,
        "split_counts": {name: dict(value) for name, value in counts.items()},
        "raw_code_counts": manifest["raw_code_counts"],
    }
    report["audit_payload_sha256"] = _stable_hash(report)
    return report


def pff_event_history_from_shard(
    shard: dict[str, Any],
    *,
    period: int,
    cutoff_frame: int,
    sequence_length: int = EVENT_HISTORY_LENGTH,
    event_type_vocab_size: int = 37,
    raw_context_dim: int = RAW_EVENT_CONTEXT_DIM,
) -> dict[str, torch.Tensor | int]:
    """Return the last causal event history at or before a tracking cutoff."""

    selected_period = shard["period"] == int(period)
    selected_indices = torch.nonzero(selected_period, as_tuple=False).flatten()
    if len(selected_indices):
        period_frames = shard["frame_id"][selected_indices]
        end = int(
            torch.searchsorted(
                period_frames,
                torch.tensor(int(cutoff_frame), dtype=period_frames.dtype),
                right=True,
            )
        )
        selected_indices = selected_indices[max(0, end - int(sequence_length)) : end]

    count = len(selected_indices)
    categorical = torch.zeros(
        (int(sequence_length), len(CATEGORICAL_FEATURES)), dtype=torch.int64
    )
    continuous = torch.zeros(
        (int(sequence_length), len(CONTINUOUS_FEATURES)), dtype=torch.float32
    )
    event_mask = torch.zeros(int(sequence_length), dtype=torch.bool)
    raw_context = torch.zeros(int(raw_context_dim), dtype=torch.float32)
    last_frame = -1
    if count:
        categorical[:count] = shard["categorical"][selected_indices]
        continuous[:count] = shard["continuous"][selected_indices]
        event_mask[:count] = True
        event_types = categorical[:count, 0].clamp(0, int(event_type_vocab_size) - 1)
        last_type = int(event_types[-1])
        raw_context[last_type] = 1.0
        counts = torch.bincount(event_types, minlength=int(event_type_vocab_size)).float()
        raw_context[event_type_vocab_size : 2 * event_type_vocab_size] = counts / count
        last_frame = int(shard["frame_id"][selected_indices[-1]])
        raw_context[2 * event_type_vocab_size] = count / float(sequence_length)
        raw_context[2 * event_type_vocab_size + 1] = min(
            max((int(cutoff_frame) - last_frame) / 1800.0, 0.0),
            1.0,
        )
        raw_context[2 * event_type_vocab_size + 2] = float((event_types == 1).sum()) / count
        raw_context[2 * event_type_vocab_size + 3] = float((event_types > 1).sum()) / count
        raw_context[2 * event_type_vocab_size + 4] = min(max(int(period) / 5.0, 0.0), 1.0)
    return {
        "event_categorical": categorical,
        "event_continuous": continuous,
        "event_mask": event_mask,
        "raw_event_context": raw_context,
        "event_history_size": int(count),
        "event_last_frame": int(last_frame),
        "event_cutoff_frame": int(cutoff_frame),
    }


class PFFTrackingEventContextDataset(Dataset):
    """Join finalized PFF TD shards to causal same-match provider event histories."""

    def __init__(
        self,
        tracking_manifest_path: str | Path,
        event_manifest_path: str | Path,
        split: str,
        *,
        tracking_cache_size: int = 1,
        event_cache_size: int = 2,
        sequence_length: int = EVENT_HISTORY_LENGTH,
        raw_context_dim: int = RAW_EVENT_CONTEXT_DIM,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("PFF tracking/event context dataset exposes only train and val.")
        self.tracking = ShardedTDJEPADataset(
            tracking_manifest_path,
            split,
            cache_size=tracking_cache_size,
        )
        self.tracking_manifest_path = Path(tracking_manifest_path)
        self.event_manifest_path = Path(event_manifest_path)
        self.event_manifest = json.loads(
            self.event_manifest_path.read_text(encoding="utf-8")
        )
        if self.event_manifest.get("test_loaded") is not False:
            raise ValueError("PFF event manifest must explicitly record test_loaded=false.")
        if split not in self.event_manifest.get("loaded_splits", []):
            raise ValueError(f"PFF event manifest contains no {split!r} event tensors.")
        tracking_split_hash = self.tracking.manifest["split_manifest_sha256"]
        if self.event_manifest["split_manifest_sha256"] != tracking_split_hash:
            raise ValueError("PFF tracking and event manifests use different split hashes.")
        self.split = split
        self.sequence_length = int(sequence_length)
        self.raw_context_dim = int(raw_context_dim)
        self.event_cache_size = max(1, int(event_cache_size))
        self.event_rows = {
            str(row["match_id"]): row
            for row in self.event_manifest["shards"]
            if row["split"] == split
        }
        tracking_match_ids = {str(row["match_id"]) for row in self.tracking.shards}
        if set(self.event_rows) != tracking_match_ids:
            raise ValueError("PFF event shards do not exactly cover tracking matches for split.")
        self._event_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.shards = self.tracking.shards
        self.shard_starts = self.tracking.shard_starts
        self.shard_ends = self.tracking.shard_ends
        self.prototype = self.tracking.prototype

    def __len__(self) -> int:
        return len(self.tracking)

    def _load_event_shard(self, match_id: str) -> dict[str, Any]:
        cached = self._event_cache.get(match_id)
        if cached is not None:
            self._event_cache.move_to_end(match_id)
            return cached
        row = self.event_rows[match_id]
        path = self.event_manifest_path.parent / row["path"]
        shard = torch.load(path, map_location="cpu", weights_only=False)
        self._event_cache[match_id] = shard
        self._event_cache.move_to_end(match_id)
        while len(self._event_cache) > self.event_cache_size:
            self._event_cache.popitem(last=False)
        return shard

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.tracking[index]
        context_frames = item["context_frame_indices"]
        if context_frames.numel() == 0:
            raise ValueError("PFF TD example has no context frame identities.")
        cutoff = int(context_frames.max())
        if cutoff < int(item["frame_t"]):
            raise ValueError("PFF event cutoff precedes the tracking sample identity.")
        history = pff_event_history_from_shard(
            self._load_event_shard(str(item["match_id"])),
            period=int(item["period"]),
            cutoff_frame=cutoff,
            sequence_length=self.sequence_length,
            raw_context_dim=self.raw_context_dim,
        )
        if int(history["event_last_frame"]) > cutoff:
            raise ValueError("PFF event history includes information after the context cutoff.")
        return {**item, **history}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_event_cache"] = OrderedDict()
        return state
