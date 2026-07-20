"""Resumable canonical sharding and quality control for PFF tracking data."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from footballq.constants import PITCH_LENGTH_M, PITCH_WIDTH_M
from footballq.io.pff import (
    discover_pff_tracking_files,
    iter_pff_records,
    pff_match_id,
    pff_record_to_tracking_rows,
)
from footballq.repro.splits import SplitManifest, load_split_manifest, stable_json_bytes
from footballq.schema import canonical_tracking_frame

PFF_CANONICAL_SHARD_VERSION = 2


class PFFRosterSlotTracker:
    """Assign changing jersey identities to eleven stable slots per team."""

    def __init__(self) -> None:
        self._jersey_to_slot: dict[str, dict[str, int]] = {"home": {}, "away": {}}

    def assign(self, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        output = [row for row in rows if row.get("agent_type") != "player"]
        dropped = 0
        for team_id in ("home", "away"):
            players = [
                row
                for row in rows
                if row.get("agent_type") == "player" and row.get("team_id") == team_id
            ]
            mapping = self._jersey_to_slot[team_id]
            current_jerseys = [str(row.get("jersey_number")) for row in players]
            occupied = {
                mapping[jersey] for jersey in current_jerseys if jersey in mapping
            }
            free_slots = [slot for slot in range(11) if slot not in occupied]
            for row in players:
                jersey = str(row.get("jersey_number"))
                slot = mapping.get(jersey)
                if slot is None:
                    if not free_slots:
                        dropped += 1
                        continue
                    slot = free_slots.pop(0)
                    for prior_jersey, prior_slot in list(mapping.items()):
                        if prior_slot == slot:
                            del mapping[prior_jersey]
                    mapping[jersey] = slot
                provider_agent_id = str(row["agent_id"])
                updated = {
                    **row,
                    "provider_agent_id": provider_agent_id,
                    "agent_id": f"{team_id}_slot_{slot:02d}",
                    "entity_id": f"{team_id}_slot_{slot:02d}",
                    "slot_assignment": "dynamic_roster_slot_v1",
                }
                output.append(updated)
        return output, dropped


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(stable_json_bytes(payload)).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _split_for_match(split: SplitManifest, match_id: str) -> str:
    if match_id in set(split.train_match_ids):
        return "train"
    if match_id in set(split.val_match_ids):
        return "val"
    if match_id in set(split.test_match_ids):
        return "test"
    raise ValueError(f"PFF match {match_id} is absent from split manifest {split.name}.")


def _duplicate_jerseys(value: object) -> bool:
    if not isinstance(value, list):
        return False
    jerseys = [str(item.get("jerseyNum")) for item in value if isinstance(item, dict)]
    return len(jerseys) != len(set(jerseys))


def _source_fingerprint(path: Path, *, hash_source: bool) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": file_sha256(path) if hash_source else None,
    }


def _resume_matches(
    manifest: dict[str, Any],
    *,
    source: Path,
    split_manifest_sha256: str,
    frames_per_shard: int,
    max_frames: int | None,
    use_smoothed: bool,
) -> bool:
    source_info = manifest.get("source", {})
    stat = source.stat()
    return bool(
        manifest.get("status") == "complete"
        and manifest.get("version") == PFF_CANONICAL_SHARD_VERSION
        and source_info.get("size_bytes") == stat.st_size
        and source_info.get("mtime_ns") == stat.st_mtime_ns
        and manifest.get("split_manifest_sha256") == split_manifest_sha256
        and manifest.get("frames_per_shard") == frames_per_shard
        and manifest.get("max_frames") == max_frames
        and manifest.get("coordinate_variant") == ("smoothed" if use_smoothed else "raw")
    )


def prepare_pff_match_shards(
    source: str | Path,
    output_root: str | Path,
    *,
    split_name: str,
    split_manifest_sha256: str,
    frames_per_shard: int = 6_000,
    max_frames: int | None = None,
    use_smoothed: bool = True,
    hash_source: bool = True,
    resume: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Stream one match into period-aware Parquet shards and a QC manifest."""

    if frames_per_shard < 1:
        raise ValueError("frames_per_shard must be positive.")
    source_path = Path(source)
    match_id = pff_match_id(source_path)
    root = Path(output_root)
    final_dir = root / split_name / match_id
    manifest_path = final_dir / "manifest.json"
    if manifest_path.exists() and resume and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _resume_matches(
            existing,
            source=source_path,
            split_manifest_sha256=split_manifest_sha256,
            frames_per_shard=frames_per_shard,
            max_frames=max_frames,
            use_smoothed=use_smoothed,
        ):
            return existing
    if final_dir.exists() and not force:
        raise FileExistsError(
            f"Existing PFF output does not match this run: {final_dir}. "
            "Use force=True to replace it."
        )

    incomplete_dir = root / split_name / f".{match_id}.incomplete"
    if incomplete_dir.exists():
        shutil.rmtree(incomplete_dir)
    incomplete_dir.mkdir(parents=True, exist_ok=False)

    records_read = 0
    unique_frames = 0
    duplicate_records = 0
    duplicated_player_array_records = 0
    dropped_extra_player_rows = 0
    missing_ball_frames = 0
    non_23_entity_frames = 0
    out_of_bounds_rows = 0
    frame_gap_count = 0
    missing_frame_count = 0
    time_regression_count = 0
    visibility_counts: Counter[str] = Counter()
    rows_by_period: Counter[int] = Counter()
    frames_by_period: Counter[int] = Counter()
    period_frame_bounds: dict[int, list[int]] = {}
    period_time_bounds: dict[int, list[float]] = {}
    previous_frame: dict[int, int] = {}
    previous_time: dict[int, float] = {}
    seen_frames: set[tuple[str, int, int]] = set()
    shard_rows: list[dict[str, Any]] = []
    shard_frame_count = 0
    shard_period: int | None = None
    shard_indices: defaultdict[int, int] = defaultdict(int)
    shards: list[dict[str, Any]] = []
    slot_tracker = PFFRosterSlotTracker()

    def flush_shard() -> None:
        nonlocal shard_rows, shard_frame_count
        if not shard_rows or shard_period is None:
            return
        shard_index = shard_indices[shard_period]
        shard_indices[shard_period] += 1
        frame_ids = [int(row["frame_id"]) for row in shard_rows]
        filename = f"tracking_p{shard_period}_s{shard_index:04d}.parquet"
        destination = incomplete_dir / filename
        temporary = incomplete_dir / f".{filename}.tmp.parquet"
        frame = canonical_tracking_frame(pd.DataFrame(shard_rows))
        frame.to_parquet(temporary, index=False)
        temporary.replace(destination)
        shards.append(
            {
                "path": filename,
                "period": shard_period,
                "shard_index": shard_index,
                "frame_count": shard_frame_count,
                "row_count": len(frame),
                "start_frame": min(frame_ids),
                "end_frame": max(frame_ids),
                "sha256": file_sha256(destination),
            }
        )
        shard_rows = []
        shard_frame_count = 0

    try:
        fallback_match_id = match_id
        for record in iter_pff_records(source_path):
            records_read += 1
            if _duplicate_jerseys(record.get("homePlayersSmoothed")) or _duplicate_jerseys(
                record.get("awayPlayersSmoothed")
            ):
                duplicated_player_array_records += 1
            record_match_id = str(record.get("gameRefId") or fallback_match_id)
            period = int(record.get("period", 1))
            frame_id = int(record.get("frameNum", -1))
            frame_key = (record_match_id, period, frame_id)
            if frame_key in seen_frames:
                duplicate_records += 1
                continue
            if max_frames is not None and unique_frames >= max_frames:
                break
            seen_frames.add(frame_key)

            if shard_period is not None and period != shard_period:
                flush_shard()
            shard_period = period
            frame_rows = pff_record_to_tracking_rows(
                record,
                source_file=source_path,
                fallback_match_id=fallback_match_id,
                use_smoothed=use_smoothed,
            )
            frame_rows, dropped_rows = slot_tracker.assign(frame_rows)
            dropped_extra_player_rows += dropped_rows
            unique_frames += 1
            shard_frame_count += 1
            shard_rows.extend(frame_rows)
            frames_by_period[period] += 1
            rows_by_period[period] += len(frame_rows)
            if len(frame_rows) != 23:
                non_23_entity_frames += 1
            if not any(row["agent_type"] == "ball" for row in frame_rows):
                missing_ball_frames += 1

            time_s = pd.to_numeric(record.get("periodElapsedTime"), errors="coerce")
            bounds = period_frame_bounds.setdefault(period, [frame_id, frame_id])
            bounds[0] = min(bounds[0], frame_id)
            bounds[1] = max(bounds[1], frame_id)
            if pd.notna(time_s):
                time_value = float(time_s)
                time_bounds = period_time_bounds.setdefault(period, [time_value, time_value])
                time_bounds[0] = min(time_bounds[0], time_value)
                time_bounds[1] = max(time_bounds[1], time_value)
                if period in previous_time and time_value < previous_time[period]:
                    time_regression_count += 1
                previous_time[period] = time_value
            if period in previous_frame and frame_id > previous_frame[period] + 1:
                frame_gap_count += 1
                missing_frame_count += frame_id - previous_frame[period] - 1
            previous_frame[period] = frame_id

            for row in frame_rows:
                visibility_counts[str(row.get("provider_visibility", "UNKNOWN"))] += 1
                x_m = pd.to_numeric(row.get("x_m"), errors="coerce")
                y_m = pd.to_numeric(row.get("y_m"), errors="coerce")
                if pd.notna(x_m) and pd.notna(y_m) and (
                    float(x_m) < 0
                    or float(x_m) > PITCH_LENGTH_M
                    or float(y_m) < 0
                    or float(y_m) > PITCH_WIDTH_M
                ):
                    out_of_bounds_rows += 1
            if shard_frame_count >= frames_per_shard:
                flush_shard()
        flush_shard()

        source_info = _source_fingerprint(source_path, hash_source=hash_source)
        manifest: dict[str, Any] = {
            "status": "complete",
            "version": PFF_CANONICAL_SHARD_VERSION,
            "dataset": "pff_fc",
            "match_id": match_id,
            "split": split_name,
            "split_manifest_sha256": split_manifest_sha256,
            "source": source_info,
            "coordinate_variant": "smoothed" if use_smoothed else "raw",
            "pitch_dimensions_m": [PITCH_LENGTH_M, PITCH_WIDTH_M],
            "pitch_dimensions_source": "pff_spec_105x68_example",
            "fps_source": "inferred_from_tracking_timestamps",
            "frames_per_shard": frames_per_shard,
            "max_frames": max_frames,
            "records_read": records_read,
            "unique_frames": unique_frames,
            "duplicate_records": duplicate_records,
            "duplicated_player_array_records": duplicated_player_array_records,
            "dropped_extra_player_rows": dropped_extra_player_rows,
            "missing_ball_frames": missing_ball_frames,
            "non_23_entity_frames": non_23_entity_frames,
            "out_of_bounds_rows": out_of_bounds_rows,
            "frame_gap_count": frame_gap_count,
            "missing_frame_count": missing_frame_count,
            "time_regression_count": time_regression_count,
            "visibility_counts": dict(sorted(visibility_counts.items())),
            "frames_by_period": {
                str(key): value for key, value in sorted(frames_by_period.items())
            },
            "rows_by_period": {str(key): value for key, value in sorted(rows_by_period.items())},
            "period_frame_bounds": {
                str(key): value for key, value in sorted(period_frame_bounds.items())
            },
            "period_time_bounds_s": {
                str(key): value for key, value in sorted(period_time_bounds.items())
            },
            "shards": shards,
        }
        manifest["manifest_payload_sha256"] = _payload_sha256(manifest)
        _write_json(incomplete_dir / "manifest.json", manifest)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        if final_dir.exists():
            shutil.rmtree(final_dir)
        incomplete_dir.replace(final_dir)
        return manifest
    except Exception:
        _write_json(
            incomplete_dir / "failure.json",
            {
                "status": "failed",
                "match_id": match_id,
                "records_read": records_read,
                "unique_frames": unique_frames,
            },
        )
        raise


def prepare_pff_dataset_shards(
    raw_dir: str | Path,
    output_root: str | Path,
    split_manifest_path: str | Path,
    *,
    match_ids: list[str] | None = None,
    split_names: list[str] | None = None,
    frames_per_shard: int = 6_000,
    max_frames: int | None = None,
    use_smoothed: bool = True,
    hash_source: bool = True,
    resume: bool = True,
    force: bool = False,
    workers: int = 1,
) -> dict[str, Any]:
    """Prepare selected PFF matches and write a dataset-level manifest."""

    split = load_split_manifest(split_manifest_path)
    if split.payload["dataset"] != "pff_fc":
        raise ValueError("PFF sharding requires a split manifest with dataset='pff_fc'.")
    discovered = discover_pff_tracking_files(raw_dir)
    missing_inventory = sorted(set(split.all_match_ids) - set(discovered))
    if missing_inventory:
        raise ValueError(
            "PFF raw inventory is missing split matches: " + ", ".join(missing_inventory)
        )

    selected = list(match_ids or split.all_match_ids)
    unknown = sorted(set(selected) - set(split.all_match_ids))
    if unknown:
        raise ValueError(
            "Requested matches are absent from the split manifest: " + ", ".join(unknown)
        )
    allowed_splits = set(split_names or ["train", "val", "test"])
    selected = [
        match_id
        for match_id in selected
        if _split_for_match(split, match_id) in allowed_splits
    ]

    if workers < 1:
        raise ValueError("workers must be positive.")

    tasks = []
    for match_id in selected:
        split_name = _split_for_match(split, match_id)
        tasks.append(
            (
                discovered[match_id],
                {
                    "split_name": split_name,
                    "split_manifest_sha256": split.sha256,
                    "frames_per_shard": frames_per_shard,
                    "max_frames": max_frames,
                    "use_smoothed": use_smoothed,
                    "hash_source": hash_source,
                    "resume": resume,
                    "force": force,
                },
            )
        )
    if workers == 1:
        manifests = [
            prepare_pff_match_shards(source, output_root, **kwargs) for source, kwargs in tasks
        ]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(prepare_pff_match_shards, source, output_root, **kwargs)
                for source, kwargs in tasks
            ]
            manifests = [future.result() for future in futures]

    aggregate = {
        "status": "complete",
        "version": PFF_CANONICAL_SHARD_VERSION,
        "dataset": "pff_fc",
        "raw_dir": str(Path(raw_dir).resolve()),
        "output_root": str(Path(output_root).resolve()),
        "split_manifest_path": str(Path(split_manifest_path)),
        "split_manifest_sha256": split.sha256,
        "selected_match_ids": selected,
        "selected_match_count": len(selected),
        "complete_inventory_match_count": len(discovered),
        "frames_per_shard": frames_per_shard,
        "workers": workers,
        "max_frames": max_frames,
        "coordinate_variant": "smoothed" if use_smoothed else "raw",
        "totals": {
            key: sum(int(manifest.get(key, 0)) for manifest in manifests)
            for key in (
                "records_read",
                "unique_frames",
                "duplicate_records",
                "duplicated_player_array_records",
                "dropped_extra_player_rows",
                "missing_ball_frames",
                "non_23_entity_frames",
                "out_of_bounds_rows",
                "frame_gap_count",
                "missing_frame_count",
                "time_regression_count",
            )
        },
        "matches": [
            {
                "match_id": manifest["match_id"],
                "split": manifest["split"],
                "manifest_payload_sha256": manifest["manifest_payload_sha256"],
                "unique_frames": manifest["unique_frames"],
                "shard_count": len(manifest["shards"]),
            }
            for manifest in manifests
        ],
    }
    aggregate["manifest_payload_sha256"] = _payload_sha256(aggregate)
    _write_json(Path(output_root) / "dataset_manifest.json", aggregate)
    return aggregate
