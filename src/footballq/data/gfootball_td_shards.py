"""Prepare checksummed, split-aware GRF TD-JEPA tensor shards."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from footballq.data.synthetic_visibility import apply_pff_like_visibility
from footballq.data.td_jepa_dataset import (
    build_td_jepa_examples,
    load_td_jepa_data,
    save_td_jepa_data,
)
from footballq.data.windows import _selected_times, _with_causal_velocity
from footballq.io.gfootball import GFootballAdapter
from footballq.io.gfootball_curriculum import (
    collection_plan_sha256,
    job_match_ids,
    job_match_prefix,
    load_collection_plan,
)
from footballq.repro.feature_views import POSITION_ONLY, apply_feature_view
from footballq.repro.splits import load_split_manifest, stable_json_bytes

PROVIDER_VELOCITY = "provider"
CAUSAL_POSITION_DIFFERENCE = "causal_position_difference"
EVENT_SEGMENTED_CAUSAL_POSITION_DIFFERENCE = "event_segmented_causal_position_difference"
JUMP_SEGMENTED_CAUSAL_POSITION_DIFFERENCE_0P5S = (
    "jump_segmented_causal_position_difference_0p5s"
)
EVENT_BOUNDARY_WINDOW_FRAMES = 5
EVENT_SEGMENTATION_IMPLEMENTATION_VERSION = 1
JUMP_SEGMENTATION_IMPLEMENTATION_VERSION = 1
PLAYER_JUMP_THRESHOLD_M = 3.0
BALL_JUMP_THRESHOLD_M = 10.0
LOW_FREQUENCY_VELOCITY_LAG_FRAMES = 5
VELOCITY_MODES = {
    PROVIDER_VELOCITY,
    CAUSAL_POSITION_DIFFERENCE,
    EVENT_SEGMENTED_CAUSAL_POSITION_DIFFERENCE,
    JUMP_SEGMENTED_CAUSAL_POSITION_DIFFERENCE_0P5S,
}


def unsegmented_example_counts_by_match(
    tracking: pd.DataFrame,
    *,
    fps_out: float,
    context_seconds: float,
    stride_seconds: float,
    prediction_gap_seconds: float,
) -> dict[str, int]:
    """Count provider-window candidates without materializing a second tensor view."""

    context_steps = max(1, int(round(context_seconds * fps_out)))
    stride_steps = max(1, int(round(stride_seconds * fps_out)))
    prediction_gap_frames = max(0, int(round(prediction_gap_seconds * fps_out)))
    total_steps = context_steps + prediction_gap_frames + context_steps
    counts: dict[str, int] = {}
    for (match_id, _period), period_df in tracking.groupby(
        ["match_id", "period"], dropna=False, sort=False
    ):
        times = _selected_times(period_df["time_s"].to_numpy(dtype=float), fps_out=fps_out)
        count = max(0, (len(times) - total_steps) // stride_steps + 1)
        counts[str(match_id)] = counts.get(str(match_id), 0) + count
    return counts


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(stable_json_bytes(payload)).hexdigest()


def apply_event_boundary_segmentation(
    tracking: pd.DataFrame,
    *,
    window_frames: int = EVENT_BOUNDARY_WINDOW_FRAMES,
) -> tuple[pd.DataFrame, dict[str, Any], set[tuple[str, int, int]]]:
    """Remove event-proximate frames and label contiguous safe temporal runs."""

    required = {
        "match_id",
        "period",
        "frame_id",
        "game_mode",
        "score_home",
        "score_away",
    }
    missing = sorted(required - set(tracking.columns))
    if missing:
        raise ValueError(f"Event segmentation requires GRF frame metadata: {missing}")
    if window_frames < 0:
        raise ValueError("window_frames must be non-negative.")

    frame_columns = [
        "match_id",
        "period",
        "frame_id",
        "game_mode",
        "score_home",
        "score_away",
    ]
    frame_metadata = tracking[frame_columns].drop_duplicates(
        ["match_id", "period", "frame_id"]
    )
    assignments: list[pd.DataFrame] = []
    unsafe_keys: set[tuple[str, int, int]] = set()
    summary = {
        "event_boundary_window_frames": int(window_frames),
        "total_frame_count": 0,
        "event_signal_frame_count": 0,
        "nonzero_game_mode_frame_count": 0,
        "game_mode_change_frame_count": 0,
        "score_change_frame_count": 0,
        "unsafe_frame_count": 0,
        "retained_frame_count": 0,
        "temporal_segment_count": 0,
    }
    for (match_id, period), frames in frame_metadata.groupby(
        ["match_id", "period"], dropna=False, sort=False
    ):
        frames = frames.sort_values("frame_id", kind="mergesort").copy()
        frame_ids = pd.to_numeric(frames["frame_id"], errors="raise").to_numpy(dtype=int)
        game_modes = (
            pd.to_numeric(frames["game_mode"], errors="coerce").fillna(0).to_numpy(dtype=int)
        )
        score_home = (
            pd.to_numeric(frames["score_home"], errors="coerce").fillna(0).to_numpy(dtype=int)
        )
        score_away = (
            pd.to_numeric(frames["score_away"], errors="coerce").fillna(0).to_numpy(dtype=int)
        )
        mode_change = np.zeros(len(frames), dtype=bool)
        score_change = np.zeros(len(frames), dtype=bool)
        if len(frames) > 1:
            mode_change[1:] = game_modes[1:] != game_modes[:-1]
            score_change[1:] = (score_home[1:] != score_home[:-1]) | (
                score_away[1:] != score_away[:-1]
            )
        nonzero_mode = game_modes != 0
        event_signal = nonzero_mode | mode_change | score_change
        unsafe = np.zeros(len(frames), dtype=bool)
        for event_frame_id in frame_ids[event_signal]:
            unsafe |= np.abs(frame_ids - event_frame_id) <= int(window_frames)
        safe = ~unsafe
        segment_ids = np.full(len(frames), -1, dtype=int)
        segment_id = -1
        for index in range(len(frames)):
            if not safe[index]:
                continue
            if index == 0 or not safe[index - 1] or frame_ids[index] != frame_ids[index - 1] + 1:
                segment_id += 1
            segment_ids[index] = segment_id
        safe_assignments = frames.loc[
            safe, ["match_id", "period", "frame_id"]
        ].copy()
        safe_assignments["temporal_segment_id"] = segment_ids[safe]
        safe_assignments["temporal_stride_origin_frame_id"] = int(frame_ids.min())
        assignments.append(safe_assignments)
        unsafe_keys.update(
            (str(match_id), int(period), int(frame_id)) for frame_id in frame_ids[unsafe]
        )
        summary["total_frame_count"] += len(frames)
        summary["event_signal_frame_count"] += int(event_signal.sum())
        summary["nonzero_game_mode_frame_count"] += int(nonzero_mode.sum())
        summary["game_mode_change_frame_count"] += int(mode_change.sum())
        summary["score_change_frame_count"] += int(score_change.sum())
        summary["unsafe_frame_count"] += int(unsafe.sum())
        summary["retained_frame_count"] += int(safe.sum())
        summary["temporal_segment_count"] += int(segment_id + 1)

    if not assignments:
        raise ValueError("Event segmentation found no GRF frames.")
    assignment_frame = pd.concat(assignments, ignore_index=True)
    retained = tracking.merge(
        assignment_frame,
        on=["match_id", "period", "frame_id"],
        how="inner",
        validate="many_to_one",
    )
    total_frames = int(summary["total_frame_count"])
    summary["retained_frame_fraction"] = (
        float(summary["retained_frame_count"]) / total_frames if total_frames else 0.0
    )
    return retained, summary, unsafe_keys


def apply_actual_jump_segmentation(
    tracking: pd.DataFrame,
    *,
    player_jump_threshold_m: float = PLAYER_JUMP_THRESHOLD_M,
    ball_jump_threshold_m: float = BALL_JUMP_THRESHOLD_M,
) -> tuple[pd.DataFrame, dict[str, Any], set[tuple[str, int, int]]]:
    """Split episodes only where adjacent raw positions cross frozen jump thresholds."""

    required = {
        "match_id",
        "period",
        "frame_id",
        "time_s",
        "agent_id",
        "agent_type",
        "x_m",
        "y_m",
    }
    missing = sorted(required - set(tracking.columns))
    if missing:
        raise ValueError(f"Actual-jump segmentation requires tracking columns: {missing}")

    boundary_keys: set[tuple[str, int, int]] = set()
    player_boundary_keys: set[tuple[str, int, int]] = set()
    ball_boundary_keys: set[tuple[str, int, int]] = set()
    group_columns = ["match_id", "period", "agent_id"]
    for (match_id, period, _), entity in tracking.groupby(
        group_columns, dropna=False, sort=False
    ):
        entity = entity.sort_values(["frame_id", "time_s"], kind="mergesort")
        frame_ids = pd.to_numeric(entity["frame_id"], errors="raise").to_numpy(dtype=int)
        xy = entity[["x_m", "y_m"]].to_numpy(dtype=float)
        if len(entity) < 2:
            continue
        adjacent = frame_ids[1:] == frame_ids[:-1] + 1
        finite = np.isfinite(xy[1:]).all(axis=1) & np.isfinite(xy[:-1]).all(axis=1)
        jumps = np.linalg.norm(xy[1:] - xy[:-1], axis=1)
        entity_type = str(entity["agent_type"].iloc[0])
        threshold = (
            ball_jump_threshold_m if entity_type == "ball" else player_jump_threshold_m
        )
        for frame_id in frame_ids[1:][adjacent & finite & (jumps >= threshold)]:
            key = (str(match_id), int(period), int(frame_id))
            boundary_keys.add(key)
            if entity_type == "ball":
                ball_boundary_keys.add(key)
            else:
                player_boundary_keys.add(key)

    frame_metadata = tracking[["match_id", "period", "frame_id"]].drop_duplicates()
    assignments: list[pd.DataFrame] = []
    segment_count = 0
    for (match_id, period), frames in frame_metadata.groupby(
        ["match_id", "period"], dropna=False, sort=False
    ):
        frames = frames.sort_values("frame_id", kind="mergesort").copy()
        frame_ids = pd.to_numeric(frames["frame_id"], errors="raise").to_numpy(dtype=int)
        segment_ids = np.zeros(len(frames), dtype=int)
        segment_id = 0
        for index in range(1, len(frames)):
            key = (str(match_id), int(period), int(frame_ids[index]))
            if key in boundary_keys or frame_ids[index] != frame_ids[index - 1] + 1:
                segment_id += 1
            segment_ids[index] = segment_id
        frames["temporal_segment_id"] = segment_ids
        frames["temporal_stride_origin_frame_id"] = int(frame_ids.min())
        assignments.append(frames)
        segment_count += segment_id + 1

    if not assignments:
        raise ValueError("Actual-jump segmentation found no GRF frames.")
    assignment_frame = pd.concat(assignments, ignore_index=True)
    segmented = tracking.merge(
        assignment_frame,
        on=["match_id", "period", "frame_id"],
        how="inner",
        validate="many_to_one",
    )
    frame_count = len(frame_metadata)
    summary = {
        "player_jump_threshold_m": float(player_jump_threshold_m),
        "ball_jump_threshold_m": float(ball_jump_threshold_m),
        "total_frame_count": int(frame_count),
        "boundary_frame_count": len(boundary_keys),
        "player_boundary_frame_count": len(player_boundary_keys),
        "ball_boundary_frame_count": len(ball_boundary_keys),
        "retained_frame_count": int(frame_count),
        "temporal_segment_count": int(segment_count),
    }
    return segmented, summary, boundary_keys


def _with_causal_lag_velocity(
    tracking: pd.DataFrame,
    *,
    lag_frames: int = LOW_FREQUENCY_VELOCITY_LAG_FRAMES,
) -> pd.DataFrame:
    """Calculate truly causal lagged velocity independently inside temporal segments."""

    if lag_frames < 1:
        raise ValueError("lag_frames must be positive.")
    out = tracking.copy()
    out["vx_mps"] = 0.0
    out["vy_mps"] = 0.0
    group_columns = ["match_id", "period"]
    if "temporal_segment_id" in out.columns:
        group_columns.append("temporal_segment_id")
    group_columns.append("agent_id")
    for _, entity in out.groupby(group_columns, dropna=False, sort=False):
        entity = entity.sort_values(["frame_id", "time_s"], kind="mergesort")
        indices = entity.index.to_numpy()
        times = entity["time_s"].to_numpy(dtype=float)
        x = entity["x_m"].to_numpy(dtype=float)
        y = entity["y_m"].to_numpy(dtype=float)
        vx = np.zeros(len(entity), dtype=np.float64)
        vy = np.zeros(len(entity), dtype=np.float64)
        for index in range(1, len(entity)):
            previous = max(0, index - int(lag_frames))
            dt = times[index] - times[previous]
            if np.isfinite(dt) and dt > 0:
                vx[index] = (x[index] - x[previous]) / dt
                vy[index] = (y[index] - y[previous]) / dt
        out.loc[indices, "vx_mps"] = vx
        out.loc[indices, "vy_mps"] = vy
    return out


def apply_velocity_mode(
    tracking: Any,
    velocity_mode: str,
    *,
    materialize_velocity: bool = True,
) -> Any:
    """Apply the frozen provider or provider-neutral velocity construction."""

    if velocity_mode not in VELOCITY_MODES:
        raise ValueError(f"velocity_mode must be one of {sorted(VELOCITY_MODES)}.")
    if velocity_mode == PROVIDER_VELOCITY:
        return tracking
    event_summary = None
    jump_summary = None
    unsafe_frame_keys: set[tuple[str, int, int]] = set()
    boundary_frame_keys: set[tuple[str, int, int]] = set()
    if velocity_mode == EVENT_SEGMENTED_CAUSAL_POSITION_DIFFERENCE:
        out, event_summary, unsafe_frame_keys = apply_event_boundary_segmentation(tracking)
    elif velocity_mode == JUMP_SEGMENTED_CAUSAL_POSITION_DIFFERENCE_0P5S:
        out, jump_summary, boundary_frame_keys = apply_actual_jump_segmentation(tracking)
    else:
        out = tracking.copy()
    if not materialize_velocity:
        out["vx_mps"] = 0.0
        out["vy_mps"] = 0.0
        result = out
    else:
        out["vx_mps"] = float("nan")
        out["vy_mps"] = float("nan")
        if velocity_mode == JUMP_SEGMENTED_CAUSAL_POSITION_DIFFERENCE_0P5S:
            result = _with_causal_lag_velocity(out)
        else:
            result = _with_causal_velocity(out)
    if event_summary is not None:
        result.attrs["event_boundary_summary"] = event_summary
        result.attrs["unsafe_frame_keys"] = unsafe_frame_keys
    if jump_summary is not None:
        result.attrs["jump_boundary_summary"] = jump_summary
        result.attrs["boundary_frame_keys"] = boundary_frame_keys
    return result


def _unsafe_tensor_frame_reference_count(
    data: Any,
    unsafe_frame_keys: set[tuple[str, int, int]],
) -> int:
    count = 0
    for index, (match_id, period) in enumerate(zip(data.match_id, data.period, strict=True)):
        frame_indices = [
            *data.context_frame_indices[index].tolist(),
            *data.target_frame_indices[index].tolist(),
        ]
        count += sum(
            (str(match_id), int(period), int(frame_id)) in unsafe_frame_keys
            for frame_id in frame_indices
        )
    return count


def _boundary_crossing_tensor_example_count(
    data: Any,
    boundary_frame_keys: set[tuple[str, int, int]],
) -> int:
    boundaries: dict[tuple[str, int], list[int]] = {}
    for match_id, period, frame_id in boundary_frame_keys:
        boundaries.setdefault((match_id, period), []).append(frame_id)
    count = 0
    for index, (match_id, period) in enumerate(zip(data.match_id, data.period, strict=True)):
        frame_indices = [
            *data.context_frame_indices[index].tolist(),
            *data.target_frame_indices[index].tolist(),
        ]
        minimum = min(frame_indices)
        maximum = max(frame_indices)
        count += any(
            minimum < boundary <= maximum
            for boundary in boundaries.get((str(match_id), int(period)), [])
        )
    return count


def prepare_gfootball_td_jepa_shards(
    plan_path: str | Path,
    raw_root: str | Path,
    output_root: str | Path,
    split_manifest_path: str | Path,
    *,
    visibility_profile_path: str | Path | None = None,
    visibility_seed: int = 20260713,
    fps_out: float = 10.0,
    context_seconds: float = 1.0,
    delta_seconds: float = 0.2,
    stride_seconds: float = 0.2,
    prediction_gap_seconds: float = 1.0,
    velocity_mode: str = PROVIDER_VELOCITY,
    feature_view: str = "geometry_only",
    included_splits: set[str] | None = None,
    resume_existing: bool = False,
) -> Path:
    """Convert each GRF collection job independently and write a lazy manifest."""

    plan_file = Path(plan_path)
    plan = load_collection_plan(plan_file)
    split_manifest = load_split_manifest(split_manifest_path)
    expected_plan_hash = split_manifest.payload.get("source_collection_plan_sha256")
    actual_plan_hash = collection_plan_sha256(plan)
    if expected_plan_hash != actual_plan_hash:
        raise ValueError("GRF split manifest collection-plan hash mismatch.")
    if velocity_mode not in VELOCITY_MODES:
        raise ValueError(f"velocity_mode must be one of {sorted(VELOCITY_MODES)}.")
    selected_splits = set(included_splits or {"train", "val", "test"})
    unknown_splits = selected_splits - {"train", "val", "test"}
    if unknown_splits:
        raise ValueError(f"Unknown requested GRF splits: {sorted(unknown_splits)}")

    profile = None
    profile_path = None
    if visibility_profile_path is not None:
        profile_path = Path(visibility_profile_path)
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        profile_name = str(profile.get("profile", "pff_train_visibility_v1"))
    else:
        profile_name = "all_available"

    raw = Path(raw_root)
    out = Path(output_root)
    dataset_root = out / profile_name
    shard_entries = []
    all_sample_ids: set[str] = set()
    split_example_counts = {"train": 0, "val": 0, "test": 0}
    unsegmented_example_count = 0
    event_boundary_totals: dict[str, int] = {}
    jump_boundary_totals: dict[str, int] = {}

    for job_index, job in enumerate(plan["jobs"]):
        split = str(job["split"])
        if split not in selected_splits:
            continue
        source_path = raw / str(job.get("source_path", Path(split) / f"{job['id']}.jsonl"))
        if not source_path.exists():
            raise FileNotFoundError(f"Missing GRF collection job output: {source_path}")
        tensor_path = dataset_root / split / str(job["id"]) / "td_jepa.pt"
        if resume_existing and tensor_path.exists():
            data = load_td_jepa_data(tensor_path)
            metadata = data.metadata or {}
            expected_metadata = {
                "collection_plan_sha256": actual_plan_hash,
                "collection_job_id": job["id"],
                "collection_split": split,
                "visibility_profile_sha256": (
                    profile.get("profile_payload_sha256") if profile is not None else None
                ),
                "visibility_seed": int(visibility_seed) + job_index,
                "velocity_mode": velocity_mode,
                "velocity_features_materialized": feature_view != POSITION_ONLY,
                "feature_view": feature_view,
                "event_segmentation_implementation_version": (
                    EVENT_SEGMENTATION_IMPLEMENTATION_VERSION
                    if velocity_mode == EVENT_SEGMENTED_CAUSAL_POSITION_DIFFERENCE
                    else None
                ),
                "jump_segmentation_implementation_version": (
                    JUMP_SEGMENTATION_IMPLEMENTATION_VERSION
                    if velocity_mode == JUMP_SEGMENTED_CAUSAL_POSITION_DIFFERENCE_0P5S
                    else None
                ),
            }
            mismatches = [
                name
                for name, expected in expected_metadata.items()
                if metadata.get(name) != expected
            ]
            if mismatches:
                raise ValueError(
                    f"Existing GRF shard {job['id']} does not match requested provenance: "
                    f"{mismatches}"
                )
            unsafe_reference_count = int(
                metadata.get("unsafe_tensor_frame_reference_count", -1)
            )
            if unsafe_reference_count != 0:
                raise ValueError(
                    f"Existing GRF shard {job['id']} has unsafe tensor references."
                )
            boundary_crossing_count = int(
                metadata.get("boundary_crossing_tensor_example_count", 0)
            )
            if boundary_crossing_count != 0:
                raise ValueError(
                    f"Existing GRF shard {job['id']} crosses jump boundaries."
                )
            event_boundary_summary = metadata.get("event_boundary_summary")
            jump_boundary_summary = metadata.get("jump_boundary_summary")
            shard_unsegmented_count = int(metadata.get("unsegmented_example_count", -1))
            if shard_unsegmented_count < 0:
                raise ValueError(
                    f"Existing GRF shard {job['id']} lacks unsegmented-example provenance."
                )
            unsegmented_example_count += shard_unsegmented_count
            actual_match_ids = set(map(str, data.match_id))
            expected_match_ids = set(job_match_ids(plan, job))
            if actual_match_ids != expected_match_ids:
                raise ValueError(
                    f"Existing GRF shard {job['id']} episode identities do not match the plan."
                )
            duplicate_ids = all_sample_ids.intersection(data.sample_id or [])
            if duplicate_ids:
                raise ValueError(f"Duplicate GRF TD-JEPA sample ID: {sorted(duplicate_ids)[0]}")
            all_sample_ids.update(data.sample_id or [])
            relative_path = tensor_path.relative_to(out)
            example_count = len(data.match_id)
            split_example_counts[split] += example_count
            shard_entries.append(
                {
                    "path": str(relative_path),
                    "split": split,
                    "job_id": job["id"],
                    "scenario": job["env_name"],
                    "action_policy": job.get("action_policy", "builtin_ai"),
                    "match_ids": sorted(actual_match_ids),
                    "example_count": example_count,
                    "source_path": str(source_path),
                    "source_sha256": _file_sha256(source_path),
                    "tensor_sha256": _file_sha256(tensor_path),
                    "velocity_mode": velocity_mode,
                    "velocity_features_materialized": feature_view != POSITION_ONLY,
                    "event_boundary_summary": event_boundary_summary,
                    "jump_boundary_summary": jump_boundary_summary,
                    "unsafe_tensor_frame_reference_count": unsafe_reference_count,
                    "boundary_crossing_tensor_example_count": boundary_crossing_count,
                    "unsegmented_example_count": shard_unsegmented_count,
                    "resumed_existing": True,
                }
            )
            if event_boundary_summary is not None:
                for name, value in event_boundary_summary.items():
                    if name.endswith("_count") and isinstance(value, int):
                        event_boundary_totals[name] = (
                            event_boundary_totals.get(name, 0) + value
                        )
            if jump_boundary_summary is not None:
                for name, value in jump_boundary_summary.items():
                    if name.endswith("_count") and isinstance(value, int):
                        jump_boundary_totals[name] = (
                            jump_boundary_totals.get(name, 0) + value
                        )
            continue
        tracking = GFootballAdapter(
            raw_dir=source_path,
            match_id=job_match_prefix(plan, job),
            fps=fps_out,
        ).load_tracking()
        unsegmented_counts = unsegmented_example_counts_by_match(
            tracking,
            fps_out=fps_out,
            context_seconds=context_seconds,
            stride_seconds=stride_seconds,
            prediction_gap_seconds=prediction_gap_seconds,
        )
        shard_unsegmented_count = sum(unsegmented_counts.values())
        unsegmented_example_count += shard_unsegmented_count
        actual_match_ids = set(tracking["match_id"].astype(str).unique())
        expected_match_ids = set(job_match_ids(plan, job))
        if actual_match_ids != expected_match_ids:
            raise ValueError(
                f"GRF job {job['id']} episode identities do not match the collection plan."
            )
        velocity_features_materialized = feature_view != POSITION_ONLY
        tracking = apply_velocity_mode(
            tracking,
            velocity_mode,
            materialize_velocity=velocity_features_materialized,
        )
        event_boundary_summary = tracking.attrs.get("event_boundary_summary")
        unsafe_frame_keys = tracking.attrs.get("unsafe_frame_keys", set())
        jump_boundary_summary = tracking.attrs.get("jump_boundary_summary")
        boundary_frame_keys = tracking.attrs.get("boundary_frame_keys", set())
        if profile is not None:
            tracking = apply_pff_like_visibility(
                tracking,
                profile,
                seed=int(visibility_seed) + job_index,
            )
        data = build_td_jepa_examples(
            tracking,
            fps_out=fps_out,
            context_seconds=context_seconds,
            delta_seconds=delta_seconds,
            stride_seconds=stride_seconds,
            objective_mode="future_nonoverlap_context_only",
            prediction_gap_seconds=prediction_gap_seconds,
            feature_view=feature_view,
            split_manifest_path=split_manifest_path,
            scientific_mode=True,
        )
        if not data.match_id:
            raise ValueError(f"GRF job {job['id']} produced no TD-JEPA examples.")
        unsafe_reference_count = _unsafe_tensor_frame_reference_count(
            data, unsafe_frame_keys
        )
        if unsafe_reference_count:
            raise ValueError(
                f"GRF job {job['id']} tensors reference {unsafe_reference_count} unsafe frames."
            )
        boundary_crossing_count = _boundary_crossing_tensor_example_count(
            data, boundary_frame_keys
        )
        if boundary_crossing_count:
            raise ValueError(
                f"GRF job {job['id']} has {boundary_crossing_count} boundary-crossing examples."
            )
        duplicate_ids = all_sample_ids.intersection(data.sample_id or [])
        if duplicate_ids:
            raise ValueError(f"Duplicate GRF TD-JEPA sample ID: {sorted(duplicate_ids)[0]}")
        all_sample_ids.update(data.sample_id or [])
        data.metadata = {
            **(data.metadata or {}),
            "collection_plan_path": str(plan_file),
            "collection_plan_sha256": actual_plan_hash,
            "collection_job_id": job["id"],
            "collection_split": split,
            "visibility_profile_path": str(profile_path) if profile_path else None,
            "visibility_profile_sha256": (
                profile.get("profile_payload_sha256") if profile is not None else None
            ),
            "visibility_seed": int(visibility_seed) + job_index,
            "velocity_mode": velocity_mode,
            "velocity_features_materialized": velocity_features_materialized,
            "feature_view": feature_view,
            "event_segmentation_implementation_version": (
                EVENT_SEGMENTATION_IMPLEMENTATION_VERSION
                if velocity_mode == EVENT_SEGMENTED_CAUSAL_POSITION_DIFFERENCE
                else None
            ),
            "jump_segmentation_implementation_version": (
                JUMP_SEGMENTATION_IMPLEMENTATION_VERSION
                if velocity_mode == JUMP_SEGMENTED_CAUSAL_POSITION_DIFFERENCE_0P5S
                else None
            ),
            "event_boundary_summary": event_boundary_summary,
            "jump_boundary_summary": jump_boundary_summary,
            "unsafe_tensor_frame_reference_count": unsafe_reference_count,
            "boundary_crossing_tensor_example_count": boundary_crossing_count,
            "unsegmented_example_counts_by_match": unsegmented_counts,
            "unsegmented_example_count": shard_unsegmented_count,
        }
        save_td_jepa_data(data, tensor_path)
        relative_path = tensor_path.relative_to(out)
        example_count = len(data.match_id)
        split_example_counts[split] += example_count
        shard_entries.append(
            {
                "path": str(relative_path),
                "split": split,
                "job_id": job["id"],
                "scenario": job["env_name"],
                "action_policy": job.get("action_policy", "builtin_ai"),
                "match_ids": sorted(actual_match_ids),
                "example_count": example_count,
                "source_path": str(source_path),
                "source_sha256": _file_sha256(source_path),
                "tensor_sha256": _file_sha256(tensor_path),
                "velocity_mode": velocity_mode,
                "velocity_features_materialized": velocity_features_materialized,
                "feature_view": feature_view,
                "event_segmentation_implementation_version": (
                    EVENT_SEGMENTATION_IMPLEMENTATION_VERSION
                    if velocity_mode == EVENT_SEGMENTED_CAUSAL_POSITION_DIFFERENCE
                    else None
                ),
                "jump_segmentation_implementation_version": (
                    JUMP_SEGMENTATION_IMPLEMENTATION_VERSION
                    if velocity_mode == JUMP_SEGMENTED_CAUSAL_POSITION_DIFFERENCE_0P5S
                    else None
                ),
                "event_boundary_summary": event_boundary_summary,
                "jump_boundary_summary": jump_boundary_summary,
                "unsafe_tensor_frame_reference_count": unsafe_reference_count,
                "boundary_crossing_tensor_example_count": boundary_crossing_count,
                "unsegmented_example_count": shard_unsegmented_count,
                "resumed_existing": False,
            }
        )
        if event_boundary_summary is not None:
            for name, value in event_boundary_summary.items():
                if name.endswith("_count") and isinstance(value, int):
                    event_boundary_totals[name] = event_boundary_totals.get(name, 0) + value
        if jump_boundary_summary is not None:
            for name, value in jump_boundary_summary.items():
                if name.endswith("_count") and isinstance(value, int):
                    jump_boundary_totals[name] = jump_boundary_totals.get(name, 0) + value

    manifest: dict[str, Any] = {
        "status": "complete",
        "version": 1,
        "dataset": "gfootball",
        "profile": profile_name,
        "collection_plan_path": str(plan_file),
        "collection_plan_file_sha256": _file_sha256(plan_file),
        "collection_plan_sha256": actual_plan_hash,
        "split_manifest_path": str(split_manifest.path),
        "split_manifest_sha256": split_manifest.sha256,
        "visibility_profile_path": str(profile_path) if profile_path else None,
        "visibility_profile_file_sha256": _file_sha256(profile_path) if profile_path else None,
        "visibility_profile_sha256": (
            profile.get("profile_payload_sha256") if profile is not None else None
        ),
        "visibility_seed": int(visibility_seed),
        "velocity_mode": velocity_mode,
        "velocity_features_materialized": feature_view != POSITION_ONLY,
        "included_splits": sorted(selected_splits),
        "event_boundary_window_frames": (
            EVENT_BOUNDARY_WINDOW_FRAMES
            if velocity_mode == EVENT_SEGMENTED_CAUSAL_POSITION_DIFFERENCE
            else None
        ),
        "event_segmentation_implementation_version": (
            EVENT_SEGMENTATION_IMPLEMENTATION_VERSION
            if velocity_mode == EVENT_SEGMENTED_CAUSAL_POSITION_DIFFERENCE
            else None
        ),
        "jump_segmentation_implementation_version": (
            JUMP_SEGMENTATION_IMPLEMENTATION_VERSION
            if velocity_mode == JUMP_SEGMENTED_CAUSAL_POSITION_DIFFERENCE_0P5S
            else None
        ),
        "player_jump_threshold_m": (
            PLAYER_JUMP_THRESHOLD_M
            if velocity_mode == JUMP_SEGMENTED_CAUSAL_POSITION_DIFFERENCE_0P5S
            else None
        ),
        "ball_jump_threshold_m": (
            BALL_JUMP_THRESHOLD_M
            if velocity_mode == JUMP_SEGMENTED_CAUSAL_POSITION_DIFFERENCE_0P5S
            else None
        ),
        "event_boundary_totals": event_boundary_totals or None,
        "jump_boundary_totals": jump_boundary_totals or None,
        "causal_velocity_lag_frames": (
            LOW_FREQUENCY_VELOCITY_LAG_FRAMES
            if velocity_mode == JUMP_SEGMENTED_CAUSAL_POSITION_DIFFERENCE_0P5S
            else None
        ),
        "resume_existing": bool(resume_existing),
        "config": {
            "fps_out": float(fps_out),
            "context_seconds": float(context_seconds),
            "delta_seconds": float(delta_seconds),
            "stride_seconds": float(stride_seconds),
            "objective_mode": "future_nonoverlap_context_only",
            "prediction_gap_seconds": float(prediction_gap_seconds),
            "feature_view": feature_view,
            "velocity_mode": velocity_mode,
            "velocity_features_materialized": feature_view != POSITION_ONLY,
            "included_splits": sorted(selected_splits),
            "event_boundary_window_frames": (
                EVENT_BOUNDARY_WINDOW_FRAMES
                if velocity_mode == EVENT_SEGMENTED_CAUSAL_POSITION_DIFFERENCE
                else None
            ),
            "event_segmentation_implementation_version": (
                EVENT_SEGMENTATION_IMPLEMENTATION_VERSION
                if velocity_mode == EVENT_SEGMENTED_CAUSAL_POSITION_DIFFERENCE
                else None
            ),
            "jump_segmentation_implementation_version": (
                JUMP_SEGMENTATION_IMPLEMENTATION_VERSION
                if velocity_mode == JUMP_SEGMENTED_CAUSAL_POSITION_DIFFERENCE_0P5S
                else None
            ),
            "player_jump_threshold_m": (
                PLAYER_JUMP_THRESHOLD_M
                if velocity_mode == JUMP_SEGMENTED_CAUSAL_POSITION_DIFFERENCE_0P5S
                else None
            ),
            "ball_jump_threshold_m": (
                BALL_JUMP_THRESHOLD_M
                if velocity_mode == JUMP_SEGMENTED_CAUSAL_POSITION_DIFFERENCE_0P5S
                else None
            ),
            "resume_existing": bool(resume_existing),
            "causal_velocity_lag_frames": (
                LOW_FREQUENCY_VELOCITY_LAG_FRAMES
                if velocity_mode == JUMP_SEGMENTED_CAUSAL_POSITION_DIFFERENCE_0P5S
                else None
            ),
        },
        "example_count": sum(split_example_counts.values()),
        "unsegmented_example_count": unsegmented_example_count,
        "example_retention_fraction": (
            sum(split_example_counts.values()) / unsegmented_example_count
            if unsegmented_example_count
            else None
        ),
        "unique_sample_id_count": len(all_sample_ids),
        "split_example_counts": split_example_counts,
        "shards": shard_entries,
        "tensor_hashes_complete": True,
    }
    manifest["manifest_payload_sha256"] = _payload_sha256(manifest)
    manifest_path = dataset_root / "dataset_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def project_gfootball_feature_view(
    source_manifest_path: str | Path,
    output_root: str | Path,
    *,
    target_feature_view: str,
) -> Path:
    """Project existing GRF shards to a narrower feature view without changing examples."""

    source_path = Path(source_manifest_path)
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    source_root = source_path.parent.parent
    out = Path(output_root)
    dataset_root = out / str(source_manifest["profile"])
    shard_entries = []
    all_sample_ids: set[str] = set()
    selected_names: list[str] | None = None
    for shard in source_manifest["shards"]:
        data = load_td_jepa_data(source_root / shard["path"])
        source_names = list(data.feature_names)
        data.state_t, names = apply_feature_view(
            data.state_t, source_names, target_feature_view
        )
        data.state_t_plus_delta, _ = apply_feature_view(
            data.state_t_plus_delta, source_names, target_feature_view
        )
        data.delta_state, _ = apply_feature_view(
            data.delta_state, source_names, target_feature_view
        )
        data.feature_names = list(names)
        data.feature_view = target_feature_view
        data.metadata = {
            **(data.metadata or {}),
            "feature_view": target_feature_view,
            "source_feature_manifest_path": str(source_path),
            "source_feature_manifest_sha256": _file_sha256(source_path),
            "source_feature_manifest_payload_sha256": source_manifest.get(
                "manifest_payload_sha256"
            ),
        }
        selected_names = list(names)
        duplicate_ids = all_sample_ids.intersection(data.sample_id or [])
        if duplicate_ids:
            raise ValueError(f"Duplicate projected GRF sample ID: {sorted(duplicate_ids)[0]}")
        all_sample_ids.update(data.sample_id or [])
        tensor_path = (
            dataset_root / str(shard["split"]) / str(shard["job_id"]) / "td_jepa.pt"
        )
        save_td_jepa_data(data, tensor_path)
        projected_shard = deepcopy(shard)
        projected_shard.update(
            {
                "path": str(tensor_path.relative_to(out)),
                "tensor_sha256": _file_sha256(tensor_path),
                "feature_view": target_feature_view,
                "source_tensor_sha256": shard["tensor_sha256"],
            }
        )
        shard_entries.append(projected_shard)

    manifest = deepcopy(source_manifest)
    manifest.pop("manifest_payload_sha256", None)
    manifest.update(
        {
            "feature_view": target_feature_view,
            "feature_names": selected_names or [],
            "source_feature_manifest_path": str(source_path),
            "source_feature_manifest_file_sha256": _file_sha256(source_path),
            "source_feature_manifest_payload_sha256": source_manifest.get(
                "manifest_payload_sha256"
            ),
            "unique_sample_id_count": len(all_sample_ids),
            "shards": shard_entries,
            "tensor_hashes_complete": True,
        }
    )
    manifest["config"] = {
        **manifest["config"],
        "feature_view": target_feature_view,
    }
    manifest["manifest_payload_sha256"] = _payload_sha256(manifest)
    manifest_path = dataset_root / "dataset_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path
