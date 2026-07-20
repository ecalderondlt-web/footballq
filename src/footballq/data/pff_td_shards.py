"""Build leakage-controlled TD-JEPA shards from canonical PFF Parquet shards."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from footballq.data.td_jepa_dataset import (
    FUTURE_NONOVERLAP_CONTEXT_ONLY,
    TDJEPAData,
    build_td_jepa_examples,
)
from footballq.repro.splits import load_split_manifest, stable_json_bytes

PFF_TD_SHARD_VERSION = 1


def _payload_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(stable_json_bytes(payload)).hexdigest()


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _subset(data: TDJEPAData, indices: list[int], metadata: dict[str, Any]) -> TDJEPAData:
    tensor_indices = torch.tensor(indices, dtype=torch.long)
    return TDJEPAData(
        state_t=data.state_t[tensor_indices],
        state_t_plus_delta=data.state_t_plus_delta[tensor_indices],
        delta_state=data.delta_state[tensor_indices],
        mask_t=data.mask_t[tensor_indices],
        mask_t_plus_delta=data.mask_t_plus_delta[tensor_indices],
        delta_mask=data.delta_mask[tensor_indices],
        entity_type=data.entity_type[tensor_indices],
        team_id=data.team_id[tensor_indices],
        match_id=[data.match_id[index] for index in indices],
        period=[data.period[index] for index in indices],
        frame_t=[data.frame_t[index] for index in indices],
        sample_id=[data.sample_id[index] for index in indices],
        delta_frames=data.delta_frames,
        feature_names=list(data.feature_names),
        fps=data.fps,
        context_seconds=data.context_seconds,
        delta_seconds=data.delta_seconds,
        stride_seconds=data.stride_seconds,
        objective_mode=data.objective_mode,
        prediction_gap_frames=data.prediction_gap_frames,
        feature_view=data.feature_view,
        context_frame_indices=data.context_frame_indices[tensor_indices],
        target_frame_indices=data.target_frame_indices[tensor_indices],
        delta_frame_indices=data.delta_frame_indices[tensor_indices],
        metadata=metadata,
    )


def _save_td_shard(data: TDJEPAData, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(data.to_dict(), temporary)
    temporary.replace(path)


def _apply_visibility_mode(frame: pd.DataFrame, visibility_mode: str) -> pd.DataFrame:
    if visibility_mode == "all_available":
        return frame
    if visibility_mode != "observed_only":
        raise ValueError("visibility_mode must be 'all_available' or 'observed_only'.")
    if "is_observed" not in frame.columns:
        raise ValueError("observed_only requires canonical PFF is_observed provenance.")
    out = frame.copy()
    available = out["visible"].fillna(False).astype(bool)
    observed = out["is_observed"].fillna(False).astype(bool)
    out["visible"] = available & observed
    out["is_visible"] = out["visible"]
    return out


def _prepare_match_td_shards(
    canonical_path: Path,
    output_path: Path,
    split_manifest_path: Path,
    *,
    match_id: str,
    split_name: str,
    canonical_dataset_manifest_sha256: str,
    config: dict[str, Any],
    resume: bool,
) -> list[dict[str, Any]]:
    split = load_split_manifest(split_manifest_path)
    canonical_match_dir = canonical_path / split_name / match_id
    match_manifest_path = canonical_match_dir / "manifest.json"
    match_manifest = json.loads(match_manifest_path.read_text(encoding="utf-8"))
    destination_dir = output_path / config["visibility_mode"] / split_name / match_id
    destination_dir.mkdir(parents=True, exist_ok=True)
    source_shards = match_manifest["shards"]
    output_shards: list[dict[str, Any]] = []
    match_sample_ids: set[str] = set()
    for index, source_shard in enumerate(source_shards):
        filename = f"td_p{source_shard['period']}_s{source_shard['shard_index']:04d}.pt"
        destination = destination_dir / filename
        shard_manifest_path = destination.with_suffix(".json")
        expected_key = {
            "canonical_shard_sha256": source_shard["sha256"],
            "split_manifest_sha256": split.sha256,
            **config,
        }
        if resume and destination.exists() and shard_manifest_path.exists():
            existing = json.loads(shard_manifest_path.read_text(encoding="utf-8"))
            if existing.get("resume_key") == _payload_sha256(expected_key):
                duplicates = match_sample_ids.intersection(existing["sample_ids"])
                if duplicates:
                    raise ValueError("Duplicate PFF TD sample IDs found while resuming.")
                match_sample_ids.update(existing["sample_ids"])
                output_shards.append(existing)
                continue

        frames = [pd.read_parquet(canonical_match_dir / source_shard["path"])]
        next_source = None
        if index + 1 < len(source_shards):
            candidate = source_shards[index + 1]
            if int(candidate["period"]) == int(source_shard["period"]):
                next_source = candidate
                frames.append(pd.read_parquet(canonical_match_dir / candidate["path"]))
        tracking = _apply_visibility_mode(
            pd.concat(frames, ignore_index=True), config["visibility_mode"]
        )
        data = build_td_jepa_examples(
            tracking,
            fps_out=config["fps_out"],
            context_seconds=config["context_seconds"],
            delta_seconds=config["delta_seconds"],
            stride_seconds=config["stride_seconds"],
            objective_mode=FUTURE_NONOVERLAP_CONTEXT_ONLY,
            prediction_gap_seconds=config["prediction_gap_seconds"],
            feature_view=config["feature_view"],
            split_manifest_path=split_manifest_path,
            scientific_mode=True,
        )
        owned_indices = [
            row_index
            for row_index, frame_t in enumerate(data.frame_t)
            if int(source_shard["start_frame"])
            <= int(frame_t)
            <= int(source_shard["end_frame"])
        ]
        owned_ids = [data.sample_id[row_index] for row_index in owned_indices]
        duplicates = match_sample_ids.intersection(owned_ids)
        if duplicates:
            raise ValueError("Duplicate PFF TD sample IDs found across shard boundaries.")
        match_sample_ids.update(owned_ids)
        metadata = {
            **(data.metadata or {}),
            "pff_td_shard_version": PFF_TD_SHARD_VERSION,
            "canonical_dataset_manifest_sha256": canonical_dataset_manifest_sha256,
            "canonical_match_manifest_sha256": match_manifest["manifest_payload_sha256"],
            "canonical_shard_sha256": source_shard["sha256"],
            "lookahead_shard_sha256": next_source["sha256"] if next_source else None,
            "visibility_mode": config["visibility_mode"],
            "owned_start_frame": source_shard["start_frame"],
            "owned_end_frame": source_shard["end_frame"],
        }
        owned = _subset(data, owned_indices, metadata)
        _save_td_shard(owned, destination)
        tensor_sha256 = _file_sha256(destination)
        shard_manifest = {
            "status": "complete",
            "version": PFF_TD_SHARD_VERSION,
            "match_id": match_id,
            "split": split_name,
            "period": source_shard["period"],
            "path": str(destination.relative_to(output_path)),
            "example_count": len(owned.match_id),
            "sample_ids": owned.sample_id,
            "start_frame_min": min(owned.frame_t) if owned.frame_t else None,
            "start_frame_max": max(owned.frame_t) if owned.frame_t else None,
            "resume_key": _payload_sha256(expected_key),
            "tensor_sha256": tensor_sha256,
            **expected_key,
        }
        shard_manifest["manifest_payload_sha256"] = _payload_sha256(shard_manifest)
        _write_json(shard_manifest_path, shard_manifest)
        output_shards.append(shard_manifest)
    return output_shards


def prepare_pff_td_jepa_shards(
    canonical_root: str | Path,
    output_root: str | Path,
    split_manifest_path: str | Path,
    *,
    match_ids: list[str] | None = None,
    split_names: list[str] | None = None,
    fps_out: float = 10.0,
    context_seconds: float = 1.0,
    delta_seconds: float = 0.2,
    stride_seconds: float = 0.2,
    prediction_gap_seconds: float = 1.0,
    feature_view: str = "geometry_only",
    visibility_mode: str = "all_available",
    resume: bool = True,
    workers: int = 1,
) -> dict[str, Any]:
    """Build one owned TD shard per canonical shard, with next-shard lookahead."""

    canonical_path = Path(canonical_root)
    canonical_manifest_path = canonical_path / "dataset_manifest.json"
    if not canonical_manifest_path.exists():
        raise FileNotFoundError(f"Canonical PFF manifest not found: {canonical_manifest_path}")
    canonical_manifest = json.loads(canonical_manifest_path.read_text(encoding="utf-8"))
    split = load_split_manifest(split_manifest_path)
    if canonical_manifest.get("split_manifest_sha256") != split.sha256:
        raise ValueError("Canonical PFF shards and requested split manifest have different hashes.")

    selected = list(match_ids or canonical_manifest["selected_match_ids"])
    allowed_splits = set(split_names or ["train", "val", "test"])
    canonical_matches = {
        item["match_id"]: item["split"] for item in canonical_manifest.get("matches", [])
    }
    missing = sorted(set(selected) - set(canonical_matches))
    if missing:
        raise ValueError(
            "Canonical PFF shards are missing requested matches: " + ", ".join(missing)
        )
    selected = [match_id for match_id in selected if canonical_matches[match_id] in allowed_splits]

    config = {
        "fps_out": fps_out,
        "context_seconds": context_seconds,
        "delta_seconds": delta_seconds,
        "stride_seconds": stride_seconds,
        "objective_mode": FUTURE_NONOVERLAP_CONTEXT_ONLY,
        "prediction_gap_seconds": prediction_gap_seconds,
        "feature_view": feature_view,
        "visibility_mode": visibility_mode,
    }
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    if workers < 1:
        raise ValueError("workers must be positive.")
    tasks = [
        {
            "match_id": match_id,
            "split_name": canonical_matches[match_id],
            "canonical_dataset_manifest_sha256": canonical_manifest[
                "manifest_payload_sha256"
            ],
            "config": config,
            "resume": resume,
        }
        for match_id in selected
    ]
    if workers == 1:
        match_outputs = [
            _prepare_match_td_shards(
                canonical_path,
                output_path,
                Path(split_manifest_path),
                **task,
            )
            for task in tasks
        ]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _prepare_match_td_shards,
                    canonical_path,
                    output_path,
                    Path(split_manifest_path),
                    **task,
                )
                for task in tasks
            ]
            match_outputs = [future.result() for future in futures]
    output_shards = [shard for match_output in match_outputs for shard in match_output]
    all_sample_ids = {
        sample_id for shard in output_shards for sample_id in shard["sample_ids"]
    }
    if sum(len(shard["sample_ids"]) for shard in output_shards) != len(all_sample_ids):
        raise ValueError("Duplicate PFF TD sample IDs found across matches.")

    manifest = {
        "status": "complete",
        "version": PFF_TD_SHARD_VERSION,
        "dataset": "pff_fc",
        "canonical_root": str(canonical_path.resolve()),
        "canonical_dataset_manifest_sha256": canonical_manifest["manifest_payload_sha256"],
        "split_manifest_path": str(Path(split_manifest_path)),
        "split_manifest_sha256": split.sha256,
        "selected_match_ids": selected,
        "selected_match_count": len(selected),
        "example_count": sum(item["example_count"] for item in output_shards),
        "unique_sample_id_count": len(all_sample_ids),
        "config": config,
        "workers": workers,
        "shards": [
            {
                key: item[key]
                for key in (
                    "path",
                    "match_id",
                    "split",
                    "period",
                    "example_count",
                    "tensor_sha256",
                    "manifest_payload_sha256",
                )
            }
            for item in output_shards
        ],
    }
    manifest["manifest_payload_sha256"] = _payload_sha256(manifest)
    _write_json(output_path / visibility_mode / "dataset_manifest.json", manifest)
    return manifest


def finalize_pff_td_jepa_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Hash every tensor and rewrite sidecar plus dataset manifests atomically."""

    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent.parent
    finalized_shards: list[dict[str, Any]] = []
    for item in manifest["shards"]:
        tensor_path = root / item["path"]
        if not tensor_path.exists():
            raise FileNotFoundError(f"PFF TD tensor shard is missing: {tensor_path}")
        sidecar_path = tensor_path.with_suffix(".json")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar["tensor_sha256"] = _file_sha256(tensor_path)
        sidecar.pop("manifest_payload_sha256", None)
        sidecar["manifest_payload_sha256"] = _payload_sha256(sidecar)
        _write_json(sidecar_path, sidecar)
        finalized_shards.append(
            {
                key: sidecar[key]
                for key in (
                    "path",
                    "match_id",
                    "split",
                    "period",
                    "example_count",
                    "tensor_sha256",
                    "manifest_payload_sha256",
                )
            }
        )
    manifest["shards"] = finalized_shards
    manifest["tensor_hashes_complete"] = True
    manifest.pop("manifest_payload_sha256", None)
    manifest["manifest_payload_sha256"] = _payload_sha256(manifest)
    _write_json(path, manifest)
    return manifest
