"""Episode-prefix tensor subsets for nested GRF scaling experiments."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from footballq.data.td_jepa_dataset import TDJEPAData, load_td_jepa_data, save_td_jepa_data
from footballq.io.gfootball_curriculum import (
    collection_plan_sha256,
    job_match_ids,
    load_collection_plan,
)
from footballq.repro.splits import load_split_manifest, stable_json_bytes


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(stable_json_bytes(payload)).hexdigest()


def _slice_td_data(
    data: TDJEPAData,
    indices: list[int],
    *,
    metadata: dict[str, Any],
) -> TDJEPAData:
    selected = torch.tensor(indices, dtype=torch.long)
    for field in (
        "state_t",
        "state_t_plus_delta",
        "delta_state",
        "mask_t",
        "mask_t_plus_delta",
        "delta_mask",
        "entity_type",
        "team_id",
        "context_frame_indices",
        "target_frame_indices",
        "delta_frame_indices",
    ):
        value = getattr(data, field)
        if value is not None:
            setattr(data, field, value[selected].contiguous())
    for field in ("match_id", "period", "frame_t", "sample_id"):
        value = getattr(data, field)
        if value is not None:
            setattr(data, field, [value[index] for index in indices])
    data.metadata = metadata
    return data


def derive_gfootball_td_episode_subset(
    master_manifest_path: str | Path,
    master_plan_path: str | Path,
    subset_plan_path: str | Path,
    subset_collection_manifest_path: str | Path,
    output_root: str | Path,
    split_manifest_path: str | Path,
) -> Path:
    """Derive exact tensor prefixes without rebuilding shared master episodes."""

    master_manifest_path = Path(master_manifest_path)
    master_plan_path = Path(master_plan_path)
    subset_plan_path = Path(subset_plan_path)
    subset_collection_manifest_path = Path(subset_collection_manifest_path)
    master_manifest = json.loads(master_manifest_path.read_text(encoding="utf-8"))
    if not master_manifest.get("tensor_hashes_complete"):
        raise ValueError("Master GRF tensor manifest has incomplete hashes.")
    master_plan = load_collection_plan(master_plan_path)
    subset_plan = load_collection_plan(subset_plan_path)
    if master_plan["name"] != subset_plan["name"]:
        raise ValueError("Master and subset plans must share a match-identity namespace.")
    if master_manifest.get("collection_plan_sha256") != collection_plan_sha256(master_plan):
        raise ValueError("Master tensor manifest collection-plan hash mismatch.")

    split_manifest = load_split_manifest(split_manifest_path)
    expected_subset_plan_hash = split_manifest.payload.get("source_collection_plan_sha256")
    subset_plan_hash = collection_plan_sha256(subset_plan)
    if expected_subset_plan_hash != subset_plan_hash:
        raise ValueError("Subset split manifest collection-plan hash mismatch.")

    collection_manifest = json.loads(
        subset_collection_manifest_path.read_text(encoding="utf-8")
    )
    if collection_manifest.get("collection_plan_sha256") != subset_plan_hash:
        raise ValueError("Subset raw collection manifest collection-plan hash mismatch.")
    collection_jobs = {str(item["id"]): item for item in collection_manifest["jobs"]}
    master_shards = {str(item["job_id"]): item for item in master_manifest["shards"]}
    if set(master_shards) != {str(job["id"]) for job in subset_plan["jobs"]}:
        raise ValueError("Master tensor shards do not match subset jobs.")

    output_root = Path(output_root)
    profile = str(master_manifest["profile"])
    dataset_root = output_root / profile
    master_root = master_manifest_path.parent.parent
    shard_entries: list[dict[str, Any]] = []
    all_sample_ids: set[str] = set()
    unsegmented_total = 0
    for job in subset_plan["jobs"]:
        job_id = str(job["id"])
        source_shard = master_shards[job_id]
        source_tensor_path = master_root / str(source_shard["path"])
        data = load_td_jepa_data(source_tensor_path)
        planned_match_ids = set(job_match_ids(subset_plan, job))
        indices = [
            index
            for index, match_id in enumerate(data.match_id)
            if str(match_id) in planned_match_ids
        ]
        if not indices:
            raise ValueError(f"Subset job {job_id!r} retained no tensor examples.")
        source_counts = dict(
            (data.metadata or {}).get("unsegmented_example_counts_by_match", {})
        )
        missing_counts = planned_match_ids - set(source_counts)
        if missing_counts:
            raise ValueError(f"Master shard {job_id!r} lacks per-match baseline counts.")
        subset_counts = {
            match_id: int(source_counts[match_id]) for match_id in sorted(planned_match_ids)
        }
        shard_unsegmented_count = sum(subset_counts.values())
        unsegmented_total += shard_unsegmented_count
        source_metadata = dict(data.metadata or {})
        metadata = {
            **source_metadata,
            **split_manifest.metadata(),
            "collection_plan_path": str(subset_plan_path),
            "collection_plan_sha256": subset_plan_hash,
            "unsegmented_example_counts_by_match": subset_counts,
            "unsegmented_example_count": shard_unsegmented_count,
            "derivation": "episode_prefix_tensor_subset",
            "master_tensor_manifest_path": str(master_manifest_path),
            "master_tensor_manifest_payload_sha256": master_manifest.get(
                "manifest_payload_sha256"
            ),
            "master_tensor_sha256": source_shard["tensor_sha256"],
            "source_master_jump_boundary_summary": source_metadata.get(
                "jump_boundary_summary"
            ),
            "jump_boundary_summary": None,
        }
        data = _slice_td_data(data, indices, metadata=metadata)
        duplicate_ids = all_sample_ids.intersection(data.sample_id or [])
        if duplicate_ids:
            raise ValueError(f"Duplicate derived sample ID: {sorted(duplicate_ids)[0]}")
        all_sample_ids.update(data.sample_id or [])

        tensor_path = dataset_root / "train" / job_id / "td_jepa.pt"
        save_td_jepa_data(data, tensor_path)
        raw_job = collection_jobs[job_id]
        shard = deepcopy(source_shard)
        shard.update(
            {
                "path": str(tensor_path.relative_to(output_root)),
                "split": "train",
                "match_ids": sorted(planned_match_ids),
                "example_count": len(data.match_id),
                "source_path": raw_job["path"],
                "source_sha256": raw_job["sha256"],
                "tensor_sha256": _file_sha256(tensor_path),
                "source_tensor_sha256": source_shard["tensor_sha256"],
                "unsegmented_example_count": shard_unsegmented_count,
                "jump_boundary_summary": None,
                "jump_boundary_summary_inherited_from_master": True,
                "derived_episode_prefix_subset": True,
            }
        )
        shard_entries.append(shard)

    manifest = deepcopy(master_manifest)
    manifest.pop("manifest_payload_sha256", None)
    manifest.update(
        {
            "collection_plan_path": str(subset_plan_path),
            "collection_plan_file_sha256": _file_sha256(subset_plan_path),
            "collection_plan_sha256": subset_plan_hash,
            "split_manifest_path": str(split_manifest.path),
            "split_manifest_sha256": split_manifest.sha256,
            "included_splits": ["train"],
            "derivation": "episode_prefix_tensor_subset",
            "master_tensor_manifest_path": str(master_manifest_path),
            "master_tensor_manifest_file_sha256": _file_sha256(master_manifest_path),
            "master_tensor_manifest_payload_sha256": master_manifest.get(
                "manifest_payload_sha256"
            ),
            "subset_collection_manifest_path": str(subset_collection_manifest_path),
            "subset_collection_manifest_sha256": _file_sha256(
                subset_collection_manifest_path
            ),
            "example_count": len(all_sample_ids),
            "unique_sample_id_count": len(all_sample_ids),
            "split_example_counts": {
                "train": len(all_sample_ids),
                "val": 0,
                "test": 0,
            },
            "unsegmented_example_count": unsegmented_total,
            "example_retention_fraction": len(all_sample_ids) / unsegmented_total,
            "jump_boundary_totals": None,
            "jump_boundary_aggregation": "inherited_master_segmentation_not_reaggregated",
            "shards": shard_entries,
            "tensor_hashes_complete": True,
        }
    )
    manifest["config"] = {
        **manifest.get("config", {}),
        "included_splits": ["train"],
        "derivation": "episode_prefix_tensor_subset",
    }
    manifest["manifest_payload_sha256"] = _payload_sha256(manifest)
    manifest_path = dataset_root / "dataset_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path
