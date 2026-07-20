"""Deterministic feature-view projection for finalized TD-JEPA shard manifests."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from footballq.data.td_jepa_dataset import load_td_jepa_data, save_td_jepa_data
from footballq.repro.feature_views import apply_feature_view


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _manifest_root(manifest_path: Path, first_shard_path: str) -> Path:
    relative = Path(first_shard_path)
    for candidate in (manifest_path.parent, manifest_path.parent.parent):
        if (candidate / relative).exists():
            return candidate
    raise FileNotFoundError(
        f"Cannot resolve shard {first_shard_path!r} from manifest {manifest_path}."
    )


def project_td_jepa_feature_view(
    source_manifest_path: str | Path,
    output_root: str | Path,
    *,
    target_feature_view: str,
    included_splits: set[str] | None = None,
) -> Path:
    """Project selected splits without opening excluded split tensors."""

    source_path = Path(source_manifest_path)
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    source_shards = list(source_manifest.get("shards", []))
    if not source_shards:
        raise ValueError("Source TD-JEPA manifest contains no shards.")
    if not source_manifest.get("tensor_hashes_complete"):
        raise ValueError("Source TD-JEPA manifest has not finalized tensor hashes.")

    available_splits = {str(shard["split"]) for shard in source_shards}
    selected_splits = set(included_splits or available_splits)
    unknown = selected_splits - available_splits
    if unknown:
        raise ValueError(f"Requested splits are absent from source manifest: {sorted(unknown)}")
    selected_shards = [
        shard for shard in source_shards if str(shard["split"]) in selected_splits
    ]
    if not selected_shards:
        raise ValueError("Feature projection selected zero source shards.")

    source_root = _manifest_root(source_path, str(selected_shards[0]["path"]))
    output_root = Path(output_root)
    dataset_name = str(source_manifest.get("profile") or source_path.parent.name)
    dataset_root = output_root / dataset_name
    source_manifest_file_sha256 = _file_sha256(source_path)
    source_manifest_payload_sha256 = source_manifest.get("manifest_payload_sha256")

    shard_entries: list[dict[str, Any]] = []
    all_sample_ids: set[str] = set()
    selected_match_ids: set[str] = set()
    selected_names: list[str] | None = None
    split_example_counts: Counter[str] = Counter()
    for shard in selected_shards:
        source_tensor_path = source_root / str(shard["path"])
        data = load_td_jepa_data(source_tensor_path)
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
            "source_feature_manifest_file_sha256": source_manifest_file_sha256,
            "source_feature_manifest_payload_sha256": source_manifest_payload_sha256,
            "projection_included_splits": sorted(selected_splits),
        }
        selected_names = list(names)

        sample_ids = list(data.sample_id or [])
        duplicate_ids = all_sample_ids.intersection(sample_ids)
        if duplicate_ids:
            raise ValueError(f"Duplicate projected sample ID: {sorted(duplicate_ids)[0]}")
        all_sample_ids.update(sample_ids)
        selected_match_ids.update(str(value) for value in data.match_id)
        split_example_counts[str(shard["split"])] += len(data.match_id)

        source_relative = Path(str(shard["path"]))
        if source_relative.parts and source_relative.parts[0] == source_path.parent.name:
            source_relative = Path(*source_relative.parts[1:])
        tensor_path = dataset_root / source_relative
        save_td_jepa_data(data, tensor_path)

        projected_shard = deepcopy(shard)
        projected_shard.update(
            {
                "path": str(tensor_path.relative_to(output_root)),
                "tensor_sha256": _file_sha256(tensor_path),
                "feature_view": target_feature_view,
                "source_tensor_sha256": shard["tensor_sha256"],
            }
        )
        shard_entries.append(projected_shard)

    excluded_counts = Counter(str(shard["split"]) for shard in source_shards)
    for split in selected_splits:
        excluded_counts.pop(split, None)

    manifest = deepcopy(source_manifest)
    manifest.pop("manifest_payload_sha256", None)
    manifest.update(
        {
            "feature_view": target_feature_view,
            "feature_names": selected_names or [],
            "source_feature_manifest_path": str(source_path),
            "source_feature_manifest_file_sha256": source_manifest_file_sha256,
            "source_feature_manifest_payload_sha256": source_manifest_payload_sha256,
            "included_splits": sorted(selected_splits),
            "excluded_source_shard_counts": dict(sorted(excluded_counts.items())),
            "selected_match_ids": sorted(selected_match_ids),
            "selected_match_count": len(selected_match_ids),
            "example_count": len(all_sample_ids),
            "unique_sample_id_count": len(all_sample_ids),
            "split_example_counts": dict(sorted(split_example_counts.items())),
            "shards": shard_entries,
            "tensor_hashes_complete": True,
        }
    )
    manifest["config"] = {
        **manifest.get("config", {}),
        "feature_view": target_feature_view,
        "included_splits": sorted(selected_splits),
    }
    manifest["manifest_payload_sha256"] = _payload_sha256(manifest)
    manifest_path = dataset_root / "dataset_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path
