"""Audit exact tensor nesting for the frozen 1x/4x/8x GRF scale datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

TENSOR_KEYS = (
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
)
LIST_KEYS = ("match_id", "period", "frame_t", "sample_id")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_root(path: Path) -> Path:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    first = Path(manifest["shards"][0]["path"])
    root = path.parent
    while first.parts and root.name == first.parts[0]:
        root = root.parent
        first = Path(*first.parts[1:])
    return root


def compare_nested_manifests(
    parent_manifest_path: str | Path,
    child_manifest_path: str | Path,
) -> dict[str, Any]:
    parent_path = Path(parent_manifest_path)
    child_path = Path(child_manifest_path)
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    child = json.loads(child_path.read_text(encoding="utf-8"))
    parent_root = _manifest_root(parent_path)
    child_root = _manifest_root(child_path)
    parent_shards = {str(item["job_id"]): item for item in parent["shards"]}
    child_shards = {str(item["job_id"]): item for item in child["shards"]}
    checks: dict[str, bool] = {
        "job_ids_match": set(parent_shards) == set(child_shards),
        "parent_train_only": parent.get("included_splits") == ["train"],
        "child_train_only": child.get("included_splits") == ["train"],
        "parent_has_zero_holdout": all(
            int(parent["split_example_counts"].get(split, 0)) == 0
            for split in ("val", "test")
        ),
        "child_has_zero_holdout": all(
            int(child["split_example_counts"].get(split, 0)) == 0
            for split in ("val", "test")
        ),
        "visibility_profile_matches": (
            parent.get("visibility_profile_sha256")
            == child.get("visibility_profile_sha256")
        ),
        "feature_view_matches": (
            parent.get("config", {}).get("feature_view")
            == child.get("config", {}).get("feature_view")
            == "position_only"
        ),
        "velocity_mode_matches": parent.get("velocity_mode") == child.get("velocity_mode"),
    }
    parent_examples = 0
    child_examples = 0
    for job_id in sorted(set(parent_shards) & set(child_shards)):
        parent_payload = torch.load(
            parent_root / parent_shards[job_id]["path"],
            map_location="cpu",
            weights_only=False,
        )
        child_payload = torch.load(
            child_root / child_shards[job_id]["path"],
            map_location="cpu",
            weights_only=False,
        )
        parent_ids = list(map(str, parent_payload["sample_id"]))
        child_ids = list(map(str, child_payload["sample_id"]))
        parent_examples += len(parent_ids)
        child_examples += len(child_ids)
        parent_index = {sample_id: index for index, sample_id in enumerate(parent_ids)}
        checks[f"{job_id}:child_ids_unique"] = len(child_ids) == len(set(child_ids))
        checks[f"{job_id}:sample_id_subset"] = set(child_ids) <= set(parent_ids)
        if not checks[f"{job_id}:sample_id_subset"]:
            continue
        selected_list = [parent_index[sample_id] for sample_id in child_ids]
        selected = torch.tensor(selected_list, dtype=torch.long)
        checks[f"{job_id}:order_preserved"] = selected_list == sorted(selected_list)
        for key in LIST_KEYS:
            checks[f"{job_id}:{key}"] = [
                parent_payload[key][index] for index in selected_list
            ] == child_payload[key]
        for key in TENSOR_KEYS:
            checks[f"{job_id}:{key}"] = torch.equal(
                parent_payload[key][selected], child_payload[key]
            )
        checks[f"{job_id}:feature_names"] = (
            parent_payload["feature_names"] == child_payload["feature_names"]
        )

    checks["parent_manifest_count_matches_loaded"] = (
        int(parent["example_count"]) == parent_examples
    )
    checks["child_manifest_count_matches_loaded"] = int(child["example_count"]) == child_examples
    checks["child_is_strict_subset"] = 0 < child_examples < parent_examples
    checks["child_retention_above_frozen_floor"] = (
        float(child["example_retention_fraction"]) >= 0.75
    )
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "passed": not failed,
        "parent_manifest_path": str(parent_path),
        "parent_manifest_sha256": _sha256(parent_path),
        "child_manifest_path": str(child_path),
        "child_manifest_sha256": _sha256(child_path),
        "parent_examples": parent_examples,
        "child_examples": child_examples,
        "checks": checks,
        "failed_checks": failed,
    }


def audit_scale_nesting(
    one_x: str | Path,
    four_x: str | Path,
    eight_x: str | Path,
) -> dict[str, Any]:
    comparisons = {
        "1x_in_4x": compare_nested_manifests(four_x, one_x),
        "4x_in_8x": compare_nested_manifests(eight_x, four_x),
        "1x_in_8x": compare_nested_manifests(eight_x, one_x),
    }
    failed = [name for name, result in comparisons.items() if not result["passed"]]
    return {
        "status": "passed" if not failed else "blocked",
        "comparisons": comparisons,
        "failed_comparisons": failed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--one-x", type=Path, required=True)
    parser.add_argument("--four-x", type=Path, required=True)
    parser.add_argument("--eight-x", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = audit_scale_nesting(args.one_x, args.four_x, args.eight_x)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"status: {audit['status']}")
    print(f"audit: {args.out}")
    for failure in audit["failed_comparisons"]:
        print(f"failed: {failure}")


if __name__ == "__main__":
    main()
