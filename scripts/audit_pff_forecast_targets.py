"""Audit PFF forecast target lineage and the train/validation-only access boundary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.repro.manifest import file_sha256  # noqa: E402
from footballq.repro.splits import load_split_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify-tensors", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    split = load_split_manifest(manifest["split_manifest_path"])
    source_manifest_path = Path(manifest["source_manifest_path"])
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_by_path = {entry["path"]: entry for entry in source_manifest["shards"]}
    root = args.manifest.parent
    checks = {
        "status_complete": manifest.get("status") == "complete",
        "tensor_hashes_complete": manifest.get("tensor_hashes_complete") is True,
        "split_hash_matches": manifest.get("split_manifest_sha256") == split.sha256,
        "source_split_hash_matches": source_manifest.get("split_manifest_sha256") == split.sha256,
        "included_splits_exact": manifest.get("included_splits") == ["train", "val"],
        "test_not_included": manifest.get("test_included") is False,
        "no_test_entries": all(entry["split"] != "test" for entry in manifest["shards"]),
        "no_test_paths": all(
            "test" not in Path(entry["path"]).parts for entry in manifest["shards"]
        ),
        "match_inventory_exact": set(manifest["included_match_ids"])
        == set(split.train_match_ids + split.val_match_ids),
        "source_links_complete": all(
            entry["source_path"] in source_by_path for entry in manifest["shards"]
        ),
        "source_hash_links_match": all(
            entry["source_tensor_sha256"]
            == source_by_path[entry["source_path"]]["tensor_sha256"]
            for entry in manifest["shards"]
        ),
        "example_count_matches": manifest["example_count"]
        == sum(int(entry["example_count"]) for entry in manifest["shards"]),
        "valid_endpoint_count_matches": manifest["valid_endpoint_count"]
        == sum(int(entry["valid_endpoint_count"]) for entry in manifest["shards"]),
    }
    tensor_failures = []
    if args.verify_tensors:
        for entry in manifest["shards"]:
            path = root / entry["path"]
            if not path.exists() or file_sha256(path) != entry["target_tensor_sha256"]:
                tensor_failures.append(entry["path"])
    checks["target_tensor_hashes_match"] = not tensor_failures if args.verify_tensors else True
    failed = sorted(name for name, value in checks.items() if not value)
    report = {
        "version": 1,
        "study": "pff_trajectory_forecast_v1",
        "status": "passed" if not failed else "blocked",
        "checks": checks,
        "failed_checks": failed,
        "tensor_hash_verification_requested": args.verify_tensors,
        "tensor_hash_failures": tensor_failures,
        "manifest_file_sha256": file_sha256(args.manifest),
        "manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "source_manifest_file_sha256": file_sha256(source_manifest_path),
        "shard_count": len(manifest["shards"]),
        "example_count": manifest["example_count"],
        "valid_endpoint_count": manifest["valid_endpoint_count"],
        "loaded_splits": ["train", "val"],
        "test_loaded": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"status: {report['status']}")
    print(f"failed_checks: {','.join(failed) if failed else 'none'}")
    print(f"report: {args.out}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
