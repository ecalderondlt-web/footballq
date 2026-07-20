"""Audit local FOOTPASS tactical data and write provenance-controlled reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.io.footpass import audit_footpass_tactical_data  # noqa: E402
from footballq.repro.manifest import (  # noqa: E402
    build_run_manifest,
    file_sha256,
    write_run_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--hash-source", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    report = audit_footpass_tactical_data(
        args.h5,
        split_manifest_path=args.split_manifest,
        full_scan=not args.inventory_only,
        hash_source=args.hash_source,
    )
    if args.archive is not None:
        if not args.archive.is_file():
            raise FileNotFoundError(f"FOOTPASS archive not found: {args.archive}")
        report["source_archive"] = {
            "path": str(args.archive.resolve()),
            "size_bytes": args.archive.stat().st_size,
            "mtime_ns": args.archive.stat().st_mtime_ns,
            "sha256": file_sha256(args.archive) if args.hash_source else None,
        }
    _write_json(args.out, report)

    warnings = [
        "Data audit only; no representation or tactical-understanding claim is permitted.",
        "FOOTPASS tactical rows do not contain ball coordinates.",
        "The frozen split is an internal development split of the official training release.",
    ]
    dataset_paths = {"footpass_tactical_h5": args.h5}
    if args.archive is not None:
        dataset_paths["footpass_archive"] = args.archive
    manifest = build_run_manifest(
        command=sys.argv,
        config_path=None,
        split_manifest_path=args.split_manifest,
        evaluation_protocol="footpass_train48_source_availability_audit_v1",
        feature_view="source_inventory_not_model_input",
        objective_mode="not_applicable_data_audit",
        dataset_paths=dataset_paths,
        output_paths={"availability_report": args.out},
        warnings=warnings,
    )
    write_run_manifest(args.out.parent / "run_manifest.json", manifest)

    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(f"status: {report['status']}")
    print(f"match_count: {report['match_count']}")
    print(f"half_count: {report['half_count']}")
    print(f"total_rows: {report['total_rows']}")
    print(f"event_rows: {report.get('event_rows', 'not_scanned')}")
    print(f"ball_coordinates_present: {report['ball_coordinates_present']}")
    print(f"split_manifest_sha256: {report['split_manifest_sha256']}")
    print(f"report: {args.out}")


if __name__ == "__main__":
    main()
