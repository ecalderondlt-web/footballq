"""Validate blinded diagnostic annotation packages before human review."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PRIVATE_FIELDS = {
    "cluster_id",
    "latent_residual_score",
    "positive_control",
    "rank_source",
    "control_group",
    "control_match_reason",
}
ANNOTATOR_REQUIRED_FIELDS = {
    "blind_id",
    "match_id",
    "period",
    "frame_t",
    "clip_path",
    "annotation",
}
KEY_REQUIRED_FIELDS = {"blind_id"} | PRIVATE_FIELDS


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _nonempty(value: object) -> bool:
    return bool(str(value or "").strip())


def _resolve_clip_path(value: str, *, cwd: Path, annotator_csv: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    annotator_relative = annotator_csv.parent / path
    if annotator_relative.exists():
        return annotator_relative
    return cwd / path


def validate_blinded_annotation_package(
    *,
    annotator_csv: str | Path,
    key_csv: str | Path,
    manifest_json: str | Path | None = None,
    require_clip_paths: bool = True,
    require_blank_annotations: bool = True,
    cwd: str | Path = ".",
) -> dict[str, Any]:
    """Return a machine-readable validation report for a blinded package."""

    cwd_path = Path(cwd)
    annotator_path = Path(annotator_csv)
    key_path = Path(key_csv)
    issues: list[str] = []
    warnings: list[str] = []

    if not annotator_path.exists():
        issues.append(f"annotator_csv does not exist: {annotator_path}")
        annotator_fields: list[str] = []
        annotator_rows: list[dict[str, str]] = []
    else:
        annotator_fields, annotator_rows = _read_csv(annotator_path)
    if not key_path.exists():
        issues.append(f"key_csv does not exist: {key_path}")
        key_fields: list[str] = []
        key_rows: list[dict[str, str]] = []
    else:
        key_fields, key_rows = _read_csv(key_path)

    missing_annotator_fields = sorted(ANNOTATOR_REQUIRED_FIELDS - set(annotator_fields))
    if missing_annotator_fields:
        issues.append("annotator_csv missing fields: " + ", ".join(missing_annotator_fields))
    leaked_fields = sorted(PRIVATE_FIELDS & set(annotator_fields))
    if leaked_fields:
        issues.append("annotator_csv exposes private fields: " + ", ".join(leaked_fields))

    missing_key_fields = sorted(KEY_REQUIRED_FIELDS - set(key_fields))
    if missing_key_fields:
        issues.append("key_csv missing fields: " + ", ".join(missing_key_fields))
    if _is_relative_to(key_path, annotator_path.parent):
        issues.append("key_csv must not be stored inside the annotator directory.")

    annotator_ids = [row.get("blind_id", "") for row in annotator_rows]
    key_ids = [row.get("blind_id", "") for row in key_rows]
    if len(annotator_ids) != len(set(annotator_ids)):
        issues.append("annotator_csv contains duplicate blind_id values.")
    if len(key_ids) != len(set(key_ids)):
        issues.append("key_csv contains duplicate blind_id values.")
    if set(annotator_ids) != set(key_ids):
        missing_in_key = sorted(set(annotator_ids) - set(key_ids))
        missing_in_annotator = sorted(set(key_ids) - set(annotator_ids))
        if missing_in_key:
            issues.append("blind_id values missing from key_csv: " + ", ".join(missing_in_key))
        if missing_in_annotator:
            issues.append(
                "blind_id values missing from annotator_csv: " + ", ".join(missing_in_annotator)
            )

    blank_clip_ids = [
        row.get("blind_id", "") for row in annotator_rows if not _nonempty(row.get("clip_path"))
    ]
    missing_clip_files: list[str] = []
    for row in annotator_rows:
        clip_path = str(row.get("clip_path", "")).strip()
        if not clip_path:
            continue
        resolved = _resolve_clip_path(clip_path, cwd=cwd_path, annotator_csv=annotator_path)
        if not resolved.exists():
            missing_clip_files.append(clip_path)
    if require_clip_paths and blank_clip_ids:
        issues.append("annotator rows have blank clip_path: " + ", ".join(blank_clip_ids))
    elif blank_clip_ids:
        warnings.append("annotator rows have blank clip_path: " + ", ".join(blank_clip_ids))
    if missing_clip_files:
        issues.append("clip_path files do not exist: " + ", ".join(sorted(missing_clip_files)))

    filled_annotation_ids = [
        row.get("blind_id", "") for row in annotator_rows if _nonempty(row.get("annotation"))
    ]
    if require_blank_annotations and filled_annotation_ids:
        issues.append(
            "annotation cells should be blank before review: " + ", ".join(filled_annotation_ids)
        )

    manifest_payload: dict[str, Any] | None = None
    manifest_path = Path(manifest_json) if manifest_json is not None else None
    if manifest_path is not None:
        if not manifest_path.exists():
            issues.append(f"manifest_json does not exist: {manifest_path}")
        else:
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_payload.get("claim_status") != "diagnostic_only":
                issues.append("manifest claim_status must be diagnostic_only.")
            if int(manifest_payload.get("rows", -1)) != len(annotator_rows):
                issues.append("manifest row count does not match annotator_csv.")
            expected_clip_count = sum(_nonempty(row.get("clip_path")) for row in annotator_rows)
            if int(manifest_payload.get("rows_with_clip_path", -1)) != expected_clip_count:
                issues.append("manifest rows_with_clip_path does not match annotator_csv.")
            render_stats = manifest_payload.get("render_stats", {})
            if require_clip_paths and int(render_stats.get("missing_windows", 0)) != 0:
                issues.append("manifest render_stats.missing_windows must be 0.")

    report = {
        "validation_status": "passed" if not issues else "failed",
        "annotator_csv": str(annotator_path),
        "key_csv": str(key_path),
        "manifest_json": str(manifest_path) if manifest_path is not None else "",
        "row_count": len(annotator_rows),
        "key_row_count": len(key_rows),
        "rows_with_clip_path": sum(_nonempty(row.get("clip_path")) for row in annotator_rows),
        "rows_without_clip_path": len(blank_clip_ids),
        "filled_annotation_count": len(filled_annotation_ids),
        "claim_status": (
            str(manifest_payload.get("claim_status", "")) if manifest_payload is not None else ""
        ),
        "issues": issues,
        "warnings": warnings,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator-csv", type=Path, required=True)
    parser.add_argument("--key-csv", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--allow-missing-clips", action="store_true")
    parser.add_argument("--allow-filled-annotations", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_blinded_annotation_package(
        annotator_csv=args.annotator_csv,
        key_csv=args.key_csv,
        manifest_json=args.manifest_json,
        require_clip_paths=not args.allow_missing_clips,
        require_blank_annotations=not args.allow_filled_annotations,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"validation_status: {report['validation_status']}")
        print(f"rows: {report['row_count']}")
        print(f"rows_with_clip_path: {report['rows_with_clip_path']}")
        print(f"rows_without_clip_path: {report['rows_without_clip_path']}")
        print(f"filled_annotation_count: {report['filled_annotation_count']}")
        for issue in report["issues"]:
            print(f"issue: {issue}")
        for warning in report["warnings"]:
            print(f"warning: {warning}")
    if report["validation_status"] != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
