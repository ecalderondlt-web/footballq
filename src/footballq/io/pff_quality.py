"""Dataset-level quality summaries for canonical PFF tracking shards."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _frame_shapes(path: Path) -> Counter[str]:
    frame = pd.read_parquet(path, columns=["frame_id", "agent_type", "team_id"])
    frame_ids = frame["frame_id"].drop_duplicates().sort_values()

    def counts(mask: pd.Series) -> pd.Series:
        return frame.loc[mask].groupby("frame_id").size().reindex(frame_ids, fill_value=0)

    balls = counts(frame["agent_type"] == "ball")
    home = counts((frame["agent_type"] == "player") & (frame["team_id"] == "home"))
    away = counts((frame["agent_type"] == "player") & (frame["team_id"] == "away"))
    return Counter(
        f"home={home_count},away={away_count},ball={ball_count}"
        for home_count, away_count, ball_count in zip(home, away, balls, strict=True)
    )


def summarize_pff_canonical_quality(
    canonical_root: str | Path,
    *,
    scan_frame_shapes: bool = True,
    write: bool = True,
) -> dict[str, Any]:
    """Aggregate match manifests and optionally scan exact per-frame entity shapes."""

    root = Path(canonical_root)
    dataset_manifest_path = root / "dataset_manifest.json"
    if not dataset_manifest_path.exists():
        raise FileNotFoundError(f"PFF dataset manifest not found: {dataset_manifest_path}")
    dataset = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    match_manifests = [
        json.loads(path.read_text(encoding="utf-8")) for path in root.rglob("manifest.json")
    ]
    if len(match_manifests) != int(dataset["selected_match_count"]):
        raise ValueError("PFF completed match manifests do not match the dataset manifest count.")

    visibility: Counter[str] = Counter()
    periods: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    total_rows = 0
    source_hash_missing: list[str] = []
    per_match: list[dict[str, Any]] = []
    shape_counts: Counter[str] = Counter()
    for manifest in match_manifests:
        rows = sum(int(value) for value in manifest["rows_by_period"].values())
        frames = int(manifest["unique_frames"])
        total_rows += rows
        visibility.update(manifest["visibility_counts"])
        periods.update({key: int(value) for key, value in manifest["frames_by_period"].items()})
        split_counts[manifest["split"]] += 1
        if not manifest["source"].get("sha256"):
            source_hash_missing.append(manifest["match_id"])
        if scan_frame_shapes:
            match_dir = root / manifest["split"] / manifest["match_id"]
            for shard in manifest["shards"]:
                shape_counts.update(_frame_shapes(match_dir / shard["path"]))
        per_match.append(
            {
                "match_id": manifest["match_id"],
                "split": manifest["split"],
                "unique_frames": frames,
                "rows": rows,
                "missing_ball_rate": manifest["missing_ball_frames"] / frames,
                "non_23_entity_rate": manifest["non_23_entity_frames"] / frames,
                "out_of_bounds_row_rate": manifest["out_of_bounds_rows"] / rows,
                "estimated_coordinate_rate": (
                    manifest["visibility_counts"].get("ESTIMATED", 0) / rows
                ),
                "frame_gap_count": manifest["frame_gap_count"],
                "time_regression_count": manifest["time_regression_count"],
            }
        )

    totals = dataset["totals"]
    total_frames = int(totals["unique_frames"])
    shape_total = sum(shape_counts.values())
    standard_with_ball = shape_counts.get("home=11,away=11,ball=1", 0)
    standard_without_ball = shape_counts.get("home=11,away=11,ball=0", 0)
    report = {
        "status": "complete",
        "dataset": "pff_fc",
        "canonical_dataset_manifest_sha256": dataset["manifest_payload_sha256"],
        "match_count": len(match_manifests),
        "split_match_counts": dict(sorted(split_counts.items())),
        "shard_count": sum(len(manifest["shards"]) for manifest in match_manifests),
        "total_frames": total_frames,
        "total_rows": total_rows,
        "frames_by_period": dict(sorted(periods.items(), key=lambda item: int(item[0]))),
        "duplicate_record_rate": totals["duplicate_records"] / totals["records_read"],
        "missing_ball_frame_rate": totals["missing_ball_frames"] / total_frames,
        "non_23_entity_frame_rate": totals["non_23_entity_frames"] / total_frames,
        "out_of_bounds_row_rate": totals["out_of_bounds_rows"] / total_rows,
        "visibility_counts": dict(sorted(visibility.items())),
        "estimated_coordinate_rate": visibility.get("ESTIMATED", 0) / total_rows,
        "visible_coordinate_rate": visibility.get("VISIBLE", 0) / total_rows,
        "frame_gap_count": totals["frame_gap_count"],
        "missing_frame_count": totals["missing_frame_count"],
        "time_regression_count": totals["time_regression_count"],
        "source_hash_missing_match_ids": sorted(source_hash_missing),
        "frame_shape_scan_complete": scan_frame_shapes and shape_total == total_frames,
        "frame_shape_counts": dict(shape_counts.most_common()),
        "standard_22_players_with_ball_rate": (
            standard_with_ball / shape_total if shape_total else None
        ),
        "standard_22_players_without_ball_rate": (
            standard_without_ball / shape_total if shape_total else None
        ),
        "other_frame_shape_rate": (
            (shape_total - standard_with_ball - standard_without_ball) / shape_total
            if shape_total
            else None
        ),
        "per_match": sorted(per_match, key=lambda item: int(item["match_id"])),
    }
    if write:
        _write_json(root / "quality_report.json", report)
    return report
