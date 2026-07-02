"""Summarize discovery cluster controls across features and seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


QUALITY_FIELDS = [
    "average_within_cluster_distance",
    "centroid_margin_proxy",
    "cluster_size_entropy",
    "empty_cluster_count",
    "max_cluster_size_fraction",
]

NUISANCE_FIELDS = [
    "max_cluster_top_match_fraction",
    "mean_cluster_top_match_fraction",
    "max_delta_norm_top_fraction",
    "mean_delta_norm_top_fraction",
    "min_heldout_examples_per_cluster",
]


def parse_summary_spec(spec: str) -> tuple[str, str, Path]:
    """Parse feature:seed:path summary specs."""

    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise ValueError("Cluster summary specs must use feature:seed:path.")
    feature, seed, path = parts
    if not feature or not seed or not path:
        raise ValueError("Cluster summary specs require nonempty feature, seed, and path.")
    return feature, seed, Path(path)


def _finite_summary(values: list[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = sum(finite) / len(finite)
    variance = sum((value - mean) ** 2 for value in finite) / len(finite)
    return {
        "count": len(finite),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(finite),
        "max": max(finite),
    }


def _empty_nuisance_fields() -> dict[str, float | int | None]:
    return {field: None for field in NUISANCE_FIELDS}


def _cluster_csv_path(summary_path: Path, cluster: dict[str, Any]) -> Path | None:
    raw_path = cluster.get("clusters_csv")
    if raw_path is None:
        return None
    path = Path(str(raw_path))
    if path.exists():
        return path
    fallback = summary_path.parent / path.name
    if fallback.exists():
        return fallback
    return path


def _cluster_csv_nuisance_fields(summary_path: Path, cluster: dict[str, Any]) -> dict[str, Any]:
    clusters_csv = _cluster_csv_path(summary_path, cluster)
    if clusters_csv is None or not clusters_csv.exists():
        return _empty_nuisance_fields()

    top_match_fractions = []
    delta_top_fractions = []
    heldout_counts = []
    with clusters_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cluster_size = int(float(row.get("n_examples", 0) or 0))
            if cluster_size <= 0:
                continue
            match_counts = json.loads(row.get("match_id_counts", "{}") or "{}")
            if match_counts:
                top_match = max(int(value) for value in match_counts.values())
                top_match_fractions.append(top_match / cluster_size)
            delta_top = row.get("delta_norm_top_fraction")
            if delta_top not in {None, ""}:
                delta_top_fractions.append(float(delta_top))
            val_count = int(float(row.get("val_count", 0) or 0))
            test_count = int(float(row.get("test_count", 0) or 0))
            heldout_counts.append(val_count + test_count)

    top_match_summary = _finite_summary(top_match_fractions)
    delta_top_summary = _finite_summary(delta_top_fractions)
    heldout_summary = _finite_summary([float(value) for value in heldout_counts])
    return {
        "max_cluster_top_match_fraction": top_match_summary["max"],
        "mean_cluster_top_match_fraction": top_match_summary["mean"],
        "max_delta_norm_top_fraction": delta_top_summary["max"],
        "mean_delta_norm_top_fraction": delta_top_summary["mean"],
        "min_heldout_examples_per_cluster": heldout_summary["min"],
    }


def row_from_cluster_summary(feature: str, seed: str, path: Path, k: int) -> dict[str, Any]:
    """Return one normalized row from a cluster summary JSON."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    clusters = payload.get("clusters", [])
    matches = [item for item in clusters if int(item.get("quality", {}).get("k", -1)) == int(k)]
    if not matches:
        raise ValueError(f"No k={k} cluster quality found in {path}.")
    cluster = matches[0]
    quality = dict(cluster["quality"])
    num_examples = int(quality["num_examples"])
    nuisance_fields = _cluster_csv_nuisance_fields(path, cluster)
    return {
        "feature": feature,
        "seed": seed,
        "path": str(path),
        "k": int(k),
        "delta_seconds": payload.get("delta_seconds"),
        "num_examples": num_examples,
        "assignment_protocol": cluster.get("assignment_protocol"),
        "scientific_mode": payload.get("scientific_mode"),
        "split_manifest_sha256": payload.get("split_manifest_sha256"),
        "average_within_cluster_distance": quality.get("average_within_cluster_distance"),
        "centroid_margin_proxy": quality.get("centroid_margin_proxy"),
        "cluster_size_entropy": quality.get("cluster_size_entropy"),
        "empty_cluster_count": quality.get("empty_cluster_count"),
        "max_cluster_size_fraction": float(quality.get("max_cluster_size", 0)) / max(
            1,
            num_examples,
        ),
        **nuisance_fields,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate rows by feature."""

    by_feature: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_feature.setdefault(str(row["feature"]), []).append(row)
    return {
        feature: {
            field: _finite_summary(
                [float(row[field]) for row in feature_rows if row.get(field) is not None]
            )
            for field in [*QUALITY_FIELDS, *NUISANCE_FIELDS]
        }
        | {
            "seeds": sorted(str(row["seed"]) for row in feature_rows),
            "assignment_protocols": sorted(
                set(str(row.get("assignment_protocol")) for row in feature_rows)
            ),
        }
        for feature, feature_rows in sorted(by_feature.items())
    }


def write_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write per-feature per-seed rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "feature",
        "seed",
        "k",
        "delta_seconds",
        "num_examples",
        "assignment_protocol",
        "scientific_mode",
        "split_manifest_sha256",
        *QUALITY_FIELDS,
        *NUISANCE_FIELDS,
        "path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-summary", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        row_from_cluster_summary(feature, seed, path, k=args.k)
        for feature, seed, path in [parse_summary_spec(spec) for spec in args.cluster_summary]
    ]
    summary = {
        "k": int(args.k),
        "num_rows": len(rows),
        "features": summarize_rows(rows),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    rows_csv = args.out / "discovery_control_rows.csv"
    summary_json = args.out / "discovery_control_summary.json"
    write_rows_csv(rows, rows_csv)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"rows_csv: {rows_csv}")
    print(f"summary_json: {summary_json}")


if __name__ == "__main__":
    main()
