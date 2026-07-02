"""Summarize probe incremental value across seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

DEFAULT_CONTRASTS = [
    ("raw_plus_td_jepa", "raw_state_summary", "raw_plus_td_jepa_vs_raw"),
    (
        "raw_plus_td_jepa_zscore",
        "raw_state_summary_zscore",
        "raw_plus_td_jepa_zscore_vs_raw_zscore",
    ),
    ("td_jepa", "random_same_shape", "td_jepa_vs_random"),
    ("td_jepa_zscore", "random_same_shape", "td_jepa_zscore_vs_random"),
]


def parse_suite_spec(spec: str) -> tuple[str, Path]:
    """Parse seed:path probe-suite summary specs."""

    parts = spec.split(":", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("Probe suite specs must use seed:path.")
    return parts[0], Path(parts[1])


def _finite_summary(values: list[float]) -> dict[str, float | int | bool | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "all_positive": None,
        }
    mean = sum(finite) / len(finite)
    variance = sum((value - mean) ** 2 for value in finite) / len(finite)
    return {
        "count": len(finite),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(finite),
        "max": max(finite),
        "all_positive": all(value > 0 for value in finite),
    }


def _metric_for_task(task_type: str) -> tuple[str, str]:
    if task_type == "classification":
        return "test_macro_f1", "higher_is_better"
    if task_type == "regression":
        return "test_rmse", "lower_is_better"
    raise ValueError(f"Unsupported probe task_type {task_type!r}.")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _match_level_fields(run_dir: str | None) -> dict[str, Any]:
    if not run_dir:
        return {
            "match_level_primary_metric": None,
            "match_level_count": None,
            "match_level_mean": None,
            "match_level_std": None,
            "match_level_min": None,
            "match_level_max": None,
        }
    eval_path = Path(run_dir) / "eval_test.json"
    if not eval_path.exists():
        return {
            "match_level_primary_metric": None,
            "match_level_count": None,
            "match_level_mean": None,
            "match_level_std": None,
            "match_level_min": None,
            "match_level_max": None,
        }
    match_level = _read_json(eval_path).get("match_level", {})
    summary = match_level.get("summary", {})
    return {
        "match_level_primary_metric": match_level.get("primary_metric"),
        "match_level_count": summary.get("count"),
        "match_level_mean": summary.get("mean"),
        "match_level_std": summary.get("std"),
        "match_level_min": summary.get("min"),
        "match_level_max": summary.get("max"),
    }


def rows_from_suite(seed: str, path: Path) -> list[dict[str, Any]]:
    """Return normalized probe rows from one suite results JSON."""

    payload = _read_json(path)
    rows = []
    for item in payload.get("results", []):
        if item.get("error"):
            continue
        metric_name, direction = _metric_for_task(str(item["task_type"]))
        metric_value = item.get(metric_name)
        if metric_value is None:
            continue
        run_dir = item.get("run_dir")
        rows.append(
            {
                "seed": seed,
                "suite_path": str(path),
                "target": item["target"],
                "task_type": item["task_type"],
                "feature_source": item["feature_source"],
                "probe_type": item["probe_type"],
                "metric_name": metric_name.replace("test_", ""),
                "metric_direction": direction,
                "metric_value": float(metric_value),
                "num_train": item.get("num_train"),
                "num_val": item.get("num_val"),
                "num_test": item.get("num_test"),
                "run_dir": run_dir,
                **_match_level_fields(run_dir),
            }
        )
    return rows


def _signed_improvement(candidate: float, baseline: float, direction: str) -> float:
    if direction == "higher_is_better":
        return candidate - baseline
    if direction == "lower_is_better":
        return baseline - candidate
    raise ValueError(f"Unsupported metric direction {direction!r}.")


def contrast_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return signed feature-source contrast rows by seed and target."""

    index = {
        (
            str(row["seed"]),
            str(row["target"]),
            str(row["probe_type"]),
            str(row["feature_source"]),
        ): row
        for row in rows
    }
    out = []
    for row in rows:
        seed = str(row["seed"])
        target = str(row["target"])
        probe_type = str(row["probe_type"])
        for candidate, baseline, contrast_name in DEFAULT_CONTRASTS:
            if str(row["feature_source"]) != candidate:
                continue
            baseline_row = index.get((seed, target, probe_type, baseline))
            if baseline_row is None:
                continue
            direction = str(row["metric_direction"])
            improvement = _signed_improvement(
                float(row["metric_value"]),
                float(baseline_row["metric_value"]),
                direction,
            )
            match_delta = None
            if (
                row.get("match_level_mean") is not None
                and baseline_row.get("match_level_mean") is not None
            ):
                match_delta = _signed_improvement(
                    float(row["match_level_mean"]),
                    float(baseline_row["match_level_mean"]),
                    direction,
                )
            out.append(
                {
                    "contrast": contrast_name,
                    "seed": seed,
                    "target": target,
                    "task_type": row["task_type"],
                    "probe_type": probe_type,
                    "candidate_feature_source": candidate,
                    "baseline_feature_source": baseline,
                    "metric_name": row["metric_name"],
                    "metric_direction": direction,
                    "candidate_metric": row["metric_value"],
                    "baseline_metric": baseline_row["metric_value"],
                    "signed_improvement": improvement,
                    "match_level_signed_improvement": match_delta,
                    "match_level_count": row.get("match_level_count"),
                    "candidate_run_dir": row.get("run_dir"),
                    "baseline_run_dir": baseline_row.get("run_dir"),
                }
            )
    return out


def summarize_feature_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate probe metric rows by target and feature source."""

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["target"]), str(row["probe_type"]), str(row["feature_source"]))
        grouped.setdefault(key, []).append(row)
    return {
        f"{target}|{probe_type}|{feature_source}": {
            "target": target,
            "probe_type": probe_type,
            "feature_source": feature_source,
            "metric_name": group_rows[0]["metric_name"],
            "metric_direction": group_rows[0]["metric_direction"],
            "metric_value": _finite_summary(
                [float(row["metric_value"]) for row in group_rows]
            ),
            "match_level_count": _finite_summary(
                [
                    float(row["match_level_count"])
                    for row in group_rows
                    if row.get("match_level_count") is not None
                ]
            ),
            "seeds": sorted(str(row["seed"]) for row in group_rows),
        }
        for (target, probe_type, feature_source), group_rows in sorted(grouped.items())
    }


def summarize_contrasts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate signed incremental contrasts."""

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["target"]), str(row["probe_type"]), str(row["contrast"]))
        grouped.setdefault(key, []).append(row)
    return {
        f"{target}|{probe_type}|{contrast}": {
            "target": target,
            "probe_type": probe_type,
            "contrast": contrast,
            "metric_name": group_rows[0]["metric_name"],
            "metric_direction": group_rows[0]["metric_direction"],
            "signed_improvement": _finite_summary(
                [float(row["signed_improvement"]) for row in group_rows]
            ),
            "match_level_signed_improvement": _finite_summary(
                [
                    float(row["match_level_signed_improvement"])
                    for row in group_rows
                    if row.get("match_level_signed_improvement") is not None
                ]
            ),
            "seeds": sorted(str(row["seed"]) for row in group_rows),
        }
        for (target, probe_type, contrast), group_rows in sorted(grouped.items())
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        row
        for seed, path in [parse_suite_spec(spec) for spec in args.suite]
        for row in rows_from_suite(seed, path)
    ]
    contrasts = contrast_rows(rows)
    summary = {
        "num_rows": len(rows),
        "num_contrasts": len(contrasts),
        "features": summarize_feature_rows(rows),
        "contrasts": summarize_contrasts(contrasts),
        "claim_status": "diagnostic_only",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    rows_csv = args.out / "probe_metric_rows.csv"
    contrasts_csv = args.out / "probe_incremental_rows.csv"
    summary_json = args.out / "probe_incremental_summary.json"
    _write_csv(rows, rows_csv)
    _write_csv(contrasts, contrasts_csv)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"rows_csv: {rows_csv}")
    print(f"contrasts_csv: {contrasts_csv}")
    print(f"summary_json: {summary_json}")


if __name__ == "__main__":
    main()
