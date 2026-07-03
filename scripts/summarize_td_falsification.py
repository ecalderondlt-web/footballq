"""Summarize TD-JEPA falsification controls across seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

CONTROL_EXPECTATIONS = {
    "shuffled_future_within_batch": "higher_than_correct",
    "future_from_another_match": "higher_than_correct",
    "reversed_time_context": "higher_than_correct",
    "masked_ball": "higher_than_correct",
    "team_swap": "higher_than_correct",
    "team_label_swap": "higher_than_correct",
    "target_team_label_swap": "higher_than_correct",
    "pitch_reflection": "higher_than_correct",
    "consistent_player_slot_permutation": "higher_than_correct",
    "target_consistent_player_slot_permutation": "higher_than_correct",
    "no_motion_predictor": "higher_than_correct",
}


def parse_summary_spec(spec: str) -> tuple[str, Path]:
    """Parse seed:path falsification summary specs."""

    parts = spec.split(":", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("Falsification summary specs must use seed:path.")
    return parts[0], Path(parts[1])


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


def rows_from_summary(seed: str, path: Path, metric: str = "td_loss") -> list[dict[str, Any]]:
    """Return normalized rows for one falsification summary."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload["results"]
    correct = float(results["correct_temporal_pairing"][metric])
    rows = []
    for condition, metrics in sorted(results.items()):
        metric_value = metrics.get(metric)
        td_loss = metrics.get("td_loss")
        total_loss = metrics.get("total_loss")
        slot_reconstruction = metrics.get("slot_reconstruction_loss")
        context_reconstruction = metrics.get("context_reconstruction_loss")
        cosine = metrics.get("cosine_similarity")
        if metric_value is None:
            ratio = None
            margin = None
        else:
            ratio = float(metric_value) / max(correct, 1e-12)
            margin = float(metric_value) - correct
        rows.append(
            {
                "seed": seed,
                "condition": condition,
                "gate_metric": metric,
                "metric_value": metric_value,
                "td_loss": td_loss,
                "total_loss": total_loss,
                "slot_reconstruction_loss": slot_reconstruction,
                "context_reconstruction_loss": context_reconstruction,
                "cosine_similarity": cosine,
                "metric_ratio_vs_correct": ratio,
                "metric_margin_vs_correct": margin,
                "num_examples": metrics.get("num_examples"),
                "summary_path": str(path),
            }
        )
    return rows


def _status_for_ratio(
    ratio_summary: dict[str, Any],
    pass_ratio: float,
    caution_ratio: float,
) -> str:
    minimum = ratio_summary.get("min")
    if minimum is None:
        return "unavailable"
    if float(minimum) >= pass_ratio:
        return "pass"
    if float(minimum) >= caution_ratio:
        return "caution"
    return "fail"


def _row_value(row: dict[str, Any], key: str, fallback: str) -> Any:
    return row.get(key, row.get(fallback))


def summarize_rows(
    rows: list[dict[str, Any]],
    pass_ratio: float = 1.25,
    caution_ratio: float = 1.05,
) -> dict[str, Any]:
    """Summarize falsification ratios by condition."""

    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_condition.setdefault(str(row["condition"]), []).append(row)
    conditions = {}
    for condition, condition_rows in sorted(by_condition.items()):
        ratio_summary = _finite_summary(
            [
                float(_row_value(row, "metric_ratio_vs_correct", "td_loss_ratio_vs_correct"))
                for row in condition_rows
                if _row_value(row, "metric_ratio_vs_correct", "td_loss_ratio_vs_correct")
                is not None
            ]
        )
        margin_summary = _finite_summary(
            [
                float(_row_value(row, "metric_margin_vs_correct", "td_loss_margin_vs_correct"))
                for row in condition_rows
                if _row_value(row, "metric_margin_vs_correct", "td_loss_margin_vs_correct")
                is not None
            ]
        )
        status = (
            "reference"
            if condition == "correct_temporal_pairing"
            else _status_for_ratio(ratio_summary, pass_ratio, caution_ratio)
        )
        conditions[condition] = {
            "expectation": "reference"
            if condition == "correct_temporal_pairing"
            else CONTROL_EXPECTATIONS.get(condition, "higher_than_correct"),
            "status": status,
            "td_loss_ratio_vs_correct": ratio_summary,
            "td_loss_margin_vs_correct": margin_summary,
            "gate_metric_ratio_vs_correct": ratio_summary,
            "gate_metric_margin_vs_correct": margin_summary,
            "seeds": sorted(str(row["seed"]) for row in condition_rows),
        }
    blocking = [
        condition
        for condition, summary in conditions.items()
        if summary["status"] in {"fail", "caution"}
        and condition
        not in {
            "correct_temporal_pairing",
            "reversed_time_context",
            "masked_ball",
        }
    ]
    return {
        "pass_ratio": pass_ratio,
        "caution_ratio": caution_ratio,
        "conditions": conditions,
        "blocking_conditions": blocking,
        "scientific_claim_status": "blocked" if blocking else "controls_passed",
    }


def write_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write one row per condition per seed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "condition",
        "gate_metric",
        "metric_value",
        "td_loss",
        "total_loss",
        "slot_reconstruction_loss",
        "context_reconstruction_loss",
        "cosine_similarity",
        "metric_ratio_vs_correct",
        "metric_margin_vs_correct",
        "num_examples",
        "summary_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pass-ratio", type=float, default=1.25)
    parser.add_argument("--caution-ratio", type=float, default=1.05)
    parser.add_argument("--metric", default="td_loss", choices=["td_loss", "total_loss"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        row
        for seed, path in [parse_summary_spec(spec) for spec in args.summary]
        for row in rows_from_summary(seed, path, metric=str(args.metric))
    ]
    summary = summarize_rows(
        rows,
        pass_ratio=float(args.pass_ratio),
        caution_ratio=float(args.caution_ratio),
    )
    summary["gate_metric"] = str(args.metric)
    args.out.mkdir(parents=True, exist_ok=True)
    rows_csv = args.out / "td_falsification_rows.csv"
    summary_json = args.out / "td_falsification_gate_summary.json"
    write_rows_csv(rows, rows_csv)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"rows_csv: {rows_csv}")
    print(f"summary_json: {summary_json}")
    print(f"scientific_claim_status: {summary['scientific_claim_status']}")
    print(f"blocking_conditions: {', '.join(summary['blocking_conditions'])}")


if __name__ == "__main__":
    main()
