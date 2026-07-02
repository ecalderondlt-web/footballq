"""Compare TD-JEPA falsification gate summaries across representation variants."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_gate_spec(spec: str) -> tuple[str, Path]:
    """Parse label:path gate summary specs."""

    parts = spec.split(":", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("Gate specs must use label:path.")
    return parts[0], Path(parts[1])


def rows_from_gate(label: str, path: Path) -> list[dict[str, Any]]:
    """Return one row per falsification condition from a gate summary."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for condition, summary in sorted(payload.get("conditions", {}).items()):
        ratio = summary.get("td_loss_ratio_vs_correct", {})
        margin = summary.get("td_loss_margin_vs_correct", {})
        rows.append(
            {
                "gate": label,
                "condition": condition,
                "status": summary.get("status"),
                "ratio_mean": ratio.get("mean"),
                "ratio_min": ratio.get("min"),
                "ratio_max": ratio.get("max"),
                "margin_mean": margin.get("mean"),
                "summary_path": str(path),
            }
        )
    return rows


def compare_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare each gate with the first gate in row order."""

    gate_order = []
    by_gate_condition = {}
    for row in rows:
        gate = str(row["gate"])
        if gate not in gate_order:
            gate_order.append(gate)
        by_gate_condition[(gate, str(row["condition"]))] = row
    if len(gate_order) < 2:
        return {"reference_gate": gate_order[0] if gate_order else None, "comparisons": {}}

    reference_gate = gate_order[0]
    comparisons = {}
    conditions = sorted({str(row["condition"]) for row in rows})
    for condition in conditions:
        reference = by_gate_condition.get((reference_gate, condition))
        if reference is None:
            continue
        comparison_rows = []
        for gate in gate_order[1:]:
            candidate = by_gate_condition.get((gate, condition))
            if candidate is None:
                continue
            comparison_rows.append(
                {
                    "gate": gate,
                    "status": candidate.get("status"),
                    "ratio_mean_delta": _delta(
                        candidate.get("ratio_mean"),
                        reference.get("ratio_mean"),
                    ),
                    "ratio_min_delta": _delta(
                        candidate.get("ratio_min"),
                        reference.get("ratio_min"),
                    ),
                    "ratio_max_delta": _delta(
                        candidate.get("ratio_max"),
                        reference.get("ratio_max"),
                    ),
                }
            )
        comparisons[condition] = {
            "reference_status": reference.get("status"),
            "reference_ratio_mean": reference.get("ratio_mean"),
            "reference_ratio_min": reference.get("ratio_min"),
            "candidates": comparison_rows,
        }
    return {"reference_gate": reference_gate, "comparisons": comparisons}


def _delta(candidate: Any, reference: Any) -> float | None:
    if candidate is None or reference is None:
        return None
    return float(candidate) - float(reference)


def write_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write comparison rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "gate",
        "condition",
        "status",
        "ratio_mean",
        "ratio_min",
        "ratio_max",
        "margin_mean",
        "summary_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        row
        for label, path in [parse_gate_spec(spec) for spec in args.gate]
        for row in rows_from_gate(label, path)
    ]
    comparison = compare_rows(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    rows_csv = args.out / "td_falsification_gate_rows.csv"
    comparison_json = args.out / "td_falsification_gate_comparison.json"
    write_rows_csv(rows, rows_csv)
    comparison_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"rows_csv: {rows_csv}")
    print(f"comparison_json: {comparison_json}")


if __name__ == "__main__":
    main()
