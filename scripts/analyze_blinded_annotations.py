"""Analyze completed blinded annotation CSVs against their private keys."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_ALLOWED_LABELS = (
    "tactical_pattern",
    "routine_motion",
    "tracking_artifact",
    "ambiguous",
)
DEFAULT_POSITIVE_LABELS = ("tactical_pattern",)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalize_label(value: object) -> str:
    text = str(value or "").strip().lower()
    return " ".join(text.split())


def _is_filled(value: object) -> bool:
    return bool(_normalize_label(value))


def _key_by_blind_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row.get("blind_id", "")): row for row in rows}


def _control_group(value: object) -> str:
    text = _normalize_label(value)
    if text in {"true", "1", "yes", "positive"}:
        return "positive"
    if text in {"false", "0", "no", "negative", "control"}:
        return "control"
    return "unlabeled"


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _fisher_greater_pvalue(
    *,
    positive_successes: int,
    positive_total: int,
    control_successes: int,
    control_total: int,
) -> float | None:
    """One-sided Fisher exact p-value for positive group enrichment."""

    total = positive_total + control_total
    successes = positive_successes + control_successes
    if total == 0 or positive_total == 0 or control_total == 0:
        return None
    denominator = math.comb(total, positive_total)
    max_successes = min(successes, positive_total)
    min_successes = max(0, positive_total - (total - successes))
    probability = 0.0
    for observed in range(max(positive_successes, min_successes), max_successes + 1):
        probability += (
            math.comb(successes, observed)
            * math.comb(total - successes, positive_total - observed)
            / denominator
        )
    return probability


def analyze_blinded_annotations(
    *,
    annotator_csv: str | Path,
    key_csv: str | Path,
    manifest_json: str | Path | None = None,
    positive_labels: list[str] | None = None,
    allowed_labels: list[str] | None = None,
) -> dict[str, Any]:
    """Return diagnostic annotation-completion and enrichment summaries."""

    from scripts.validate_blinded_annotation_package import validate_blinded_annotation_package

    validation = validate_blinded_annotation_package(
        annotator_csv=annotator_csv,
        key_csv=key_csv,
        manifest_json=manifest_json,
        require_clip_paths=True,
        require_blank_annotations=False,
    )
    if validation["validation_status"] != "passed":
        return {
            "claim_status": "diagnostic_only",
            "annotation_status": "invalid_package",
            "validation": validation,
            "issues": validation["issues"],
        }

    annotator_rows = _read_csv(Path(annotator_csv))
    key_rows = _read_csv(Path(key_csv))
    key_lookup = _key_by_blind_id(key_rows)
    positive_label_set = {
        _normalize_label(value)
        for value in (positive_labels or DEFAULT_POSITIVE_LABELS)
        if _normalize_label(value)
    }
    allowed_label_set = {
        _normalize_label(value)
        for value in (allowed_labels or DEFAULT_ALLOWED_LABELS)
        if _normalize_label(value)
    }
    completed_rows: list[dict[str, str]] = []
    joined_rows: list[dict[str, str]] = []
    for row in annotator_rows:
        annotation = _normalize_label(row.get("annotation"))
        key_row = key_lookup.get(str(row.get("blind_id", "")), {})
        joined = {**key_row, **row, "annotation_normalized": annotation}
        joined_rows.append(joined)
        if _is_filled(annotation):
            completed_rows.append(joined)

    invalid_labels = sorted(
        {
            row["annotation_normalized"]
            for row in completed_rows
            if row["annotation_normalized"] not in allowed_label_set
        }
    )
    if invalid_labels:
        return {
            "claim_status": "diagnostic_only",
            "annotation_status": "invalid_labels",
            "annotator_csv": str(annotator_csv),
            "key_csv": str(key_csv),
            "manifest_json": str(manifest_json) if manifest_json is not None else "",
            "row_count": len(joined_rows),
            "completed_count": len(completed_rows),
            "completion_rate": _safe_rate(len(completed_rows), len(joined_rows)),
            "allowed_labels": sorted(allowed_label_set),
            "invalid_labels": invalid_labels,
            "validation": validation,
            "issues": [
                "annotation column contains labels outside the controlled vocabulary: "
                + ", ".join(invalid_labels)
            ],
        }

    label_counts = Counter(row["annotation_normalized"] for row in completed_rows)
    groups: dict[str, dict[str, Any]] = {}
    for group in ["positive", "control", "unlabeled"]:
        group_rows = [
            row for row in completed_rows if _control_group(row.get("positive_control")) == group
        ]
        positive_hits = sum(
            row["annotation_normalized"] in positive_label_set for row in group_rows
        )
        groups[group] = {
            "completed_count": len(group_rows),
            "positive_label_count": positive_hits,
            "positive_label_rate": _safe_rate(positive_hits, len(group_rows)),
            "label_counts": dict(
                sorted(Counter(row["annotation_normalized"] for row in group_rows).items())
            ),
        }

    positive_group = groups["positive"]
    control_group = groups["control"]
    pos_count = int(positive_group["positive_label_count"])
    ctrl_count = int(control_group["positive_label_count"])
    pos_total = int(positive_group["completed_count"])
    ctrl_total = int(control_group["completed_count"])
    pos_rate = _safe_rate(pos_count, pos_total)
    ctrl_rate = _safe_rate(ctrl_count, ctrl_total)
    risk_difference = None if pos_rate is None or ctrl_rate is None else pos_rate - ctrl_rate
    risk_ratio = None if pos_rate is None or not ctrl_rate else pos_rate / ctrl_rate
    enrichment = {
        "checked": bool(positive_label_set) and pos_total > 0 and ctrl_total > 0,
        "positive_labels": sorted(positive_label_set),
        "positive_group_positive_label_rate": pos_rate,
        "control_group_positive_label_rate": ctrl_rate,
        "risk_difference": risk_difference,
        "risk_ratio": risk_ratio,
        "fisher_greater_pvalue": (
            _fisher_greater_pvalue(
                positive_successes=pos_count,
                positive_total=pos_total,
                control_successes=ctrl_count,
                control_total=ctrl_total,
            )
            if positive_label_set
            else None
        ),
    }

    completed_count = len(completed_rows)
    status = "incomplete" if completed_count == 0 else "analyzed"
    return {
        "claim_status": "diagnostic_only",
        "annotation_status": status,
        "annotator_csv": str(annotator_csv),
        "key_csv": str(key_csv),
        "manifest_json": str(manifest_json) if manifest_json is not None else "",
        "row_count": len(joined_rows),
        "completed_count": completed_count,
        "completion_rate": _safe_rate(completed_count, len(joined_rows)),
        "allowed_labels": sorted(allowed_label_set),
        "label_counts": dict(sorted(label_counts.items())),
        "groups": groups,
        "enrichment": enrichment,
        "validation": validation,
        "note": (
            "Annotation summaries are diagnostic until probe/discovery gates pass and "
            "annotation enrichment is reviewed against matched controls."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator-csv", type=Path, required=True)
    parser.add_argument("--key-csv", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--positive-labels",
        nargs="*",
        default=list(DEFAULT_POSITIVE_LABELS),
        help="Annotation labels counted as positive/enriched.",
    )
    parser.add_argument(
        "--allowed-labels",
        nargs="*",
        default=list(DEFAULT_ALLOWED_LABELS),
        help="Controlled annotation labels accepted in filled rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze_blinded_annotations(
        annotator_csv=args.annotator_csv,
        key_csv=args.key_csv,
        manifest_json=args.manifest_json,
        positive_labels=args.positive_labels,
        allowed_labels=args.allowed_labels,
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"annotation_status: {summary['annotation_status']}")
    print(f"claim_status: {summary['claim_status']}")
    print(f"completed_count: {summary.get('completed_count', 0)}")
    print(f"completion_rate: {summary.get('completion_rate')}")
    if args.out is not None:
        print(f"summary_json: {args.out}")
    if summary["annotation_status"] in {"invalid_package", "invalid_labels"}:
        for issue in summary.get("issues", []):
            print(f"issue: {issue}")
        sys.exit(1)


if __name__ == "__main__":
    main()
