"""Aggregate multiple independent blinded annotators into panel statistics.

Takes several filled copies of the same blinded annotations.csv (one per
annotator) and reports label distributions, pairwise Cohen's kappa, Fleiss'
kappa, and per-item majority labels. It never reads private key files; the
enrichment of majority labels against hidden controls is computed separately by
scripts/analyze_blinded_annotations.py on the majority-vote CSV this script can
emit.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

ALLOWED_LABELS = ["tactical_pattern", "routine_motion", "tracking_artifact", "ambiguous"]


def read_annotations(path: Path) -> dict[str, str]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    labels: dict[str, str] = {}
    for row in rows:
        blind_id = row.get("blind_id", "").strip()
        annotation = row.get("annotation", "").strip()
        if blind_id:
            labels[blind_id] = annotation
    return labels


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    counts_a = Counter(a)
    counts_b = Counter(b)
    expected = sum(
        (counts_a[label] / n) * (counts_b[label] / n) for label in set(counts_a) | set(counts_b)
    )
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def fleiss_kappa(items: list[list[str]]) -> float:
    if not items:
        return float("nan")
    n_raters = len(items[0])
    categories = sorted({label for item in items for label in item})
    if n_raters < 2 or not categories:
        return float("nan")
    p_j = []
    for category in categories:
        total = sum(item.count(category) for item in items)
        p_j.append(total / (len(items) * n_raters))
    p_i = []
    for item in items:
        counts = Counter(item)
        agree = sum(c * (c - 1) for c in counts.values())
        p_i.append(agree / (n_raters * (n_raters - 1)))
    p_bar = sum(p_i) / len(p_i)
    p_e = sum(p * p for p in p_j)
    if p_e >= 1.0:
        return 1.0
    return (p_bar - p_e) / (1.0 - p_e)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        action="append",
        required=True,
        metavar="NAME:PATH",
        help="annotator name and filled annotations.csv path",
    )
    parser.add_argument("--template-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    annotators: dict[str, dict[str, str]] = {}
    for spec in args.annotations:
        name, _, path = spec.partition(":")
        if not path:
            raise SystemExit(f"expected NAME:PATH, got {spec!r}")
        annotators[name] = read_annotations(Path(path))

    with args.template_csv.open() as handle:
        template_rows = list(csv.DictReader(handle))
        fieldnames = list(template_rows[0].keys()) if template_rows else []
    blind_ids = [row["blind_id"] for row in template_rows]

    names = sorted(annotators)
    label_matrix: dict[str, dict[str, str]] = {}
    invalid: dict[str, list[str]] = {name: [] for name in names}
    for blind_id in blind_ids:
        row_labels = {}
        for name in names:
            label = annotators[name].get(blind_id, "")
            if label not in ALLOWED_LABELS:
                invalid[name].append(blind_id)
            row_labels[name] = label
        label_matrix[blind_id] = row_labels

    complete_ids = [
        blind_id
        for blind_id in blind_ids
        if all(label_matrix[blind_id][name] in ALLOWED_LABELS for name in names)
    ]
    pairwise = {}
    for left, right in combinations(names, 2):
        a = [label_matrix[i][left] for i in complete_ids]
        b = [label_matrix[i][right] for i in complete_ids]
        pairwise[f"{left}|{right}"] = {
            "cohen_kappa": cohen_kappa(a, b),
            "raw_agreement": (
                sum(1 for x, y in zip(a, b) if x == y) / len(a) if a else float("nan")
            ),
        }
    fleiss = fleiss_kappa([[label_matrix[i][name] for name in names] for i in complete_ids])

    majority_rows = []
    for row in template_rows:
        blind_id = row["blind_id"]
        labels = [label_matrix[blind_id][name] for name in names]
        counts = Counter(label for label in labels if label in ALLOWED_LABELS)
        majority_label = ""
        agreement_count = 0
        if counts:
            top = counts.most_common()
            if len(top) == 1 or top[0][1] > top[1][1]:
                majority_label = top[0][0]
                agreement_count = top[0][1]
        majority_rows.append(
            {**row, "annotation": majority_label, "panel_agreement_count": agreement_count}
        )

    args.out.mkdir(parents=True, exist_ok=True)
    majority_csv = args.out / "panel_majority_annotations.csv"
    majority_fields = [*fieldnames, "panel_agreement_count"]
    with majority_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=majority_fields)
        writer.writeheader()
        writer.writerows(majority_rows)

    summary = {
        "annotators": names,
        "num_items": len(blind_ids),
        "num_complete_items": len(complete_ids),
        "invalid_or_blank_by_annotator": {k: len(v) for k, v in invalid.items()},
        "label_distribution_by_annotator": {
            name: dict(
                Counter(
                    label_matrix[i][name]
                    for i in blind_ids
                    if label_matrix[i][name] in ALLOWED_LABELS
                )
            )
            for name in names
        },
        "pairwise": pairwise,
        "fleiss_kappa": fleiss,
        "majority_blank_items": sum(1 for row in majority_rows if not row["annotation"]),
        "claim_status": "model_annotator_diagnostic_only",
    }
    summary_path = args.out / "panel_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ["annotators", "num_complete_items", "fleiss_kappa"]}))
    print(f"panel summary: {summary_path}")
    print(f"majority csv: {majority_csv}")


if __name__ == "__main__":
    main()
