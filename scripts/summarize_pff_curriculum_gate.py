"""Aggregate the frozen easy-only versus GRF V2 PFF validation gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _parse_seed_paths(values: list[str]) -> dict[int, Path]:
    parsed: dict[int, Path] = {}
    for value in values:
        seed_text, separator, path_text = value.partition(":")
        if not separator or not seed_text or not path_text:
            raise ValueError(f"Expected SEED:PATH, got {value!r}")
        seed = int(seed_text)
        if seed in parsed:
            raise ValueError(f"Duplicate seed: {seed}")
        parsed[seed] = Path(path_text)
    return parsed


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_metrics(path: Path) -> dict[str, Any]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Metric file is empty: {path}")
    return json.loads(lines[-1])


def summarize_curriculum_gate(
    baseline_paths: dict[int, Path],
    candidate_paths: dict[int, Path],
    *,
    baseline_label: str = "easy_only",
    candidate_label: str = "balanced_v2",
    min_total_wins: int = 2,
    min_mean_total_improvement: float = 0.02,
    max_mean_td_relative_change: float = 0.0,
    min_z_online_std_mean: float = 0.05,
) -> dict[str, Any]:
    if set(baseline_paths) != set(candidate_paths):
        raise ValueError("Baseline and candidate seeds must match exactly.")
    seeds = sorted(baseline_paths)
    if not seeds:
        raise ValueError("At least one paired seed is required.")
    if not baseline_label or not candidate_label or baseline_label == candidate_label:
        raise ValueError("Baseline and candidate labels must be non-empty and distinct.")

    rows = []
    for seed in seeds:
        baseline = _read_metrics(baseline_paths[seed])
        candidate = _read_metrics(candidate_paths[seed])
        rows.append(
            {
                "seed": seed,
                baseline_label: {
                    "metrics_path": str(baseline_paths[seed]),
                    "metrics_sha256": _sha256(baseline_paths[seed]),
                    "total_loss": float(baseline["total_loss"]),
                    "td_loss": float(baseline["td_loss"]),
                    "z_online_std_mean": float(baseline["z_online_std_mean"]),
                },
                candidate_label: {
                    "metrics_path": str(candidate_paths[seed]),
                    "metrics_sha256": _sha256(candidate_paths[seed]),
                    "total_loss": float(candidate["total_loss"]),
                    "td_loss": float(candidate["td_loss"]),
                    "z_online_std_mean": float(candidate["z_online_std_mean"]),
                },
            }
        )

    finite = all(
        math.isfinite(row[family][metric])
        for row in rows
        for family in (baseline_label, candidate_label)
        for metric in ("total_loss", "td_loss", "z_online_std_mean")
    )
    mean_baseline_total = sum(row[baseline_label]["total_loss"] for row in rows) / len(rows)
    mean_candidate_total = sum(row[candidate_label]["total_loss"] for row in rows) / len(rows)
    mean_total_improvement = (
        mean_baseline_total - mean_candidate_total
    ) / mean_baseline_total
    total_wins = sum(
        row[candidate_label]["total_loss"] < row[baseline_label]["total_loss"] for row in rows
    )
    mean_baseline_td = sum(row[baseline_label]["td_loss"] for row in rows) / len(rows)
    mean_candidate_td = sum(row[candidate_label]["td_loss"] for row in rows) / len(rows)
    mean_td_relative_change = (mean_candidate_td - mean_baseline_td) / mean_baseline_td
    minimum_spread = min(
        row[family]["z_online_std_mean"]
        for row in rows
        for family in (baseline_label, candidate_label)
    )

    criteria = {
        "finite_metrics": {"passed": finite},
        f"{candidate_label}_total_wins": {
            "value": total_wins,
            "minimum": min_total_wins,
            "passed": total_wins >= min_total_wins,
        },
        "mean_total_relative_improvement": {
            "value": mean_total_improvement,
            "minimum": min_mean_total_improvement,
            "passed": mean_total_improvement >= min_mean_total_improvement,
        },
        "mean_td_relative_change": {
            "value": mean_td_relative_change,
            "maximum": max_mean_td_relative_change,
            "passed": mean_td_relative_change <= max_mean_td_relative_change,
        },
        "minimum_z_online_std_mean": {
            "value": minimum_spread,
            "minimum": min_z_online_std_mean,
            "passed": minimum_spread > min_z_online_std_mean,
        },
    }
    blockers = [name for name, criterion in criteria.items() if not criterion["passed"]]
    return {
        "status": "controls_passed" if not blockers else "blocked",
        "comparison": {"baseline": baseline_label, "candidate": candidate_label},
        "seeds": seeds,
        "rows": rows,
        "means": {
            f"{baseline_label}_total_loss": mean_baseline_total,
            f"{candidate_label}_total_loss": mean_candidate_total,
            f"{baseline_label}_td_loss": mean_baseline_td,
            f"{candidate_label}_td_loss": mean_candidate_td,
        },
        "criteria": criteria,
        "blocking_conditions": blockers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", nargs="+", required=True, metavar="SEED:PATH")
    parser.add_argument("--candidate", nargs="+", required=True, metavar="SEED:PATH")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--baseline-label", default="easy_only")
    parser.add_argument("--candidate-label", default="balanced_v2")
    parser.add_argument("--min-total-wins", type=int, default=2)
    parser.add_argument("--min-mean-total-improvement", type=float, default=0.02)
    parser.add_argument("--max-mean-td-relative-change", type=float, default=0.0)
    parser.add_argument("--min-z-online-std-mean", type=float, default=0.05)
    args = parser.parse_args()

    summary = summarize_curriculum_gate(
        _parse_seed_paths(args.baseline),
        _parse_seed_paths(args.candidate),
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        min_total_wins=args.min_total_wins,
        min_mean_total_improvement=args.min_mean_total_improvement,
        max_mean_td_relative_change=args.max_mean_td_relative_change,
        min_z_online_std_mean=args.min_z_online_std_mean,
    )
    summary["protocol_path"] = str(args.protocol)
    summary["protocol_sha256"] = _sha256(args.protocol)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"status: {summary['status']}")
    print(f"summary: {args.out}")
    for blocker in summary["blocking_conditions"]:
        print(f"blocker: {blocker}")


if __name__ == "__main__":
    main()
