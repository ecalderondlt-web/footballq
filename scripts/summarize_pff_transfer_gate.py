"""Aggregate a prespecified paired PFF transfer validation gate."""

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


def summarize_transfer_gate(
    scratch_paths: dict[int, Path],
    transfer_paths: dict[int, Path],
    *,
    min_total_wins: int = 2,
    min_mean_total_improvement: float = 0.05,
    max_mean_td_relative_change: float = 0.0,
    min_z_online_std_mean: float = 0.05,
) -> dict[str, Any]:
    if set(scratch_paths) != set(transfer_paths):
        raise ValueError("Scratch and transfer seeds must match exactly.")
    seeds = sorted(scratch_paths)
    if not seeds:
        raise ValueError("At least one paired seed is required.")

    rows = []
    for seed in seeds:
        scratch = _read_metrics(scratch_paths[seed])
        transfer = _read_metrics(transfer_paths[seed])
        rows.append(
            {
                "seed": seed,
                "scratch": {
                    "metrics_path": str(scratch_paths[seed]),
                    "metrics_sha256": _sha256(scratch_paths[seed]),
                    "total_loss": float(scratch["total_loss"]),
                    "td_loss": float(scratch["td_loss"]),
                    "z_online_std_mean": float(scratch["z_online_std_mean"]),
                },
                "transfer": {
                    "metrics_path": str(transfer_paths[seed]),
                    "metrics_sha256": _sha256(transfer_paths[seed]),
                    "total_loss": float(transfer["total_loss"]),
                    "td_loss": float(transfer["td_loss"]),
                    "z_online_std_mean": float(transfer["z_online_std_mean"]),
                },
            }
        )

    finite = all(
        math.isfinite(row[family][metric])
        for row in rows
        for family in ("scratch", "transfer")
        for metric in ("total_loss", "td_loss", "z_online_std_mean")
    )
    mean_scratch_total = sum(row["scratch"]["total_loss"] for row in rows) / len(rows)
    mean_transfer_total = sum(row["transfer"]["total_loss"] for row in rows) / len(rows)
    mean_total_improvement = (mean_scratch_total - mean_transfer_total) / mean_scratch_total
    total_wins = sum(
        row["transfer"]["total_loss"] < row["scratch"]["total_loss"] for row in rows
    )
    mean_scratch_td = sum(row["scratch"]["td_loss"] for row in rows) / len(rows)
    mean_transfer_td = sum(row["transfer"]["td_loss"] for row in rows) / len(rows)
    mean_td_relative_change = (mean_transfer_td - mean_scratch_td) / mean_scratch_td
    minimum_spread = min(
        row[family]["z_online_std_mean"] for row in rows for family in ("scratch", "transfer")
    )

    criteria = {
        "finite_metrics": {"passed": finite},
        "transfer_total_wins": {
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
        "seeds": seeds,
        "rows": rows,
        "means": {
            "scratch_total_loss": mean_scratch_total,
            "transfer_total_loss": mean_transfer_total,
            "scratch_td_loss": mean_scratch_td,
            "transfer_td_loss": mean_transfer_td,
        },
        "criteria": criteria,
        "blocking_conditions": blockers,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", nargs="+", required=True, metavar="SEED:PATH")
    parser.add_argument("--transfer", nargs="+", required=True, metavar="SEED:PATH")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=None)
    parser.add_argument("--min-total-wins", type=int, default=2)
    parser.add_argument("--min-mean-total-improvement", type=float, default=0.05)
    parser.add_argument("--max-mean-td-relative-change", type=float, default=0.0)
    parser.add_argument("--min-z-online-std-mean", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_transfer_gate(
        _parse_seed_paths(args.scratch),
        _parse_seed_paths(args.transfer),
        min_total_wins=args.min_total_wins,
        min_mean_total_improvement=args.min_mean_total_improvement,
        max_mean_td_relative_change=args.max_mean_td_relative_change,
        min_z_online_std_mean=args.min_z_online_std_mean,
    )
    if args.protocol is not None:
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
