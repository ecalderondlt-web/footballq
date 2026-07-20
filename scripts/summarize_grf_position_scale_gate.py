"""Aggregate the frozen GRF position-scale PFF validation gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

REQUIRED_TRANSFER_FAMILIES = ("1x", "1x_replay", "4x", "8x")
METRICS = ("total_loss", "td_loss", "z_online_std_mean")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_last_jsonl(path: Path) -> dict[str, Any]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Metric file is empty: {path}")
    return json.loads(lines[-1])


def _read_curve(metrics_path: Path) -> list[dict[str, Any]]:
    path = metrics_path.with_name("metrics_val_curve.jsonl")
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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


def _parse_family_paths(values: list[str]) -> dict[str, dict[int, Path]]:
    parsed: dict[str, dict[int, Path]] = {}
    for value in values:
        parts = value.split(":", 2)
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"Expected FAMILY:SEED:PATH, got {value!r}")
        family, seed_text, path_text = parts
        seed = int(seed_text)
        if seed in parsed.setdefault(family, {}):
            raise ValueError(f"Duplicate {family} seed: {seed}")
        parsed[family][seed] = Path(path_text)
    return parsed


def _relative_improvement(baseline: float, candidate: float) -> float:
    if baseline == 0.0:
        raise ValueError("Cannot calculate relative improvement from a zero baseline.")
    return (baseline - candidate) / baseline


def summarize_position_scale_gate(
    scratch_paths: dict[int, Path],
    family_paths: dict[str, dict[int, Path]],
    *,
    min_total_wins: int = 2,
    min_scratch_improvement: float = 0.05,
    min_replay_improvement: float = 0.02,
    min_z_online_std_mean: float = 0.05,
) -> dict[str, Any]:
    missing = sorted(set(REQUIRED_TRANSFER_FAMILIES) - set(family_paths))
    extra = sorted(set(family_paths) - set(REQUIRED_TRANSFER_FAMILIES))
    if missing or extra:
        raise ValueError(f"Transfer families mismatch; missing={missing}, extra={extra}.")
    seeds = sorted(scratch_paths)
    if not seeds:
        raise ValueError("At least one paired seed is required.")
    if any(set(paths) != set(seeds) for paths in family_paths.values()):
        raise ValueError("Every family must use exactly the scratch seeds.")

    all_paths = {"scratch": scratch_paths, **family_paths}
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        row: dict[str, Any] = {"seed": seed, "families": {}}
        for family, paths in all_paths.items():
            path = paths[seed]
            metrics = _read_last_jsonl(path)
            row["families"][family] = {
                "metrics_path": str(path),
                "metrics_sha256": _sha256(path),
                **{metric: float(metrics[metric]) for metric in METRICS},
                "validation_curve": _read_curve(path),
            }
        rows.append(row)

    means = {
        family: {
            metric: sum(row["families"][family][metric] for row in rows) / len(rows)
            for metric in METRICS
        }
        for family in all_paths
    }
    finite = all(
        math.isfinite(row["families"][family][metric])
        for row in rows
        for family in all_paths
        for metric in METRICS
    )
    total_wins = sum(
        row["families"]["8x"]["total_loss"]
        < row["families"]["scratch"]["total_loss"]
        for row in rows
    )
    scratch_improvement = _relative_improvement(
        means["scratch"]["total_loss"], means["8x"]["total_loss"]
    )
    replay_improvement = _relative_improvement(
        means["1x_replay"]["total_loss"], means["8x"]["total_loss"]
    )
    minimum_spread = min(
        row["families"][family]["z_online_std_mean"]
        for row in rows
        for family in all_paths
    )
    criteria = {
        "finite_metrics": {"passed": finite},
        "eight_x_total_wins_vs_scratch": {
            "value": total_wins,
            "minimum": min_total_wins,
            "passed": total_wins >= min_total_wins,
        },
        "eight_x_mean_total_improvement_vs_scratch": {
            "value": scratch_improvement,
            "minimum": min_scratch_improvement,
            "passed": scratch_improvement >= min_scratch_improvement,
        },
        "eight_x_mean_td_no_worse_than_scratch": {
            "eight_x": means["8x"]["td_loss"],
            "scratch": means["scratch"]["td_loss"],
            "passed": means["8x"]["td_loss"] <= means["scratch"]["td_loss"],
        },
        "minimum_z_online_std_mean": {
            "value": minimum_spread,
            "minimum_exclusive": min_z_online_std_mean,
            "passed": minimum_spread > min_z_online_std_mean,
        },
        "eight_x_mean_total_improvement_vs_replay": {
            "value": replay_improvement,
            "minimum": min_replay_improvement,
            "passed": replay_improvement >= min_replay_improvement,
        },
        "eight_x_mean_td_no_worse_than_replay": {
            "eight_x": means["8x"]["td_loss"],
            "replay": means["1x_replay"]["td_loss"],
            "passed": means["8x"]["td_loss"] <= means["1x_replay"]["td_loss"],
        },
    }
    blockers = [name for name, criterion in criteria.items() if not criterion["passed"]]
    return {
        "status": "controls_passed" if not blockers else "blocked",
        "seeds": seeds,
        "rows": rows,
        "means": means,
        "criteria": criteria,
        "blocking_conditions": blockers,
        "descriptive_total_loss_order": sorted(
            all_paths, key=lambda family: means[family]["total_loss"]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", nargs="+", required=True, metavar="SEED:PATH")
    parser.add_argument(
        "--family",
        nargs="+",
        required=True,
        metavar="FAMILY:SEED:PATH",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_position_scale_gate(
        _parse_seed_paths(args.scratch),
        _parse_family_paths(args.family),
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
