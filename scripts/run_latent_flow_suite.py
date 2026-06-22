"""Run an Experiment 4A latent rollout comparison suite."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.latent_flow.eval import (  # noqa: E402
    evaluate_latent_baseline,
    evaluate_latent_checkpoint,
)
from footballq.latent_flow.io import save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=None)
    return parser.parse_args()


def _row(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": metrics.get("model"),
        "split": metrics.get("split"),
        "latent_ADE": metrics.get("latent_ADE"),
        "latent_FDE": metrics.get("latent_FDE"),
        "latent_step_mse": metrics.get("latent_step_mse"),
        "latent_cosine_similarity": metrics.get("latent_cosine_similarity"),
        "minADE_8": metrics.get("minADE_8"),
        "minFDE_8": metrics.get("minFDE_8"),
        "diversity_mean_pairwise_distance": metrics.get("diversity_mean_pairwise_distance"),
        "checkpoint_or_config": metrics.get("checkpoint_or_config"),
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "split",
        "latent_ADE",
        "latent_FDE",
        "latent_step_mse",
        "latent_cosine_similarity",
        "minADE_8",
        "minFDE_8",
        "diversity_mean_pairwise_distance",
        "checkpoint_or_config",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for baseline in ["last_latent", "constant_latent_velocity"]:
        result = evaluate_latent_baseline(
            args.dataset,
            baseline=baseline,
            split=args.split,
            device=args.device,
        )
        rows.append(_row(result["metrics"]))
    for checkpoint in args.checkpoint:
        result = evaluate_latent_checkpoint(
            checkpoint,
            dataset=args.dataset,
            split=args.split,
            device=args.device,
            num_samples=args.num_samples,
            num_steps=args.num_steps,
        )
        rows.append(_row(result["metrics"]))
    args.out.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, args.out / "results.csv")
    save_json({"results": rows}, args.out / "results.json")
    print(f"results_csv: {args.out / 'results.csv'}")
    for row in rows:
        print(
            f"{row['model']} | ADE={row['latent_ADE']} | "
            f"FDE={row['latent_FDE']} | cosine={row['latent_cosine_similarity']}"
        )


if __name__ == "__main__":
    main()
