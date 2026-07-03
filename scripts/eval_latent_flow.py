"""Evaluate latent flow checkpoints or deterministic latent baselines."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.latent_flow.eval import (  # noqa: E402
    evaluate_latent_baseline,
    evaluate_latent_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--baseline",
        choices=["last_latent", "constant_latent_velocity"],
        default=None,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--noise-scale", type=float, default=None)
    return parser.parse_args()


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.6f}"
    return str(value)


def main() -> None:
    args = parse_args()
    if args.baseline:
        if args.dataset is None:
            raise SystemExit("--dataset is required when evaluating a baseline.")
        result = evaluate_latent_baseline(
            args.dataset,
            baseline=args.baseline,
            split=args.split,
            device=args.device,
        )
    elif args.checkpoint:
        result = evaluate_latent_checkpoint(
            args.checkpoint,
            dataset=args.dataset,
            split=args.split,
            device=args.device,
            num_samples=args.num_samples,
            num_steps=args.num_steps,
            noise_scale=args.noise_scale,
        )
    else:
        raise SystemExit("Provide either --checkpoint or --baseline.")
    metrics = result["metrics"]
    for key in [
        "model",
        "split",
        "latent_ADE",
        "latent_FDE",
        "latent_RMSE",
        "latent_step_mse",
        "latent_cosine_similarity",
        "one_step_error",
        "multi_step_rollout_error",
        "delta_ADE",
        "residual_ADE",
        "minADE_4",
        "minFDE_4",
        "minADE_8",
        "minFDE_8",
        "diversity_mean_pairwise_distance",
        "noise_scale",
        "num_sampling_steps",
        "num_examples",
    ]:
        if key in metrics:
            print(f"{key}: {_fmt(metrics[key])}")


if __name__ == "__main__":
    main()
