"""Analyze latent residual scores."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.discovery.surprise import write_surprise_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--delta-seconds", type=float, default=None)
    parser.add_argument("--assignments", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = write_surprise_outputs(
        args.dataset,
        args.out,
        delta_seconds=args.delta_seconds,
        assignments=args.assignments,
        top_n=args.top_n,
    )
    summary = result["summary"]
    print(f"latent_residual_examples: {result['surprise_examples']}")
    print(f"latent_residual_summary: {result['surprise_summary']}")
    print(f"score_name: {summary['score_name']}")
    print(f"num_examples: {summary['num_examples']}")
    print(f"high_latent_residual_threshold: {summary['high_latent_residual_threshold']:.6f}")


if __name__ == "__main__":
    main()
