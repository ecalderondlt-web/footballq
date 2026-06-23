"""Cluster latent transition vectors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.discovery.clustering import cluster_transition_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--delta-seconds", type=float, default=None)
    parser.add_argument("--k", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument(
        "--feature",
        choices=["raw_delta_z", "normalized_delta_z", "z_t_delta_z"],
        default="normalized_delta_z",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--fit-sample-size", type=int, default=50000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = cluster_transition_file(
        args.dataset,
        args.out,
        k_values=args.k,
        delta_seconds=args.delta_seconds,
        feature=args.feature,
        seed=args.seed,
        max_iter=args.max_iter,
        fit_sample_size=args.fit_sample_size,
    )
    print(f"cluster_summary: {args.out / 'cluster_summary.json'}")
    print(f"num_examples: {summary['num_examples']}")
    for cluster in summary["clusters"]:
        q = cluster["quality"]
        print(
            f"k={q['k']} avg_distance={q['average_within_cluster_distance']:.6f} "
            f"entropy={q['cluster_size_entropy']:.3f} empty={q['empty_cluster_count']}"
        )


if __name__ == "__main__":
    main()
