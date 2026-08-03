from __future__ import annotations

import argparse
import json

from footballq.training.train_matchup import overfit_matchup_subset_from_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the train-only 5,000-sample RLCS memorization preflight."
    )
    parser.add_argument("--config", default="configs/rlcs_identity_matchup_v1.yaml")
    parser.add_argument("--condition", default="full")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--sample-count", type=int, default=5_000)
    parser.add_argument("--maximum-steps", type=int, default=3_000)
    parser.add_argument("--evaluation-interval", type=int, default=100)
    parser.add_argument("--target-joint-nll", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--output", default="runs/rlcs_identity_matchup_v1/preflight/overfit_5000.json"
    )
    args = parser.parse_args()
    report = overfit_matchup_subset_from_config(
        args.config,
        condition=args.condition,
        seed=args.seed,
        sample_count=args.sample_count,
        maximum_steps=args.maximum_steps,
        evaluation_interval=args.evaluation_interval,
        target_joint_nll=args.target_joint_nll,
        learning_rate=args.learning_rate,
        output_report=args.output,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
