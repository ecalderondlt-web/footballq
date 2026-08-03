from __future__ import annotations

import argparse

import yaml

from footballq.models.identity_matchup_transformer import IDENTITY_CONDITIONS
from footballq.training.train_matchup import train_matchup_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train matched RLCS identity ablations.")
    parser.add_argument("--config", default="configs/rlcs_identity_matchup_v1.yaml")
    parser.add_argument("--condition", choices=IDENTITY_CONDITIONS)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    conditions = [args.condition] if args.condition else list(config["training"]["conditions"])
    seeds = [args.seed] if args.seed is not None else list(config["training"]["seeds"])
    for seed in seeds:
        for condition in conditions:
            result = train_matchup_from_config(
                args.config,
                condition=condition,
                seed=int(seed),
            )
            print(
                f"condition={condition} seed={seed} run_dir={result['run_dir']} "
                f"val_nll={result['best_validation_factorized_joint_nll']:.6f}"
            )


if __name__ == "__main__":
    main()
