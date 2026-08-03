from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from footballq.models.player_matchup_value import VALUE_CONDITIONS
from footballq.training.train_rlcs_value import train_value_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train matched RLCS V2 critical-value models.")
    parser.add_argument("--config", default="configs/rlcs_player_matchup_value_v2.yaml")
    parser.add_argument("--condition", choices=VALUE_CONDITIONS)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    conditions = [args.condition] if args.condition else list(config["training"]["conditions"])
    seeds = [args.seed] if args.seed is not None else list(config["training"]["seeds"])
    for seed in seeds:
        for condition in conditions:
            result = train_value_from_config(
                args.config, condition=condition, seed=int(seed)
            )
            print(
                f"condition={condition} seed={seed} run_dir={result['run_dir']} "
                f"internal_log_loss={result['best_internal_development_log_loss']:.6f}"
            )


if __name__ == "__main__":
    main()
