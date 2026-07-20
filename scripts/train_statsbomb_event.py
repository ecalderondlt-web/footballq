"""Train a causal StatsBomb event encoder from YAML configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.training.train_statsbomb_event import (  # noqa: E402
    train_statsbomb_event_from_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = args.config
    if args.seed is not None:
        from footballq.training.train_statsbomb_event import load_statsbomb_event_config

        config = load_statsbomb_event_config(args.config)
        config["seed"] = int(args.seed)
        config.setdefault("training", {})["seed"] = int(args.seed)
    result = train_statsbomb_event_from_config(config)
    print(f"run_dir: {result['run_dir']}")
    print(f"latest: {result['latest_checkpoint']}")
    print(f"best: {result['best_checkpoint']}")
    print(f"best_metric: {result['best_metric']:.6f}")
    print(f"step: {result['step']}")


if __name__ == "__main__":
    main()
