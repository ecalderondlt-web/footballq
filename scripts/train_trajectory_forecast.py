"""Train one matched downstream trajectory forecast family."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.training.train_trajectory_forecast import (  # noqa: E402
    train_trajectory_forecast_from_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--family", choices=["raw", "frozen", "finetuned"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tracking-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    result = train_trajectory_forecast_from_config(
        args.config,
        family=args.family,
        seed=args.seed,
        tracking_checkpoint=args.tracking_checkpoint,
    )
    print(f"run_dir: {result['run_dir']}")
    print(f"player_ADE_m: {result['metrics']['player_ADE_m']:.6f}")
    print(f"ball_ADE_m: {result['metrics']['ball_ADE_m']:.6f}")


if __name__ == "__main__":
    main()
