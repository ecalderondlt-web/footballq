"""Train a configured Phase 1 baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.training.train import train_from_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_from_config(args.config)
    print(f"run_dir: {result['run_dir']}")
    print(f"latest: {result['latest_checkpoint']}")
    print(f"best: {result['best_checkpoint']}")
    player_ade = result["test_metrics"].get("player_ADE_m")
    if player_ade is not None:
        print(f"test player_ADE_m: {player_ade:.4f}")


if __name__ == "__main__":
    main()
