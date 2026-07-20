"""CLI for frozen tracking plus PFF event-context residual studies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.training.train_event_context_residual import (  # noqa: E402
    train_event_context_residual_from_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--family", choices=("tracking", "raw", "random", "pretrained"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--tracking-checkpoint")
    parser.add_argument("--event-checkpoint")
    args = parser.parse_args()
    result = train_event_context_residual_from_config(
        args.config,
        family=args.family,
        seed=args.seed,
        tracking_checkpoint=args.tracking_checkpoint,
        event_checkpoint=args.event_checkpoint,
    )
    print(f"run_dir: {result['run_dir']}")
    print(f"latest: {result['latest_checkpoint']}")
    print(f"td_loss: {result['final']['td_loss']:.8f}")


if __name__ == "__main__":
    main()
