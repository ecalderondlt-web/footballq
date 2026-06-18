"""Evaluate a saved Phase 1 baseline checkpoint."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.training.eval import evaluate_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_checkpoint(args.checkpoint, split=args.split, device=args.device)
    metrics = result["metrics"]
    print(f"run_dir: {result['run_dir']}")
    print(f"{'Metric':<32} Value")
    for key in [
        "player_ADE_m",
        "player_FDE_m",
        "ball_ADE_m",
        "ball_FDE_m",
        "all_entity_ADE_m",
        "all_entity_FDE_m",
        "team_centroid_error_m",
        "team_width_error_m",
        "team_length_error_m",
        "team_stretch_index_error_m",
    ]:
        value = metrics.get(key)
        text = (
            "nan"
            if value is None or (isinstance(value, float) and math.isnan(value))
            else f"{value:.4f}"
        )
        print(f"{key:<32} {text}")


if __name__ == "__main__":
    main()
