"""Evaluate an Experiment 4C coordinate decoder checkpoint."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.decoding.eval import evaluate_decoder_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.6f}"
    return str(value)


def main() -> None:
    args = parse_args()
    result = evaluate_decoder_checkpoint(
        args.checkpoint,
        dataset=args.dataset,
        split=args.split,
        device=args.device,
    )
    metrics = result["metrics"]
    for key in [
        "mode",
        "decoder_type",
        "split",
        "current_all_entity_error_m",
        "current_player_error_m",
        "current_ball_error_m",
        "player_ADE_m",
        "player_FDE_m",
        "ball_ADE_m",
        "ball_FDE_m",
        "all_entity_ADE_m",
        "all_entity_FDE_m",
        "team_centroid_error_m",
        "team_width_error_m",
        "team_length_error_m",
        "stretch_index_error_m",
        "loss",
        "num_examples",
    ]:
        if key in metrics:
            print(f"{key}: {_fmt(metrics[key])}")


if __name__ == "__main__":
    main()
