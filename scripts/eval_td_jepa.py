"""Evaluate a TD-JEPA checkpoint."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.training.eval_td_jepa import evaluate_td_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_td_checkpoint(
        args.checkpoint,
        split=args.split,
        data_path=args.data,
        device=args.device,
    )
    metrics = result["metrics"]
    print(f"run_dir: {result['run_dir']}")
    print(f"{'Metric':<32} Value")
    for key in [
        "total_loss",
        "td_loss",
        "anti_collapse_loss",
        "cosine_similarity",
        "z_online_std_mean",
        "z_online_std_min",
        "z_target_std_mean",
        "z_target_std_min",
        "num_examples",
    ]:
        value = metrics.get(key)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            text = "nan"
        elif isinstance(value, int):
            text = str(value)
        else:
            text = f"{float(value):.6f}"
        print(f"{key:<32} {text}")


if __name__ == "__main__":
    main()
