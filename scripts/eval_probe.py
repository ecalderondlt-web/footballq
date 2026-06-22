"""Evaluate one Experiment 3 frozen-probe checkpoint."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.probes.training import evaluate_probe_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _format(value: object) -> str:
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.6f}"
    return str(value)


def main() -> None:
    args = parse_args()
    result = evaluate_probe_checkpoint(
        args.checkpoint,
        split=args.split,
        data_path=args.data,
        device=args.device,
    )
    metrics = result["metrics"]
    print(f"run_dir: {result['run_dir']}")
    for key in [
        "target_name",
        "feature_source",
        "probe_type",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "mae",
        "rmse",
        "r2",
        "num_examples",
    ]:
        if key in metrics:
            print(f"{key}: {_format(metrics[key])}")


if __name__ == "__main__":
    main()
