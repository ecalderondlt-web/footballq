"""Run the Experiment 4C coordinate-decoder comparison suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.decoding.suite import run_decoder_suite  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-batches", type=int, default=20)
    parser.add_argument("--max-eval-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_decoder_suite(
        args.dataset,
        args.out,
        split=args.split,
        device=args.device,
        epochs=args.epochs,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        batch_size=args.batch_size,
        run_root=args.run_root,
    )
    print(f"results_csv: {result['results_csv']}")
    print(f"summary_json: {result['summary_json']}")
    best = result["summary"].get("best_by_all_entity_ADE_m")
    if best:
        print(
            "best_all_entity_ADE_m: "
            f"{best['model']}={best['all_entity_ADE_m']}"
        )


if __name__ == "__main__":
    main()
