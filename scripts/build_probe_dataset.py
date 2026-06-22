"""Build Experiment 3 frozen-probe datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.probes.dataset import build_probe_dataset, save_probe_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = build_probe_dataset(
        embeddings_path=args.embeddings,
        windows_path=args.windows,
        target_names=args.targets,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    save_probe_dataset(data, args.out)
    print(f"probe_dataset: {args.out}")
    print(f"examples: {data.metadata['num_examples']}")
    print(f"targets: {', '.join(data.metadata['targets'])}")
    skipped = data.metadata.get("skipped_targets", [])
    if skipped:
        print(f"skipped_targets: {', '.join(skipped)}")
    for warning in data.metadata.get("warnings", []):
        print(f"warning: {warning}")


if __name__ == "__main__":
    main()
