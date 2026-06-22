"""Export TD-JEPA embeddings for downstream probe experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.training.export_td_embeddings import export_td_embeddings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = export_td_embeddings(
        checkpoint=args.checkpoint,
        data_path=args.data,
        out=args.out,
        split=args.split,
        device=args.device,
    )
    print(f"embeddings: {out}")


if __name__ == "__main__":
    main()
