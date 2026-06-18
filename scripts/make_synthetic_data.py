"""Create synthetic 23-entity tracking data for Phase 1 tests and demos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.synthetic.generate import generate_synthetic_tracking  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-matches", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=1000)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    duration_s = max(args.num_frames - 1, 1) / args.fps
    frames = []
    for match_idx in range(args.num_matches):
        match_id = f"synthetic_match_{match_idx + 1}"
        df = generate_synthetic_tracking(
            match_id=match_id,
            duration_s=duration_s,
            fps=args.fps,
            seed=args.seed + match_idx,
        )
        df = df[df["frame_id"] < args.num_frames].copy()
        df["fps"] = args.fps
        frames.append(df)
    tracking = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix.lower() == ".csv":
        tracking.to_csv(args.out, index=False)
    else:
        tracking.to_parquet(args.out, index=False)
    print(f"wrote {len(tracking):,} rows to {args.out}")


if __name__ == "__main__":
    main()
