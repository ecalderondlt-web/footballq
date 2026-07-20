"""Summarize tournament-wide quality from canonical PFF shards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.io.pff_quality import summarize_pff_canonical_quality  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=Path("data/processed/pff_wc2022_canonical_v1"),
    )
    parser.add_argument("--skip-frame-shapes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = summarize_pff_canonical_quality(
        args.canonical_root,
        scan_frame_shapes=not args.skip_frame_shapes,
    )
    print(f"matches: {report['match_count']}")
    print(f"frames: {report['total_frames']}")
    print(f"missing_ball_frame_rate: {report['missing_ball_frame_rate']:.6f}")
    print(f"estimated_coordinate_rate: {report['estimated_coordinate_rate']:.6f}")
    print(f"other_frame_shape_rate: {report['other_frame_shape_rate']:.6f}")
    print(f"frame_gaps: {report['frame_gap_count']}")
    print(f"time_regressions: {report['time_regression_count']}")
    print(f"quality_report: {args.canonical_root / 'quality_report.json'}")


if __name__ == "__main__":
    main()
