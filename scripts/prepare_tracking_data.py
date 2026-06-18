"""Normalize tracking data and export Phase 1 Torch windows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.data.windows import build_tracking_windows, save_windows_pt  # noqa: E402
from footballq.io.idsse import IDSSEAdapter  # noqa: E402
from footballq.io.metrica import MetricaAdapter  # noqa: E402
from footballq.io.skillcorner import SkillCornerAdapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["synthetic", "skillcorner", "idsse", "metrica"],
        required=True,
    )
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--match-id", default="match_1")
    parser.add_argument("--fps-out", type=float, default=10.0)
    parser.add_argument("--context-seconds", type=float, default=2.0)
    parser.add_argument("--horizon-seconds", type=float, default=2.0)
    parser.add_argument("--stride-seconds", type=float, default=0.2)
    return parser.parse_args()


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table extension: {path.suffix}")


def load_source(source: str, raw: Path, match_id: str) -> pd.DataFrame:
    if source == "synthetic":
        return _read_table(raw)
    if source == "skillcorner":
        return SkillCornerAdapter(raw_dir=raw, match_id=match_id).load_tracking()
    if source == "idsse":
        return IDSSEAdapter(raw_dir=raw, match_id=match_id).load_tracking()
    if source == "metrica":
        return MetricaAdapter(raw_dir=raw, match_id=match_id).load_tracking()
    raise ValueError(f"Unknown source: {source}")


def main() -> None:
    args = parse_args()
    tracking = load_source(args.source, args.raw, args.match_id)
    windows = build_tracking_windows(
        tracking,
        fps_out=args.fps_out,
        context_seconds=args.context_seconds,
        horizon_seconds=args.horizon_seconds,
        stride_seconds=args.stride_seconds,
    )
    if len(windows.match_id) == 0:
        raise RuntimeError(
            "No windows were produced. Check that the input has enough consecutive frames for "
            f"context={args.context_seconds}s and horizon={args.horizon_seconds}s at "
            f"fps_out={args.fps_out}, and that player/ball rows have visible x/y positions."
        )
    out = save_windows_pt(windows, args.out)
    print(
        f"wrote {len(windows.match_id):,} windows to {out} "
        f"with past={tuple(windows.past.shape)} future={tuple(windows.future_xy.shape)}"
    )


if __name__ == "__main__":
    main()
