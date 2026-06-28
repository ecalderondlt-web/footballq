"""Prepare shifted temporal examples for TD-JEPA pretraining."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.data.td_jepa_dataset import build_td_jepa_examples, save_td_jepa_data  # noqa: E402
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
    parser.add_argument("--context-seconds", type=float, default=1.0)
    parser.add_argument("--delta-seconds", type=float, default=0.2)
    parser.add_argument("--stride-seconds", type=float, default=0.2)
    parser.add_argument(
        "--objective-mode",
        choices=["legacy_shifted_overlap", "future_nonoverlap_context_only"],
        default="legacy_shifted_overlap",
    )
    parser.add_argument("--prediction-gap-seconds", type=float, default=0.0)
    parser.add_argument(
        "--feature-view",
        choices=[
            "full_state_legacy",
            "geometry_only",
            "missingness_only_control",
            "raw_kinematics_control",
        ],
        default="full_state_legacy",
    )
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--scientific-mode", action="store_true")
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
    data = build_td_jepa_examples(
        tracking,
        fps_out=args.fps_out,
        context_seconds=args.context_seconds,
        delta_seconds=args.delta_seconds,
        stride_seconds=args.stride_seconds,
        objective_mode=args.objective_mode,
        prediction_gap_seconds=args.prediction_gap_seconds,
        feature_view=args.feature_view,
        split_manifest_path=args.split_manifest,
        scientific_mode=args.scientific_mode,
    )
    if len(data.match_id) == 0:
        raise RuntimeError(
            "No TD-JEPA examples were produced. Check that the input has enough consecutive "
            "visible tracking frames for the requested context and delta settings."
        )
    out = save_td_jepa_data(data, args.out)
    print(
        f"wrote {len(data.match_id):,} TD-JEPA examples to {out} "
        f"with state_t={tuple(data.state_t.shape)} delta_state={tuple(data.delta_state.shape)}"
    )


if __name__ == "__main__":
    main()
