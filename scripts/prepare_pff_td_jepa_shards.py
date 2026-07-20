"""Build boundary-aware geometry-only TD-JEPA shards from canonical PFF data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.data.pff_td_shards import prepare_pff_td_jepa_shards  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=Path("data/processed/pff_wc2022_canonical_v1"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/processed/pff_wc2022_td_jepa_v1")
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("splits/pff_wc2022_64match_inductive_v1.json"),
    )
    parser.add_argument("--match-ids", nargs="*", default=None)
    parser.add_argument("--splits", nargs="*", choices=["train", "val", "test"], default=None)
    parser.add_argument("--fps-out", type=float, default=10.0)
    parser.add_argument("--context-seconds", type=float, default=1.0)
    parser.add_argument("--delta-seconds", type=float, default=0.2)
    parser.add_argument("--stride-seconds", type=float, default=0.2)
    parser.add_argument("--prediction-gap-seconds", type=float, default=1.0)
    parser.add_argument("--feature-view", default="geometry_only")
    parser.add_argument(
        "--visibility-mode",
        choices=["all_available", "observed_only"],
        default="all_available",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare_pff_td_jepa_shards(
        args.canonical_root,
        args.out,
        args.split_manifest,
        match_ids=args.match_ids,
        split_names=args.splits,
        fps_out=args.fps_out,
        context_seconds=args.context_seconds,
        delta_seconds=args.delta_seconds,
        stride_seconds=args.stride_seconds,
        prediction_gap_seconds=args.prediction_gap_seconds,
        feature_view=args.feature_view,
        visibility_mode=args.visibility_mode,
        resume=not args.no_resume,
        workers=args.workers,
    )
    print(f"prepared_matches: {manifest['selected_match_count']}")
    print(f"examples: {manifest['example_count']}")
    print(f"unique_sample_ids: {manifest['unique_sample_id_count']}")
    print(f"shards: {len(manifest['shards'])}")
    print(f"dataset_manifest: {args.out / args.visibility_mode / 'dataset_manifest.json'}")


if __name__ == "__main__":
    main()
