"""Prepare resumable canonical PFF tracking shards and quality manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.io.pff_shards import prepare_pff_dataset_shards  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("wc2022datav2"))
    parser.add_argument(
        "--out", type=Path, default=Path("data/processed/pff_wc2022_canonical_v1")
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("splits/pff_wc2022_64match_inductive_v1.json"),
    )
    parser.add_argument("--match-ids", nargs="*", default=None)
    parser.add_argument("--splits", nargs="*", choices=["train", "val", "test"], default=None)
    parser.add_argument("--frames-per-shard", type=int, default=6_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--raw-coordinates", action="store_true")
    parser.add_argument("--no-source-hash", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare_pff_dataset_shards(
        args.raw,
        args.out,
        args.split_manifest,
        match_ids=args.match_ids,
        split_names=args.splits,
        frames_per_shard=args.frames_per_shard,
        max_frames=args.max_frames,
        use_smoothed=not args.raw_coordinates,
        hash_source=not args.no_source_hash,
        resume=not args.no_resume,
        force=args.force,
        workers=args.workers,
    )
    totals = manifest["totals"]
    print(f"prepared_matches: {manifest['selected_match_count']}")
    print(f"unique_frames: {totals['unique_frames']}")
    print(f"duplicate_records_removed: {totals['duplicate_records']}")
    print(f"missing_ball_frames: {totals['missing_ball_frames']}")
    print(f"non_23_entity_frames: {totals['non_23_entity_frames']}")
    print(f"out_of_bounds_rows: {totals['out_of_bounds_rows']}")
    print(f"dataset_manifest: {args.out / 'dataset_manifest.json'}")


if __name__ == "__main__":
    main()
