"""Prepare train/validation PFF event histories for frozen context studies."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.data.pff_event_context import (  # noqa: E402
    prepare_pff_event_context_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="wc2022datav2")
    parser.add_argument("--canonical", default="data/processed/pff_wc2022_canonical_v2")
    parser.add_argument(
        "--split-manifest",
        default="splits/pff_wc2022_64match_inductive_v1.json",
    )
    parser.add_argument(
        "--statsbomb-manifest",
        default="data/processed/statsbomb_event_sequence_v1/manifest.json",
    )
    parser.add_argument(
        "--out",
        default="data/processed/pff_statsbomb_event_context_v1",
    )
    parser.add_argument("--splits", nargs="+", choices=("train", "val"), default=["train"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--snapshot-manifest")
    args = parser.parse_args()
    manifest = prepare_pff_event_context_dataset(
        args.raw,
        args.canonical,
        args.split_manifest,
        args.statsbomb_manifest,
        args.out,
        include_splits=tuple(args.splits),
        workers=args.workers,
    )
    manifest_path = Path(args.out) / "manifest.json"
    if args.snapshot_manifest:
        snapshot = Path(args.snapshot_manifest)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, snapshot)
    print(f"manifest: {manifest_path}")
    print(f"loaded_splits: {manifest['loaded_splits']}")
    print(f"events: {manifest['quality_totals']['event_count']}")
    print(f"payload_sha256: {manifest['manifest_payload_sha256']}")


if __name__ == "__main__":
    main()
