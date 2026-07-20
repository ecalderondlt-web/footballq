"""Prepare sharded StatsBomb causal event windows without loading the test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.data.statsbomb_event_dataset import (  # noqa: E402
    prepare_statsbomb_event_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--split",
        type=Path,
        default=ROOT / "splits" / "statsbomb_open_data_b0bc9f2_match_inductive_v1.json",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT / "runs" / "integrity" / "statsbomb_open_data_b0bc9f2_source_manifest.json",
    )
    parser.add_argument(
        "--schema-audit",
        type=Path,
        default=ROOT / "runs" / "integrity" / "statsbomb_open_data_b0bc9f2_train_schema.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    args = parser.parse_args()

    limits = {
        split: limit
        for split, limit in (("train", args.limit_train), ("val", args.limit_val))
        if limit is not None
    }
    manifest = prepare_statsbomb_event_dataset(
        args.raw_root,
        args.split,
        args.source_manifest,
        args.schema_audit,
        args.out,
        sequence_length=args.sequence_length,
        stride=args.stride,
        match_limits=limits,
    )
    print(f"loaded_splits: {manifest['loaded_splits']}")
    print(f"test_loaded: {manifest['test_loaded']}")
    for split_name, counts in manifest["split_counts"].items():
        print(
            f"{split_name}: matches={counts['matches']} events={counts['events']} "
            f"windows={counts['windows']}"
        )
    print(f"manifest_payload_sha256: {manifest['manifest_payload_sha256']}")


if __name__ == "__main__":
    main()
