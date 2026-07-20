"""Pin StatsBomb Open Data coverage, file hashes, and a match-level split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.data.statsbomb_events import (  # noqa: E402
    build_statsbomb_source_manifest,
    build_statsbomb_split,
    load_statsbomb_match_catalog,
    write_immutable_json,
)
from footballq.repro.splits import split_manifest_sha256  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument(
        "--split-out",
        type=Path,
        default=ROOT / "splits" / "statsbomb_open_data_b0bc9f2_match_inductive_v1.json",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=ROOT / "runs" / "integrity" / "statsbomb_open_data_b0bc9f2_source_manifest.json",
    )
    parser.add_argument("--skip-file-hashes", action="store_true")
    args = parser.parse_args()

    catalog = load_statsbomb_match_catalog(args.raw_root)
    split = build_statsbomb_split(catalog)
    write_immutable_json(args.split_out, split)
    manifest = build_statsbomb_source_manifest(
        args.raw_root,
        catalog,
        split,
        archive_path=args.archive,
        hash_files=not args.skip_file_hashes,
    )
    write_immutable_json(args.manifest_out, manifest)

    print(f"matches: {manifest['coverage']['matches']}")
    print(f"matches_with_360: {manifest['coverage']['matches_with_360']}")
    print(
        "split_counts: "
        f"{len(split['train_match_ids'])}/"
        f"{len(split['val_match_ids'])}/"
        f"{len(split['test_match_ids'])}"
    )
    print(f"split_sha256: {split_manifest_sha256(split)}")
    print(f"source_manifest_sha256: {manifest['manifest_payload_sha256']}")


if __name__ == "__main__":
    main()
