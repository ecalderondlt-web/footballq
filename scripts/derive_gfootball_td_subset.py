"""Derive an exact episode-prefix tensor subset from a master GRF manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.data.gfootball_td_subset import (  # noqa: E402
    derive_gfootball_td_episode_subset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-manifest", type=Path, required=True)
    parser.add_argument("--master-plan", type=Path, required=True)
    parser.add_argument("--subset-plan", type=Path, required=True)
    parser.add_argument("--subset-collection-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = derive_gfootball_td_episode_subset(
        args.master_manifest,
        args.master_plan,
        args.subset_plan,
        args.subset_collection_manifest,
        args.out,
        args.split_manifest,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"dataset_manifest: {manifest_path}")
    print(f"examples: {manifest['example_count']}")
    print(f"retention: {manifest['example_retention_fraction']:.6f}")
    print(f"manifest_sha256: {manifest['manifest_payload_sha256']}")


if __name__ == "__main__":
    main()
