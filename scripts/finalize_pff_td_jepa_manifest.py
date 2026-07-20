"""Finalize PFF TD-JEPA lineage by hashing every tensor shard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.data.pff_td_shards import finalize_pff_td_jepa_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = finalize_pff_td_jepa_manifest(args.manifest)
    print(f"tensor_shards_hashed: {len(manifest['shards'])}")
    print(f"examples: {manifest['example_count']}")
    print(f"manifest_sha256: {manifest['manifest_payload_sha256']}")


if __name__ == "__main__":
    main()
