"""Validate all processed StatsBomb train/validation event tensors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.data.statsbomb_event_dataset import (  # noqa: E402
    audit_statsbomb_event_dataset,
)
from footballq.data.statsbomb_events import write_immutable_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "runs" / "integrity" / "statsbomb_event_sequence_v1_tensor_audit.json",
    )
    parser.add_argument("--skip-hashes", action="store_true")
    args = parser.parse_args()

    audit = audit_statsbomb_event_dataset(
        args.manifest,
        verify_hashes=not args.skip_hashes,
    )
    write_immutable_json(args.out, audit)
    print(f"status: {audit['status']}")
    for split_name, counts in audit["split_counts"].items():
        print(
            f"{split_name}: matches={counts['matches']} events={counts['events']} "
            f"windows={counts['windows']}"
        )
    print(f"audit_payload_sha256: {audit['audit_payload_sha256']}")


if __name__ == "__main__":
    main()
