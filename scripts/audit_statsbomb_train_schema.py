"""Audit StatsBomb event and 360 schemas using the training partition only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.data.statsbomb_events import (  # noqa: E402
    audit_statsbomb_training_schema,
    write_immutable_json,
)
from footballq.repro.splits import load_split_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--split",
        type=Path,
        default=ROOT / "splits" / "statsbomb_open_data_b0bc9f2_match_inductive_v1.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "runs" / "integrity" / "statsbomb_open_data_b0bc9f2_train_schema.json",
    )
    args = parser.parse_args()

    split = load_split_manifest(args.split)
    audit = audit_statsbomb_training_schema(args.raw_root, split.payload)
    write_immutable_json(args.out, audit)
    summary = {key: audit[key] for key in ("scope", "train_match_count", "maxima")}
    print(json.dumps(summary, indent=2))
    print(f"events: {audit['counts']['events']}")
    print(f"malformed_three_sixty: {len(audit['malformed_three_sixty'])}")
    print(f"vocabulary_payload_sha256: {audit['vocabulary_payload_sha256']}")
    print(f"audit_payload_sha256: {audit['audit_payload_sha256']}")


if __name__ == "__main__":
    main()
