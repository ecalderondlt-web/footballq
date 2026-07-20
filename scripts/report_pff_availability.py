"""Report local PFF FC tracking inventory and sampled format health."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.io.pff import audit_pff_match, discover_pff_tracking_files  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--match-ids", nargs="*", default=None)
    parser.add_argument("--max-records", type=int, default=10_000)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    discovered = discover_pff_tracking_files(args.raw)
    requested = args.match_ids or list(discovered)
    missing = [match_id for match_id in requested if match_id not in discovered]
    if missing:
        raise ValueError("PFF match IDs not found: " + ", ".join(missing))
    audits = [
        audit_pff_match(discovered[match_id], max_records=args.max_records)
        for match_id in requested
    ]
    report = {
        "raw_root": str(args.raw),
        "unique_match_count": len(discovered),
        "match_ids": list(discovered),
        "metadata_present": any(
            path.name.lower().startswith("metadata")
            for path in args.raw.rglob("*")
            if path.is_file()
        ),
        "audits": audits,
    }
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(f"unique_match_count: {report['unique_match_count']}")
    print(f"match_ids: {', '.join(report['match_ids'])}")
    print(f"metadata_present: {report['metadata_present']}")
    for audit in audits:
        fps = audit["inferred_fps"]
        fps_text = f"{fps:.5f}" if fps is not None else "unknown"
        print(
            "match: "
            f"{audit['match_id']} records={audit['records_sampled']} "
            f"unique_frames={audit['unique_frames_sampled']} "
            f"duplicate_records={audit['duplicate_records']} "
            f"duplicated_player_arrays={audit['duplicated_player_array_records']} "
            f"fps={fps_text}"
        )


if __name__ == "__main__":
    main()
