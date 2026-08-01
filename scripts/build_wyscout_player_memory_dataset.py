"""Build compact, provenance-verified pass tables from public Wyscout events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from footballq.data.wyscout_public import (
    ALL_COMPETITIONS,
    build_competition_pass_frame,
    file_sha256,
    verify_source_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="data/raw/wyscout_public")
    parser.add_argument(
        "--provenance",
        default="provenance/wyscout_public_source_v1.json",
    )
    parser.add_argument(
        "--output-root",
        default="data/processed/wyscout_player_memory_v1",
    )
    parser.add_argument(
        "--competitions",
        nargs="+",
        default=list(ALL_COMPETITIONS),
        choices=list(ALL_COMPETITIONS),
    )
    parser.add_argument("--horizon-seconds", type=float, default=20.0)
    parser.add_argument("--horizon-events", type=int, default=10)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    source = verify_source_manifest(raw_root, args.provenance)
    outputs: list[dict[str, object]] = []
    for competition in args.competitions:
        frame = build_competition_pass_frame(
            raw_root / "extracted",
            raw_root / "players.json",
            competition,
            horizon_seconds=args.horizon_seconds,
            horizon_events=args.horizon_events,
        )
        output_path = output_root / f"passes_{competition}.parquet"
        frame.to_parquet(output_path, index=False)
        outputs.append(
            {
                "competition": competition,
                "path": str(output_path),
                "rows": int(len(frame)),
                "matches": int(frame["match_id"].nunique()),
                "players": int(frame["player_id"].nunique()),
                "sha256": file_sha256(output_path),
            }
        )
        print(
            json.dumps(
                {
                    "competition": competition,
                    "rows": len(frame),
                    "matches": frame["match_id"].nunique(),
                }
            )
        )

    manifest = {
        "name": "wyscout_player_memory_dataset_v1",
        "version": 1,
        "source": source,
        "definition": {
            "unit": "valid pass event",
            "sample_identity": "match_id:period:event_id",
            "coordinates": "attacking-team perspective, percentage [0,100]",
            "start_grid": [5, 4],
            "destination_grid": [6, 5],
            "shot_horizon_seconds": args.horizon_seconds,
            "shot_horizon_subsequent_events": args.horizon_events,
            "shot_label": "same team records a Shot within both horizons",
            "key_pass_tags": [301, 302],
            "accurate_tag": 1801,
        },
        "outputs": outputs,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
