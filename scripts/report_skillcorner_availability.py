"""Report local SkillCorner raw, window, decoder, and embedding availability."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.io.skillcorner_report import build_skillcorner_availability_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw/skillcorner"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/processed/skillcorner_td_embeddings_all.pt"),
    )
    parser.add_argument("--horizon-seconds", nargs="*", type=float, default=[2.0, 4.0, 6.0])
    parser.add_argument("--windows-prefix", default="skillcorner_windows")
    parser.add_argument("--decoder-prefix", default="skillcorner_decoder_dataset")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_skillcorner_availability_report(
        args.raw,
        args.processed_dir,
        embeddings=args.embeddings,
        horizons=args.horizon_seconds,
        windows_prefix=args.windows_prefix,
        decoder_prefix=args.decoder_prefix,
    )
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(f"raw_match_count: {report['raw_match_count']}")
    print(f"raw_match_ids: {', '.join(report['raw_match_ids'])}")
    print(f"raw_periods: {','.join(str(value) for value in report['raw_periods']) or 'none'}")
    print(f"embedding_match_count: {report['embedding_match_count']}")
    print(f"embedding_match_ids: {', '.join(report['embedding_match_ids'])}")
    for match in report["raw_matches"]:
        print(
            "raw_match: "
            f"{match['match_id']} tracking={match['has_tracking']} "
            f"metadata={match['has_metadata']} events={match['has_events']} "
            f"raw_periods={','.join(str(value) for value in match['raw_periods']) or 'none'}"
        )
    for horizon in report["horizons"]:
        alignment = horizon["embedding_alignment"]
        print(
            "horizon: "
            f"{horizon['horizon_label']} windows={horizon['windows_exists']} "
            f"window_count={horizon['window_count']} "
            f"periods={','.join(str(value) for value in horizon['window_periods']) or 'none'} "
            "missing_processed_periods="
            f"{','.join(str(value) for value in horizon['missing_processed_periods']) or 'none'} "
            f"decoder={horizon['decoder_dataset_exists']} "
            f"decoder_examples={horizon['decoder_example_count']} "
            f"matching_embedding_keys={alignment['matching_window_keys']}/"
            f"{alignment['window_keys']}"
        )
        if alignment["missing_window_matches_in_embeddings"]:
            print(
                "missing_embedding_matches: "
                + ", ".join(alignment["missing_window_matches_in_embeddings"])
            )
        if horizon["window_count_by_match"]:
            print("windows_per_match:")
            for match_id, count in horizon["window_count_by_match"].items():
                print(f"- {match_id}: {count}")
        if horizon["window_count_by_period"]:
            print("windows_per_period:")
            for period, count in horizon["window_count_by_period"].items():
                print(f"- {period}: {count}")
        if horizon["missing_processed_periods_by_match"]:
            print("missing_processed_periods_by_match:")
            for match_id, periods in horizon["missing_processed_periods_by_match"].items():
                print(f"- {match_id}: {','.join(str(value) for value in periods)}")
        if horizon["decoder_example_count_by_match"]:
            print("decoder_examples_per_match:")
            for match_id, count in horizon["decoder_example_count_by_match"].items():
                print(f"- {match_id}: {count}")


if __name__ == "__main__":
    main()
