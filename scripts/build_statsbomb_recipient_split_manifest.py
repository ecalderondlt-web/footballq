"""Build an immutable split manifest from configured StatsBomb cohorts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from footballq.analysis.statsbomb_player_history_signal import (
    load_cohorts,
    load_match_records,
)
from footballq.analysis.statsbomb_recipient_history import (
    load_config,
    recipient_cohort_for_match,
)

SPLIT_KEYS = {
    "train": "train_match_ids",
    "validation": "val_match_ids",
    "development_test": "test_match_ids",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    cohorts = load_cohorts(config)
    records = load_match_records(config["data"]["statsbomb_root"])
    split_ids = {key: [] for key in SPLIT_KEYS.values()}
    cohort_counts: dict[str, int] = {}
    for record in records:
        cohort = recipient_cohort_for_match(record, cohorts)
        if cohort is None:
            continue
        split_ids[SPLIT_KEYS[cohort.split]].append(record.match_id)
        cohort_counts[cohort.name] = cohort_counts.get(cohort.name, 0) + 1

    all_ids = [
        *split_ids["train_match_ids"],
        *split_ids["val_match_ids"],
        *split_ids["test_match_ids"],
    ]
    payload = {
        "name": Path(args.out).stem,
        "version": 1,
        "dataset": "statsbomb",
        "protocol": str(config["experiment_protocol"]),
        **split_ids,
        "all_match_ids": all_ids,
        "expected_count": len(all_ids),
        "cohort_match_counts": dict(sorted(cohort_counts.items())),
        "source_commit": str(config["data"]["source_commit"]),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), **payload["cohort_match_counts"]}, indent=2))


if __name__ == "__main__":
    main()
