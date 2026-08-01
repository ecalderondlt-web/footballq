"""Freeze chronological match manifests for cross-team fingerprint retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from footballq.repro.splits import validate_split_manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _catalog(root: Path, competitions: list[str]) -> pd.DataFrame:
    frames = [
        pd.read_parquet(
            root / f"passes_{competition}.parquet",
            columns=["match_id", "dateutc", "competition"],
        )
        for competition in competitions
    ]
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("match_id")
        .sort_values(["dateutc", "match_id"])
    )


def _payload(
    *,
    name: str,
    protocol: str,
    status: str,
    support: pd.DataFrame,
    query: pd.DataFrame,
    dataset_manifest: Path,
    support_competitions: list[str],
    query_competitions: list[str],
) -> dict[str, Any]:
    support_ids = [str(value) for value in support["match_id"].tolist()]
    query_ids = [str(value) for value in query["match_id"].tolist()]
    if not set(support_ids).isdisjoint(query_ids):
        raise ValueError("Fingerprint support and query matches overlap.")
    support_end = str(support["dateutc"].max())
    query_start = str(query["dateutc"].min())
    if support_end >= query_start:
        raise ValueError("Fingerprint support is not strictly earlier than query.")
    payload = {
        "name": name,
        "version": 1,
        "dataset": "wyscout",
        "protocol": protocol,
        "train_match_ids": [],
        "val_match_ids": [],
        "test_match_ids": query_ids,
        "all_match_ids": query_ids,
        "expected_count": len(query_ids),
        "status": status,
        "support_match_ids": support_ids,
        "support_expected_count": len(support_ids),
        "support_competitions": support_competitions,
        "query_competitions": query_competitions,
        "latest_support_dateutc": support_end,
        "earliest_query_dateutc": query_start,
        "dataset_manifest_path": str(dataset_manifest),
        "dataset_manifest_sha256": _sha256(dataset_manifest),
    }
    validate_split_manifest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/wyscout_player_fingerprint_v1.yaml",
    )
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    root = Path(config["data"]["dataset_root"])
    dataset_manifest = Path(config["data"]["dataset_manifest"])
    protocol = str(config["experiment_protocol"])
    for key, manifest_key in (
        ("development", "development_manifest"),
        ("confirmatory", "confirmatory_manifest"),
    ):
        cohort = config[key]
        support_competitions = [
            str(value) for value in cohort["support_competitions"]
        ]
        query_competitions = [
            str(value) for value in cohort["query_competitions"]
        ]
        payload = _payload(
            name=Path(config["data"][manifest_key]).stem,
            protocol=f"{protocol}_{cohort['name']}",
            status=str(cohort["status"]),
            support=_catalog(root, support_competitions),
            query=_catalog(root, query_competitions),
            dataset_manifest=dataset_manifest,
            support_competitions=support_competitions,
            query_competitions=query_competitions,
        )
        path = Path(config["data"][manifest_key])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "cohort": key,
                    "manifest": str(path),
                    "support_matches": payload["support_expected_count"],
                    "query_matches": payload["expected_count"],
                    "latest_support": payload["latest_support_dateutc"],
                    "earliest_query": payload["earliest_query_dateutc"],
                }
            )
        )


if __name__ == "__main__":
    main()
