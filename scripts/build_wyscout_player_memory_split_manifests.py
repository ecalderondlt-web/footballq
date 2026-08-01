"""Freeze development and World Cup confirmatory match manifests."""

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


def _match_catalog(dataset_root: Path, competition: str) -> pd.DataFrame:
    path = dataset_root / f"passes_{competition}.parquet"
    frame = pd.read_parquet(
        path,
        columns=["match_id", "competition", "dateutc", "gameweek"],
    )
    return (
        frame.drop_duplicates("match_id")
        .sort_values(["dateutc", "match_id"])
        .reset_index(drop=True)
    )


def _manifest_payload(
    *,
    name: str,
    protocol: str,
    train: list[str],
    validation: list[str],
    test: list[str],
    extra: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "name": name,
        "version": 1,
        "dataset": "wyscout",
        "protocol": protocol,
        "train_match_ids": train,
        "val_match_ids": validation,
        "test_match_ids": test,
        "all_match_ids": [*train, *validation, *test],
        "expected_count": len(train) + len(validation) + len(test),
        **extra,
    }
    validate_split_manifest(payload)
    return payload


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    dataset_root = Path(config["data"]["dataset_root"])
    development = config["development"]
    protocol = str(config["experiment_protocol"])
    catalogs = {
        competition: _match_catalog(dataset_root, competition)
        for competition in development["support_competitions"]
    }
    support_end = str(development["support_end_dateutc_exclusive"])
    query_start = str(development["query_start_dateutc_inclusive"])
    if support_end >= query_start:
        raise ValueError("Support cutoff must be strictly earlier than query start.")
    support_ids = sorted(
        str(match_id)
        for catalog in catalogs.values()
        for match_id in catalog.loc[
            catalog["dateutc"] < support_end,
            "match_id",
        ].tolist()
    )

    def query_ids(competitions: list[str]) -> list[str]:
        return sorted(
            str(match_id)
            for competition in competitions
            for match_id in catalogs[competition].loc[
                catalogs[competition]["dateutc"] >= query_start,
                "match_id",
            ].tolist()
        )

    train_ids = query_ids(development["train_query_competitions"])
    validation_ids = query_ids(development["validation_query_competitions"])
    development_ids = query_ids(development["development_query_competitions"])
    query_union = set(train_ids) | set(validation_ids) | set(development_ids)
    if not set(support_ids).isdisjoint(query_union):
        raise ValueError("Wyscout support and development query matches overlap.")
    support_dates = [
        str(value)
        for catalog in catalogs.values()
        for value in catalog.loc[
            catalog["match_id"].astype(str).isin(support_ids),
            "dateutc",
        ].tolist()
    ]
    query_dates = [
        str(value)
        for catalog in catalogs.values()
        for value in catalog.loc[
            catalog["match_id"].astype(str).isin(query_union),
            "dateutc",
        ].tolist()
    ]
    if max(support_dates) >= min(query_dates):
        raise ValueError("Development support is not strictly earlier than all queries.")
    dataset_manifest_path = Path(config["data"]["dataset_manifest"])
    development_payload = _manifest_payload(
        name=Path(config["data"]["development_split_manifest"]).stem,
        protocol=protocol,
        train=train_ids,
        validation=validation_ids,
        test=development_ids,
        extra={
            "status": "development_only",
            "support_match_ids": support_ids,
            "support_expected_count": len(support_ids),
            "support_end_dateutc_exclusive": support_end,
            "query_start_dateutc_inclusive": query_start,
            "latest_support_dateutc": max(support_dates),
            "earliest_query_dateutc": min(query_dates),
            "train_query_competitions": development[
                "train_query_competitions"
            ],
            "validation_query_competitions": development[
                "validation_query_competitions"
            ],
            "development_query_competitions": development[
                "development_query_competitions"
            ],
            "dataset_manifest_path": str(dataset_manifest_path),
            "dataset_manifest_sha256": _sha256(dataset_manifest_path),
        },
    )
    development_path = Path(config["data"]["development_split_manifest"])
    _write(development_path, development_payload)

    confirmatory_competition = str(config["confirmatory"]["competition"])
    confirmatory_catalog = _match_catalog(dataset_root, confirmatory_competition)
    confirmatory_ids = [
        str(value) for value in confirmatory_catalog["match_id"].tolist()
    ]
    all_domestic_ids = sorted(
        str(match_id)
        for catalog in catalogs.values()
        for match_id in catalog["match_id"].tolist()
    )
    latest_support_date = max(
        str(catalog["dateutc"].max()) for catalog in catalogs.values()
    )
    earliest_confirmatory_date = str(confirmatory_catalog["dateutc"].min())
    if latest_support_date >= earliest_confirmatory_date:
        raise ValueError(
            "Confirmatory profile source is not strictly earlier than query cohort."
        )
    confirmatory_payload = _manifest_payload(
        name=Path(config["data"]["confirmatory_split_manifest"]).stem,
        protocol=f"{protocol}_confirmatory",
        train=[],
        validation=[],
        test=confirmatory_ids,
        extra={
            "status": "metric_sealed",
            "outcome_access_note": (
                "Aggregate label prevalence was audited before freezing; no model "
                "comparison metric may be loaded until allow_metric_unseal is true."
            ),
            "support_match_ids": all_domestic_ids,
            "support_expected_count": len(all_domestic_ids),
            "profile_source_competitions": config["confirmatory"][
                "profile_source_competitions"
            ],
            "latest_support_dateutc": latest_support_date,
            "earliest_query_dateutc": earliest_confirmatory_date,
            "dataset_manifest_path": str(dataset_manifest_path),
            "dataset_manifest_sha256": _sha256(dataset_manifest_path),
        },
    )
    confirmatory_path = Path(config["data"]["confirmatory_split_manifest"])
    _write(confirmatory_path, confirmatory_payload)
    print(
        json.dumps(
            {
                "development_manifest": str(development_path),
                "support_matches": len(support_ids),
                "train_matches": len(train_ids),
                "validation_matches": len(validation_ids),
                "development_matches": len(development_ids),
                "confirmatory_manifest": str(confirmatory_path),
                "confirmatory_matches": len(confirmatory_ids),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
