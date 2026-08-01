"""Validate externally resolved FOOTPASS identities against extracted lineups."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

PARTITIONS = (
    "profile_support_only",
    "development_train",
    "development_validation",
    "confirmatory_reserve_do_not_read_until_frozen",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signature-report", type=Path, required=True)
    parser.add_argument("--identity-manifest", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def validate_manifest(
    manifest_path: Path,
    observed: dict[str, set[int]],
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    matches = list(manifest["matches"])
    match_ids = [str(match["footpass_match_id"]) for match in matches]
    if len(match_ids) != len(set(match_ids)):
        raise ValueError(f"Duplicate match IDs in {manifest_path}.")

    dates = [date.fromisoformat(str(match["match_date"])) for match in matches]
    if dates != sorted(dates):
        raise ValueError(f"Matches are not chronological in {manifest_path}.")

    verification: list[dict[str, Any]] = []
    for match in matches:
        match_id = str(match["footpass_match_id"])
        focal_side = int(match["focal_team_index"])
        opponent_side = 1 - focal_side
        focal_key = f"{match_id}:{focal_side}"
        opponent_key = f"{match_id}:{opponent_side}"
        expected_focal = {int(value) for value in match["focal_starting_shirts"]}
        expected_opponent = {
            int(value) for value in match["opponent_starting_shirts"]
        }
        actual_focal = observed.get(focal_key)
        actual_opponent = observed.get(opponent_key)
        if actual_focal != expected_focal:
            raise ValueError(
                f"Focal lineup mismatch for {focal_key}: "
                f"expected={sorted(expected_focal)}, actual={sorted(actual_focal or [])}."
            )
        if actual_opponent != expected_opponent:
            raise ValueError(
                f"Opponent lineup mismatch for {opponent_key}: "
                f"expected={sorted(expected_opponent)}, "
                f"actual={sorted(actual_opponent or [])}."
            )
        verification.append(
            {
                "footpass_match_id": match_id,
                "focal_appearance_id": focal_key,
                "opponent_appearance_id": opponent_key,
                "focal_exact": True,
                "opponent_exact": True,
            }
        )

    partition = dict(manifest["research_partition"])
    partition_sets = {
        name: {str(value) for value in partition[name]} for name in PARTITIONS
    }
    for left_index, left_name in enumerate(PARTITIONS):
        for right_name in PARTITIONS[left_index + 1 :]:
            overlap = partition_sets[left_name] & partition_sets[right_name]
            if overlap:
                raise ValueError(
                    f"Partition overlap in {manifest_path}: "
                    f"{left_name}/{right_name}={sorted(overlap)}."
                )
    partition_union = set().union(*partition_sets.values())
    if partition_union != set(match_ids):
        raise ValueError(
            f"Partition coverage mismatch in {manifest_path}: "
            f"expected={sorted(match_ids)}, actual={sorted(partition_union)}."
        )

    players = list(manifest["players"])
    player_shirts = [int(player["shirt_number"]) for player in players]
    player_ids = [str(player["player_id"]) for player in players]
    if len(player_shirts) != len(set(player_shirts)):
        raise ValueError(f"Duplicate player shirt mappings in {manifest_path}.")
    if len(player_ids) != len(set(player_ids)):
        raise ValueError(f"Duplicate persistent player IDs in {manifest_path}.")
    starter_shirts = {
        int(shirt) for match in matches for shirt in match["focal_starting_shirts"]
    }
    missing_players = starter_shirts - set(player_shirts)
    if missing_players:
        raise ValueError(
            f"Starting shirts lack player mappings in {manifest_path}: "
            f"{sorted(missing_players)}."
        )

    return {
        "manifest": str(manifest_path),
        "name": manifest["name"],
        "team_id": manifest["team"]["team_id"],
        "match_count": len(matches),
        "first_match_date": dates[0].isoformat(),
        "last_match_date": dates[-1].isoformat(),
        "persistent_player_count": len(players),
        "exact_lineup_matches": len(verification),
        "partition_counts": {
            name: len(partition_sets[name]) for name in PARTITIONS
        },
        "verification": verification,
    }


def main() -> None:
    args = parse_args()
    signature_report = _read_json(args.signature_report)
    observed = {
        str(item["appearance_id"]): {
            int(value) for value in item["shirt_numbers"]
        }
        for item in signature_report["lineup_signatures"]
    }
    reports = [
        validate_manifest(path, observed) for path in args.identity_manifest
    ]
    payload = {
        "version": 1,
        "status": "passed",
        "claim_status": "identity_and_chronology_validation_only",
        "signature_report": str(args.signature_report),
        "identity_manifest_count": len(reports),
        "total_team_appearances": sum(
            int(report["match_count"]) for report in reports
        ),
        "reports": reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"status: {payload['status']}")
    print(f"identity_manifests: {len(reports)}")
    print(f"team_appearances: {payload['total_team_appearances']}")
    print(f"report: {args.out}")


if __name__ == "__main__":
    main()
