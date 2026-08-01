"""Verified loading and compact pass-table construction for public Wyscout data."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

DOMESTIC_COMPETITIONS = ("England", "France", "Germany", "Italy", "Spain")
ALL_COMPETITIONS = (
    *DOMESTIC_COMPETITIONS,
    "European_Championship",
    "World_Cup",
)
PERIOD_ORDER = {"1H": 1, "2H": 2, "E1": 3, "E2": 4, "P": 5}
ROLE_CODES = {"Goalkeeper": 0, "Defender": 1, "Midfielder": 2, "Forward": 3}
ACCURATE_TAG = 1801
KEY_PASS_TAGS = {301, 302}


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def file_sha256(path: str | Path) -> str:
    """Hash a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_manifest(
    raw_root: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Verify source file sizes and hashes against the checked-in provenance record."""

    root = Path(raw_root)
    manifest = read_json(manifest_path)
    verified: list[dict[str, Any]] = []
    for expected in manifest["files"]:
        path = root / str(expected["name"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing Wyscout source file: {path}")
        actual_size = path.stat().st_size
        if actual_size != int(expected["bytes"]):
            raise ValueError(
                f"Wyscout source size mismatch for {path.name}: "
                f"expected {expected['bytes']}, got {actual_size}."
            )
        actual_sha = file_sha256(path)
        if actual_sha.lower() != str(expected["sha256"]).lower():
            raise ValueError(
                f"Wyscout source SHA-256 mismatch for {path.name}: "
                f"expected {expected['sha256']}, got {actual_sha}."
            )
        verified.append(
            {
                "name": path.name,
                "bytes": actual_size,
                "sha256": actual_sha,
            }
        )
    return {
        "provenance_name": str(manifest["name"]),
        "provenance_path": str(Path(manifest_path)),
        "collection_doi": str(manifest["collection_doi"]),
        "license": str(manifest["license"]["name"]),
        "verified_files": verified,
    }


def _zone_index(x: float, y: float, x_bins: int, y_bins: int) -> int:
    x_index = min(x_bins - 1, max(0, int(float(x) * x_bins / 100.0)))
    y_index = min(y_bins - 1, max(0, int(float(y) * y_bins / 100.0)))
    return x_index * y_bins + y_index


def _event_tags(event: dict[str, Any]) -> set[int]:
    return {int(tag["id"]) for tag in event.get("tags") or []}


def _opponent_by_team(match: dict[str, Any]) -> dict[int, int]:
    team_ids = [int(value) for value in match.get("teamsData", {})]
    if len(team_ids) != 2:
        return {}
    return {team_ids[0]: team_ids[1], team_ids[1]: team_ids[0]}


def _player_roles(players_path: str | Path) -> dict[int, int]:
    roles: dict[int, int] = {}
    for player in read_json(players_path):
        role_name = str((player.get("role") or {}).get("name") or "")
        roles[int(player["wyId"])] = ROLE_CODES.get(role_name, 4)
    return roles


def _shot_within_horizon(
    rows: list[dict[str, Any]],
    event_index: int,
    *,
    horizon_seconds: float,
    horizon_events: int,
) -> bool:
    event = rows[event_index]
    period = str(event.get("matchPeriod") or "")
    event_second = float(event.get("eventSec") or 0.0)
    team_id = int(event.get("teamId") or 0)
    for candidate in rows[event_index + 1 : event_index + 1 + horizon_events]:
        if str(candidate.get("matchPeriod") or "") != period:
            break
        if float(candidate.get("eventSec") or 0.0) - event_second > horizon_seconds:
            break
        if (
            int(candidate.get("teamId") or 0) == team_id
            and str(candidate.get("eventName") or "") == "Shot"
        ):
            return True
    return False


def build_competition_pass_frame(
    extracted_root: str | Path,
    players_path: str | Path,
    competition: str,
    *,
    horizon_seconds: float = 20.0,
    horizon_events: int = 10,
    start_x_bins: int = 5,
    start_y_bins: int = 4,
    destination_x_bins: int = 6,
    destination_y_bins: int = 5,
) -> pd.DataFrame:
    """Build one row per pass with pre-pass context and future outcome labels."""

    if competition not in ALL_COMPETITIONS:
        raise ValueError(f"Unknown Wyscout competition: {competition}")
    root = Path(extracted_root)
    matches = read_json(root / f"matches_{competition}.json")
    events = read_json(root / f"events_{competition}.json")
    roles = _player_roles(players_path)
    match_by_id = {int(match["wyId"]): match for match in matches}
    events_by_match: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        events_by_match[int(event["matchId"])].append(event)

    records: list[dict[str, Any]] = []
    for match_id, rows in events_by_match.items():
        match = match_by_id.get(match_id)
        if match is None:
            raise ValueError(f"Event data references absent match ID {match_id}.")
        rows.sort(
            key=lambda event: (
                PERIOD_ORDER.get(str(event.get("matchPeriod") or ""), 99),
                float(event.get("eventSec") or 0.0),
                int(event.get("id") or 0),
            )
        )
        opponents = _opponent_by_team(match)
        for event_index, event in enumerate(rows):
            positions = event.get("positions") or []
            player_id = int(event.get("playerId") or 0)
            if (
                str(event.get("eventName") or "") != "Pass"
                or player_id <= 0
                or len(positions) < 2
            ):
                continue
            start = positions[0]
            destination = positions[1]
            start_x = float(start["x"])
            start_y = float(start["y"])
            destination_x = float(destination["x"])
            destination_y = float(destination["y"])
            team_id = int(event.get("teamId") or 0)
            period_name = str(event.get("matchPeriod") or "")
            event_id = int(event["id"])
            tags = _event_tags(event)
            records.append(
                {
                    "sample_id": f"{match_id}:{period_name}:{event_id}",
                    "match_id": match_id,
                    "event_id": event_id,
                    "competition": competition,
                    "dateutc": str(match["dateutc"]),
                    "gameweek": int(match.get("gameweek") or 0),
                    "period": PERIOD_ORDER.get(period_name, 99),
                    "event_sec": float(event.get("eventSec") or 0.0),
                    "player_id": player_id,
                    "team_id": team_id,
                    "opponent_team_id": opponents.get(team_id, 0),
                    "role": roles.get(player_id, 4),
                    "subevent_id": int(event.get("subEventId") or 0),
                    "start_x": start_x,
                    "start_y": start_y,
                    "destination_x": destination_x,
                    "destination_y": destination_y,
                    "start_zone": _zone_index(
                        start_x,
                        start_y,
                        start_x_bins,
                        start_y_bins,
                    ),
                    "destination_zone": _zone_index(
                        destination_x,
                        destination_y,
                        destination_x_bins,
                        destination_y_bins,
                    ),
                    "accurate": int(ACCURATE_TAG in tags),
                    "key_pass": int(bool(KEY_PASS_TAGS & tags)),
                    "shot_within_horizon": int(
                        _shot_within_horizon(
                            rows,
                            event_index,
                            horizon_seconds=horizon_seconds,
                            horizon_events=horizon_events,
                        )
                    ),
                }
            )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError(f"No valid passes found for Wyscout competition {competition}.")
    return frame
