from __future__ import annotations

import json
from pathlib import Path

import pytest

from footballq.data.wyscout_public import (
    build_competition_pass_frame,
    file_sha256,
    verify_source_manifest,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_verify_source_manifest_checks_size_and_hash(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"verified")
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {
            "name": "fixture",
            "collection_doi": "fixture-doi",
            "license": {"name": "CC BY 4.0"},
            "files": [
                {
                    "name": source.name,
                    "bytes": source.stat().st_size,
                    "sha256": file_sha256(source),
                }
            ],
        },
    )

    result = verify_source_manifest(tmp_path, manifest)

    assert result["provenance_name"] == "fixture"
    assert result["verified_files"][0]["sha256"] == file_sha256(source)


def test_verify_source_manifest_rejects_modified_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"before")
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {
            "name": "fixture",
            "collection_doi": "fixture-doi",
            "license": {"name": "CC BY 4.0"},
            "files": [
                {
                    "name": source.name,
                    "bytes": source.stat().st_size,
                    "sha256": file_sha256(source),
                }
            ],
        },
    )
    source.write_bytes(b"after!")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_source_manifest(tmp_path, manifest)


def test_pass_frame_has_period_aware_ids_and_strict_future_label(
    tmp_path: Path,
) -> None:
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    _write_json(
        tmp_path / "players.json",
        [
            {
                "wyId": 10,
                "role": {"name": "Midfielder"},
            }
        ],
    )
    _write_json(
        extracted / "matches_England.json",
        [
            {
                "wyId": 1,
                "dateutc": "2018-01-01 12:00:00",
                "gameweek": 1,
                "teamsData": {"100": {}, "200": {}},
            }
        ],
    )
    _write_json(
        extracted / "events_England.json",
        [
            {
                "id": 1,
                "matchId": 1,
                "matchPeriod": "1H",
                "eventSec": 5.0,
                "eventName": "Pass",
                "subEventId": 85,
                "playerId": 10,
                "teamId": 100,
                "positions": [{"x": 50, "y": 25}, {"x": 75, "y": 50}],
                "tags": [{"id": 1801}, {"id": 302}],
            },
            {
                "id": 2,
                "matchId": 1,
                "matchPeriod": "1H",
                "eventSec": 18.0,
                "eventName": "Shot",
                "subEventId": 0,
                "playerId": 10,
                "teamId": 100,
                "positions": [{"x": 90, "y": 50}],
                "tags": [],
            },
        ],
    )

    frame = build_competition_pass_frame(
        extracted,
        tmp_path / "players.json",
        "England",
    )

    assert frame.loc[0, "sample_id"] == "1:1H:1"
    assert frame.loc[0, "role"] == 2
    assert frame.loc[0, "start_zone"] == 9
    assert frame.loc[0, "destination_zone"] == 22
    assert frame.loc[0, "accurate"] == 1
    assert frame.loc[0, "key_pass"] == 1
    assert frame.loc[0, "shot_within_horizon"] == 1


def test_future_shot_outside_time_horizon_is_negative(tmp_path: Path) -> None:
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    _write_json(
        tmp_path / "players.json",
        [{"wyId": 10, "role": {"name": "Forward"}}],
    )
    _write_json(
        extracted / "matches_England.json",
        [
            {
                "wyId": 1,
                "dateutc": "2018-01-01 12:00:00",
                "gameweek": 1,
                "teamsData": {"100": {}, "200": {}},
            }
        ],
    )
    _write_json(
        extracted / "events_England.json",
        [
            {
                "id": 1,
                "matchId": 1,
                "matchPeriod": "1H",
                "eventSec": 5.0,
                "eventName": "Pass",
                "subEventId": 85,
                "playerId": 10,
                "teamId": 100,
                "positions": [{"x": 10, "y": 10}, {"x": 20, "y": 20}],
                "tags": [],
            },
            {
                "id": 2,
                "matchId": 1,
                "matchPeriod": "1H",
                "eventSec": 25.1,
                "eventName": "Shot",
                "subEventId": 0,
                "playerId": 10,
                "teamId": 100,
                "positions": [{"x": 90, "y": 50}],
                "tags": [],
            },
        ],
    )

    frame = build_competition_pass_frame(
        extracted,
        tmp_path / "players.json",
        "England",
    )

    assert frame.loc[0, "shot_within_horizon"] == 0
