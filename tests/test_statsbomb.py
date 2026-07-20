import json

import pandas as pd

from footballq.data.statsbomb_events import (
    audit_statsbomb_training_schema,
    build_statsbomb_source_manifest,
    build_statsbomb_split,
    load_statsbomb_match_catalog,
)
from footballq.io.statsbomb import StatsBombAdapter
from footballq.repro.splits import validate_split_manifest


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _tiny_open_data(tmp_path, match_count=10, three_sixty_count=4):
    data = tmp_path / "data"
    _write_json(data / "competitions.json", [])
    matches = []
    for match_id in range(1, match_count + 1):
        matches.append(
            {
                "match_id": match_id,
                "match_date": "2026-01-01",
                "competition": {
                    "competition_id": 1,
                    "competition_name": "Test League",
                    "country_name": "Testland",
                },
                "season": {"season_id": 2, "season_name": "2025/2026"},
                "home_team": {"home_team_id": 10},
                "away_team": {"away_team_id": 20},
                "match_status": "available",
                "match_status_360": "available" if match_id <= three_sixty_count else "scheduled",
            }
        )
        _write_json(data / "events" / f"{match_id}.json", [])
        _write_json(data / "lineups" / f"{match_id}.json", [])
        if match_id <= three_sixty_count:
            _write_json(data / "three-sixty" / f"{match_id}.json", [])
    _write_json(data / "matches" / "1" / "2.json", matches)
    return data


def test_statsbomb_catalog_and_split_are_match_disjoint_and_360_stratified(tmp_path):
    data = _tiny_open_data(tmp_path)
    catalog = load_statsbomb_match_catalog(data)
    split = build_statsbomb_split(catalog)

    validate_split_manifest(split)
    assert len(catalog) == 10
    assert sum(row["has_360"] for row in catalog) == 4
    assert [len(split[f"{name}_match_ids"]) for name in ("train", "val", "test")] == [8, 2, 0]
    assert split["strata"]["metadata_with_360"] == {
        "total": 4,
        "train": 3,
        "val": 1,
        "test": 0,
    }


def test_statsbomb_source_manifest_hashes_without_parsing_event_payloads(tmp_path):
    data = _tiny_open_data(tmp_path, match_count=3, three_sixty_count=1)
    catalog = load_statsbomb_match_catalog(data)
    split = build_statsbomb_split(catalog)
    manifest = build_statsbomb_source_manifest(data, catalog, split)

    assert manifest["coverage"]["matches"] == 3
    assert manifest["coverage"]["orphaned_event_matches"] == 0
    assert manifest["coverage"]["event_files"] == 3
    assert manifest["coverage"]["three_sixty_files"] == 1
    assert manifest["file_hashes_complete"] is True
    assert all("sha256" in row for row in manifest["files"])


def test_statsbomb_adapter_preserves_semantics_and_loads_only_requested_match(tmp_path):
    data = _tiny_open_data(tmp_path, match_count=2, three_sixty_count=1)
    event = {
        "id": "event-1",
        "index": 5,
        "period": 1,
        "timestamp": "00:01:02.500",
        "minute": 1,
        "second": 2,
        "duration": 1.25,
        "type": {"id": 30, "name": "Pass"},
        "team": {"id": 10, "name": "Home"},
        "player": {"id": 100, "name": "Player"},
        "possession": 3,
        "possession_team": {"id": 10, "name": "Home"},
        "play_pattern": {"id": 1, "name": "Regular Play"},
        "position": {"id": 6, "name": "Left Back"},
        "location": [60.0, 40.0],
        "under_pressure": True,
        "pass": {
            "end_location": [120.0, 80.0],
            "type": {"id": 65, "name": "Kick Off"},
            "outcome": {"id": 9, "name": "Incomplete"},
        },
    }
    _write_json(data / "events" / "1.json", [event])
    _write_json(data / "events" / "2.json", [{**event, "id": "wrong-match"}])
    _write_json(
        data / "three-sixty" / "1.json",
        [
            {
                "event_uuid": "event-1",
                "visible_area": [0.0, 0.0, 120.0, 0.0, 120.0, 80.0],
                "freeze_frame": [
                    {
                        "location": [60.0, 40.0],
                        "teammate": True,
                        "actor": True,
                        "keeper": False,
                    }
                ],
            }
        ],
    )

    adapter = StatsBombAdapter(tmp_path, "1")
    events = adapter.load_events()
    frames = adapter.load_360()

    assert len(events) == 1
    assert events.loc[0, "event_id"] == "event-1"
    assert events.loc[0, "team_id"] == 10
    assert events.loc[0, "event_subtype"] == "Kick Off"
    assert events.loc[0, "outcome"] == "Incomplete"
    assert events.loc[0, "x_m"] == 52.5
    assert events.loc[0, "end_y_m"] == 68.0
    assert bool(events.loc[0, "under_pressure"]) is True
    assert len(frames) == 1
    assert frames.loc[0, "event_id"] == "event-1"
    assert frames.loc[0, "freeze_frame_count"] == 1
    assert isinstance(events, pd.DataFrame)


def test_statsbomb_catalog_retains_event_lineup_pairs_missing_match_metadata(tmp_path):
    data = _tiny_open_data(tmp_path, match_count=3, three_sixty_count=1)
    _write_json(data / "events" / "99.json", [])
    _write_json(data / "lineups" / "99.json", [])

    catalog = load_statsbomb_match_catalog(data)
    orphan = next(row for row in catalog if row["match_id"] == "99")
    split = build_statsbomb_split(catalog)

    assert len(catalog) == 4
    assert orphan["metadata_available"] is False
    assert orphan["competition_id"] is None
    assert "orphaned_without_360" in split["strata"]


def test_statsbomb_training_schema_audit_does_not_open_val_or_test_events(tmp_path):
    data = _tiny_open_data(tmp_path, match_count=30, three_sixty_count=0)
    catalog = load_statsbomb_match_catalog(data)
    split = build_statsbomb_split(catalog)
    held_out_id = split["test_match_ids"][0]
    (data / "events" / f"{held_out_id}.json").write_text("not json", encoding="utf-8")

    audit = audit_statsbomb_training_schema(data, split)

    assert audit["scope"] == "train_only"
    assert audit["loaded_splits"] == ["train"]
    assert audit["train_match_count"] == len(split["train_match_ids"])
    assert audit["counts"]["events"] == 0
