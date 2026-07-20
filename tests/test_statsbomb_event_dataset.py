import json

import torch

from footballq.data.statsbomb_event_dataset import (
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    FREEZE_FRAME_FEATURES,
    ShardedStatsBombEventDataset,
    audit_statsbomb_event_dataset,
    prepare_statsbomb_event_dataset,
    prepare_statsbomb_match_shard,
)
from footballq.data.statsbomb_events import (
    audit_statsbomb_training_schema,
    build_statsbomb_source_manifest,
    build_statsbomb_split,
    load_statsbomb_match_catalog,
)
from footballq.repro.splits import split_manifest_sha256


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _event(event_index, period=1, event_id=None):
    return {
        "id": event_id or f"event-{event_index}",
        "index": event_index,
        "period": period,
        "minute": 0,
        "second": event_index,
        "type": {"id": 30, "name": "Pass"},
        "play_pattern": {"id": 1, "name": "Regular Play"},
        "position": {"id": 6, "name": "Left Back"},
        "team": {"id": 10, "name": "Home"},
        "possession": 1,
        "possession_team": {"id": 10, "name": "Home"},
        "location": [60.0, 40.0],
        "pass": {"end_location": [61.0, 41.0], "outcome": {"id": 8, "name": "Complete"}},
    }


def _category_maps():
    return {
        "event_type": {30: 2},
        "play_pattern": {1: 2},
        "position": {6: 2},
        "subtype": {},
        "outcome": {8: 2},
    }


def test_statsbomb_match_shard_windows_do_not_cross_periods(tmp_path):
    data = tmp_path / "data"
    events = [_event(index, period=1) for index in range(1, 8)] + [
        _event(index + 20, period=2) for index in range(1, 8)
    ]
    _write_json(data / "events" / "1.json", events)

    shard, quality = prepare_statsbomb_match_shard(
        data,
        "1",
        "train",
        _category_maps(),
        sequence_length=4,
        stride=2,
    )

    assert shard["categorical"].shape == (14, len(CATEGORICAL_FEATURES))
    assert shard["continuous"].shape == (14, len(CONTINUOUS_FEATURES))
    assert shard["window_starts"].tolist() == [0, 2, 7, 9]
    for start in shard["window_starts"].tolist():
        assert len(set(shard["period"][start : start + 5].tolist())) == 1
    assert quality["matched_three_sixty_rows"] == 0


def test_statsbomb_match_shard_joins_360_and_marks_out_of_bounds_coordinates(tmp_path):
    data = tmp_path / "data"
    events = [_event(index) for index in range(1, 7)]
    _write_json(data / "events" / "1.json", events)
    _write_json(
        data / "three-sixty" / "1.json",
        [
            {
                "event_uuid": "event-1",
                "visible_area": [0, 0, 120, 0, 120, 80, 0, 80],
                "freeze_frame": [
                    {
                        "location": [125.0, 40.0],
                        "teammate": True,
                        "actor": False,
                        "keeper": False,
                    }
                ],
            },
            {"event_uuid": "stale-event", "freeze_frame": [], "visible_area": []},
        ],
    )

    shard, quality = prepare_statsbomb_match_shard(
        data,
        "1",
        "train",
        _category_maps(),
        sequence_length=4,
        stride=2,
    )

    assert shard["freeze_frame"].shape == (1, 22, len(FREEZE_FRAME_FEATURES))
    assert shard["freeze_frame"][0, 0, 0] == 1.0
    assert shard["freeze_frame"][0, 0, 5] == 0.0
    assert quality["matched_three_sixty_rows"] == 1
    assert quality["unmatched_three_sixty_rows"] == 1


def test_statsbomb_sharded_dataset_exposes_train_val_only(tmp_path):
    raw = tmp_path / "raw"
    data = raw / "data"
    _write_json(data / "competitions.json", [])
    matches = []
    for match_id in range(1, 31):
        matches.append(
            {
                "match_id": match_id,
                "competition": {"competition_id": 1, "competition_name": "League"},
                "season": {"season_id": 1, "season_name": "2026"},
                "home_team": {"home_team_id": 10},
                "away_team": {"away_team_id": 20},
            }
        )
        _write_json(data / "events" / f"{match_id}.json", [_event(i) for i in range(1, 8)])
        _write_json(data / "lineups" / f"{match_id}.json", [])
    _write_json(data / "matches" / "1" / "1.json", matches)
    catalog = load_statsbomb_match_catalog(raw)
    split = build_statsbomb_split(catalog)
    audit = audit_statsbomb_training_schema(raw, split)
    source = build_statsbomb_source_manifest(raw, catalog, split)
    split_path = tmp_path / "split.json"
    source_path = tmp_path / "source.json"
    audit_path = tmp_path / "audit.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    source_path.write_text(json.dumps(source), encoding="utf-8")
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    out = tmp_path / "processed"

    manifest = prepare_statsbomb_event_dataset(
        raw,
        split_path,
        source_path,
        audit_path,
        out,
        sequence_length=4,
        stride=2,
        match_limits={"train": 2, "val": 1},
    )
    train = ShardedStatsBombEventDataset(out / "manifest.json", "train")
    batch = train[0]

    assert manifest["loaded_splits"] == ["train", "val"]
    assert manifest["test_loaded"] is False
    assert manifest["split_manifest_sha256"] == split_manifest_sha256(split)
    assert batch["categorical"].shape == (4, len(CATEGORICAL_FEATURES))
    assert batch["target_event_type"].shape == (4,)
    assert batch["freeze_frame"].shape == (4, 22, len(FREEZE_FRAME_FEATURES))
    assert torch.all(batch["target_location_mask"])

    tensor_audit = audit_statsbomb_event_dataset(out / "manifest.json")
    assert tensor_audit["status"] == "passed"
    assert tensor_audit["test_loaded"] is False
