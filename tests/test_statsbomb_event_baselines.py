import json

from footballq.analysis.statsbomb_event_baselines import compute_statsbomb_event_baselines
from footballq.data.statsbomb_event_dataset import prepare_statsbomb_event_dataset
from footballq.data.statsbomb_events import (
    audit_statsbomb_training_schema,
    build_statsbomb_source_manifest,
    build_statsbomb_split,
    load_statsbomb_match_catalog,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_statsbomb_event_baselines_fit_train_and_score_validation_only(tmp_path):
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
        events = []
        for index in range(1, 9):
            event_type = 30 if index % 2 else 42
            events.append(
                {
                    "id": f"{match_id}-{index}",
                    "index": index,
                    "period": 1,
                    "minute": 0,
                    "second": index,
                    "type": {"id": event_type, "name": str(event_type)},
                    "location": [float(index * 10), 40.0],
                }
            )
        _write_json(data / "events" / f"{match_id}.json", events)
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
    output = tmp_path / "processed"
    prepare_statsbomb_event_dataset(
        raw,
        split_path,
        source_path,
        audit_path,
        output,
        sequence_length=4,
        stride=2,
    )

    report = compute_statsbomb_event_baselines(output / "manifest.json")

    assert report["loaded_splits"] == ["train", "val"]
    assert report["test_loaded"] is False
    assert report["validation_target_weight"] > 0
    assert report["first_order_markov_event_type_nll"] < report[
        "global_frequency_event_type_nll"
    ]
    assert report["copy_current_location_mae"] > 0
