import json

import pandas as pd
from test_pff import _record

from footballq.io.pff_shards import PFFRosterSlotTracker, prepare_pff_dataset_shards
from footballq.repro.splits import load_split_manifest


def _split(path) -> None:
    path.write_text(
        json.dumps(
            {
                "name": "pff_test_split",
                "version": 1,
                "dataset": "pff_fc",
                "protocol": "inductive_match_holdout",
                "train_match_ids": ["10502"],
                "val_match_ids": ["10503"],
                "test_match_ids": ["10504"],
                "all_match_ids": ["10502", "10503", "10504"],
                "expected_count": 3,
            }
        ),
        encoding="utf-8",
    )


def test_pff_sharding_is_period_aware_deduplicated_and_resumable(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    records = [_record(0), _record(1), _record(1), _record(2)]
    records.extend(
        [{**_record(frame), "period": 2, "periodElapsedTime": frame - 3} for frame in (3, 4)]
    )
    source = raw / "10502.jsonl"
    source.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    for match_id in ("10503", "10504"):
        (raw / f"{match_id}.jsonl").write_text(json.dumps(_record(0)) + "\n")
    split_path = tmp_path / "split.json"
    _split(split_path)
    out = tmp_path / "out"

    manifest = prepare_pff_dataset_shards(
        raw,
        out,
        split_path,
        match_ids=["10502"],
        frames_per_shard=2,
        hash_source=True,
    )
    resumed = prepare_pff_dataset_shards(
        raw,
        out,
        split_path,
        match_ids=["10502"],
        frames_per_shard=2,
        hash_source=True,
    )

    assert manifest["totals"]["unique_frames"] == 5
    assert manifest["totals"]["duplicate_records"] == 1
    assert resumed["matches"] == manifest["matches"]
    match_manifest = json.loads((out / "train" / "10502" / "manifest.json").read_text())
    assert [shard["period"] for shard in match_manifest["shards"]] == [1, 1, 2]
    shard_frames = [
        pd.read_parquet(out / "train" / "10502" / shard["path"])["frame_id"].nunique()
        for shard in match_manifest["shards"]
    ]
    assert shard_frames == [2, 1, 2]
    assert load_split_manifest(split_path).payload["dataset"] == "pff_fc"


def test_pff_sharding_records_nonstandard_entity_counts(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    record = _record(0)
    record["ballsSmoothed"] = None
    (raw / "10502.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    for match_id in ("10503", "10504"):
        (raw / f"{match_id}.jsonl").write_text(json.dumps(_record(0)) + "\n")
    split_path = tmp_path / "split.json"
    _split(split_path)

    manifest = prepare_pff_dataset_shards(
        raw,
        tmp_path / "out",
        split_path,
        match_ids=["10502"],
        hash_source=False,
    )

    assert manifest["totals"]["missing_ball_frames"] == 1
    assert manifest["totals"]["non_23_entity_frames"] == 1


def test_pff_sharding_supports_parallel_matches(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for match_id in ("10502", "10503", "10504"):
        record = {**_record(0), "gameRefId": int(match_id)}
        (raw / f"{match_id}.jsonl").write_text(json.dumps(record) + "\n")
    split_path = tmp_path / "split.json"
    _split(split_path)

    manifest = prepare_pff_dataset_shards(
        raw,
        tmp_path / "out",
        split_path,
        hash_source=False,
        workers=2,
    )

    assert manifest["selected_match_count"] == 3
    assert manifest["totals"]["unique_frames"] == 3


def test_pff_roster_slots_transfer_only_after_a_player_vacates():
    tracker = PFFRosterSlotTracker()
    initial = _record(0)["homePlayersSmoothed"]

    def rows(players):
        return [
            {
                "agent_type": "player",
                "team_id": "home",
                "agent_id": f"home_{player['jerseyNum']}",
                "entity_id": f"home_{player['jerseyNum']}",
                "jersey_number": player["jerseyNum"],
            }
            for player in players
        ]

    assigned, dropped = tracker.assign(rows(initial))
    slot_for_one = next(row["agent_id"] for row in assigned if row["jersey_number"] == "1")
    overlap = [*initial, {**initial[0], "jerseyNum": "99"}]
    assigned, dropped = tracker.assign(rows(overlap))
    assert dropped == 1
    assert all(row["jersey_number"] != "99" for row in assigned)

    substituted = [{**initial[0], "jerseyNum": "99"}, *initial[1:]]
    assigned, dropped = tracker.assign(rows(substituted))
    slot_for_99 = next(row["agent_id"] for row in assigned if row["jersey_number"] == "99")
    assert dropped == 0
    assert slot_for_99 == slot_for_one
