import json

import torch

from footballq.data.pff_event_context import (
    build_pff_event_match_shard,
    pff_event_history_from_shard,
    pff_event_mapping_payload,
)


def _statsbomb_manifest():
    names = [
        "Duel",
        "Clearance",
        "Shot",
        "Substitution",
        "Player On",
        "Player Off",
        "Pass",
        "Half End",
        "Half Start",
        "Ball Receipt*",
        "Carry",
    ]
    return {
        "manifest_payload_sha256": "manifest",
        "vocabulary_payload_sha256": "vocabulary",
        "categorical_vocabularies": {
            "event_type": {
                "entries": [
                    {"name": name, "index": index + 2}
                    for index, name in enumerate(names)
                ]
            }
        },
    }


def test_pff_event_mapping_is_stable_and_explicit():
    first = pff_event_mapping_payload()
    second = pff_event_mapping_payload()

    assert first == second
    assert first["possession_event_map"]["PA"] == "Pass"
    assert first["possession_event_map"]["BC"] == "Carry"
    assert first["excluded_game_events"] == ["OTB"]


def test_pff_event_shard_deduplicates_and_retains_unknown(tmp_path):
    rows = [
        {
            "gameRefId": 1,
            "period": 1,
            "frameNum": 100,
            "periodElapsedTime": 1.0,
            "game_event_id": 10,
            "game_event": {"game_event_type": "OTB", "home_team": True},
            "possession_event_id": 20,
            "possession_event": {"possession_event_type": "PA", "start_frame": 100},
            "ballsSmoothed": [{"x": 0.0, "y": 0.0}],
        },
        {
            "gameRefId": 1,
            "period": 1,
            "frameNum": 101,
            "periodElapsedTime": 1.1,
            "game_event_id": 10,
            "game_event": {"game_event_type": "OTB", "home_team": True},
            "possession_event_id": 20,
            "possession_event": {"possession_event_type": "PA", "start_frame": 100},
            "ballsSmoothed": [{"x": 1.0, "y": 0.0}],
        },
        {
            "gameRefId": 1,
            "period": 1,
            "frameNum": 120,
            "periodElapsedTime": 3.0,
            "game_event_id": 11,
            "game_event": {"game_event_type": "OUT"},
            "ballsSmoothed": [{"x": 2.0, "y": 1.0}],
        },
    ]
    source = tmp_path / "1.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    shard = build_pff_event_match_shard(
        source,
        match_id="1",
        split="train",
        statsbomb_manifest=_statsbomb_manifest(),
        source_sha256="source",
    )

    assert shard["provider_code"] == ["PA", "OUT"]
    assert shard["quality"]["excluded_code_counts"] == {"OTB": 1}
    assert shard["quality"]["mapped_event_count"] == 1
    assert shard["quality"]["unknown_event_count"] == 1
    assert shard["categorical"][0, 0] > 1
    assert shard["categorical"][1, 0] == 1
    assert torch.equal(shard["frame_id"], torch.tensor([100, 120]))
    assert shard["continuous"][0, 4] == 1.0


def test_pff_event_history_is_period_aware_and_causal():
    shard = {
        "period": torch.tensor([1, 1, 1, 2]),
        "frame_id": torch.tensor([90, 100, 130, 10]),
        "categorical": torch.tensor(
            [[2, 0, 0, 0, 0], [3, 0, 0, 0, 0], [4, 0, 0, 0, 0], [5, 0, 0, 0, 0]]
        ),
        "continuous": torch.zeros(4, 17),
    }

    history = pff_event_history_from_shard(
        shard,
        period=1,
        cutoff_frame=110,
        sequence_length=4,
    )

    assert history["event_history_size"] == 2
    assert history["event_last_frame"] == 100
    assert history["event_cutoff_frame"] == 110
    assert torch.equal(history["event_categorical"][:2, 0], torch.tensor([2, 3]))
    assert not bool(history["event_mask"][2:].any())
    assert history["raw_event_context"][3] == 1.0
