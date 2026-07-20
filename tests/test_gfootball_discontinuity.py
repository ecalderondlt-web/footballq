import hashlib
import json
from pathlib import Path

import pytest

from footballq.analysis.gfootball_discontinuity import (
    run_gfootball_position_discontinuity_audit,
)
from footballq.repro.splits import split_manifest_sha256


def _observation(x: float, *, game_mode: int = 0, score=(0, 0)):
    return [
        {
            "ball": [0.0, 0.0, 0.0],
            "left_team": [[x, 0.0]],
            "left_team_active": [1],
            "right_team": [[0.4, 0.0]],
            "right_team_active": [1],
            "game_mode": game_mode,
            "score": list(score),
            "steps_left": 100,
        }
    ]


def _fixture(tmp_path: Path, *, event_near_jump: bool = True):
    match_id = "pilot_train_episode_0"
    source = tmp_path / "data" / "raw" / "gfootball" / "v2_pilot" / "train" / "job.jsonl"
    source.parent.mkdir(parents=True)
    positions = [0.0, 0.001, 0.002, 0.5, 0.501, 0.502]
    records = []
    for frame_id, x in enumerate(positions):
        records.append(
            {
                "collection_job_id": "job_train",
                "env_name": "scenario",
                "split": "train",
                "match_id": match_id,
                "episode_id": 0,
                "frame_id": frame_id,
                "time_s": frame_id / 10.0,
                "observation": _observation(
                    x,
                    game_mode=1 if event_near_jump and frame_id == 3 else 0,
                ),
            }
        )
    source.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    collection = {
        "collection_plan_sha256": "plan-sha",
        "jobs": [
            {
                "id": "job_train",
                "split": "train",
                "env_name": "scenario",
                "path": str(source.relative_to(tmp_path)),
                "sha256": source_sha256,
            },
            {
                "id": "job_val",
                "split": "val",
                "env_name": "held_out",
                "path": "missing-val-file.jsonl",
                "sha256": "not-read",
            },
        ],
    }
    split = {
        "name": "test",
        "version": 1,
        "dataset": "gfootball",
        "protocol": "inductive_episode_holdout",
        "train_match_ids": [match_id],
        "val_match_ids": ["pilot_val_episode_0"],
        "test_match_ids": [],
        "creation_timestamp_utc": "2026-07-13T00:00:00Z",
        "seed": 1,
        "source_description": "unit test",
    }
    collection_path = tmp_path / "collection.json"
    split_path = tmp_path / "split.json"
    collection_path.write_text(json.dumps(collection), encoding="utf-8")
    split_path.write_text(json.dumps(split), encoding="utf-8")
    return collection_path, split_path, split_manifest_sha256(split), records


def test_discontinuity_audit_is_train_only_and_attributes_event_spike(tmp_path):
    collection, split, split_sha256, _ = _fixture(tmp_path)

    result = run_gfootball_position_discontinuity_audit(
        collection,
        split,
        repo_root=tmp_path,
        expected_collection_plan_sha256="plan-sha",
        expected_split_manifest_sha256=split_sha256,
    )

    assert result["scope"] == "train_only_raw_grf"
    assert result["inputs"]["held_out_jobs_read"] == []
    assert result["inputs"]["train_frame_count"] == 6
    assert result["global_event_attribution"]["players"]["extreme_count"] >= 1
    assert result["global_event_attribution"]["players"]["event_proximate_mass_share"] == 1.0
    assert (
        result["decision"]["selected_next_candidate"]
        == "event_boundary_segmentation_or_masking"
    )
    assert result["top_player_accelerations"][0]["game_mode_change_nearby"]


def test_discontinuity_audit_selects_generic_jump_without_recorded_event(tmp_path):
    collection, split, split_sha256, _ = _fixture(tmp_path, event_near_jump=False)

    result = run_gfootball_position_discontinuity_audit(
        collection,
        split,
        repo_root=tmp_path,
        expected_collection_plan_sha256="plan-sha",
        expected_split_manifest_sha256=split_sha256,
    )

    assert result["global_event_attribution"]["players"]["event_proximate_mass_share"] == 0.0
    assert result["global_event_attribution"]["players"]["jump_associated_mass_share"] == 1.0
    assert result["decision"]["selected_next_candidate"] == "generic_jump_boundary_mask"


def test_discontinuity_audit_rejects_non_train_record(tmp_path):
    collection, split, split_sha256, records = _fixture(tmp_path)
    manifest = json.loads(collection.read_text(encoding="utf-8"))
    source = tmp_path / manifest["jobs"][0]["path"]
    records[2]["split"] = "val"
    source.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    manifest["jobs"][0]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    collection.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Non-train record"):
        run_gfootball_position_discontinuity_audit(
            collection,
            split,
            repo_root=tmp_path,
            expected_collection_plan_sha256="plan-sha",
            expected_split_manifest_sha256=split_sha256,
        )
