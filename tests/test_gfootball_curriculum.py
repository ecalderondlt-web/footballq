import json

import pytest

from footballq.io.gfootball_curriculum import (
    build_episode_split_manifest,
    collection_plan_sha256,
    validate_collection_plan,
)
from scripts.collect_gfootball_tracking import _seed_environment
from scripts.derive_gfootball_collection_subset import derive_collection_subset


def _plan():
    return {
        "name": "grf_plan",
        "version": 1,
        "dataset": "gfootball",
        "creation_timestamp_utc": "2026-07-13T00:00:00Z",
        "jobs": [
            {
                "id": "train_job",
                "split": "train",
                "env_name": "11_vs_11_stochastic",
                "episodes": 2,
                "max_steps": 100,
                "seed": 1,
                "action_policy": "builtin_ai",
            },
            {
                "id": "val_job",
                "split": "val",
                "env_name": "11_vs_11_easy_stochastic",
                "episodes": 1,
                "max_steps": 100,
                "seed": 2,
                "action_policy": "builtin_ai",
            },
            {
                "id": "test_job",
                "split": "test",
                "env_name": "11_vs_11_hard_stochastic",
                "episodes": 1,
                "max_steps": 100,
                "seed": 3,
                "action_policy": "builtin_ai_perturbed",
            },
        ],
    }


def test_collection_plan_builds_disjoint_episode_split():
    plan = _plan()
    split = build_episode_split_manifest(plan)

    assert split["train_match_ids"] == [
        "grf_plan_train_job_episode_0",
        "grf_plan_train_job_episode_1",
    ]
    assert split["val_match_ids"] == ["grf_plan_val_job_episode_0"]
    assert split["test_match_ids"] == ["grf_plan_test_job_episode_0"]
    assert split["expected_count"] == 4
    assert split["source_collection_plan_sha256"] == collection_plan_sha256(plan)


def test_collection_plan_rejects_duplicate_jobs():
    plan = _plan()
    plan["jobs"].append(plan["jobs"][0].copy())

    with pytest.raises(ValueError, match="Duplicate collection job"):
        validate_collection_plan(plan)


def test_collector_seeds_environment_and_action_space():
    class SeedTarget:
        def __init__(self):
            self.value = None

        def seed(self, value):
            self.value = value

    class Environment(SeedTarget):
        def __init__(self):
            super().__init__()
            self.action_space = SeedTarget()

    env = Environment()
    _seed_environment(env, 123)

    assert env.value == 123
    assert env.action_space.value == 123


def test_episode_prefix_subset_preserves_exact_master_lines(tmp_path):
    master_plan = _plan()
    master_plan["name"] = "nested_scale"
    master_plan["jobs"] = [master_plan["jobs"][0]]
    master_plan["jobs"][0]["episodes"] = 4
    subset_plan = json.loads(json.dumps(master_plan))
    subset_plan["jobs"][0]["episodes"] = 3

    master_plan_path = tmp_path / "master_plan.json"
    subset_plan_path = tmp_path / "subset_plan.json"
    master_plan_path.write_text(json.dumps(master_plan), encoding="utf-8")
    subset_plan_path.write_text(json.dumps(subset_plan), encoding="utf-8")
    master_root = tmp_path / "master"
    source_path = master_root / "train" / "train_job.jsonl"
    source_path.parent.mkdir(parents=True)
    lines = []
    for episode_id in range(4):
        for frame_id in range(2):
            lines.append(
                json.dumps(
                    {
                        "episode_id": episode_id,
                        "frame_id": frame_id,
                        "match_id": f"nested_scale_train_job_episode_{episode_id}",
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    source_path.write_text("".join(lines), encoding="utf-8")
    (master_root / "collection_manifest.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )

    output_root = tmp_path / "subset"
    split_path = tmp_path / "subset_split.json"
    manifest_path = derive_collection_subset(
        master_plan_path,
        subset_plan_path,
        master_root,
        output_root,
        split_path,
    )

    target_lines = (output_root / "train" / "train_job.jsonl").read_text(
        encoding="utf-8"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    assert target_lines == "".join(lines[:6])
    assert manifest["derivation"] == "episode_prefix_subset"
    assert manifest["total_frames"] == 6
    assert split["train_match_ids"] == [
        "nested_scale_train_job_episode_0",
        "nested_scale_train_job_episode_1",
        "nested_scale_train_job_episode_2",
    ]
