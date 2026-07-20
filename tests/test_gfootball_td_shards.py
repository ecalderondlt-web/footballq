import json

import torch
from test_gfootball import _obs

from footballq.data.gfootball_td_shards import (
    prepare_gfootball_td_jepa_shards,
    project_gfootball_feature_view,
)
from footballq.data.gfootball_td_subset import derive_gfootball_td_episode_subset
from footballq.data.sharded_td_dataset import ShardedTDJEPADataset
from footballq.data.td_jepa_dataset import TDJEPAData, save_td_jepa_data
from footballq.io.gfootball_curriculum import (
    collection_plan_sha256,
    write_episode_split_manifest,
)
from scripts.evaluate_provider_neutral_preflight import (
    compare_train_tensor_subset_invariants,
)


def test_gfootball_curriculum_prepares_hashed_lazy_shards(tmp_path):
    plan = {
        "name": "grf_shard_test",
        "version": 1,
        "dataset": "gfootball",
        "creation_timestamp_utc": "2026-07-13T00:00:00Z",
        "jobs": [
            {
                "id": f"{split}_job",
                "split": split,
                "env_name": "11_vs_11_stochastic",
                "episodes": 1,
                "max_steps": 30,
                "seed": seed,
                "action_policy": "builtin_ai",
            }
            for split, seed in (("train", 1), ("val", 2), ("test", 3))
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    split_path = tmp_path / "split.json"
    write_episode_split_manifest(plan, split_path)
    raw_root = tmp_path / "raw"
    for job in plan["jobs"]:
        path = raw_root / job["split"] / f"{job['id']}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for frame_id in range(30):
            record = _obs(frame_id)
            record["match_id"] = f"grf_shard_test_{job['id']}_episode_0"
            if job["split"] == "train" and frame_id == 14:
                record["observation"]["game_mode"] = 1
                record["observation"]["left_team"][0][0] = 0.8
            record["observation"]["ball_direction"] = [0.01, 0.0, 0.0]
            record["observation"]["left_team_direction"] = [[0.01, 0.0]] * 11
            record["observation"]["right_team_direction"] = [[-0.01, 0.0]] * 11
            rows.append(json.dumps(record))
        path.write_text("\n".join(rows), encoding="utf-8")
    profile = {
        "profile": "test_visibility",
        "profile_payload_sha256": "profile-hash",
        "observed_player_count_probabilities": {"8": 1.0},
        "player_observed_rate": 0.4,
        "ball_observed_rate": 0.5,
        "distance_bin_edges_m": [0.0, 20.0, 110.0],
        "distance_bin_observed_probabilities": [0.8, 0.1],
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    manifest_path = prepare_gfootball_td_jepa_shards(
        plan_path,
        raw_root,
        tmp_path / "processed",
        split_path,
        visibility_profile_path=profile_path,
        context_seconds=0.2,
        delta_seconds=0.1,
        stride_seconds=0.2,
        prediction_gap_seconds=0.1,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = ShardedTDJEPADataset(manifest_path, "train", verify_hashes_on_load=True)

    assert manifest["status"] == "complete"
    assert manifest["example_count"] == manifest["unique_sample_id_count"]
    assert manifest["split_example_counts"]["train"] > 0
    assert manifest["unsegmented_example_count"] == manifest["example_count"]
    assert manifest["example_retention_fraction"] == 1.0
    assert len(manifest["shards"]) == 3
    assert dataset[0]["state_t"].shape[-2:] == (23, 7)
    assert dataset[0]["mask_t"].sum() <= 18

    neutral_manifest_path = prepare_gfootball_td_jepa_shards(
        plan_path,
        raw_root,
        tmp_path / "processed_neutral",
        split_path,
        visibility_profile_path=profile_path,
        context_seconds=0.2,
        delta_seconds=0.1,
        stride_seconds=0.2,
        prediction_gap_seconds=0.1,
        velocity_mode="causal_position_difference",
    )
    neutral_manifest = json.loads(neutral_manifest_path.read_text(encoding="utf-8"))
    provider_payload = torch.load(
        tmp_path / "processed" / "test_visibility" / "train" / "train_job" / "td_jepa.pt",
        map_location="cpu",
        weights_only=False,
    )
    neutral_payload = torch.load(
        tmp_path
        / "processed_neutral"
        / "test_visibility"
        / "train"
        / "train_job"
        / "td_jepa.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert neutral_manifest["velocity_mode"] == "causal_position_difference"
    assert neutral_manifest["config"]["velocity_mode"] == "causal_position_difference"
    assert provider_payload["sample_id"] == neutral_payload["sample_id"]
    assert torch.equal(provider_payload["mask_t"], neutral_payload["mask_t"])
    assert torch.equal(
        provider_payload["context_frame_indices"], neutral_payload["context_frame_indices"]
    )
    assert torch.equal(provider_payload["state_t"][..., :2], neutral_payload["state_t"][..., :2])
    assert not torch.equal(
        provider_payload["state_t"][..., 2:4], neutral_payload["state_t"][..., 2:4]
    )

    event_manifest_path = prepare_gfootball_td_jepa_shards(
        plan_path,
        raw_root,
        tmp_path / "processed_event_segmented",
        split_path,
        visibility_profile_path=profile_path,
        context_seconds=0.2,
        delta_seconds=0.1,
        stride_seconds=0.2,
        prediction_gap_seconds=0.1,
        velocity_mode="event_segmented_causal_position_difference",
        included_splits={"train"},
    )
    event_manifest = json.loads(event_manifest_path.read_text(encoding="utf-8"))
    event_payload = torch.load(
        tmp_path
        / "processed_event_segmented"
        / "test_visibility"
        / "train"
        / "train_job"
        / "td_jepa.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert event_manifest["included_splits"] == ["train"]
    assert len(event_manifest["shards"]) == 1
    assert event_manifest["split_example_counts"]["val"] == 0
    assert event_manifest["split_example_counts"]["test"] == 0
    assert event_manifest["event_boundary_totals"]["unsafe_frame_count"] == 12
    assert event_manifest["unsegmented_example_count"] == manifest["split_example_counts"]["train"]
    assert event_manifest["shards"][0]["unsafe_tensor_frame_reference_count"] == 0
    assert set(event_payload["sample_id"]) < set(provider_payload["sample_id"])
    provider_indices = {
        sample_id: index for index, sample_id in enumerate(provider_payload["sample_id"])
    }
    selected_provider_indices = torch.tensor(
        [provider_indices[sample_id] for sample_id in event_payload["sample_id"]]
    )
    assert torch.equal(
        provider_payload["mask_t"][selected_provider_indices], event_payload["mask_t"]
    )
    assert torch.equal(
        provider_payload["state_t"][selected_provider_indices, ..., :2],
        event_payload["state_t"][..., :2],
    )
    referenced_frames = torch.cat(
        [event_payload["context_frame_indices"], event_payload["target_frame_indices"]],
        dim=1,
    )
    assert not torch.isin(referenced_frames, torch.arange(9, 21)).any()
    subset_invariants = compare_train_tensor_subset_invariants(
        manifest_path, event_manifest_path
    )
    assert subset_invariants["passed"]
    assert 0.0 < subset_invariants["retention_fraction"] < 1.0

    resumed_manifest_path = prepare_gfootball_td_jepa_shards(
        plan_path,
        raw_root,
        tmp_path / "processed_event_segmented",
        split_path,
        visibility_profile_path=profile_path,
        context_seconds=0.2,
        delta_seconds=0.1,
        stride_seconds=0.2,
        prediction_gap_seconds=0.1,
        velocity_mode="event_segmented_causal_position_difference",
        included_splits={"train"},
        resume_existing=True,
    )
    resumed_manifest = json.loads(resumed_manifest_path.read_text(encoding="utf-8"))
    assert resumed_manifest["shards"][0]["resumed_existing"]
    assert resumed_manifest["split_example_counts"] == event_manifest["split_example_counts"]

    lagged_manifest_path = prepare_gfootball_td_jepa_shards(
        plan_path,
        raw_root,
        tmp_path / "processed_jump_lagged",
        split_path,
        visibility_profile_path=profile_path,
        context_seconds=0.2,
        delta_seconds=0.1,
        stride_seconds=0.2,
        prediction_gap_seconds=0.1,
        velocity_mode="jump_segmented_causal_position_difference_0p5s",
        included_splits={"train"},
    )
    lagged_manifest = json.loads(lagged_manifest_path.read_text(encoding="utf-8"))
    lagged_payload = torch.load(
        tmp_path
        / "processed_jump_lagged"
        / "test_visibility"
        / "train"
        / "train_job"
        / "td_jepa.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert lagged_manifest["jump_boundary_totals"]["boundary_frame_count"] == 2
    assert lagged_manifest["unsegmented_example_count"] == manifest["split_example_counts"]["train"]
    assert lagged_manifest["causal_velocity_lag_frames"] == 5
    assert lagged_manifest["shards"][0]["boundary_crossing_tensor_example_count"] == 0
    assert torch.equal(lagged_payload["state_t"][0, 0, :, 2:4], torch.zeros(23, 2))
    lagged_invariants = compare_train_tensor_subset_invariants(
        manifest_path, lagged_manifest_path
    )
    assert lagged_invariants["passed"]

    position_manifest_path = project_gfootball_feature_view(
        lagged_manifest_path,
        tmp_path / "processed_jump_position_only",
        target_feature_view="position_only",
    )
    position_manifest = json.loads(position_manifest_path.read_text(encoding="utf-8"))
    position_payload = torch.load(
        tmp_path
        / "processed_jump_position_only"
        / "test_visibility"
        / "train"
        / "train_job"
        / "td_jepa.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert position_manifest["feature_names"] == [
        "x_norm",
        "y_norm",
        "is_ball",
        "is_home",
        "is_away",
    ]
    assert position_payload["sample_id"] == lagged_payload["sample_id"]
    assert torch.equal(position_payload["mask_t"], lagged_payload["mask_t"])
    assert position_payload["state_t"].shape[-1] == 5
    position_invariants = compare_train_tensor_subset_invariants(
        manifest_path, position_manifest_path
    )
    assert position_invariants["passed"]

    direct_position_manifest_path = prepare_gfootball_td_jepa_shards(
        plan_path,
        raw_root,
        tmp_path / "processed_jump_position_only_direct",
        split_path,
        visibility_profile_path=profile_path,
        context_seconds=0.2,
        delta_seconds=0.1,
        stride_seconds=0.2,
        prediction_gap_seconds=0.1,
        velocity_mode="jump_segmented_causal_position_difference_0p5s",
        feature_view="position_only",
        included_splits={"train"},
    )
    direct_position_manifest = json.loads(
        direct_position_manifest_path.read_text(encoding="utf-8")
    )
    direct_position_payload = torch.load(
        tmp_path
        / "processed_jump_position_only_direct"
        / "test_visibility"
        / "train"
        / "train_job"
        / "td_jepa.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert not direct_position_manifest["velocity_features_materialized"]
    assert not direct_position_payload["metadata"]["velocity_features_materialized"]
    for tensor_name in (
        "state_t",
        "state_t_plus_delta",
        "delta_state",
        "mask_t",
        "mask_t_plus_delta",
        "delta_mask",
        "entity_type",
        "team_id",
        "context_frame_indices",
        "target_frame_indices",
        "delta_frame_indices",
    ):
        assert torch.equal(position_payload[tensor_name], direct_position_payload[tensor_name])
    assert position_payload["sample_id"] == direct_position_payload["sample_id"]


def test_gfootball_td_episode_subset_is_an_exact_tensor_prefix(tmp_path):
    master_plan = {
        "name": "nested_td_scale",
        "version": 1,
        "dataset": "gfootball",
        "creation_timestamp_utc": "2026-07-15T00:00:00Z",
        "jobs": [
            {
                "id": "train_job",
                "split": "train",
                "env_name": "11_vs_11_stochastic",
                "episodes": 4,
                "max_steps": 30,
                "seed": 1,
                "action_policy": "builtin_ai",
            }
        ],
    }
    subset_plan = json.loads(json.dumps(master_plan))
    subset_plan["jobs"][0]["episodes"] = 3
    master_plan_path = tmp_path / "master_plan.json"
    subset_plan_path = tmp_path / "subset_plan.json"
    master_plan_path.write_text(json.dumps(master_plan), encoding="utf-8")
    subset_plan_path.write_text(json.dumps(subset_plan), encoding="utf-8")
    split_path = tmp_path / "subset_split.json"
    write_episode_split_manifest(subset_plan, split_path)

    match_ids = [
        f"nested_td_scale_train_job_episode_{episode}"
        for episode in range(4)
        for _ in range(2)
    ]
    sample_ids = [f"sample-{index}" for index in range(8)]
    state = torch.arange(8 * 2 * 23 * 5, dtype=torch.float32).reshape(8, 2, 23, 5)
    data = TDJEPAData(
        state_t=state,
        state_t_plus_delta=state + 1,
        delta_state=torch.zeros((8, 1, 23, 5)),
        mask_t=torch.ones((8, 2, 23), dtype=torch.bool),
        mask_t_plus_delta=torch.ones((8, 2, 23), dtype=torch.bool),
        delta_mask=torch.zeros((8, 1, 23), dtype=torch.bool),
        entity_type=torch.zeros((8, 23), dtype=torch.long),
        team_id=torch.zeros((8, 23), dtype=torch.long),
        match_id=match_ids,
        period=[1] * 8,
        frame_t=list(range(8)),
        sample_id=sample_ids,
        delta_frames=1,
        feature_names=["x_norm", "y_norm", "is_ball", "is_home", "is_away"],
        fps=10.0,
        context_seconds=0.2,
        delta_seconds=0.1,
        stride_seconds=0.1,
        objective_mode="future_nonoverlap_context_only",
        prediction_gap_frames=1,
        feature_view="position_only",
        context_frame_indices=torch.arange(16).reshape(8, 2),
        target_frame_indices=torch.arange(16, 32).reshape(8, 2),
        delta_frame_indices=torch.full((8, 1), -1),
        metadata={
            "unsegmented_example_counts_by_match": {
                f"nested_td_scale_train_job_episode_{episode}": 2
                for episode in range(4)
            },
            "unsegmented_example_count": 8,
            "jump_boundary_summary": {"boundary_frame_count": 1},
        },
    )
    master_root = tmp_path / "master_tensors"
    tensor_path = master_root / "profile" / "train" / "train_job" / "td_jepa.pt"
    save_td_jepa_data(data, tensor_path)
    master_manifest_path = master_root / "profile" / "dataset_manifest.json"
    master_manifest = {
        "status": "complete",
        "version": 1,
        "dataset": "gfootball",
        "profile": "profile",
        "collection_plan_sha256": collection_plan_sha256(master_plan),
        "included_splits": ["train"],
        "velocity_mode": "jump_segmented_causal_position_difference_0p5s",
        "config": {"feature_view": "position_only", "included_splits": ["train"]},
        "example_count": 8,
        "unique_sample_id_count": 8,
        "split_example_counts": {"train": 8, "val": 0, "test": 0},
        "unsegmented_example_count": 8,
        "example_retention_fraction": 1.0,
        "jump_boundary_totals": {"boundary_frame_count": 1},
        "shards": [
            {
                "path": "profile/train/train_job/td_jepa.pt",
                "split": "train",
                "job_id": "train_job",
                "tensor_sha256": "master-tensor-hash",
                "unsafe_tensor_frame_reference_count": 0,
                "boundary_crossing_tensor_example_count": 0,
            }
        ],
        "tensor_hashes_complete": True,
        "manifest_payload_sha256": "master-manifest-hash",
    }
    master_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    master_manifest_path.write_text(json.dumps(master_manifest), encoding="utf-8")
    collection_manifest_path = tmp_path / "subset_collection.json"
    collection_manifest_path.write_text(
        json.dumps(
            {
                "collection_plan_sha256": collection_plan_sha256(subset_plan),
                "jobs": [
                    {
                        "id": "train_job",
                        "path": "subset/train/train_job.jsonl",
                        "sha256": "subset-raw-hash",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    subset_manifest_path = derive_gfootball_td_episode_subset(
        master_manifest_path,
        master_plan_path,
        subset_plan_path,
        collection_manifest_path,
        tmp_path / "subset_tensors",
        split_path,
    )
    subset_manifest = json.loads(subset_manifest_path.read_text(encoding="utf-8"))
    subset_data = ShardedTDJEPADataset(subset_manifest_path, "train").prototype

    assert subset_manifest["example_count"] == 6
    assert subset_manifest["unsegmented_example_count"] == 6
    assert subset_manifest["shards"][0]["source_tensor_sha256"] == "master-tensor-hash"
    assert subset_data.sample_id == sample_ids[:6]
    assert torch.equal(subset_data.state_t, state[:6])
