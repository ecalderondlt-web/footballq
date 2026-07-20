import json

import torch

from scripts.audit_gfootball_td_nesting import compare_nested_manifests


def _write_manifest(root, name, sample_ids):
    tensor_root = root / name / "profile" / "train" / "job"
    tensor_root.mkdir(parents=True)
    values = torch.tensor([[float(ord(sample_id))] for sample_id in sample_ids])
    payload = {
        "state_t": values,
        "state_t_plus_delta": values + 1,
        "delta_state": values + 2,
        "mask_t": values.bool(),
        "mask_t_plus_delta": values.bool(),
        "delta_mask": values.bool(),
        "entity_type": values.long(),
        "team_id": values.long(),
        "context_frame_indices": values.long(),
        "target_frame_indices": values.long(),
        "delta_frame_indices": values.long(),
        "match_id": ["match"] * len(sample_ids),
        "period": [1] * len(sample_ids),
        "frame_t": list(range(len(sample_ids))),
        "sample_id": sample_ids,
        "feature_names": ["x_norm", "y_norm", "is_ball", "is_home", "is_away"],
    }
    torch.save(payload, tensor_root / "td_jepa.pt")
    manifest = {
        "included_splits": ["train"],
        "split_example_counts": {"train": len(sample_ids), "val": 0, "test": 0},
        "visibility_profile_sha256": "visibility",
        "velocity_mode": "jump_segmented_causal_position_difference_0p5s",
        "config": {"feature_view": "position_only"},
        "example_count": len(sample_ids),
        "example_retention_fraction": 0.9,
        "shards": [
            {
                "job_id": "job",
                "split": "train",
                "path": "profile/train/job/td_jepa.pt",
            }
        ],
    }
    path = root / name / "profile" / "dataset_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_nested_manifest_audit_checks_every_tensor_field(tmp_path):
    parent = _write_manifest(tmp_path, "parent", ["a", "b", "c"])
    child = _write_manifest(tmp_path, "child", ["a", "b"])

    result = compare_nested_manifests(parent, child)

    assert result["passed"]
    assert result["child_examples"] == 2

    child_tensor = tmp_path / "child" / "profile" / "train" / "job" / "td_jepa.pt"
    payload = torch.load(child_tensor, map_location="cpu", weights_only=False)
    payload["team_id"][0] += 1
    torch.save(payload, child_tensor)

    corrupted = compare_nested_manifests(parent, child)
    assert not corrupted["passed"]
    assert "job:team_id" in corrupted["failed_checks"]
