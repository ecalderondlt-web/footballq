import pandas as pd

from footballq.data.windows import build_tracking_windows, save_windows_pt
from footballq.probes.dataset import build_probe_dataset, split_probe_indices_by_match
from footballq.probes.labels import derive_probe_targets
from footballq.synthetic.generate import generate_synthetic_tracking

import torch


def _save_embeddings(path, windows):
    z = torch.randn(len(windows.match_id), 8)
    torch.save(
        {
            "z": z,
            "match_id": list(windows.match_id),
            "frame_t": list(windows.start_frame),
            "delta_frames": [2 for _ in windows.match_id],
            "source_split": "synthetic",
            "config": {},
        },
        path,
    )


def test_probe_split_by_match_id():
    match_ids = ["m0", "m0", "m1", "m1", "m2", "m2"]
    splits, warnings = split_probe_indices_by_match(match_ids, seed=7)
    assert warnings == []
    train = set(splits["train_match_ids"])
    val = set(splits["val_match_ids"])
    test = set(splits["test_match_ids"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


def test_probe_label_derivation_has_finite_supported_targets():
    tracking = generate_synthetic_tracking(duration_s=5.0, fps=10.0)
    windows = build_tracking_windows(
        tracking,
        fps_out=10.0,
        context_seconds=1.0,
        horizon_seconds=1.0,
        stride_seconds=0.2,
    )
    derived = derive_probe_targets(
        windows,
        [
            "future_ball_progression_bucket",
            "team_shape_change_bucket",
            "future_ball_displacement_m",
            "phase",
        ],
    )
    assert "phase" not in derived.targets
    assert any("phase" in warning for warning in derived.warnings)
    assert {"future_ball_progression_bucket", "team_shape_change_bucket"}.issubset(
        derived.label_maps
    )
    assert derived.masks["future_ball_displacement_m"].any()


def test_probe_possession_labels_use_window_metadata():
    tracking = generate_synthetic_tracking(duration_s=5.0, fps=10.0)
    tracking["possession_team_id"] = "home"
    windows = build_tracking_windows(
        tracking,
        fps_out=10.0,
        context_seconds=1.0,
        horizon_seconds=1.0,
        stride_seconds=0.2,
    )
    derived = derive_probe_targets(
        windows,
        ["possession_team", "has_ball_or_possession_available"],
    )
    assert set(derived.targets["possession_team"].tolist()) == {0}
    assert set(derived.targets["has_ball_or_possession_available"].tolist()) == {1}


def test_probe_dataset_with_three_matches_has_disjoint_match_splits(tmp_path):
    frames = [
        generate_synthetic_tracking(match_id=f"probe_m{idx}", duration_s=5.0, fps=10.0, seed=idx)
        for idx in range(3)
    ]
    windows = build_tracking_windows(
        pd.concat(frames, ignore_index=True),
        fps_out=10.0,
        context_seconds=1.0,
        horizon_seconds=1.0,
        stride_seconds=0.2,
    )
    windows_path = save_windows_pt(windows, tmp_path / "windows.pt")
    embeddings_path = tmp_path / "embeddings.pt"
    _save_embeddings(embeddings_path, windows)
    data = build_probe_dataset(
        embeddings_path,
        windows_path,
        target_names=["future_ball_progression_bucket"],
    )
    train = set(data.splits["train_match_ids"])
    val = set(data.splits["val_match_ids"])
    test = set(data.splits["test_match_ids"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
