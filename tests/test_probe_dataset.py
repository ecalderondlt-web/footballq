import torch

from footballq.data.windows import build_tracking_windows, save_windows_pt
from footballq.probes.dataset import ProbeDataset, ProbeDatasetData, build_probe_dataset
from footballq.synthetic.generate import generate_synthetic_tracking


def _synthetic_windows(tmp_path):
    tracking = generate_synthetic_tracking(
        match_id="probe_m0",
        duration_s=5.0,
        fps=10.0,
        seed=11,
    )
    windows = build_tracking_windows(
        tracking,
        fps_out=10.0,
        context_seconds=1.0,
        horizon_seconds=1.0,
        stride_seconds=0.2,
    )
    windows_path = save_windows_pt(windows, tmp_path / "windows.pt")
    return windows, windows_path


def _save_embeddings(path, windows, order=None):
    order = list(range(len(windows.match_id))) if order is None else order
    z = torch.arange(len(order) * 8, dtype=torch.float32).view(len(order), 8)
    torch.save(
        {
            "z": z,
            "match_id": [windows.match_id[idx] for idx in order],
            "frame_t": [windows.start_frame[idx] for idx in order],
            "delta_frames": [2 for _ in order],
            "source_split": "synthetic",
            "config": {},
        },
        path,
    )


def test_probe_dataset_shapes(tmp_path):
    windows, windows_path = _synthetic_windows(tmp_path)
    embeddings_path = tmp_path / "embeddings.pt"
    _save_embeddings(embeddings_path, windows)
    data = build_probe_dataset(
        embeddings_path,
        windows_path,
        target_names=[
            "future_ball_progression_bucket",
            "team_shape_change_bucket",
            "future_ball_displacement_m",
        ],
    )
    n = data.num_examples
    assert data.examples["z"].shape == (n, 8)
    assert data.examples["raw_state_summary"].shape[0] == n
    assert data.metadata["feature_dim"] == 8
    assert set(data.metadata["targets"]) == {
        "future_ball_displacement_m",
        "future_ball_progression_bucket",
        "team_shape_change_bucket",
    }
    for target in data.metadata["targets"]:
        assert data.examples["targets"][target].shape[0] == n
        assert data.examples["target_masks"][target].shape[0] == n
    assert torch.isfinite(data.examples["z"]).all()
    assert torch.isfinite(data.examples["raw_state_summary"]).all()


def test_probe_label_alignment(tmp_path):
    windows, windows_path = _synthetic_windows(tmp_path)
    order = list(reversed(range(min(len(windows.match_id), 5))))
    embeddings_path = tmp_path / "embeddings_reversed.pt"
    _save_embeddings(embeddings_path, windows, order=order)
    data = build_probe_dataset(
        embeddings_path,
        windows_path,
        target_names=["future_ball_displacement_m"],
    )
    assert data.metadata["alignment"] == "sample_id"
    assert data.examples["frame_t"].tolist() == [windows.start_frame[idx] for idx in order]
    assert data.examples["sample_id"] == [windows.sample_id[idx] for idx in order]


def test_probe_builder_preserves_window_metadata_after_alignment(tmp_path):
    tracking = generate_synthetic_tracking(
        match_id="probe_meta",
        duration_s=5.0,
        fps=10.0,
        seed=13,
    )
    tracking["possession_team_id"] = "away"
    windows = build_tracking_windows(
        tracking,
        fps_out=10.0,
        context_seconds=1.0,
        horizon_seconds=1.0,
        stride_seconds=0.2,
    )
    windows_path = save_windows_pt(windows, tmp_path / "windows_meta.pt")
    embeddings_path = tmp_path / "embeddings_meta.pt"
    _save_embeddings(embeddings_path, windows)
    data = build_probe_dataset(
        embeddings_path,
        windows_path,
        target_names=["possession_team"],
    )
    assert set(data.examples["targets"]["possession_team"].tolist()) == {1}


def test_probe_zscore_uses_train_split_only():
    data = ProbeDatasetData(
        metadata={"target_types": {"target": "regression"}},
        examples={
            "z": torch.tensor([[1.0], [3.0], [100.0]]),
            "raw_state_summary": torch.tensor([[2.0], [4.0], [200.0]]),
            "targets": {"target": torch.tensor([0.0, 1.0, 2.0])},
            "target_masks": {"target": torch.tensor([True, True, True])},
        },
        label_maps={},
        splits={"train_indices": [0, 1], "test_indices": [2]},
    )
    dataset = ProbeDataset(data, "target", feature_source="td_jepa_zscore", split="test")
    assert torch.allclose(dataset.features[:2].mean(dim=0), torch.zeros(1))
    assert torch.allclose(dataset.features[:2].std(dim=0, unbiased=False), torch.ones(1))
    assert dataset.features[2].item() > 90.0
