import pandas as pd
import pytest
import torch

from footballq.data.windows import build_tracking_windows, save_windows_pt
from footballq.decoding.dataset import DecoderDataset, build_decoder_dataset
from footballq.models.constant_velocity import predict_constant_velocity
from footballq.synthetic.generate import generate_synthetic_tracking


def _windows_and_embeddings(tmp_path, matches=3, dim=8):
    frames = []
    for idx in range(matches):
        frame = generate_synthetic_tracking(
            match_id=f"decoder_m{idx}",
            duration_s=5.0,
            fps=5.0,
            seed=idx,
        )
        frame["possession_team_id"] = "home" if idx % 2 == 0 else "away"
        frames.append(frame)
    windows = build_tracking_windows(
        pd.concat(frames, ignore_index=True),
        fps_out=5.0,
        context_seconds=1.0,
        horizon_seconds=1.0,
        stride_seconds=0.2,
    )
    windows_path = save_windows_pt(windows, tmp_path / "windows.pt")
    z = torch.arange(len(windows.match_id) * dim, dtype=torch.float32).view(
        len(windows.match_id), dim
    )
    embeddings_path = tmp_path / "embeddings.pt"
    torch.save(
        {
            "z": z,
            "match_id": windows.match_id,
            "frame_t": windows.start_frame,
            "delta_frames": [1 for _ in windows.match_id],
            "source_split": ["synthetic" for _ in windows.match_id],
            "config": {},
        },
        embeddings_path,
    )
    return windows, windows_path, embeddings_path


def test_decoder_dataset_builds_and_preserves_shapes(tmp_path):
    windows, windows_path, embeddings_path = _windows_and_embeddings(tmp_path)
    data = build_decoder_dataset(
        embeddings_path,
        windows_path,
        horizon_steps=4,
        context_z_steps=3,
        rollout_steps=2,
    )
    assert data.num_examples == len(windows.match_id)
    assert data.examples["current_xy"].shape[1:] == (23, 2)
    assert data.examples["future_xy"].shape[1:] == (4, 23, 2)
    assert data.examples["z_context"].shape[1:] == (3, 8)
    assert data.examples["z_rollout"].shape[1:] == (2, 8)
    assert data.examples["past_context"].shape[0] == data.num_examples
    assert data.examples["z_past_context"].shape[1] == (
        data.examples["past_context"].shape[1] + data.latent_dim
    )
    assert data.examples["match_id"][0] == windows.match_id[0]
    assert data.examples["label_frame"][0].item() == windows.label_frame[0]
    assert data.metadata["alignment"] == "sample_id"
    assert data.examples["sample_id"][0] == windows.sample_id[0]


def test_decoder_dataset_match_splits_are_disjoint_when_possible(tmp_path):
    _, windows_path, embeddings_path = _windows_and_embeddings(tmp_path, matches=3)
    data = build_decoder_dataset(embeddings_path, windows_path)
    train = set(data.splits["train_match_ids"])
    val = set(data.splits["val_match_ids"])
    test = set(data.splits["test_match_ids"])
    assert train
    assert val
    assert test
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


def test_no_silent_match_drop_when_embeddings_missing(tmp_path):
    windows, windows_path, embeddings_path = _windows_and_embeddings(tmp_path, matches=3)
    kept = [idx for idx, match_id in enumerate(windows.match_id) if match_id != "decoder_m2"]
    payload = torch.load(embeddings_path, map_location="cpu", weights_only=False)
    torch.save(
        {
            **payload,
            "z": payload["z"][kept],
            "match_id": [payload["match_id"][idx] for idx in kept],
            "frame_t": [payload["frame_t"][idx] for idx in kept],
            "source_split": [payload["source_split"][idx] for idx in kept],
        },
        tmp_path / "embeddings_missing_match.pt",
    )
    with pytest.raises(ValueError, match="Missing match IDs"):
        build_decoder_dataset(tmp_path / "embeddings_missing_match.pt", windows_path)


def test_decoder_dataset_modes_return_expected_targets(tmp_path):
    _, windows_path, embeddings_path = _windows_and_embeddings(tmp_path)
    data = build_decoder_dataset(embeddings_path, windows_path, horizon_steps=4, rollout_steps=2)
    current = DecoderDataset(data, mode="reconstruct_current")[0]
    future = DecoderDataset(data, mode="future_from_z")[0]
    context = DecoderDataset(data, mode="future_from_context")[0]
    rollout = DecoderDataset(data, mode="rollout_from_latents")[0]
    context_only = DecoderDataset(data, mode="future_from_past_context")[0]
    z_context = DecoderDataset(data, mode="future_from_z_past_context")[0]
    residual = DecoderDataset(data, mode="residual_future_from_z_past_context")[0]
    assert current["target_xy"].shape == (23, 2)
    assert future["target_xy"].shape == (4, 23, 2)
    assert context["x"].ndim == 2
    assert rollout["x"].shape[0] == 2
    assert rollout["target_xy"].shape == (2, 23, 2)
    assert context_only["x"].ndim == 1
    assert z_context["x"].shape[0] > context_only["x"].shape[0]
    assert residual["coordinate_baseline_xy"].shape == (4, 23, 2)


def test_coordinate_constant_velocity_baseline_alignment(tmp_path):
    _, windows_path, embeddings_path = _windows_and_embeddings(tmp_path)
    data = build_decoder_dataset(embeddings_path, windows_path, horizon_steps=4)
    expected = predict_constant_velocity(
        data.examples["past"],
        data.examples["past_mask"],
        horizon_steps=4,
        dt=1.0 / float(data.metadata["fps"]),
        feature_names=data.metadata["feature_names"],
    )
    assert torch.allclose(data.examples["coordinate_baseline_xy"], expected)


def test_decoder_dataset_supports_multiple_horizon_lengths(tmp_path):
    frames = []
    for idx in range(3):
        frames.append(
            generate_synthetic_tracking(
                match_id=f"decoder_horizon_{idx}",
                duration_s=8.0,
                fps=5.0,
                seed=idx,
            )
        )
    for horizon_steps, horizon_seconds in [(4, 0.8), (8, 1.6), (12, 2.4)]:
        windows = build_tracking_windows(
            pd.concat(frames, ignore_index=True),
            fps_out=5.0,
            context_seconds=1.0,
            horizon_seconds=horizon_seconds,
            stride_seconds=0.2,
        )
        windows_path = save_windows_pt(windows, tmp_path / f"windows_{horizon_steps}.pt")
        embeddings_path = tmp_path / f"embeddings_{horizon_steps}.pt"
        torch.save(
            {
                "z": torch.randn(len(windows.match_id), 8),
                "match_id": windows.match_id,
                "frame_t": windows.start_frame,
                "source_split": ["synthetic" for _ in windows.match_id],
                "config": {},
            },
            embeddings_path,
        )
        data = build_decoder_dataset(
            embeddings_path,
            windows_path,
            horizon_steps=horizon_steps,
        )
        assert data.horizon_steps == horizon_steps
        assert data.examples["future_xy"].shape[1] == horizon_steps
        assert data.examples["coordinate_baseline_xy"].shape[1] == horizon_steps
