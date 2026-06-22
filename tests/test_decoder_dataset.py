import pandas as pd
import torch

from footballq.data.windows import build_tracking_windows, save_windows_pt
from footballq.decoding.dataset import DecoderDataset, build_decoder_dataset
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
    z = torch.arange(len(windows.match_id) * dim, dtype=torch.float32).view(len(windows.match_id), dim)
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
    assert data.examples["match_id"][0] == windows.match_id[0]
    assert data.examples["label_frame"][0].item() == windows.label_frame[0]
    assert data.metadata["alignment"] == "match_id_frame_t"


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


def test_decoder_dataset_modes_return_expected_targets(tmp_path):
    _, windows_path, embeddings_path = _windows_and_embeddings(tmp_path)
    data = build_decoder_dataset(embeddings_path, windows_path, horizon_steps=4, rollout_steps=2)
    current = DecoderDataset(data, mode="reconstruct_current")[0]
    future = DecoderDataset(data, mode="future_from_z")[0]
    context = DecoderDataset(data, mode="future_from_context")[0]
    rollout = DecoderDataset(data, mode="rollout_from_latents")[0]
    assert current["target_xy"].shape == (23, 2)
    assert future["target_xy"].shape == (4, 23, 2)
    assert context["x"].ndim == 2
    assert rollout["x"].shape[0] == 2
    assert rollout["target_xy"].shape == (2, 23, 2)
