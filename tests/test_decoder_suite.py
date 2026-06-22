import csv

import pandas as pd
import torch

from footballq.data.windows import build_tracking_windows, save_windows_pt
from footballq.decoding.dataset import build_decoder_dataset, save_decoder_dataset
from footballq.decoding.suite import run_decoder_suite
from footballq.synthetic.generate import generate_synthetic_tracking


def _decoder_dataset_path(tmp_path):
    frames = []
    for idx in range(3):
        frames.append(
            generate_synthetic_tracking(
                match_id=f"decoder_suite_{idx}",
                duration_s=5.0,
                fps=5.0,
                seed=idx,
            )
        )
    windows = build_tracking_windows(
        pd.concat(frames, ignore_index=True),
        fps_out=5.0,
        context_seconds=1.0,
        horizon_seconds=1.0,
        stride_seconds=0.2,
    )
    windows_path = save_windows_pt(windows, tmp_path / "windows.pt")
    torch.save(
        {
            "z": torch.randn(len(windows.match_id), 8),
            "match_id": windows.match_id,
            "frame_t": windows.start_frame,
            "delta_frames": [1 for _ in windows.match_id],
            "source_split": ["synthetic" for _ in windows.match_id],
            "config": {},
        },
        tmp_path / "embeddings.pt",
    )
    data = build_decoder_dataset(
        tmp_path / "embeddings.pt",
        windows_path,
        horizon_steps=4,
        context_z_steps=3,
        rollout_steps=2,
    )
    return save_decoder_dataset(data, tmp_path / "decoder_dataset.pt")


def test_decoder_suite_writes_results_and_summary(tmp_path):
    dataset_path = _decoder_dataset_path(tmp_path)
    result = run_decoder_suite(
        dataset_path,
        tmp_path / "suite",
        split="test",
        device="cpu",
        epochs=1,
        max_train_batches=1,
        max_eval_batches=1,
        batch_size=4,
        run_root=tmp_path / "runs",
    )
    assert result["results_csv"].exists()
    assert result["summary_json"].exists()
    with result["results_csv"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    models = {row["model"] for row in rows}
    assert "coordinate_constant_velocity" in models
    assert "linear_future_from_z" in models
    assert "last_latent_rollout_decoded" in models
    for row in rows:
        assert torch.isfinite(torch.tensor(float(row["all_entity_ADE_m"])))
