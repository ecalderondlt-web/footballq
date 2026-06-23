import csv

import pandas as pd
import torch

from footballq.data.windows import build_tracking_windows, save_windows_pt
from footballq.decoding.dataset import (
    build_decoder_dataset,
    decoder_split_diagnostics,
    save_decoder_dataset,
    subset_decoder_dataset,
)
from footballq.decoding.learning_curve import run_decoder_learning_curve
from footballq.synthetic.generate import generate_synthetic_tracking


def _decoder_dataset(tmp_path, matches=3):
    frames = []
    for idx in range(matches):
        frames.append(
            generate_synthetic_tracking(
                match_id=f"decoder_curve_{idx}",
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
    return build_decoder_dataset(
        tmp_path / "embeddings.pt",
        windows_path,
        horizon_steps=4,
        context_z_steps=3,
        rollout_steps=2,
    )


def test_decoder_split_diagnostics_are_disjoint_with_three_matches(tmp_path):
    data = _decoder_dataset(tmp_path, matches=3)
    diagnostics = decoder_split_diagnostics(data)
    assert diagnostics["num_matches"] == 3
    assert diagnostics["disjoint_match_split"] is True
    assert diagnostics["smoke_split"] is False


def test_subset_smoke_split_warns_with_one_match(tmp_path):
    data = _decoder_dataset(tmp_path, matches=3)
    first_match = data.splits["train_match_ids"][0]
    indices = [
        idx for idx, match_id in enumerate(data.examples["match_id"]) if match_id == first_match
    ]
    subset = subset_decoder_dataset(data, indices)
    diagnostics = decoder_split_diagnostics(subset)
    assert diagnostics["num_matches"] == 1
    assert diagnostics["smoke_split"] is True
    assert subset.metadata["subset_warnings"]


def test_decoder_learning_curve_writes_results(tmp_path):
    dataset_path = save_decoder_dataset(_decoder_dataset(tmp_path, matches=3), tmp_path / "decoder.pt")
    result = run_decoder_learning_curve(
        dataset_path,
        tmp_path / "learning_curve",
        match_counts=[1, 3],
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
    assert "raw_past_summary_mlp" in models
    assert "context_only_decoder" in models
    assert "z_plus_context_decoder" in models
    assert "residual_context_only_decoder" in models
    assert "residual_z_plus_context_decoder" in models
    assert any(row["smoke_split"] == "True" for row in rows)
    for row in rows:
        assert row["finite_metrics"] == "True"
        assert "current_team_centroid_error_m" in row
    assert result["summary"]["subset_diagnostics"]
