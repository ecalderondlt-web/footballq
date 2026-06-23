import json

import pandas as pd
import torch

from footballq.data.windows import build_tracking_windows, save_windows_pt
from footballq.decoding.dataset import build_decoder_dataset, save_decoder_dataset
from footballq.decoding.eval import evaluate_decoder_checkpoint
from footballq.decoding.train import train_coordinate_decoder_from_config
from footballq.synthetic.generate import generate_synthetic_tracking


def _decoder_dataset_path(tmp_path):
    frames = []
    for idx in range(3):
        frames.append(
            generate_synthetic_tracking(
                match_id=f"decoder_train_{idx}",
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


def _config(dataset_path, tmp_path, mode="future_from_z", name="linear"):
    return {
        "seed": 123,
        "data": {"decoder_dataset": str(dataset_path), "batch_size": 4, "num_workers": 0},
        "target": {"mode": mode},
        "model": {
            "name": name,
            "hidden_sizes": [16] if name != "linear" else [],
            "dropout": 0.0,
            "pooling": "mean" if name == "context_mlp" else "flatten",
        },
        "training": {
            "epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "device": "cpu",
            "run_root": str(tmp_path / "runs"),
            "max_train_batches": 1,
            "max_eval_batches": 1,
        },
    }


def test_reconstruction_decoder_train_eval_reload(tmp_path):
    dataset_path = _decoder_dataset_path(tmp_path)
    result = train_coordinate_decoder_from_config(
        _config(dataset_path, tmp_path, mode="reconstruct_current", name="linear")
    )
    assert result["best_checkpoint"].exists()
    assert (result["run_dir"] / "eval_test.json").exists()
    metrics = evaluate_decoder_checkpoint(
        result["best_checkpoint"],
        dataset=dataset_path,
        split="test",
        device="cpu",
    )["metrics"]
    assert torch.isfinite(torch.tensor(metrics["current_all_entity_error_m"]))
    with (result["run_dir"] / "eval_test.json").open("r", encoding="utf-8") as handle:
        assert json.load(handle)


def test_future_decoder_train_eval_reload(tmp_path):
    dataset_path = _decoder_dataset_path(tmp_path)
    result = train_coordinate_decoder_from_config(
        _config(dataset_path, tmp_path, mode="future_from_context", name="context_mlp")
    )
    assert result["latest_checkpoint"].exists()
    metrics = evaluate_decoder_checkpoint(
        result["best_checkpoint"],
        dataset=dataset_path,
        split="test",
        device="cpu",
    )["metrics"]
    assert torch.isfinite(torch.tensor(metrics["all_entity_ADE_m"]))
    payload = torch.load(result["best_checkpoint"], map_location="cpu", weights_only=False)
    assert payload["encoder_frozen"] is True


def test_residual_context_future_decoder_train_eval_reload(tmp_path):
    dataset_path = _decoder_dataset_path(tmp_path)
    result = train_coordinate_decoder_from_config(
        _config(
            dataset_path,
            tmp_path,
            mode="residual_future_from_z_past_context",
            name="residual_context_mlp",
        )
    )
    metrics = evaluate_decoder_checkpoint(
        result["best_checkpoint"],
        dataset=dataset_path,
        split="test",
        device="cpu",
    )["metrics"]
    assert torch.isfinite(torch.tensor(metrics["all_entity_ADE_m"]))
    sample = torch.load(result["run_dir"] / "predictions_sample.pt", map_location="cpu", weights_only=False)
    assert "raw_decoder_output_xy_norm" in sample
