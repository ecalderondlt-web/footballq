import json

import pandas as pd
import torch

from footballq.data.windows import build_tracking_windows, save_windows_pt
from footballq.probes.dataset import build_probe_dataset, save_probe_dataset
from footballq.probes.training import train_probe_from_config
from footballq.synthetic.generate import generate_synthetic_tracking


def _probe_dataset_path(tmp_path):
    frames = [
        generate_synthetic_tracking(
            match_id=f"probe_smoke_{idx}",
            duration_s=4.0,
            fps=5.0,
            seed=idx,
        )
        for idx in range(3)
    ]
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
            "z": torch.randn(len(windows.match_id), 16),
            "match_id": list(windows.match_id),
            "frame_t": list(windows.start_frame),
            "delta_frames": [1 for _ in windows.match_id],
            "source_split": "synthetic",
            "config": {},
        },
        tmp_path / "embeddings.pt",
    )
    data = build_probe_dataset(
        tmp_path / "embeddings.pt",
        windows_path,
        target_names=[
            "future_ball_progression_bucket",
            "future_ball_displacement_m",
        ],
    )
    out = tmp_path / "probe_dataset.pt"
    save_probe_dataset(data, out)
    return out


def _config(tmp_path, dataset_path, target, task_type):
    return {
        "seed": 123,
        "data": {"probe_dataset": str(dataset_path)},
        "target": {"name": target, "task_type": task_type},
        "features": {"source": "td_jepa", "random_seed": 123},
        "model": {"probe_type": "linear", "hidden_dim": 16, "dropout": 0.0},
        "training": {
            "batch_size": 4,
            "max_epochs": 2,
            "patience": 1,
            "max_train_batches": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "device": "cpu",
            "seed": 123,
            "run_root": str(tmp_path / "runs"),
        },
    }


def test_probe_train_smoke_classification(tmp_path):
    dataset_path = _probe_dataset_path(tmp_path)
    result = train_probe_from_config(
        _config(
            tmp_path,
            dataset_path,
            "future_ball_progression_bucket",
            "classification",
        )
    )
    assert result["best_checkpoint"].exists()
    assert (result["run_dir"] / "eval_test.json").exists()
    metrics = json.loads((result["run_dir"] / "eval_test.json").read_text())
    assert metrics["match_level"]["group_key"] == "match_id"
    assert metrics["match_level"]["summary"]["count"] >= 1


def test_probe_train_smoke_regression(tmp_path):
    dataset_path = _probe_dataset_path(tmp_path)
    result = train_probe_from_config(
        _config(
            tmp_path,
            dataset_path,
            "future_ball_displacement_m",
            "regression",
        )
    )
    assert result["best_checkpoint"].exists()
    assert (result["run_dir"] / "predictions_sample.pt").exists()


def test_no_encoder_finetuning(tmp_path):
    dataset_path = _probe_dataset_path(tmp_path)
    result = train_probe_from_config(
        _config(
            tmp_path,
            dataset_path,
            "future_ball_progression_bucket",
            "classification",
        )
    )
    payload = torch.load(result["best_checkpoint"], map_location="cpu", weights_only=False)
    forbidden = {"online_encoder", "target_encoder", "motion_encoder"}
    assert forbidden.isdisjoint(payload)
    assert payload["encoder_frozen"] is True
