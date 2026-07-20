import json
from pathlib import Path

import torch
from test_pff_forecasting import build_source_manifest

from footballq.data.pff_forecasting import prepare_pff_forecast_targets
from footballq.models.td_jepa import SoccerTDJEPA
from footballq.training.train_trajectory_forecast import train_trajectory_forecast_from_config


def test_trajectory_forecast_training_is_validation_only(tmp_path):
    source_manifest, split_path = build_source_manifest(tmp_path)
    forecast_root = tmp_path / "forecast"
    prepare_pff_forecast_targets(
        source_manifest,
        forecast_root,
        split_path,
        horizons_seconds=(0.1, 0.2),
        included_splits=("train",),
    )
    model_cfg = {
        "z_dim": 8,
        "d_model": 8,
        "n_heads": 2,
        "n_layers": 1,
        "dropout": 0.0,
        "motion_hidden_dim": 16,
    }
    source_model = SoccerTDJEPA(
        context_steps=2,
        delta_steps=1,
        n_entities=23,
        n_features=5,
        z_dim=8,
        d_model=8,
        n_heads=2,
        n_layers=1,
        dropout=0.0,
        motion_hidden_dim=16,
    )
    checkpoint = tmp_path / "tracking.pt"
    torch.save(
        {
            "config": {"model": model_cfg},
            "online_encoder": source_model.online_encoder.state_dict(),
            "step": 10,
        },
        checkpoint,
    )
    config = {
        "data": {
            "manifest": str(forecast_root / "dataset_manifest.json"),
            "split_manifest": str(split_path),
        },
        "sources": {"tracking_checkpoint": str(checkpoint)},
        "model": {"hidden_dim": 16, "dropout": 0.0},
        "training": {
            "seed": 7,
            "batch_size": 2,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "max_train_updates": 2,
            "validation_curve_steps": [],
            "max_val_batches": 2,
            "max_epochs": 2,
            "num_workers": 0,
            "device": "cpu",
            "validation_split": "train",
            "run_root": str(tmp_path / "runs"),
        },
    }
    result = train_trajectory_forecast_from_config(config, family="frozen")
    manifest = json.loads(Path(result["run_manifest"]).read_text(encoding="utf-8"))

    assert manifest["loaded_splits"] == ["train"]
    assert manifest["test_loaded"] is False
    assert result["metrics"]["num_examples"] == 3
    assert torch.isfinite(torch.tensor(result["metrics"]["player_ADE_m"]))
