from copy import deepcopy

import pandas as pd
import pytest
import torch

from footballq.data.td_jepa_dataset import build_td_jepa_examples, save_td_jepa_data
from footballq.synthetic.generate import generate_synthetic_tracking
from footballq.training.eval_td_jepa import evaluate_td_checkpoint
from footballq.training.export_td_embeddings import export_td_embeddings
from footballq.training.train_td_jepa import train_td_jepa_from_config


def test_td_jepa_one_batch_train_eval_and_export(tmp_path):
    frames = [
        generate_synthetic_tracking(match_id=f"td{idx}", duration_s=2.5, fps=10.0, seed=idx)
        for idx in range(2)
    ]
    data = build_td_jepa_examples(
        pd.concat(frames, ignore_index=True),
        fps_out=10.0,
        context_seconds=0.5,
        delta_seconds=0.2,
        stride_seconds=0.2,
    )
    data_path = save_td_jepa_data(data, tmp_path / "td.pt")
    cfg = {
        "experiment": "td_jepa_smoke",
        "seed": 7,
        "data": {"path": str(data_path), "source": "synthetic"},
        "split": {"val_fraction": 0.5, "test_fraction": 0.5},
        "model": {
            "z_dim": 16,
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 4,
            "dropout": 0.0,
            "motion_hidden_dim": 32,
            "state_decoder_hidden_dim": 32,
            "temporal_motion_head_hidden_dim": 32,
            "transition_decoder_hidden_dim": 32,
        },
        "ema": {"momentum": 0.9},
        "loss": {
            "variance_weight": 0.05,
            "variance_threshold": 0.2,
            "slot_reconstruction_weight": 0.1,
            "context_reconstruction_weight": 0.1,
            "no_motion_margin_weight": 0.1,
            "no_motion_margin": 0.01,
            "temporal_motion_weight": 0.25,
            "match_invariance_weight": 0.05,
            "transition_reconstruction_weight": 0.25,
        },
        "training": {
            "seed": 7,
            "batch_size": 8,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "max_epochs": 1,
            "max_train_batches": 1,
            "num_workers": 0,
            "device": "cpu",
            "run_root": str(tmp_path / "runs"),
        },
    }
    result = train_td_jepa_from_config(cfg)
    assert result["latest_checkpoint"].exists()
    metrics = evaluate_td_checkpoint(result["latest_checkpoint"], split="test", device="cpu")
    assert torch.isfinite(torch.tensor(metrics["metrics"]["total_loss"]))
    assert "slot_reconstruction_loss" in metrics["metrics"]
    assert "context_reconstruction_loss" in metrics["metrics"]
    assert "no_motion_margin_loss" in metrics["metrics"]
    assert "base_total_loss" in metrics["metrics"]
    assert "temporal_motion_loss" in metrics["metrics"]
    assert "temporal_motion_cosine_similarity" in metrics["metrics"]
    assert "match_invariance_loss" in metrics["metrics"]
    assert "transition_reconstruction_loss" in metrics["metrics"]
    out = export_td_embeddings(
        result["latest_checkpoint"],
        data_path,
        tmp_path / "embeddings.pt",
        split="test",
        device="cpu",
    )
    payload = torch.load(out, map_location="cpu", weights_only=False)
    assert payload["z"].shape[1] == 16
    assert len(payload["match_id"]) == payload["z"].shape[0]
    assert len(payload["frame_t"]) == payload["z"].shape[0]
    assert payload["source_split"] == "test"

    all_out = export_td_embeddings(
        result["latest_checkpoint"],
        data_path,
        tmp_path / "embeddings_all.pt",
        split="all",
        device="cpu",
    )
    all_payload = torch.load(all_out, map_location="cpu", weights_only=False)
    assert all_payload["z"].shape[0] == len(data.match_id)
    assert len(all_payload["source_split"]) == all_payload["z"].shape[0]
    assert set(all_payload["source_split"]).issubset({"train", "val", "test"})

    finetune_cfg = deepcopy(cfg)
    finetune_cfg["experiment"] = "td_jepa_transfer_smoke"
    finetune_cfg["training"]["run_root"] = str(tmp_path / "finetune_runs")
    finetune_result = train_td_jepa_from_config(
        finetune_cfg,
        init_checkpoint=result["best_checkpoint"],
    )
    finetune_payload = torch.load(
        finetune_result["latest_checkpoint"], map_location="cpu", weights_only=False
    )
    initialization = finetune_payload["initialization"]
    assert initialization["mode"] == "pretrained_weights_fresh_optimizer"
    assert initialization["source_dataset"] == "synthetic"
    assert initialization["checkpoint_sha256"]
    assert initialization["loaded_components"] == [
        "online_encoder",
        "target_encoder",
        "motion_encoder",
        "state_decoder",
        "temporal_motion_head",
        "transition_decoder",
    ]
    assert finetune_payload["step"] == 1

    incompatible_cfg = deepcopy(cfg)
    incompatible_cfg["model"]["n_heads"] = 2
    with pytest.raises(ValueError, match="n_heads"):
        train_td_jepa_from_config(
            incompatible_cfg,
            init_checkpoint=result["best_checkpoint"],
        )


def test_td_jepa_honors_global_update_budget_and_drops_partial_train_batch(tmp_path):
    frames = [
        generate_synthetic_tracking(match_id=f"budget{idx}", duration_s=2.5, fps=10.0, seed=idx)
        for idx in range(4)
    ]
    data = build_td_jepa_examples(
        pd.concat(frames, ignore_index=True),
        fps_out=10.0,
        context_seconds=0.5,
        delta_seconds=0.2,
        stride_seconds=0.2,
    )
    data_path = save_td_jepa_data(data, tmp_path / "budget.pt")
    cfg = {
        "experiment": "td_jepa_update_budget_smoke",
        "seed": 7,
        "data": {"path": str(data_path), "source": "synthetic"},
        "split": {"val_fraction": 0.25, "test_fraction": 0.25},
        "model": {
            "z_dim": 8,
            "d_model": 16,
            "n_layers": 1,
            "n_heads": 2,
            "dropout": 0.0,
            "motion_hidden_dim": 16,
        },
        "training": {
            "batch_size": 4,
            "max_epochs": 10,
            "max_train_updates": 3,
            "drop_last_train": True,
            "num_workers": 0,
            "device": "cpu",
            "run_root": str(tmp_path / "budget_runs"),
        },
    }

    result = train_td_jepa_from_config(cfg)
    checkpoint = torch.load(result["latest_checkpoint"], map_location="cpu", weights_only=False)

    assert checkpoint["step"] == 3
