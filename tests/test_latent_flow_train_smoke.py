import csv
import json

import torch

from footballq.latent_flow.ablation import run_latent_flow_ablation
from footballq.latent_flow.dataset import build_latent_rollout_dataset
from footballq.latent_flow.eval import evaluate_latent_baseline, evaluate_latent_checkpoint
from footballq.latent_flow.train import train_latent_flow_from_config


def _write_embeddings(path, matches=3, steps=10, dim=8):
    z_parts = []
    match_id = []
    frame_t = []
    for match_idx in range(matches):
        increments = torch.randn(steps, dim) * 0.03
        z = match_idx + torch.cumsum(increments, dim=0)
        z_parts.append(z)
        match_id.extend([f"smoke_{match_idx}" for _ in range(steps)])
        frame_t.extend(list(range(steps)))
    torch.save(
        {
            "z": torch.cat(z_parts, dim=0),
            "match_id": match_id,
            "frame_t": frame_t,
            "source_split": "synthetic",
            "config": {},
        },
        path,
    )


def _build_dataset(tmp_path):
    embeddings = tmp_path / "embeddings.pt"
    _write_embeddings(embeddings)
    dataset = tmp_path / "rollout.pt"
    build_latent_rollout_dataset(
        embeddings,
        out=dataset,
        context_steps=3,
        horizon_steps=2,
        stride_steps=1,
    )
    return dataset


def _residual_cv_config(dataset, run_root, max_epochs=1, max_train_batches=1):
    return {
        "seed": 123,
        "data": {"rollout_dataset": str(dataset)},
        "model": {
            "name": "residual_latent_flow_mlp",
            "hidden_dim": 16,
            "num_layers": 2,
            "dropout": 0.0,
            "time_embed_dim": 8,
            "conditioning": "past_z_flat",
        },
        "flow": {
            "target_mode": "residual",
            "residual_mode": "constant_latent_velocity",
            "noise_scale": 0.0,
            "num_sampling_steps": 2,
            "deterministic_mean_eval": True,
        },
        "training": {
            "batch_size": 4,
            "max_epochs": max_epochs,
            "max_train_batches": max_train_batches,
            "max_val_batches": 1,
            "save_every_steps": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "device": "cpu",
            "run_root": str(run_root),
        },
        "sampling": {"num_samples": 2, "num_steps": 2, "noise_scale": 0.0},
    }


def test_train_latent_flow_smoke(tmp_path):
    dataset = _build_dataset(tmp_path)
    result = train_latent_flow_from_config(
        {
            "seed": 123,
            "data": {"rollout_dataset": str(dataset)},
            "model": {
                "name": "latent_flow_mlp",
                "hidden_dim": 16,
                "num_layers": 2,
                "dropout": 0.0,
                "time_embed_dim": 8,
            },
            "training": {
                "batch_size": 4,
                "max_epochs": 1,
                "max_train_batches": 1,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "device": "cpu",
                "run_root": str(tmp_path / "runs"),
            },
            "sampling": {"num_samples": 2, "num_steps": 2},
        }
    )
    assert result["best_checkpoint"].exists()
    assert (result["run_dir"] / "eval_test.json").exists()
    payload = torch.load(result["best_checkpoint"], map_location="cpu", weights_only=False)
    assert payload["encoder_frozen"] is True
    assert {"online_encoder", "target_encoder", "motion_encoder"}.isdisjoint(payload)


def test_eval_latent_flow_smoke(tmp_path):
    dataset = _build_dataset(tmp_path)
    baseline = evaluate_latent_baseline(dataset, baseline="last_latent", split="test", device="cpu")
    assert torch.isfinite(torch.tensor(baseline["metrics"]["latent_ADE"]))
    result = train_latent_flow_from_config(
        {
            "seed": 123,
            "data": {"rollout_dataset": str(dataset)},
            "model": {
                "name": "latent_flow_mlp",
                "hidden_dim": 16,
                "num_layers": 2,
                "dropout": 0.0,
                "time_embed_dim": 8,
            },
            "training": {
                "batch_size": 4,
                "max_epochs": 1,
                "max_train_batches": 1,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "device": "cpu",
                "run_root": str(tmp_path / "runs_eval"),
            },
            "sampling": {"num_samples": 2, "num_steps": 2},
        }
    )
    metrics = evaluate_latent_checkpoint(
        result["best_checkpoint"],
        dataset=dataset,
        split="test",
        device="cpu",
        num_samples=2,
        num_steps=2,
    )["metrics"]
    assert torch.isfinite(torch.tensor(metrics["latent_ADE"]))


def test_residual_flow_train_smoke(tmp_path):
    dataset = _build_dataset(tmp_path)
    result = train_latent_flow_from_config(
        {
            "seed": 123,
            "data": {"rollout_dataset": str(dataset)},
            "model": {
                "name": "residual_latent_flow_mlp",
                "hidden_dim": 16,
                "num_layers": 2,
                "dropout": 0.0,
                "time_embed_dim": 8,
                "conditioning": "past_z_flat",
            },
            "flow": {
                "target_mode": "residual",
                "residual_mode": "constant_latent_velocity",
                "noise_scale": 0.0,
                "num_sampling_steps": 2,
                "deterministic_mean_eval": True,
            },
            "training": {
                "batch_size": 4,
                "max_epochs": 1,
                "max_train_batches": 1,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "device": "cpu",
                "run_root": str(tmp_path / "runs_residual"),
            },
            "sampling": {"num_samples": 2, "num_steps": 2, "noise_scale": 0.0},
        }
    )
    assert result["best_checkpoint"].exists()
    assert torch.isfinite(torch.tensor(result["test_metrics"]["latent_ADE"]))


def test_residual_flow_eval_smoke(tmp_path):
    dataset = _build_dataset(tmp_path)
    result = train_latent_flow_from_config(
        {
            "seed": 123,
            "data": {"rollout_dataset": str(dataset)},
            "model": {
                "name": "residual_latent_flow_mlp",
                "hidden_dim": 16,
                "num_layers": 2,
                "dropout": 0.0,
                "time_embed_dim": 8,
                "conditioning": "past_z_flat",
            },
            "flow": {
                "target_mode": "residual",
                "residual_mode": "last_latent",
                "noise_scale": 0.0,
                "num_sampling_steps": 2,
                "deterministic_mean_eval": True,
            },
            "training": {
                "batch_size": 4,
                "max_epochs": 1,
                "max_train_batches": 1,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "device": "cpu",
                "run_root": str(tmp_path / "runs_residual_eval"),
            },
            "sampling": {"num_samples": 2, "num_steps": 2, "noise_scale": 0.0},
        }
    )
    metrics = evaluate_latent_checkpoint(
        result["best_checkpoint"],
        dataset=dataset,
        split="test",
        device="cpu",
        num_samples=2,
        num_steps=2,
    )["metrics"]
    assert torch.isfinite(torch.tensor(metrics["latent_ADE"]))
    assert "residual_ADE" in metrics


def test_checkpoint_reload_residual_flow(tmp_path):
    dataset = _build_dataset(tmp_path)
    result = train_latent_flow_from_config(
        {
            "seed": 123,
            "data": {"rollout_dataset": str(dataset)},
            "model": {
                "name": "residual_latent_flow_mlp",
                "hidden_dim": 16,
                "num_layers": 2,
                "dropout": 0.0,
                "time_embed_dim": 8,
                "conditioning": "past_z_flat",
            },
            "flow": {
                "target_mode": "residual",
                "residual_mode": "constant_latent_velocity",
                "noise_scale": 0.0,
                "num_sampling_steps": 2,
                "deterministic_mean_eval": True,
            },
            "training": {
                "batch_size": 4,
                "max_epochs": 1,
                "max_train_batches": 1,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "device": "cpu",
                "run_root": str(tmp_path / "runs_residual_reload"),
            },
            "sampling": {"num_samples": 2, "num_steps": 2, "noise_scale": 0.0},
        }
    )
    payload = torch.load(result["best_checkpoint"], map_location="cpu", weights_only=False)
    assert payload["model_name"] == "residual_latent_flow_mlp"
    assert payload["residual_mode"] == "constant_latent_velocity"
    assert "residual_mean" in payload["normalization"]


def test_checkpoint_resume_residual_flow(tmp_path):
    dataset = _build_dataset(tmp_path)
    first = train_latent_flow_from_config(
        _residual_cv_config(dataset, tmp_path / "runs_resume", max_epochs=1)
    )
    resumed = train_latent_flow_from_config(
        _residual_cv_config(dataset, tmp_path / "runs_resume", max_epochs=2),
        resume=first["latest_checkpoint"],
    )
    assert resumed["run_dir"] == first["run_dir"]
    payload = torch.load(resumed["latest_checkpoint"], map_location="cpu", weights_only=False)
    assert payload["epoch"] == 2
    assert payload["step"] >= 2
    metrics = evaluate_latent_checkpoint(
        resumed["best_checkpoint"],
        dataset=dataset,
        split="test",
        device="cpu",
        num_samples=2,
        num_steps=2,
        noise_scale=0.0,
    )["metrics"]
    assert torch.isfinite(torch.tensor(metrics["latent_ADE"]))


def test_latent_flow_ablation_writes_results_and_summary(tmp_path):
    dataset = _build_dataset(tmp_path)
    train_result = train_latent_flow_from_config(
        _residual_cv_config(dataset, tmp_path / "runs_ablation", max_epochs=1)
    )
    out = tmp_path / "ablation"
    result = run_latent_flow_ablation(
        _residual_cv_config(dataset, tmp_path / "runs_ablation", max_epochs=1),
        out,
        checkpoint=train_result["best_checkpoint"],
        noise_scales=[0.0, 0.1],
        num_steps=[2],
        num_samples=[2],
        split="test",
        device="cpu",
    )
    assert result["results_csv"].exists()
    assert result["summary_json"].exists()
    with result["results_csv"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["model"] for row in rows} >= {
        "last_latent",
        "constant_latent_velocity",
        "residual_flow_cv",
    }
    stochastic = [
        row for row in rows if row["model"] == "residual_flow_cv" and float(row["noise_scale"]) > 0.0
    ]
    assert stochastic
    assert any(float(row["diversity_mean_pairwise_distance"]) > 0.0 for row in stochastic)
    for row in rows:
        for key in [
            "latent_ADE",
            "latent_FDE",
            "minADE",
            "minFDE",
            "cosine_similarity",
            "diversity_mean_pairwise_distance",
            "sample_std_mean",
        ]:
            assert torch.isfinite(torch.tensor(float(row[key])))
    with result["summary_json"].open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["best_deterministic_baseline_by_latent_ADE"]["model"] in {
        "last_latent",
        "constant_latent_velocity",
    }
    assert summary["best_residual_flow_config_by_minADE"]["model"] == "residual_flow_cv"
