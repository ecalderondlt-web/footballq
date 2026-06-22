import torch

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
