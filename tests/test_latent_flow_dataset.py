import torch

from footballq.latent_flow.baselines import (
    constant_latent_velocity_predict,
    denormalize_residual,
    last_latent_predict,
    normalize_residual,
)
from footballq.latent_flow.dataset import (
    add_residual_targets,
    build_latent_rollout_dataset,
    split_latent_indices_by_match,
)


def _write_latent_embeddings(path, matches=3, steps=12, dim=6):
    z_parts = []
    match_id = []
    frame_t = []
    source_split = []
    split_names = ["train", "val", "test"]
    for match_idx in range(matches):
        base = torch.randn(dim) + match_idx
        increments = torch.randn(steps, dim) * 0.05
        z = base + torch.cumsum(increments, dim=0)
        z_parts.append(z)
        match_id.extend([f"m{match_idx}" for _ in range(steps)])
        frame_t.extend(list(range(steps)))
        source_split.extend([split_names[min(match_idx, 2)] for _ in range(steps)])
    torch.save(
        {
            "z": torch.cat(z_parts, dim=0),
            "match_id": match_id,
            "frame_t": frame_t,
            "source_split": source_split,
            "config": {},
        },
        path,
    )


def test_latent_rollout_dataset_shapes(tmp_path):
    path = tmp_path / "embeddings.pt"
    _write_latent_embeddings(path, matches=2, steps=12, dim=6)
    data = build_latent_rollout_dataset(
        path,
        context_steps=3,
        horizon_steps=2,
        stride_steps=1,
    )
    assert data.examples["past_z"].shape[1:] == (3, 6)
    assert data.examples["future_z"].shape[1:] == (2, 6)
    assert data.examples["future_mask"].shape[1:] == (2,)
    assert data.metadata["latent_dim"] == 6


def test_latent_rollout_dataset_does_not_cross_match_boundaries(tmp_path):
    path = tmp_path / "embeddings.pt"
    _write_latent_embeddings(path, matches=2, steps=8, dim=4)
    data = build_latent_rollout_dataset(path, context_steps=3, horizon_steps=3)
    counts = {
        match_id: data.examples["match_id"].count(match_id)
        for match_id in set(data.examples["match_id"])
    }
    assert counts == {"m0": 3, "m1": 3}
    assert set(data.examples["frame_t"].tolist()) == {0, 1, 2}


def test_latent_rollout_split_by_match_id():
    splits, warnings = split_latent_indices_by_match(["m0", "m0", "m1", "m1", "m2", "m2"])
    assert warnings == []
    train = set(splits["train_match_ids"])
    val = set(splits["val_match_ids"])
    test = set(splits["test_match_ids"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


def test_residual_latent_dataset_shapes(tmp_path):
    path = tmp_path / "embeddings.pt"
    _write_latent_embeddings(path, matches=3, steps=12, dim=6)
    data = build_latent_rollout_dataset(
        path,
        context_steps=3,
        horizon_steps=2,
        residual_mode="constant_latent_velocity",
    )
    assert data.examples["baseline_future_z"].shape == data.examples["future_z"].shape
    assert data.examples["residual_future_z"].shape == data.examples["future_z"].shape
    assert data.metadata["normalization"]["residual_mode"] == "constant_latent_velocity"
    assert data.metadata["normalization"]["residual_mean"].shape == (6,)


def test_last_latent_baseline_residual_computation(tmp_path):
    path = tmp_path / "embeddings.pt"
    _write_latent_embeddings(path, matches=3, steps=8, dim=4)
    data = build_latent_rollout_dataset(path, context_steps=3, horizon_steps=2, residual_mode=None)
    data = add_residual_targets(data, "last_latent")
    expected = last_latent_predict(data.examples["past_z"], data.horizon_steps)
    assert torch.allclose(data.examples["baseline_future_z"], expected)
    assert torch.allclose(
        data.examples["residual_future_z"],
        data.examples["future_z"] - expected,
    )


def test_constant_velocity_baseline_residual_computation(tmp_path):
    path = tmp_path / "embeddings.pt"
    _write_latent_embeddings(path, matches=3, steps=8, dim=4)
    data = build_latent_rollout_dataset(path, context_steps=3, horizon_steps=2, residual_mode=None)
    data = add_residual_targets(data, "constant_latent_velocity")
    expected = constant_latent_velocity_predict(data.examples["past_z"], data.horizon_steps)
    assert torch.allclose(data.examples["baseline_future_z"], expected)
    assert torch.allclose(
        data.examples["residual_future_z"],
        data.examples["future_z"] - expected,
    )


def test_residual_normalization_roundtrip(tmp_path):
    path = tmp_path / "embeddings.pt"
    _write_latent_embeddings(path, matches=3, steps=12, dim=5)
    data = build_latent_rollout_dataset(
        path,
        context_steps=3,
        horizon_steps=2,
        residual_mode="last_latent",
    )
    stats = data.metadata["normalization"]
    residual = data.examples["residual_future_z"]
    norm = normalize_residual(residual, stats["residual_mean"], stats["residual_std"])
    restored = denormalize_residual(norm, stats["residual_mean"], stats["residual_std"])
    assert torch.allclose(restored, residual, atol=1e-6)
