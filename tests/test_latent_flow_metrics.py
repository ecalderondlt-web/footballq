import torch

from footballq.latent_flow.metrics import compute_latent_rollout_metrics


def test_latent_metrics_known_values():
    target = torch.zeros(2, 2, 3)
    pred = torch.ones(2, 2, 3)
    mask = torch.ones(2, 2, dtype=torch.bool)
    metrics = compute_latent_rollout_metrics(pred, target, mask)
    assert abs(metrics["latent_ADE"] - (3.0**0.5)) < 1e-6
    assert abs(metrics["latent_FDE"] - (3.0**0.5)) < 1e-6
    assert metrics["latent_step_mse"] == 1.0
    assert metrics["minADE_8"] == metrics["latent_ADE"]
    assert metrics["minFDE_8"] == metrics["latent_FDE"]


def test_minade_minfde_metrics():
    target = torch.zeros(1, 2, 2)
    bad = torch.ones(1, 1, 2, 2) * 3.0
    good = torch.zeros(1, 1, 2, 2)
    predictions = torch.cat([bad, good], dim=1)
    metrics = compute_latent_rollout_metrics(
        predictions,
        target,
        torch.ones(1, 2, dtype=torch.bool),
    )
    assert metrics["latent_ADE"] > 0.0
    assert metrics["minADE"] == 0.0
    assert metrics["minFDE"] == 0.0
    assert metrics["minADE_4"] == 0.0
    assert metrics["minFDE_4"] == 0.0
    assert metrics["sample_std_mean"] > 0.0
    assert metrics["diversity_mean_pairwise_distance"] > 0.0
    for value in metrics.values():
        if isinstance(value, float):
            assert torch.isfinite(torch.tensor(value))
