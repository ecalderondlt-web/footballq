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
