import torch

from footballq.latent_flow.baselines import (
    constant_latent_velocity_predict,
    last_latent_predict,
)
from footballq.latent_flow.flow_matching import flow_matching_loss, sample_latent_flow
from footballq.latent_flow.models import LatentFlowMLP


def test_last_latent_baseline_shapes():
    past = torch.randn(4, 3, 5)
    pred = last_latent_predict(past, horizon_steps=2)
    assert pred.shape == (4, 2, 5)
    assert torch.equal(pred[:, 0], past[:, -1])
    assert torch.equal(pred[:, 1], past[:, -1])


def test_constant_latent_velocity_baseline_shapes():
    past = torch.zeros(2, 3, 4)
    past[:, -2] = 1.0
    past[:, -1] = 3.0
    pred = constant_latent_velocity_predict(past, horizon_steps=2)
    assert pred.shape == (2, 2, 4)
    assert torch.allclose(pred[:, 0], torch.full((2, 4), 5.0))
    assert torch.allclose(pred[:, 1], torch.full((2, 4), 7.0))


def test_latent_flow_forward_shapes():
    model = LatentFlowMLP(
        latent_dim=6,
        context_steps=3,
        horizon_steps=2,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
        time_embed_dim=8,
    )
    past = torch.randn(5, 3, 6)
    x_t = torch.randn(5, 2, 6)
    t = torch.rand(5)
    out = model(past, x_t, t)
    assert out.shape == (5, 2, 6)


def test_flow_matching_loss_finite():
    model = LatentFlowMLP(
        latent_dim=6,
        context_steps=3,
        horizon_steps=2,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
        time_embed_dim=8,
    )
    loss = flow_matching_loss(
        model,
        torch.randn(5, 3, 6),
        torch.randn(5, 2, 6),
        torch.ones(5, 2, dtype=torch.bool),
    )
    assert torch.isfinite(loss)


def test_latent_flow_sampling_shapes():
    model = LatentFlowMLP(
        latent_dim=6,
        context_steps=3,
        horizon_steps=2,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
        time_embed_dim=8,
    )
    samples = sample_latent_flow(
        model,
        torch.randn(4, 3, 6),
        horizon_steps=2,
        latent_dim=6,
        num_samples=3,
        num_steps=2,
    )
    assert samples.shape == (4, 3, 2, 6)
    assert torch.isfinite(samples).all()
