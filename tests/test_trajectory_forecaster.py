import torch

from footballq.models.soccer_state_encoder import SoccerStateEncoder
from footballq.models.trajectory_forecaster import (
    MultiHorizonTrajectoryForecaster,
    last_observed_kinematics,
    predict_constant_velocity,
)
from footballq.training.train_trajectory_forecast import masked_displacement_loss_m


def _context():
    state = torch.zeros(2, 3, 4, 5)
    state[:, :, :, 0] = torch.tensor([0.0, 0.1, 0.2]).view(1, 3, 1)
    state[:, :, :, 1] = torch.tensor([0.0, -0.1, -0.2]).view(1, 3, 1)
    mask = torch.ones(2, 3, 4, dtype=torch.bool)
    return state, mask


def test_constant_velocity_uses_last_two_observations():
    state, mask = _context()
    last_xy, velocity, has_last = last_observed_kinematics(state, mask, fps=10.0)
    prediction = predict_constant_velocity(state, mask, (0.5, 1.0), fps=10.0)

    assert bool(has_last.all())
    assert torch.allclose(last_xy[0, 0], torch.tensor([0.2, -0.2]))
    assert torch.allclose(velocity[0, 0], torch.tensor([1.0, -1.0]))
    assert torch.allclose(prediction[0, :, 0], torch.tensor([[0.7, -0.7], [1.2, -1.2]]))


def test_frozen_forecaster_only_trains_decoder():
    encoder = SoccerStateEncoder(
        context_steps=3,
        n_entities=4,
        n_features=5,
        z_dim=8,
        d_model=8,
        n_heads=2,
        n_layers=1,
        dropout=0.0,
    )
    model = MultiHorizonTrajectoryForecaster(
        encoder,
        family="frozen",
        z_dim=8,
        n_entities=4,
        horizons_seconds=(0.5, 1.0),
        fps=10.0,
        hidden_dim=16,
        dropout=0.0,
    )
    state, mask = _context()
    target = torch.zeros(2, 2, 4, 2)
    future_mask = torch.ones(2, 2, 4, dtype=torch.bool)
    loss = masked_displacement_loss_m(model(state, mask), target, future_mask)
    loss.backward()

    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.decoder.parameters())
    assert torch.isfinite(loss)


def test_entity_token_forecaster_preserves_entity_axis_and_gradient_boundary():
    encoder = SoccerStateEncoder(
        context_steps=3,
        n_entities=4,
        n_features=5,
        z_dim=8,
        d_model=8,
        n_heads=2,
        n_layers=1,
        dropout=0.0,
    )
    model = MultiHorizonTrajectoryForecaster(
        encoder,
        family="frozen",
        z_dim=8,
        n_entities=4,
        horizons_seconds=(0.5, 1.0),
        fps=10.0,
        hidden_dim=16,
        dropout=0.0,
        representation_mode="entity_tokens",
        token_dim=8,
    )
    state, mask = _context()
    prediction = model(state, mask)
    loss = prediction.square().mean()
    loss.backward()

    assert encoder.encode_entity_tokens(state, mask).shape == (2, 4, 8)
    assert prediction.shape == (2, 2, 4, 2)
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.decoder.parameters())


def test_player_ball_decoder_specializes_canonical_ball_slot():
    encoder = SoccerStateEncoder(
        context_steps=3,
        n_entities=4,
        n_features=5,
        z_dim=8,
        d_model=8,
        n_heads=2,
        n_layers=1,
        dropout=0.0,
    )
    model = MultiHorizonTrajectoryForecaster(
        encoder,
        family="raw",
        z_dim=8,
        n_entities=4,
        horizons_seconds=(0.5, 1.0),
        fps=10.0,
        hidden_dim=16,
        dropout=0.0,
        representation_mode="entity_tokens",
        token_dim=8,
        decoder_mode="player_ball",
    )
    state = torch.zeros(2, 3, 4, 5)
    mask = torch.ones(2, 3, 4, dtype=torch.bool)
    with torch.no_grad():
        for parameter in model.decoder.parameters():
            parameter.zero_()
        model.decoder["ball"][-1].bias.fill_(1.0)

    prediction = model(state, mask)

    assert prediction.shape == (2, 2, 4, 2)
    assert torch.allclose(prediction[:, :, 0], torch.ones(2, 2, 2))
    assert torch.allclose(prediction[:, :, 1:], torch.zeros(2, 2, 3, 2))


def test_player_ball_decoder_trains_both_heads():
    encoder = SoccerStateEncoder(
        context_steps=3,
        n_entities=4,
        n_features=5,
        z_dim=8,
        d_model=8,
        n_heads=2,
        n_layers=1,
        dropout=0.0,
    )
    model = MultiHorizonTrajectoryForecaster(
        encoder,
        family="frozen",
        z_dim=8,
        n_entities=4,
        horizons_seconds=(0.5, 1.0),
        fps=10.0,
        hidden_dim=16,
        dropout=0.0,
        representation_mode="entity_tokens",
        token_dim=8,
        decoder_mode="player_ball",
    )
    state, mask = _context()
    model(state, mask).square().mean().backward()

    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.decoder["player"].parameters())
    assert any(parameter.grad is not None for parameter in model.decoder["ball"].parameters())


def test_hybrid_ball_decoder_receives_all_entity_kinematics():
    encoder = SoccerStateEncoder(
        context_steps=3,
        n_entities=4,
        n_features=5,
        z_dim=8,
        d_model=8,
        n_heads=2,
        n_layers=1,
        dropout=0.0,
    )
    model = MultiHorizonTrajectoryForecaster(
        encoder,
        family="raw",
        z_dim=8,
        n_entities=4,
        horizons_seconds=(0.5, 1.0),
        fps=10.0,
        hidden_dim=16,
        dropout=0.0,
        representation_mode="entity_tokens",
        token_dim=8,
        decoder_mode="player_global_ball",
    )
    state, mask = _context()
    captured = []
    hook = model.decoder["ball"].register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone())
    )

    prediction = model(state, mask)
    hook.remove()
    last_xy, velocity, has_last = last_observed_kinematics(state, mask, fps=10.0)
    expected = torch.cat(
        [last_xy, velocity, has_last.unsqueeze(-1).to(state.dtype)], dim=-1
    ).reshape(2, -1)

    assert prediction.shape == (2, 2, 4, 2)
    assert model.decoder["player"][0].in_features == 13
    assert model.decoder["ball"][0].in_features == 28
    assert captured[0].shape == (2, 1, 28)
    assert torch.allclose(captured[0][:, 0, 8:], expected)
