import torch

from footballq.models.soccer_state_encoder import SoccerStateEncoder
from footballq.models.td_jepa import MotionEncoder, SoccerTDJEPA
from footballq.training.ema import update_ema
from footballq.training.td_jepa_losses import td_jepa_loss


def _batch(batch_size: int = 4):
    state = torch.randn(batch_size, 3, 23, 10)
    mask = torch.ones(batch_size, 3, 23, dtype=torch.bool)
    delta = torch.randn(batch_size, 2, 23, 10)
    delta_mask = torch.ones(batch_size, 2, 23, dtype=torch.bool)
    return state, mask, delta, delta_mask


def test_state_and_motion_encoder_shapes():
    state, mask, delta, delta_mask = _batch()
    encoder = SoccerStateEncoder(3, 23, 10, z_dim=16, d_model=32, n_heads=4, n_layers=1)
    z = encoder(state, mask)
    assert z.shape == (4, 16)
    motion = MotionEncoder(2, 23, 10, z_dim=16, hidden_dim=32)
    delta_z = motion(delta, delta_mask, z)
    assert delta_z.shape == (4, 16)


def test_state_encoder_cls_pooling_shapes_and_rejects_bad_pooling():
    state, mask, _, _ = _batch()
    encoder = SoccerStateEncoder(
        3,
        23,
        10,
        z_dim=16,
        d_model=32,
        n_heads=4,
        n_layers=1,
        pooling="cls",
    )
    assert encoder(state, mask).shape == (4, 16)
    try:
        SoccerStateEncoder(3, 23, 10, pooling="unknown")
    except ValueError as exc:
        assert "pooling" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected invalid pooling to raise ValueError.")


def test_td_jepa_forward_and_loss_are_finite():
    state, mask, delta, delta_mask = _batch()
    model = SoccerTDJEPA(
        context_steps=3,
        delta_steps=2,
        n_entities=23,
        n_features=10,
        z_dim=16,
        d_model=32,
        n_heads=4,
        n_layers=1,
        motion_hidden_dim=32,
    )
    outputs = model(
        {
            "state_t": state,
            "mask_t": mask,
            "state_t_plus_delta": state + 0.01,
            "mask_t_plus_delta": mask,
            "delta_state": delta,
            "delta_mask": delta_mask,
        }
    )
    losses = td_jepa_loss(outputs["z_pred"], outputs["z_target"], outputs["z_t"])
    assert {"total_loss", "td_loss", "anti_collapse_loss"}.issubset(losses)
    assert all(torch.isfinite(value) for value in losses.values())


def test_td_jepa_forward_with_cls_pooling_is_finite():
    state, mask, delta, delta_mask = _batch()
    model = SoccerTDJEPA(
        context_steps=3,
        delta_steps=2,
        n_entities=23,
        n_features=10,
        z_dim=16,
        d_model=32,
        n_heads=4,
        n_layers=1,
        motion_hidden_dim=32,
        pooling="cls",
    )
    outputs = model(
        {
            "state_t": state,
            "mask_t": mask,
            "state_t_plus_delta": state + 0.01,
            "mask_t_plus_delta": mask,
            "delta_state": delta,
            "delta_mask": delta_mask,
        }
    )
    assert torch.isfinite(outputs["z_pred"]).all()
    assert torch.isfinite(outputs["z_target"]).all()


def test_td_jepa_forward_with_state_decoder_outputs_slot_reconstruction():
    state, mask, delta, delta_mask = _batch()
    model = SoccerTDJEPA(
        context_steps=3,
        delta_steps=2,
        n_entities=23,
        n_features=10,
        z_dim=16,
        d_model=32,
        n_heads=4,
        n_layers=1,
        motion_hidden_dim=32,
        state_decoder_hidden_dim=32,
    )
    outputs = model(
        {
            "state_t": state,
            "mask_t": mask,
            "state_t_plus_delta": state + 0.01,
            "mask_t_plus_delta": mask,
            "delta_state": delta,
            "delta_mask": delta_mask,
        }
    )
    assert outputs["state_reconstruction"].shape == state.shape
    assert outputs["context_reconstruction"].shape == state.shape
    assert model.decode_state(outputs["z_t"]).shape == state.shape


def test_target_encoder_has_no_grad_and_ema_updates():
    model = SoccerTDJEPA(3, 2, 23, 10, z_dim=16, d_model=32, n_heads=4, n_layers=1)
    assert not any(parameter.requires_grad for parameter in model.target_encoder.parameters())
    before = [parameter.clone() for parameter in model.target_encoder.parameters()]
    with torch.no_grad():
        for parameter in model.online_encoder.parameters():
            parameter.add_(0.1)
    update_ema(model.target_encoder, model.online_encoder, momentum=0.5)
    after = list(model.target_encoder.parameters())
    assert any(not torch.allclose(old, new) for old, new in zip(before, after, strict=True))
