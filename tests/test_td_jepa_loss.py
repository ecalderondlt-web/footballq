import torch

from footballq.training.td_jepa_losses import td_jepa_loss, variance_loss


def test_anti_collapse_loss_positive_for_constant_embeddings():
    z = torch.ones(8, 16)
    assert variance_loss(z, threshold=0.2).item() > 0.0
    losses = td_jepa_loss(z, z, z, variance_weight=0.1, variance_threshold=0.2)
    assert losses["anti_collapse_loss"].item() > 0.0
    assert losses["z_online_std_min"].item() == 0.0


def test_slot_reconstruction_loss_adds_to_total():
    z = torch.randn(4, 8)
    target = torch.zeros(4, 2, 3, 2)
    reconstruction = torch.ones_like(target)
    mask = torch.ones(4, 2, 3, dtype=torch.bool)
    without = td_jepa_loss(z, z, z, variance_weight=0.0)
    with_reconstruction = td_jepa_loss(
        z,
        z,
        z,
        variance_weight=0.0,
        state_reconstruction=reconstruction,
        state_target=target,
        state_mask=mask,
        slot_reconstruction_weight=0.5,
    )
    assert with_reconstruction["slot_reconstruction_loss"].item() == 1.0
    assert with_reconstruction["total_loss"].item() > without["total_loss"].item()


def test_no_motion_margin_loss_adds_to_total_when_prediction_matches_online():
    z = torch.randn(4, 8)
    without = td_jepa_loss(z, z, z, variance_weight=0.0)
    with_margin = td_jepa_loss(
        z,
        z,
        z,
        variance_weight=0.0,
        no_motion_margin_weight=2.0,
        no_motion_margin=0.25,
    )
    assert round(with_margin["no_motion_margin_loss"].item(), 6) == 0.25
    assert with_margin["total_loss"].item() > without["total_loss"].item()
