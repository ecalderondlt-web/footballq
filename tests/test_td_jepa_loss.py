import torch

from footballq.training.td_jepa_losses import (
    match_mean_invariance_loss,
    td_jepa_loss,
    temporal_motion_reconstruction_loss,
    variance_loss,
)


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


def test_context_reconstruction_loss_adds_to_total():
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
        context_reconstruction=reconstruction,
        context_target=target,
        context_mask=mask,
        context_reconstruction_weight=0.5,
    )
    assert with_reconstruction["context_reconstruction_loss"].item() == 1.0
    assert with_reconstruction["total_loss"].item() > without["total_loss"].item()


def test_transition_reconstruction_loss_adds_to_total():
    z = torch.randn(4, 8)
    target = torch.zeros(4, 2, 3, 2)
    reconstruction = torch.ones_like(target)
    mask = torch.ones(4, 2, 3, dtype=torch.bool)
    losses = td_jepa_loss(
        z,
        z,
        z,
        variance_weight=0.0,
        transition_reconstruction=reconstruction,
        transition_target=target,
        transition_mask=mask,
        transition_reconstruction_weight=0.5,
    )
    assert losses["transition_reconstruction_loss"].item() == 1.0
    assert losses["total_loss"].item() == 0.5


def test_ball_dynamic_reconstruction_reports_ball_only_error():
    z = torch.randn(2, 8)
    target = torch.zeros(2, 2, 23, 7)
    reconstruction = torch.zeros_like(target)
    reconstruction[:, :, 0, :4] = 2.0
    reconstruction[:, :, 1:, :4] = 10.0
    mask = torch.ones(2, 2, 23, dtype=torch.bool)
    losses = td_jepa_loss(
        z,
        z,
        z,
        variance_weight=0.0,
        state_reconstruction=reconstruction,
        state_target=target,
        state_mask=mask,
    )
    assert losses["ball_dynamic_reconstruction_loss"].item() == 4.0


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


def test_temporal_motion_reconstruction_rewards_signed_predictions():
    displacement = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    zero_loss, zero_cosine = temporal_motion_reconstruction_loss(
        torch.zeros_like(displacement),
        torch.zeros_like(displacement),
        displacement,
        valid,
    )
    signed_loss, signed_cosine = temporal_motion_reconstruction_loss(
        displacement,
        -displacement,
        displacement,
        valid,
    )
    assert signed_loss.item() < zero_loss.item()
    assert signed_loss.item() == 0.0
    assert signed_cosine.item() == 1.0
    assert zero_cosine.item() == 0.0


def test_match_mean_invariance_detects_match_specific_offsets():
    aligned = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    aligned_loss, groups = match_mean_invariance_loss(aligned, ["a", "a", "b", "b"])
    assert groups == 2
    assert aligned_loss.item() == 0.0

    separated = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    separated_loss, groups = match_mean_invariance_loss(
        separated,
        ["a", "a", "b", "b"],
    )
    assert groups == 2
    assert separated_loss.item() > 0.0
