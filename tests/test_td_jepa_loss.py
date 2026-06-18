import torch

from footballq.training.td_jepa_losses import td_jepa_loss, variance_loss


def test_anti_collapse_loss_positive_for_constant_embeddings():
    z = torch.ones(8, 16)
    assert variance_loss(z, threshold=0.2).item() > 0.0
    losses = td_jepa_loss(z, z, z, variance_weight=0.1, variance_threshold=0.2)
    assert losses["anti_collapse_loss"].item() > 0.0
    assert losses["z_online_std_min"].item() == 0.0
