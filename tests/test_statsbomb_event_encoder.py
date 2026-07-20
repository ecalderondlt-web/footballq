import copy

import torch

from footballq.models.statsbomb_event_encoder import (
    StatsBombEventEncoder,
    statsbomb_event_loss,
)


def _batch():
    generator = torch.Generator().manual_seed(7)
    categorical = torch.stack(
        [
            torch.randint(size, (2, 8), generator=generator)
            for size in (37, 11, 28, 33, 38)
        ],
        dim=-1,
    )
    return {
        "categorical": categorical,
        "continuous": torch.rand(2, 8, 17, generator=generator),
        "event_mask": torch.ones(2, 8, dtype=torch.bool),
        "freeze_frame": torch.rand(2, 8, 22, 6, generator=generator),
        "freeze_mask": torch.ones(2, 8, 22, dtype=torch.bool),
        "has_360": torch.ones(2, 8, dtype=torch.bool),
        "target_event_type": torch.randint(37, (2, 8), generator=generator),
        "target_location": torch.rand(2, 8, 2, generator=generator),
        "target_location_mask": torch.ones(2, 8, dtype=torch.bool),
    }


def _model(use_360=True):
    return StatsBombEventEncoder(
        [37, 11, 28, 33, 38],
        17,
        6,
        use_360=use_360,
        categorical_dim=8,
        d_model=32,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
        max_sequence_length=16,
    )


def test_statsbomb_event_encoder_shapes_and_backward():
    batch = _batch()
    model = _model(use_360=True)
    outputs = model(batch)
    losses = statsbomb_event_loss(outputs, batch, location_weight=0.5)
    losses["total_loss"].backward()

    assert outputs["sequence"].shape == (2, 8, 32)
    assert outputs["pooled"].shape == (2, 32)
    assert outputs["next_event_type_logits"].shape == (2, 8, 37)
    assert outputs["next_location"].shape == (2, 8, 2)
    assert torch.isfinite(losses["total_loss"])


def test_statsbomb_freeze_frame_encoding_is_permutation_invariant():
    batch = _batch()
    permuted = copy.deepcopy(batch)
    order = torch.randperm(22, generator=torch.Generator().manual_seed(11))
    permuted["freeze_frame"] = permuted["freeze_frame"][:, :, order]
    permuted["freeze_mask"] = permuted["freeze_mask"][:, :, order]
    model = _model(use_360=True).eval()

    with torch.no_grad():
        original = model(batch)["next_event_type_logits"]
        changed = model(permuted)["next_event_type_logits"]

    assert torch.allclose(original, changed, atol=1e-6)


def test_event_only_view_ignores_all_360_inputs():
    batch = _batch()
    changed = copy.deepcopy(batch)
    changed["freeze_frame"] = torch.randn_like(changed["freeze_frame"]) * 100.0
    changed["freeze_mask"] = ~changed["freeze_mask"]
    changed["has_360"] = ~changed["has_360"]
    changed["continuous"][:, :, 15:] = torch.randn_like(changed["continuous"][:, :, 15:])
    model = _model(use_360=False).eval()

    with torch.no_grad():
        original = model(batch)["next_event_type_logits"]
        without_geometry = model(changed)["next_event_type_logits"]

    assert torch.allclose(original, without_geometry, atol=1e-6)


def test_event_feature_views_have_matched_parameter_shapes():
    event_only = _model(use_360=False)
    event_plus_360 = _model(use_360=True)

    assert {
        name: tuple(parameter.shape) for name, parameter in event_only.named_parameters()
    } == {
        name: tuple(parameter.shape) for name, parameter in event_plus_360.named_parameters()
    }
