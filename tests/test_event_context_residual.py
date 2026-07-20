import torch

from footballq.models.event_context_residual import (
    FrozenTrackingEventResidual,
    event_context_residual_loss,
)
from footballq.models.statsbomb_event_encoder import StatsBombEventEncoder
from footballq.models.td_jepa import SoccerTDJEPA


def _tracking_model():
    return SoccerTDJEPA(
        context_steps=4,
        delta_steps=2,
        n_entities=3,
        n_features=5,
        z_dim=16,
        d_model=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
        motion_hidden_dim=32,
    )


def _event_encoder():
    return StatsBombEventEncoder(
        [37, 11, 28, 33, 38],
        17,
        6,
        use_360=False,
        categorical_dim=4,
        d_model=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
        max_sequence_length=8,
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
        "state_t": torch.rand(2, 4, 3, 5, generator=generator),
        "state_t_plus_delta": torch.rand(2, 4, 3, 5, generator=generator),
        "delta_state": torch.rand(2, 2, 3, 5, generator=generator),
        "mask_t": torch.ones(2, 4, 3, dtype=torch.bool),
        "mask_t_plus_delta": torch.ones(2, 4, 3, dtype=torch.bool),
        "delta_mask": torch.ones(2, 2, 3, dtype=torch.bool),
        "event_categorical": categorical,
        "event_continuous": torch.rand(2, 8, 17, generator=generator),
        "event_mask": torch.tensor(
            [[True] * 8, [False] * 8],
            dtype=torch.bool,
        ),
        "raw_event_context": torch.rand(2, 16, generator=generator),
    }


def test_frozen_event_residual_only_trains_correction_head():
    model = FrozenTrackingEventResidual(
        _tracking_model(),
        family="pretrained",
        z_dim=16,
        event_encoder=_event_encoder(),
        hidden_dim=32,
    )
    batch = _batch()
    outputs = model(batch)
    losses = event_context_residual_loss(outputs, batch["event_mask"])
    losses["td_loss"].backward()

    assert outputs["z_pred"].shape == (2, 16)
    assert all(parameter.grad is None for parameter in model.tracking_model.parameters())
    assert all(parameter.grad is None for parameter in model.event_encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.correction_head.parameters())
    assert torch.isfinite(losses["td_loss"])


def test_event_ablation_zeroes_encoded_context_but_keeps_matched_head():
    model = FrozenTrackingEventResidual(
        _tracking_model(),
        family="pretrained",
        z_dim=16,
        event_encoder=_event_encoder(),
        hidden_dim=32,
    ).eval()
    batch = _batch()

    with torch.no_grad():
        regular = model(batch)
        ablated = model(batch, ablate_event=True)

    assert not torch.equal(regular["event_context"][0], ablated["event_context"][0])
    assert torch.count_nonzero(ablated["event_context"]) == 0
    assert torch.equal(regular["z_base"], ablated["z_base"])
