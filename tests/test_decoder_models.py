import torch

from footballq.decoding.dataset import DecoderDatasetData
from footballq.decoding.models import create_coordinate_decoder


def _data():
    return DecoderDatasetData(
        metadata={},
        examples={
            "z": torch.randn(5, 8),
            "z_context": torch.randn(5, 3, 8),
            "z_rollout": torch.randn(5, 2, 8),
            "entity_type": torch.zeros(5, 23, dtype=torch.long),
            "future_xy": torch.randn(5, 4, 23, 2),
        },
        splits={},
    )


def test_linear_reconstruction_decoder_shape():
    data = _data()
    model = create_coordinate_decoder(
        {"target": {"mode": "reconstruct_current"}, "model": {"name": "linear"}},
        data,
    )
    out = model(torch.randn(6, 8))
    assert out.shape == (6, 1, 23, 2)


def test_mlp_future_decoder_shape():
    data = _data()
    model = create_coordinate_decoder(
        {
            "target": {"mode": "future_from_z"},
            "model": {"name": "mlp", "hidden_sizes": [16], "dropout": 0.0},
        },
        data,
    )
    out = model(torch.randn(6, 8))
    assert out.shape == (6, 4, 23, 2)


def test_context_and_rollout_decoder_shapes():
    data = _data()
    context_model = create_coordinate_decoder(
        {
            "target": {"mode": "future_from_context"},
            "model": {"name": "context_mlp", "hidden_sizes": [16], "pooling": "mean"},
        },
        data,
    )
    rollout_model = create_coordinate_decoder(
        {
            "target": {"mode": "rollout_from_latents"},
            "model": {"name": "rollout_mlp", "hidden_sizes": [16], "pooling": "flatten"},
        },
        data,
    )
    assert context_model(torch.randn(6, 3, 8)).shape == (6, 4, 23, 2)
    assert rollout_model(torch.randn(6, 2, 8)).shape == (6, 2, 23, 2)
