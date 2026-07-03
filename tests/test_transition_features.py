import pytest
import torch

from footballq.discovery.transitions import TransitionDatasetData


def test_transition_features_are_train_normalized():
    z_t = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    z_next = torch.tensor([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]])
    delta_z = z_next - z_t
    source_split = ["train", "train", "test"]
    from footballq.discovery.transitions import _feature_payload

    features, diagnostics = _feature_payload(
        z_t,
        z_next,
        delta_z,
        torch.tensor([1.0, 1.0, 1.0]),
        source_split,
    )
    assert features["delta_norm"].tolist() == [1.0, 2.0, 3.0]
    assert features["latent_velocity"].shape == delta_z.shape
    assert features["pca_delta_z"].shape[0] == delta_z.shape[0]
    assert features["random_encoder_delta_z"].shape == delta_z.shape
    assert diagnostics["normalization_train_rows"] == 2
    assert torch.isfinite(features["normalized_delta_z"]).all()


def test_scientific_transition_features_require_train_rows():
    from footballq.discovery.transitions import _feature_payload

    z_t = torch.zeros(3, 2)
    z_next = torch.ones(3, 2)
    delta_z = z_next - z_t

    with pytest.raises(ValueError, match="requires at least one train row"):
        _feature_payload(
            z_t,
            z_next,
            delta_z,
            torch.ones(3),
            ["val", "test", "test"],
            scientific_mode=True,
        )


def test_transition_dataset_properties():
    data = TransitionDatasetData(
        examples={
            "z_t": torch.zeros(2, 4),
            "delta_seconds": torch.tensor([0.2, 0.5]),
        },
        features={},
        metadata={},
    )
    assert data.num_examples == 2
    assert data.latent_dim == 4
    assert data.delta_seconds_values == [0.20000000298023224, 0.5]
