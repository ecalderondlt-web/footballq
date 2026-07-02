import torch

from footballq.discovery.clustering import kmeans, transition_feature_matrix
from footballq.discovery.transitions import TransitionDatasetData


def test_kmeans_is_deterministic_with_fixed_seed():
    x = torch.cat([torch.randn(20, 2) - 2.0, torch.randn(20, 2) + 2.0], dim=0)
    first = kmeans(x, k=2, seed=42, max_iter=10, fit_sample_size=40)
    second = kmeans(x, k=2, seed=42, max_iter=10, fit_sample_size=40)
    assert torch.equal(first["assignments"], second["assignments"])
    assert torch.allclose(first["centroids"], second["centroids"])
    assert torch.isfinite(first["distances"]).all()


def test_transition_feature_matrix_accepts_dataset_feature():
    data = TransitionDatasetData(
        examples={
            "z_t": torch.zeros(2, 2),
            "delta_seconds": torch.tensor([0.2, 0.2]),
            "delta_z": torch.zeros(2, 2),
        },
        features={"handcrafted_structure_metrics": torch.ones(2, 3)},
        metadata={},
    )
    x, indices = transition_feature_matrix(data, feature="handcrafted_structure_metrics")
    assert x.shape == (2, 3)
    assert indices == [0, 1]
