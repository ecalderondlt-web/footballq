import torch

from footballq.discovery.clustering import kmeans


def test_kmeans_is_deterministic_with_fixed_seed():
    x = torch.cat([torch.randn(20, 2) - 2.0, torch.randn(20, 2) + 2.0], dim=0)
    first = kmeans(x, k=2, seed=42, max_iter=10, fit_sample_size=40)
    second = kmeans(x, k=2, seed=42, max_iter=10, fit_sample_size=40)
    assert torch.equal(first["assignments"], second["assignments"])
    assert torch.allclose(first["centroids"], second["centroids"])
    assert torch.isfinite(first["distances"]).all()
