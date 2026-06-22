import torch

from footballq.probes.features import random_same_shape_features


def test_random_feature_baseline_deterministic():
    z = torch.zeros(5, 7)
    first = random_same_shape_features(z, seed=123)
    second = random_same_shape_features(z, seed=123)
    other = random_same_shape_features(z, seed=124)
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert first.shape == z.shape
