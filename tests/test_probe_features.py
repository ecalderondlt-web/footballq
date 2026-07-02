import torch

from footballq.probes.features import probe_feature_matrix, random_same_shape_features


def test_random_feature_baseline_deterministic():
    z = torch.zeros(5, 7)
    first = random_same_shape_features(z, seed=123)
    second = random_same_shape_features(z, seed=123)
    other = random_same_shape_features(z, seed=124)
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert first.shape == z.shape


def test_raw_plus_td_jepa_concatenates_raw_and_z():
    examples = {
        "z": torch.ones(3, 2),
        "raw_state_summary": torch.zeros(3, 4),
    }
    features = probe_feature_matrix(examples, "raw_plus_td_jepa")
    assert features.shape == (3, 6)
    assert torch.equal(features[:, :4], torch.zeros(3, 4))
    assert torch.equal(features[:, 4:], torch.ones(3, 2))
