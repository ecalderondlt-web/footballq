"""Feature sources for frozen probe baselines."""

from __future__ import annotations

import torch


def random_same_shape_features(z: torch.Tensor, seed: int = 123) -> torch.Tensor:
    """Generate deterministic random features with the same shape as ``z``."""

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(z.shape, generator=generator, dtype=z.dtype)


def probe_feature_matrix(
    examples: dict[str, object],
    feature_source: str,
    seed: int = 123,
) -> torch.Tensor:
    """Return the feature matrix for a named probe feature source."""

    if feature_source == "td_jepa":
        return torch.as_tensor(examples["z"]).float()
    if feature_source == "random_same_shape":
        return random_same_shape_features(torch.as_tensor(examples["z"]).float(), seed=seed)
    if feature_source == "raw_state_summary":
        return torch.as_tensor(examples["raw_state_summary"]).float()
    raise ValueError(
        f"Unknown feature source {feature_source!r}. "
        "Expected td_jepa, random_same_shape, or raw_state_summary."
    )
