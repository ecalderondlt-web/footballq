"""Placeholder interface for future generative trajectory models."""

from __future__ import annotations

from typing import Protocol

import torch


class GenerativeTrajectoryModel(Protocol):
    """Future raw-coordinate sampler interface; intentionally not implemented in Phase 1."""

    def sample(self, batch: dict[str, torch.Tensor], num_samples: int) -> torch.Tensor:
        """Return sampled trajectories shaped ``[samples, batch, horizon, entities, xy]``."""
