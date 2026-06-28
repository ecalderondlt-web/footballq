"""Latent transition discovery and residual diagnostics."""

from footballq.discovery.transitions import (
    TransitionDatasetData,
    build_transition_dataset,
    load_transition_dataset,
    save_transition_dataset,
)

__all__ = [
    "TransitionDatasetData",
    "build_transition_dataset",
    "load_transition_dataset",
    "save_transition_dataset",
]
