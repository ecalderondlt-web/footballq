"""Training and evaluation helpers for Phase 1 baselines."""

from footballq.training.eval import evaluate_checkpoint
from footballq.training.train import train_from_config

__all__ = ["evaluate_checkpoint", "train_from_config"]
