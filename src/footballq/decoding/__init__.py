"""Coordinate decoders for frozen TD-JEPA latent states."""

from footballq.decoding.dataset import (
    DecoderDataset,
    DecoderDatasetData,
    build_decoder_dataset,
    load_decoder_dataset,
    save_decoder_dataset,
)
from footballq.decoding.eval import evaluate_decoder_checkpoint
from footballq.decoding.learning_curve import run_decoder_learning_curve
from footballq.decoding.train import train_coordinate_decoder_from_config

__all__ = [
    "DecoderDataset",
    "DecoderDatasetData",
    "build_decoder_dataset",
    "evaluate_decoder_checkpoint",
    "load_decoder_dataset",
    "run_decoder_learning_curve",
    "save_decoder_dataset",
    "train_coordinate_decoder_from_config",
]
