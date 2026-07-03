"""Evaluation helpers for coordinate decoder checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from footballq.decoding.dataset import DecoderDataset, load_decoder_dataset
from footballq.decoding.models import create_coordinate_decoder
from footballq.decoding.train import decoder_mode, evaluate_decoder_model
from footballq.latent_flow.io import save_json
from footballq.training.train import resolve_device


def evaluate_decoder_checkpoint(
    checkpoint: str | Path,
    dataset: str | Path | None = None,
    split: str = "test",
    device: str | None = "auto",
) -> dict[str, Any]:
    """Reload and evaluate a coordinate decoder checkpoint."""

    checkpoint_path = Path(checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = dict(payload["config"])
    if dataset is not None:
        cfg.setdefault("data", {})["decoder_dataset"] = str(dataset)
    data_path = Path(cfg.get("data", {}).get("decoder_dataset"))
    data = load_decoder_dataset(data_path)
    mode = str(payload.get("mode", decoder_mode(cfg)))
    split_indices = payload.get("split_indices", {})
    if split not in split_indices:
        available = sorted(split_indices)
        raise ValueError(f"Split {split!r} not found in checkpoint. Available: {available}")
    model = create_coordinate_decoder(cfg, data)
    model.load_state_dict(payload["model_state_dict"])
    torch_device = resolve_device(device)
    model = model.to(torch_device)
    loader = DataLoader(
        DecoderDataset(data, mode=mode, indices=[int(value) for value in split_indices[split]]),
        batch_size=int(cfg.get("data", {}).get("batch_size", 128)),
        shuffle=False,
        num_workers=int(cfg.get("data", {}).get("num_workers", 0)),
    )
    metrics = evaluate_decoder_model(model, loader, mode, torch_device)
    metrics.update(
        {
            "mode": mode,
            "decoder_type": str(payload.get("decoder_type", cfg.get("model", {}).get("name", ""))),
            "split": split,
            "checkpoint_or_config": str(checkpoint_path),
        }
    )
    run_dir = Path(payload.get("run_dir", checkpoint_path.parent))
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, run_dir / f"eval_{split}.json")
    return {"metrics": metrics, "run_dir": run_dir}
