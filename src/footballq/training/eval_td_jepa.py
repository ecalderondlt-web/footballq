"""TD-JEPA checkpoint evaluation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from footballq.data.td_jepa_dataset import TDJEPADataset, load_td_jepa_data
from footballq.training.train_td_jepa import (
    create_td_jepa_model,
    evaluate_td_model,
    resolve_device,
)


def _jsonable(metrics: dict[str, float]) -> dict[str, float | int | None]:
    out: dict[str, float | int | None] = {}
    for key, value in metrics.items():
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            out[key] = None
        else:
            out[key] = value
    return out


def load_td_checkpoint_model(
    checkpoint: str | Path,
    data_path: str | Path | None = None,
    device: str | None = "auto",
) -> tuple[Any, Any, dict[str, Any], dict[str, Any], torch.device]:
    checkpoint_path = Path(checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    actual_data_path = data_path or cfg.get("data", {}).get("path")
    data = load_td_jepa_data(actual_data_path)
    model = create_td_jepa_model(cfg, data)
    model.online_encoder.load_state_dict(payload["online_encoder"])
    model.target_encoder.load_state_dict(payload["target_encoder"])
    model.motion_encoder.load_state_dict(payload["motion_encoder"])
    torch_device = resolve_device(device)
    model = model.to(torch_device)
    return model, data, cfg, payload, torch_device


def evaluate_td_checkpoint(
    checkpoint: str | Path,
    split: str = "test",
    data_path: str | Path | None = None,
    device: str | None = "auto",
) -> dict[str, Any]:
    model, data, cfg, payload, torch_device = load_td_checkpoint_model(
        checkpoint,
        data_path=data_path,
        device=device,
    )
    split_indices = payload.get("split_indices", {})
    if split not in split_indices:
        raise ValueError(f"Split {split!r} not found. Available: {sorted(split_indices)}")
    loader = DataLoader(
        TDJEPADataset(data, indices=split_indices[split]),
        batch_size=int(cfg.get("training", {}).get("batch_size", 64)),
        shuffle=False,
        num_workers=int(cfg.get("training", {}).get("num_workers", 0)),
    )
    metrics = evaluate_td_model(model, loader, torch_device, cfg.get("loss", {}))
    run_dir = Path(payload.get("run_dir", Path(checkpoint).parent))
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / f"eval_{split}.json").open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(metrics), handle, indent=2)
    return {"metrics": metrics, "run_dir": run_dir}
