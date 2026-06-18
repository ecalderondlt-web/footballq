"""Checkpoint evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from footballq.data.windows import TrackingWindowDataset, load_windows_pt
from footballq.training.train import (
    batch_to_device,
    create_model,
    evaluate_model,
    predict_batch,
    resolve_device,
)


def evaluate_checkpoint(
    checkpoint: str | Path,
    split: str = "test",
    device: str | None = "auto",
) -> dict[str, Any]:
    """Evaluate a saved checkpoint and write ``eval_<split>.json``."""

    checkpoint_path = Path(checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    data = load_windows_pt(
        cfg.get("data", {}).get("windows", "data/processed/synthetic_windows.pt")
    )
    split_indices = payload.get("split_indices", {})
    if split not in split_indices:
        raise ValueError(
            f"Split {split!r} not found in checkpoint. Available: {sorted(split_indices)}"
        )
    model = create_model(cfg, data)
    model.load_state_dict(payload["model_state_dict"])
    torch_device = resolve_device(device)
    model = model.to(torch_device)
    loader = DataLoader(
        TrackingWindowDataset(data, indices=split_indices[split]),
        batch_size=int(cfg.get("data", {}).get("batch_size", 64)),
        shuffle=False,
        num_workers=int(cfg.get("data", {}).get("num_workers", 0)),
    )
    metrics = evaluate_model(model, loader, torch_device)
    run_dir = Path(payload.get("run_dir", checkpoint_path.parent))
    run_dir.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: None
        if isinstance(value, float)
        and (torch.isnan(torch.tensor(value)) or torch.isinf(torch.tensor(value)))
        else value
        for key, value in metrics.items()
    }
    with (run_dir / f"eval_{split}.json").open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)

    try:
        batch = next(iter(loader))
    except StopIteration:
        return {"metrics": metrics, "run_dir": run_dir}
    model.eval()
    with torch.no_grad():
        device_batch = batch_to_device(batch, torch_device)
        pred = predict_batch(model, device_batch).detach().cpu()
    torch.save(
        {
            "prediction_xy_norm": pred[:4],
            "target_xy_norm": batch["future_xy"][:4],
            "future_mask": batch["future_mask"][:4],
            "match_id": batch["match_id"][:4],
            "start_frame": batch["start_frame"][:4],
        },
        run_dir / "predictions_sample.pt",
    )
    return {"metrics": metrics, "run_dir": run_dir}
