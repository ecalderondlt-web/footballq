"""Evaluation helpers for latent rollout models and baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from footballq.latent_flow.baselines import (
    constant_latent_velocity_predict,
    last_latent_predict,
)
from footballq.latent_flow.dataset import (
    LatentRolloutDataset,
    load_latent_rollout_dataset,
)
from footballq.latent_flow.flow_matching import sample_latent_flow
from footballq.latent_flow.io import save_json
from footballq.latent_flow.metrics import compute_latent_rollout_metrics
from footballq.latent_flow.models import create_latent_model
from footballq.training.train import resolve_device


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _collect_targets(
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    futures: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for batch in loader:
        batch = _batch_to_device(batch, device)
        futures.append(batch["future_z"].detach().cpu())
        masks.append(batch["future_mask"].detach().cpu())
    return futures, masks


def evaluate_latent_baseline(
    dataset: str | Path,
    baseline: str,
    split: str = "test",
    batch_size: int = 256,
    device: str | None = "auto",
) -> dict[str, Any]:
    """Evaluate a deterministic latent baseline without a checkpoint."""

    data = load_latent_rollout_dataset(dataset)
    torch_device = resolve_device(device)
    loader = DataLoader(
        LatentRolloutDataset(data, split=split),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    predictions: list[torch.Tensor] = []
    futures: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for batch in loader:
        batch = _batch_to_device(batch, torch_device)
        if baseline == "last_latent":
            pred = last_latent_predict(batch["past_z"], data.horizon_steps)
        elif baseline == "constant_latent_velocity":
            pred = constant_latent_velocity_predict(batch["past_z"], data.horizon_steps)
        else:
            raise ValueError(
                f"Unknown baseline {baseline!r}. Expected last_latent or constant_latent_velocity."
            )
        predictions.append(pred.detach().cpu())
        futures.append(batch["future_z"].detach().cpu())
        masks.append(batch["future_mask"].detach().cpu())
    if not predictions:
        raise ValueError(f"No examples found for split {split!r}.")
    metrics = compute_latent_rollout_metrics(
        torch.cat(predictions, dim=0),
        torch.cat(futures, dim=0),
        torch.cat(masks, dim=0),
    )
    metrics.update(
        {
            "model": baseline,
            "split": split,
            "checkpoint_or_config": str(dataset),
        }
    )
    return {"metrics": metrics, "run_dir": None}


def evaluate_latent_checkpoint(
    checkpoint: str | Path,
    dataset: str | Path | None = None,
    split: str = "test",
    device: str | None = "auto",
    num_samples: int | None = None,
    num_steps: int | None = None,
) -> dict[str, Any]:
    """Reload a latent model checkpoint and evaluate a split."""

    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    cfg = dict(payload["config"])
    data_path = Path(dataset or cfg.get("data", {}).get("rollout_dataset"))
    data = load_latent_rollout_dataset(data_path)
    batch_size = int(cfg.get("training", {}).get("batch_size", 256))
    loader = DataLoader(
        LatentRolloutDataset(data, split=split),
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg.get("training", {}).get("num_workers", 0)),
    )
    torch_device = resolve_device(device)
    model = create_latent_model(
        cfg,
        latent_dim=int(payload["latent_dim"]),
        context_steps=int(payload["context_steps"]),
        horizon_steps=int(payload["horizon_steps"]),
    )
    model.load_state_dict(payload["model_state_dict"])
    model = model.to(torch_device)
    model.eval()
    predictions: list[torch.Tensor] = []
    futures: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    model_name = str(payload["model_name"])
    sampling_cfg = cfg.get("sampling", {})
    sample_count = int(num_samples or sampling_cfg.get("num_samples", 8))
    step_count = int(num_steps or sampling_cfg.get("num_steps", 20))
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, torch_device)
            if model_name == "latent_flow_mlp":
                pred = sample_latent_flow(
                    model,
                    batch["past_z"],
                    horizon_steps=data.horizon_steps,
                    latent_dim=data.latent_dim,
                    num_samples=sample_count,
                    num_steps=step_count,
                )
            elif model_name == "mlp_latent":
                pred = model(batch["past_z"])
            else:
                raise ValueError(f"Unsupported checkpoint model_name {model_name!r}")
            predictions.append(pred.detach().cpu())
            futures.append(batch["future_z"].detach().cpu())
            masks.append(batch["future_mask"].detach().cpu())
    if not predictions:
        raise ValueError(f"No examples found for split {split!r}.")
    pred_all = torch.cat(predictions, dim=0)
    metrics = compute_latent_rollout_metrics(
        pred_all,
        torch.cat(futures, dim=0),
        torch.cat(masks, dim=0),
    )
    metrics.update(
        {
            "model": model_name,
            "split": split,
            "checkpoint_or_config": str(checkpoint),
        }
    )
    run_dir = Path(payload.get("run_dir", Path(checkpoint).parent))
    save_json(metrics, run_dir / f"eval_{split}.json")
    return {"metrics": metrics, "run_dir": run_dir}
