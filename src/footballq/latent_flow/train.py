"""Training loops for latent flow matching and latent MLP baselines."""

from __future__ import annotations

import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from footballq.latent_flow.dataset import LatentRolloutDataset, load_latent_rollout_dataset
from footballq.latent_flow.eval import evaluate_latent_checkpoint
from footballq.latent_flow.flow_matching import flow_matching_loss
from footballq.latent_flow.io import append_jsonl, load_yaml, save_yaml
from footballq.latent_flow.metrics import compute_latent_rollout_metrics
from footballq.latent_flow.models import create_latent_model
from footballq.training.train import resolve_device


def load_latent_flow_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    return load_yaml(config)


def set_latent_flow_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _model_name(cfg: dict[str, Any]) -> str:
    return str(cfg.get("model", {}).get("name", "latent_flow_mlp"))


def _dataset_path(cfg: dict[str, Any]) -> Path:
    value = cfg.get("data", {}).get("rollout_dataset", cfg.get("data", {}).get("path", ""))
    if not value:
        raise ValueError("Latent flow config must set data.rollout_dataset.")
    return Path(value)


def _supervised_mlp_loss(model: torch.nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    pred = model(batch["past_z"])
    error = (pred - batch["future_z"]).square()
    mask = batch["future_mask"].bool().unsqueeze(-1)
    denom = mask.sum().clamp_min(1).to(error.dtype) * error.shape[-1]
    return (error * mask.to(error.dtype)).sum() / denom


@torch.no_grad()
def evaluate_training_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    model_name: str,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for batch in loader:
        batch = _batch_to_device(batch, device)
        if model_name == "latent_flow_mlp":
            loss = flow_matching_loss(
                model,
                batch["past_z"],
                batch["future_z"],
                batch["future_mask"],
            )
        else:
            loss = _supervised_mlp_loss(model, batch)
            preds.append(model(batch["past_z"]).detach().cpu())
            targets.append(batch["future_z"].detach().cpu())
            masks.append(batch["future_mask"].detach().cpu())
        losses.append(float(loss.item()))
    metrics: dict[str, float] = {
        "loss": float(np.mean(losses)) if losses else math.nan,
        "num_examples": float(len(loader.dataset)),
    }
    if preds:
        rollout = compute_latent_rollout_metrics(
            torch.cat(preds, dim=0),
            torch.cat(targets, dim=0),
            torch.cat(masks, dim=0),
        )
        for key, value in rollout.items():
            if isinstance(value, float):
                metrics[key] = value
    return metrics


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    run_dir: Path,
    epoch: int,
    best_metric: float,
    data_meta: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "model_name": _model_name(cfg),
            "latent_dim": int(data_meta["latent_dim"]),
            "context_steps": int(data_meta["context_steps"]),
            "horizon_steps": int(data_meta["horizon_steps"]),
            "split_match_ids": data_meta["split_match_ids"],
            "run_dir": str(run_dir),
            "epoch": epoch,
            "best_metric": best_metric,
            "encoder_frozen": True,
        },
        path,
    )


def train_latent_flow_from_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Train a latent flow or latent MLP baseline from config."""

    cfg = load_latent_flow_config(config)
    train_cfg = cfg.get("training", {})
    seed = int(train_cfg.get("seed", cfg.get("seed", 123)))
    set_latent_flow_seed(seed)
    data = load_latent_rollout_dataset(_dataset_path(cfg))
    if data.num_examples == 0:
        raise ValueError("Latent rollout dataset contains zero examples.")
    batch_size = int(train_cfg.get("batch_size", 256))
    num_workers = int(train_cfg.get("num_workers", 0))
    loaders = {
        split: DataLoader(
            LatentRolloutDataset(data, split=split),
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
        )
        for split in ["train", "val", "test"]
    }
    if len(loaders["train"].dataset) == 0 or len(loaders["val"].dataset) == 0:
        raise ValueError("Latent rollout train and val splits must both be non-empty.")

    device = resolve_device(train_cfg.get("device", "auto"))
    model_name = _model_name(cfg)
    model = create_latent_model(
        cfg,
        latent_dim=data.latent_dim,
        context_steps=data.context_steps,
        horizon_steps=data.horizon_steps,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
    )
    run_root = Path(train_cfg.get("run_root", "runs"))
    run_dir = run_root / "latent_flow" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, run_dir / "config.yaml")
    latest_path = run_dir / "latest.pt"
    best_path = run_dir / "best.pt"
    max_epochs = int(train_cfg.get("max_epochs", train_cfg.get("epochs", 20)))
    max_train_batches = train_cfg.get("max_train_batches")
    max_train_batches = int(max_train_batches) if max_train_batches is not None else None
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))
    best_metric_name = str(train_cfg.get("best_metric", "loss"))
    best_metric = float("inf")
    data_meta = {
        "latent_dim": data.latent_dim,
        "context_steps": data.context_steps,
        "horizon_steps": data.horizon_steps,
        "split_match_ids": {
            key: value for key, value in data.splits.items() if key.endswith("_match_ids")
        },
    }

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses: list[float] = []
        for batch_idx, batch in enumerate(loaders["train"], start=1):
            batch = _batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            if model_name == "latent_flow_mlp":
                loss = flow_matching_loss(
                    model,
                    batch["past_z"],
                    batch["future_z"],
                    batch["future_mask"],
                )
            elif model_name == "mlp_latent":
                loss = _supervised_mlp_loss(model, batch)
            else:
                raise ValueError(f"Unsupported latent model name {model_name!r}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            losses.append(float(loss.item()))
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break

        train_metrics = evaluate_training_loss(model, loaders["train"], model_name, device)
        train_metrics["loss"] = float(np.mean(losses)) if losses else math.nan
        val_metrics = evaluate_training_loss(model, loaders["val"], model_name, device)
        append_jsonl(run_dir / "metrics_train.jsonl", {"epoch": epoch, **train_metrics})
        append_jsonl(run_dir / "metrics_val.jsonl", {"epoch": epoch, **val_metrics})
        current = float(val_metrics.get(best_metric_name, math.nan))
        _save_checkpoint(
            latest_path,
            model,
            optimizer,
            cfg,
            run_dir,
            epoch,
            best_metric,
            data_meta,
        )
        if not math.isnan(current) and current < best_metric:
            best_metric = current
            _save_checkpoint(
                best_path,
                model,
                optimizer,
                cfg,
                run_dir,
                epoch,
                best_metric,
                data_meta,
            )

    if not best_path.exists():
        _save_checkpoint(
            best_path,
            model,
            optimizer,
            cfg,
            run_dir,
            max_epochs,
            best_metric,
            data_meta,
        )
    test_result = evaluate_latent_checkpoint(
        best_path,
        dataset=_dataset_path(cfg),
        split="test",
        device=train_cfg.get("device", "auto"),
    )
    return {
        "run_dir": run_dir,
        "latest_checkpoint": latest_path,
        "best_checkpoint": best_path,
        "best_metric": best_metric,
        "test_metrics": test_result["metrics"],
    }
