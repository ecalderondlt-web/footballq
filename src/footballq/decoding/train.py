"""Training loop for Experiment 4C coordinate decoders."""

from __future__ import annotations

import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from footballq.decoding.dataset import DecoderDataset, DecoderDatasetData, load_decoder_dataset
from footballq.decoding.metrics import decoder_metrics
from footballq.decoding.models import create_coordinate_decoder, decoder_output_steps
from footballq.latent_flow.io import append_jsonl, load_yaml, save_json, save_yaml
from footballq.training.losses import masked_mse_loss
from footballq.training.train import resolve_device


def load_decoder_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    return load_yaml(config)


def set_decoder_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dataset_path(cfg: dict[str, Any]) -> Path:
    value = cfg.get("data", {}).get("decoder_dataset", cfg.get("data", {}).get("path", ""))
    if not value:
        raise ValueError("Coordinate decoder config must set data.decoder_dataset.")
    return Path(value)


def decoder_mode(cfg: dict[str, Any]) -> str:
    return str(cfg.get("target", {}).get("mode", cfg.get("model", {}).get("input_type", "future_from_z")))


def decoder_name(cfg: dict[str, Any]) -> str:
    return str(cfg.get("model", {}).get("name", "linear"))


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _target_for_loss(batch: dict[str, Any], mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    target = batch["target_xy"]
    mask = batch["target_mask"]
    if mode == "reconstruct_current":
        target = target.unsqueeze(1)
        mask = mask.unsqueeze(1)
    return target, mask


@torch.no_grad()
def evaluate_decoder_model(
    model: torch.nn.Module,
    loader: DataLoader,
    mode: str,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    entity_types: list[torch.Tensor] = []
    team_ids: list[torch.Tensor] = []
    losses: list[float] = []
    for batch_idx, batch in enumerate(loader, start=1):
        batch = _batch_to_device(batch, device)
        pred = model(batch["x"])
        target, mask = _target_for_loss(batch, mode)
        loss = masked_mse_loss(pred, target, mask)
        losses.append(float(loss.item()))
        if mode == "reconstruct_current":
            preds.append(pred[:, 0].detach().cpu())
            targets.append(batch["target_xy"].detach().cpu())
            masks.append(batch["target_mask"].detach().cpu())
        else:
            preds.append(pred.detach().cpu())
            targets.append(batch["target_xy"].detach().cpu())
            masks.append(batch["target_mask"].detach().cpu())
        entity_types.append(batch["entity_type"].detach().cpu())
        team_ids.append(batch["team_id"].detach().cpu())
        if max_batches is not None and batch_idx >= max_batches:
            break
    if not preds:
        return {"loss": math.nan}
    metrics = decoder_metrics(
        torch.cat(preds, dim=0),
        torch.cat(targets, dim=0),
        torch.cat(masks, dim=0),
        torch.cat(entity_types, dim=0),
        torch.cat(team_ids, dim=0),
        mode=mode,
    )
    metrics["loss"] = float(np.mean(losses)) if losses else math.nan
    metrics["num_examples"] = float(len(loader.dataset))
    return metrics


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    cfg: dict[str, Any],
    data: DecoderDatasetData,
    run_dir: Path,
    split_indices: dict[str, Any],
    epoch: int,
    best_metric: float,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "config": cfg,
        "mode": decoder_mode(cfg),
        "decoder_type": decoder_name(cfg),
        "latent_dim": data.latent_dim,
        "n_entities": data.n_entities,
        "horizon_steps": data.horizon_steps,
        "context_z_steps": data.context_z_steps,
        "rollout_steps": data.rollout_steps,
        "output_steps": decoder_output_steps(decoder_mode(cfg), data),
        "split_indices": split_indices,
        "split_match_ids": {
            key: value for key, value in data.splits.items() if key.endswith("_match_ids")
        },
        "metadata": data.metadata,
        "run_dir": str(run_dir),
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "encoder_frozen": True,
    }
    torch.save(payload, path)


def _save_prediction_sample(
    path: Path,
    model: torch.nn.Module,
    loader: DataLoader,
    mode: str,
    device: torch.device,
) -> None:
    try:
        batch = next(iter(loader))
    except StopIteration:
        return
    model.eval()
    with torch.no_grad():
        device_batch = _batch_to_device(batch, device)
        pred = model(device_batch["x"]).detach().cpu()
    torch.save(
        {
            "prediction_xy_norm": pred[:4],
            "target_xy_norm": batch["target_xy"][:4],
            "target_mask": batch["target_mask"][:4],
            "mode": mode,
            "match_id": batch["match_id"][:4],
            "frame_t": batch["frame_t"][:4],
        },
        path,
    )


def train_coordinate_decoder_from_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Train a coordinate decoder from a YAML path or config dictionary."""

    cfg = load_decoder_config(config)
    train_cfg = cfg.get("training", {})
    seed = int(train_cfg.get("seed", cfg.get("seed", 123)))
    set_decoder_seed(seed)
    data_path = _dataset_path(cfg)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Decoder dataset not found: {data_path}. Run scripts/build_decoder_dataset.py first."
        )
    data = load_decoder_dataset(data_path)
    if data.num_examples == 0:
        raise ValueError("Decoder dataset contains zero examples.")
    mode = decoder_mode(cfg)
    batch_size = int(cfg.get("data", {}).get("batch_size", 128))
    num_workers = int(cfg.get("data", {}).get("num_workers", 0))
    split_indices = {
        split: [int(value) for value in data.splits.get(f"{split}_indices", [])]
        for split in ["train", "val", "test"]
    }
    if not split_indices["train"] or not split_indices["val"]:
        raise ValueError("Decoder train and val splits must both be non-empty.")
    loaders = {
        split: DataLoader(
            DecoderDataset(data, mode=mode, indices=indices),
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
        )
        for split, indices in split_indices.items()
    }
    device = resolve_device(train_cfg.get("device", "auto"))
    model = create_coordinate_decoder(cfg, data).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    run_root = Path(train_cfg.get("run_root", "runs"))
    run_dir = (
        run_root
        / "decoders"
        / mode
        / decoder_name(cfg)
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, run_dir / "config.yaml")
    latest_path = run_dir / "latest.pt"
    best_path = run_dir / "best.pt"
    epochs = int(train_cfg.get("epochs", train_cfg.get("max_epochs", 5)))
    max_train_batches = train_cfg.get("max_train_batches")
    max_train_batches = int(max_train_batches) if max_train_batches is not None else None
    max_eval_batches = train_cfg.get("max_eval_batches")
    max_eval_batches = int(max_eval_batches) if max_eval_batches is not None else None
    metric_name = str(
        train_cfg.get(
            "best_metric",
            "current_all_entity_error_m" if mode == "reconstruct_current" else "all_entity_ADE_m",
        )
    )
    best_metric = float("inf")
    last_epoch = 0
    for epoch in range(1, epochs + 1):
        last_epoch = epoch
        model.train()
        train_losses: list[float] = []
        for batch_idx, batch in enumerate(loaders["train"], start=1):
            batch = _batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch["x"])
            target, mask = _target_for_loss(batch, mode)
            loss = masked_mse_loss(pred, target, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(train_cfg.get("grad_clip_norm", 1.0)),
            )
            optimizer.step()
            train_losses.append(float(loss.item()))
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break
        train_metrics = evaluate_decoder_model(
            model,
            loaders["train"],
            mode,
            device,
            max_batches=max_eval_batches,
        )
        train_metrics["loss"] = float(np.mean(train_losses)) if train_losses else math.nan
        val_metrics = evaluate_decoder_model(
            model,
            loaders["val"],
            mode,
            device,
            max_batches=max_eval_batches,
        )
        append_jsonl(run_dir / "metrics_train.jsonl", {"epoch": epoch, **train_metrics})
        append_jsonl(run_dir / "metrics_val.jsonl", {"epoch": epoch, **val_metrics})
        current = float(val_metrics.get(metric_name, math.inf))
        if current < best_metric:
            best_metric = current
            _save_checkpoint(
                best_path,
                model,
                optimizer,
                cfg,
                data,
                run_dir,
                split_indices,
                epoch,
                best_metric,
            )
        _save_checkpoint(
            latest_path,
            model,
            optimizer,
            cfg,
            data,
            run_dir,
            split_indices,
            epoch,
            best_metric,
        )
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.6f} "
            f"val_{metric_name}={current:.6f} latest={latest_path} best={best_path}"
        )
    if not best_path.exists():
        _save_checkpoint(
            best_path,
            model,
            optimizer,
            cfg,
            data,
            run_dir,
            split_indices,
            last_epoch,
            best_metric,
        )
    test_metrics = evaluate_decoder_model(model, loaders["test"], mode, device)
    save_json(test_metrics, run_dir / "eval_test.json")
    _save_prediction_sample(run_dir / "predictions_sample.pt", model, loaders["test"], mode, device)
    return {
        "run_dir": run_dir,
        "latest_checkpoint": latest_path,
        "best_checkpoint": best_path,
        "best_metric": best_metric,
        "test_metrics": test_metrics,
    }
