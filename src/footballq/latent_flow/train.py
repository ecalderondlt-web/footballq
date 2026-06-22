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

from footballq.latent_flow.baselines import denormalize_residual, normalize_residual
from footballq.latent_flow.dataset import (
    LatentRolloutDataset,
    ensure_residual_targets,
    load_latent_rollout_dataset,
    residual_normalization_stats,
)
from footballq.latent_flow.eval import evaluate_latent_checkpoint
from footballq.latent_flow.flow_matching import flow_matching_loss, sample_latent_flow
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


def _residual_mode(cfg: dict[str, Any]) -> str:
    flow_cfg = cfg.get("flow", {})
    return str(flow_cfg.get("residual_mode", cfg.get("model", {}).get("residual_mode", "last_latent")))


def _is_flow_model(model_name: str) -> bool:
    return model_name in {"latent_flow_mlp", "residual_latent_flow_mlp"}


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


def _residual_flow_target(batch: dict[str, Any], mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return normalize_residual(batch["residual_future_z"], mean, std)


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _format_metric(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


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
        elif model_name == "residual_latent_flow_mlp":
            raise ValueError("Residual flow training loss requires residual stats.")
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
            "residual_mode": data_meta.get("residual_mode"),
            "normalization": data_meta.get("normalization", {}),
            "run_dir": str(run_dir),
            "epoch": epoch,
            "step": int(data_meta.get("step", 0)),
            "best_metric": best_metric,
            "encoder_frozen": True,
        },
        path,
    )


def train_latent_flow_from_config(
    config: str | Path | dict[str, Any],
    resume: str | Path | None = None,
) -> dict[str, Any]:
    """Train a latent flow or latent MLP baseline from config."""

    cfg = load_latent_flow_config(config)
    train_cfg = cfg.get("training", {})
    seed = int(train_cfg.get("seed", cfg.get("seed", 123)))
    set_latent_flow_seed(seed)
    data = load_latent_rollout_dataset(_dataset_path(cfg))
    model_name = _model_name(cfg)
    if model_name == "residual_latent_flow_mlp":
        data = ensure_residual_targets(data, _residual_mode(cfg))
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
    start_epoch = 1
    step = 0
    best_metric = float("inf")
    if resume is not None:
        resume_payload = torch.load(Path(resume), map_location="cpu", weights_only=False)
        model.load_state_dict(resume_payload["model_state_dict"])
        if "optimizer" in resume_payload:
            optimizer.load_state_dict(resume_payload["optimizer"])
        run_dir = Path(resume_payload.get("run_dir", Path(resume).parent))
        start_epoch = int(resume_payload.get("epoch", 0)) + 1
        step = int(resume_payload.get("step", 0))
        best_metric = float(resume_payload.get("best_metric", float("inf")))
    else:
        run_dir = run_root / "latent_flow" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, run_dir / "config.yaml")
    latest_path = run_dir / "latest.pt"
    best_path = run_dir / "best.pt"
    max_epochs = int(train_cfg.get("max_epochs", train_cfg.get("epochs", 20)))
    max_train_batches = train_cfg.get("max_train_batches")
    max_train_batches = int(max_train_batches) if max_train_batches is not None else None
    max_val_batches = train_cfg.get("max_val_batches")
    max_val_batches = int(max_val_batches) if max_val_batches is not None else None
    max_steps = train_cfg.get("max_steps")
    max_steps = int(max_steps) if max_steps is not None else None
    save_every_steps = int(train_cfg.get("save_every_steps", 0) or 0)
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))
    best_metric_name = str(train_cfg.get("best_metric", "loss"))
    residual_mean = residual_std = None
    if model_name == "residual_latent_flow_mlp":
        residual_mean, residual_std = residual_normalization_stats(data)
        residual_mean = residual_mean.to(device)
        residual_std = residual_std.to(device)
    flow_cfg = cfg.get("flow", {})
    sampling_cfg = cfg.get("sampling", {})
    noise_scale = float(flow_cfg.get("noise_scale", cfg.get("sampling", {}).get("noise_scale", 1.0)))
    val_sample_steps = int(flow_cfg.get("num_sampling_steps", sampling_cfg.get("num_steps", 20)))
    val_noise_scale = 0.0 if bool(flow_cfg.get("deterministic_mean_eval", False)) else noise_scale
    data_meta = {
        "latent_dim": data.latent_dim,
        "context_steps": data.context_steps,
        "horizon_steps": data.horizon_steps,
        "split_match_ids": {
            key: value for key, value in data.splits.items() if key.endswith("_match_ids")
        },
    }
    if model_name == "residual_latent_flow_mlp":
        data_meta["residual_mode"] = _residual_mode(cfg)
        data_meta["normalization"] = data.metadata.get("normalization", {})
    if resume is not None:
        _optimizer_to_device(optimizer, device)

    last_epoch = max(0, start_epoch - 1)
    stop_training = bool(max_steps is not None and step >= max_steps)
    if stop_training:
        print(f"resume step={step} already satisfies max_steps={max_steps}; skipping training.")
    if start_epoch > max_epochs:
        print(f"resume epoch={start_epoch} is beyond max_epochs={max_epochs}; skipping training.")
        stop_training = True

    for epoch in range(start_epoch, max_epochs + 1):
        if stop_training:
            break
        last_epoch = epoch
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
                    noise_scale=noise_scale,
                )
            elif model_name == "residual_latent_flow_mlp":
                assert residual_mean is not None and residual_std is not None
                loss = flow_matching_loss(
                    model,
                    batch["past_z"],
                    _residual_flow_target(batch, residual_mean, residual_std),
                    batch["future_mask"],
                    noise_scale=noise_scale,
                )
            elif model_name == "mlp_latent":
                loss = _supervised_mlp_loss(model, batch)
            else:
                raise ValueError(f"Unsupported latent model name {model_name!r}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            step += 1
            losses.append(float(loss.item()))
            if save_every_steps > 0 and step % save_every_steps == 0:
                data_meta["step"] = step
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
            if max_steps is not None and step >= max_steps:
                stop_training = True
                break
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break

        if model_name == "residual_latent_flow_mlp":
            train_metrics = {"loss": float(np.mean(losses)) if losses else math.nan}
            val_losses: list[float] = []
            val_preds: list[torch.Tensor] = []
            val_targets: list[torch.Tensor] = []
            val_masks: list[torch.Tensor] = []
            model.eval()
            with torch.no_grad():
                for val_batch_idx, val_batch in enumerate(loaders["val"], start=1):
                    val_batch = _batch_to_device(val_batch, device)
                    assert residual_mean is not None and residual_std is not None
                    val_loss = flow_matching_loss(
                        model,
                        val_batch["past_z"],
                        _residual_flow_target(val_batch, residual_mean, residual_std),
                        val_batch["future_mask"],
                        noise_scale=noise_scale,
                    )
                    val_losses.append(float(val_loss.item()))
                    residual_norm = sample_latent_flow(
                        model,
                        val_batch["past_z"],
                        horizon_steps=data.horizon_steps,
                        latent_dim=data.latent_dim,
                        num_samples=1,
                        num_steps=val_sample_steps,
                        noise_scale=val_noise_scale,
                    )
                    residual = denormalize_residual(residual_norm, residual_mean, residual_std)
                    pred = val_batch["baseline_future_z"].unsqueeze(1) + residual
                    val_preds.append(pred[:, 0].detach().cpu())
                    val_targets.append(val_batch["future_z"].detach().cpu())
                    val_masks.append(val_batch["future_mask"].detach().cpu())
                    if max_val_batches is not None and val_batch_idx >= max_val_batches:
                        break
            val_metrics = {
                "loss": float(np.mean(val_losses)) if val_losses else math.nan,
                "num_examples": float(len(loaders["val"].dataset)),
            }
            if val_preds:
                rollout = compute_latent_rollout_metrics(
                    torch.cat(val_preds, dim=0),
                    torch.cat(val_targets, dim=0),
                    torch.cat(val_masks, dim=0),
                )
                for key, value in rollout.items():
                    if isinstance(value, float):
                        val_metrics[key] = value
        else:
            train_metrics = evaluate_training_loss(model, loaders["train"], model_name, device)
            val_metrics = evaluate_training_loss(model, loaders["val"], model_name, device)
        train_metrics["loss"] = float(np.mean(losses)) if losses else math.nan
        train_row = {"epoch": epoch, "step": step, **train_metrics}
        val_row = {"epoch": epoch, "step": step, **val_metrics}
        append_jsonl(run_dir / "metrics_train.jsonl", train_row)
        append_jsonl(run_dir / "metrics_val.jsonl", val_row)
        current = float(val_metrics.get(best_metric_name, math.nan))
        data_meta["step"] = step
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
        parts = [
            f"epoch={epoch}",
            f"step={step}",
            f"train_loss={_format_metric(float(train_metrics.get('loss', math.nan)))}",
            f"val_loss={_format_metric(float(val_metrics.get('loss', math.nan)))}",
        ]
        for key, label in [
            ("latent_ADE", "val_ADE"),
            ("latent_FDE", "val_FDE"),
            ("latent_cosine_similarity", "val_cosine"),
        ]:
            if key in val_metrics:
                parts.append(f"{label}={_format_metric(float(val_metrics[key]))}")
        parts.append(f"latest={latest_path}")
        parts.append(f"best={best_path}")
        print(" ".join(parts))

    if not best_path.exists():
        data_meta["step"] = step
        _save_checkpoint(
            best_path,
            model,
            optimizer,
            cfg,
            run_dir,
            last_epoch,
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
