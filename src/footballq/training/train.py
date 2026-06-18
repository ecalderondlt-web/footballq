"""Training loop for Phase 1 deterministic baselines."""

from __future__ import annotations

import json
import math
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only in minimal environments
    def tqdm(iterable: Any, *_args: Any, **_kwargs: Any) -> Any:
        return iterable

from footballq.data.windows import (
    TrackingWindowDataset,
    TrackingWindowTensorData,
    load_windows_pt,
)
from footballq.models import (
    ConstantVelocityBaseline,
    MLPBaseline,
    SpatioTemporalTransformerBaseline,
)
from footballq.training.losses import masked_mse_loss
from footballq.training.metrics import compute_metrics


def load_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    with Path(config).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str | None = None) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_indices_by_match(
    match_ids: list[str],
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 7,
) -> dict[str, list[int]]:
    """Create deterministic train/val/test splits without splitting matches."""

    unique = sorted(set(match_ids))
    rng = random.Random(seed)
    rng.shuffle(unique)
    if len(unique) == 1:
        split_matches = {"train": set(unique), "val": set(unique), "test": set(unique)}
    elif len(unique) == 2:
        split_matches = {"train": {unique[0]}, "val": {unique[1]}, "test": {unique[1]}}
    else:
        n_test = max(1, int(round(len(unique) * test_fraction)))
        n_val = max(1, int(round(len(unique) * val_fraction)))
        test = set(unique[:n_test])
        val = set(unique[n_test : n_test + n_val])
        train = set(unique[n_test + n_val :])
        if not train:
            train = {unique[-1]}
        split_matches = {"train": train, "val": val or train, "test": test or val or train}

    return {
        split: [idx for idx, match_id in enumerate(match_ids) if match_id in matches]
        for split, matches in split_matches.items()
    }


def create_model(config: dict[str, Any], data: TrackingWindowTensorData) -> torch.nn.Module:
    model_config = dict(config.get("model", {}))
    name = model_config.get("name", "constant_velocity")
    if name == "constant_velocity":
        return ConstantVelocityBaseline(
            horizon_steps=data.horizon_steps,
            fps=data.fps,
            feature_names=data.feature_names,
        )
    if name == "mlp":
        return MLPBaseline(
            history_steps=data.history_steps,
            horizon_steps=data.horizon_steps,
            n_entities=data.n_entities,
            n_features=data.n_features,
            hidden_sizes=list(model_config.get("hidden_sizes", [256, 256])),
            dropout=float(model_config.get("dropout", 0.1)),
        )
    if name == "st_transformer":
        return SpatioTemporalTransformerBaseline(
            history_steps=data.history_steps,
            horizon_steps=data.horizon_steps,
            n_entities=data.n_entities,
            n_features=data.n_features,
            d_model=int(model_config.get("d_model", 128)),
            n_heads=int(model_config.get("n_heads", 4)),
            n_layers_temporal=int(model_config.get("n_layers_temporal", 2)),
            n_layers_social=int(model_config.get("n_layers_social", 2)),
            dropout=float(model_config.get("dropout", 0.1)),
        )
    raise ValueError(f"Unknown model name: {name}")


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def predict_batch(
    model: torch.nn.Module,
    batch: dict[str, Any],
) -> torch.Tensor:
    if isinstance(model, ConstantVelocityBaseline):
        return model(batch)
    return model(batch["past"])


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    entity_types: list[torch.Tensor] = []
    team_ids: list[torch.Tensor] = []
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            pred = predict_batch(model, batch)
            loss = masked_mse_loss(pred, batch["future_xy"], batch["future_mask"])
            losses.append(float(loss.item()))
            preds.append(pred.detach().cpu())
            targets.append(batch["future_xy"].detach().cpu())
            masks.append(batch["future_mask"].detach().cpu())
            entity_types.append(batch["entity_type"].detach().cpu())
            team_ids.append(batch["team_id"].detach().cpu())

    if not preds:
        return {"loss": math.nan}
    metrics = compute_metrics(
        torch.cat(preds, dim=0),
        torch.cat(targets, dim=0),
        torch.cat(masks, dim=0),
        torch.cat(entity_types, dim=0),
        torch.cat(team_ids, dim=0),
    )
    metrics["loss"] = float(np.mean(losses)) if losses else math.nan
    return metrics


def _jsonable_metrics(metrics: dict[str, float]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key, value in metrics.items():
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            out[key] = None
        else:
            out[key] = value
    return out


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    config: dict[str, Any],
    data: TrackingWindowTensorData,
    split_indices: dict[str, list[int]],
    run_dir: Path,
    epoch: int,
    best_metric: float,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "feature_names": data.feature_names,
        "fps": data.fps,
        "history_steps": data.history_steps,
        "horizon_steps": data.horizon_steps,
        "n_entities": data.n_entities,
        "n_features": data.n_features,
        "split_indices": split_indices,
        "run_dir": str(run_dir),
        "epoch": epoch,
        "best_metric": best_metric,
    }
    torch.save(checkpoint, path)


def _save_prediction_sample(
    path: Path,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> None:
    model.eval()
    try:
        batch = next(iter(loader))
    except StopIteration:
        return
    with torch.no_grad():
        device_batch = batch_to_device(batch, device)
        pred = predict_batch(model, device_batch).detach().cpu()
    torch.save(
        {
            "prediction_xy_norm": pred[:4],
            "target_xy_norm": batch["future_xy"][:4],
            "future_mask": batch["future_mask"][:4],
            "match_id": batch["match_id"][:4],
            "start_frame": batch["start_frame"][:4],
        },
        path,
    )


def train_from_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Train or evaluate a configured Phase 1 baseline."""

    cfg = load_config(config)
    training_cfg = cfg.get("training", {})
    seed = int(training_cfg.get("seed", 7))
    set_seed(seed)
    data_path = Path(cfg.get("data", {}).get("windows", "data/processed/synthetic_windows.pt"))
    if not data_path.exists():
        raise FileNotFoundError(
            f"Window file not found: {data_path}. Run scripts/prepare_tracking_data.py first."
        )
    data = load_windows_pt(data_path)
    if len(data.match_id) == 0:
        raise ValueError("Window file contains zero windows.")

    split_cfg = cfg.get("split", {})
    split_indices = split_indices_by_match(
        data.match_id,
        val_fraction=float(split_cfg.get("val_fraction", 0.2)),
        test_fraction=float(split_cfg.get("test_fraction", 0.2)),
        seed=seed,
    )
    batch_size = int(cfg.get("data", {}).get("batch_size", 64))
    num_workers = int(cfg.get("data", {}).get("num_workers", 0))
    loaders = {
        split: DataLoader(
            TrackingWindowDataset(data, indices=indices),
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
        )
        for split, indices in split_indices.items()
    }

    device = resolve_device(training_cfg.get("device", "auto"))
    model = create_model(cfg, data).to(device)
    model_name = cfg.get("model", {}).get("name", "constant_velocity")
    run_root = Path(training_cfg.get("run_root", "runs"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = run_root / model_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    latest_path = run_dir / "latest.pt"
    best_path = run_dir / "best.pt"
    best_metric = float("inf")
    epochs = int(training_cfg.get("epochs", 1))
    max_train_batches = training_cfg.get("max_train_batches")
    max_train_batches = int(max_train_batches) if max_train_batches is not None else None

    if isinstance(model, ConstantVelocityBaseline):
        train_metrics = evaluate_model(model, loaders["train"], device)
        val_metrics = evaluate_model(model, loaders["val"], device)
        _append_jsonl(
            run_dir / "metrics_train.jsonl",
            {"epoch": 0, **_jsonable_metrics(train_metrics)},
        )
        _append_jsonl(
            run_dir / "metrics_val.jsonl",
            {"epoch": 0, **_jsonable_metrics(val_metrics)},
        )
        best_metric = float(val_metrics.get("player_ADE_m", float("inf")))
        _save_checkpoint(latest_path, model, cfg, data, split_indices, run_dir, 0, best_metric)
        shutil.copy2(latest_path, best_path)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
        )
        for epoch in range(1, epochs + 1):
            model.train()
            train_losses: list[float] = []
            iterator = tqdm(loaders["train"], desc=f"epoch {epoch}", leave=False)
            for batch_idx, batch in enumerate(iterator, start=1):
                batch = batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                pred = predict_batch(model, batch)
                loss = masked_mse_loss(pred, batch["future_xy"], batch["future_mask"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(float(loss.item()))
                if hasattr(iterator, "set_postfix"):
                    iterator.set_postfix(loss=f"{loss.item():.4f}")
                if max_train_batches is not None and batch_idx >= max_train_batches:
                    break

            train_metrics = evaluate_model(model, loaders["train"], device)
            train_metrics["loss"] = float(np.mean(train_losses)) if train_losses else math.nan
            val_metrics = evaluate_model(model, loaders["val"], device)
            _append_jsonl(
                run_dir / "metrics_train.jsonl",
                {"epoch": epoch, **_jsonable_metrics(train_metrics)},
            )
            _append_jsonl(
                run_dir / "metrics_val.jsonl",
                {"epoch": epoch, **_jsonable_metrics(val_metrics)},
            )
            current = float(val_metrics.get("player_ADE_m", float("inf")))
            _save_checkpoint(
                latest_path,
                model,
                cfg,
                data,
                split_indices,
                run_dir,
                epoch,
                best_metric,
            )
            if current < best_metric:
                best_metric = current
                _save_checkpoint(
                    best_path,
                    model,
                    cfg,
                    data,
                    split_indices,
                    run_dir,
                    epoch,
                    best_metric,
                )

    test_metrics = evaluate_model(model, loaders["test"], device)
    with (run_dir / "eval_test.json").open("w", encoding="utf-8") as handle:
        json.dump(_jsonable_metrics(test_metrics), handle, indent=2)
    _save_prediction_sample(run_dir / "predictions_sample.pt", model, loaders["test"], device)

    model_root = run_root / model_name
    shutil.copy2(latest_path, model_root / "latest.pt")
    shutil.copy2(best_path, model_root / "best.pt")
    return {
        "run_dir": run_dir,
        "latest_checkpoint": latest_path,
        "best_checkpoint": best_path,
        "test_metrics": test_metrics,
    }
