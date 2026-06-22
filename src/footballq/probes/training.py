"""Training and evaluation loops for frozen probes."""

from __future__ import annotations

import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from footballq.probes.dataset import ProbeDataset, ProbeDatasetData, load_probe_dataset
from footballq.probes.io import append_jsonl, load_yaml, save_json, save_yaml
from footballq.probes.metrics import classification_metrics, regression_metrics
from footballq.probes.models import create_probe_model
from footballq.training.train import resolve_device


def load_probe_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    return load_yaml(config)


def set_probe_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _target_name(cfg: dict[str, Any]) -> str:
    return str(cfg.get("target", {}).get("name", cfg.get("target_name", "")))


def _feature_source(cfg: dict[str, Any]) -> str:
    return str(cfg.get("features", {}).get("source", cfg.get("feature_source", "td_jepa")))


def _probe_type(cfg: dict[str, Any]) -> str:
    return str(cfg.get("model", {}).get("probe_type", cfg.get("probe_type", "linear")))


def _dataset_path(cfg: dict[str, Any]) -> Path:
    value = cfg.get("data", {}).get("probe_dataset", cfg.get("data", {}).get("path", ""))
    if not value:
        raise ValueError("Probe config must set data.probe_dataset.")
    return Path(value)


def _num_classes(data: ProbeDatasetData, target_name: str) -> int:
    if target_name in data.label_maps:
        return len(data.label_maps[target_name])
    target = torch.as_tensor(data.examples["targets"][target_name]).long()
    mask = torch.as_tensor(data.examples["target_masks"][target_name]).bool()
    if not mask.any():
        raise ValueError(f"Target {target_name!r} has no valid examples.")
    return int(target[mask].max().item()) + 1


def _build_datasets(
    data: ProbeDatasetData,
    target_name: str,
    feature_source: str,
    random_seed: int,
) -> dict[str, ProbeDataset]:
    return {
        split: ProbeDataset(
            data,
            target_name=target_name,
            feature_source=feature_source,
            split=split,
            random_seed=random_seed,
        )
        for split in ["train", "val", "test"]
    }


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def evaluate_probe_model(
    model: torch.nn.Module,
    loader: DataLoader,
    task_type: str,
    device: torch.device,
    num_classes: int | None = None,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    ys: list[torch.Tensor] = []
    outputs_all: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            output = model(batch["x"])
            if task_type == "classification":
                loss = torch.nn.functional.cross_entropy(output, batch["y"].long())
                outputs_all.append(output.detach().cpu())
                ys.append(batch["y"].detach().cpu().long())
            else:
                pred = output.view(-1)
                target = batch["y"].float().view(-1)
                loss = torch.nn.functional.mse_loss(pred, target)
                outputs_all.append(pred.detach().cpu())
                ys.append(target.detach().cpu())
            losses.append(float(loss.item()))
    if not ys:
        return {"loss": math.nan, "num_examples": 0}
    y_true = torch.cat(ys, dim=0)
    outputs = torch.cat(outputs_all, dim=0)
    if task_type == "classification":
        assert num_classes is not None
        y_pred = outputs.argmax(dim=1)
        metrics = classification_metrics(y_true, y_pred, num_classes=num_classes)
    else:
        metrics = regression_metrics(y_true, outputs)
    metrics["loss"] = float(np.mean(losses)) if losses else math.nan
    return metrics


def _is_better(task_type: str, current: float, best: float) -> bool:
    if math.isnan(current):
        return False
    if task_type == "classification":
        return current > best
    return current < best


def _best_metric_name(task_type: str, cfg: dict[str, Any]) -> str:
    default = "macro_f1" if task_type == "classification" else "rmse"
    return str(cfg.get("training", {}).get("best_metric", default))


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    data: ProbeDatasetData,
    run_dir: Path,
    epoch: int,
    best_metric: float,
    input_dim: int,
    output_dim: int,
    task_type: str,
) -> None:
    target_name = _target_name(cfg)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "target_name": target_name,
            "task_type": task_type,
            "feature_source": _feature_source(cfg),
            "probe_type": _probe_type(cfg),
            "input_dim": input_dim,
            "output_dim": output_dim,
            "label_map": data.label_maps.get(target_name),
            "split_indices": {
                key: value
                for key, value in data.splits.items()
                if key.endswith("_indices")
            },
            "split_match_ids": {
                key: value
                for key, value in data.splits.items()
                if key.endswith("_match_ids")
            },
            "run_dir": str(run_dir),
            "epoch": epoch,
            "best_metric": best_metric,
            "encoder_frozen": True,
        },
        path,
    )


def _save_prediction_sample(
    path: Path,
    model: torch.nn.Module,
    loader: DataLoader,
    task_type: str,
    device: torch.device,
) -> None:
    try:
        batch = next(iter(loader))
    except StopIteration:
        return
    model.eval()
    with torch.no_grad():
        device_batch = _batch_to_device(batch, device)
        output = model(device_batch["x"]).detach().cpu()
    payload: dict[str, Any] = {
        "indices": batch["index"][:32].detach().cpu(),
        "target": batch["y"][:32].detach().cpu(),
    }
    if task_type == "classification":
        payload["logits"] = output[:32]
        payload["prediction"] = output.argmax(dim=1)[:32]
    else:
        payload["prediction"] = output.view(-1)[:32]
    torch.save(payload, path)


def train_probe_from_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Train one frozen probe from a YAML config or config dict."""

    cfg = load_probe_config(config)
    train_cfg = cfg.get("training", {})
    seed = int(train_cfg.get("seed", cfg.get("seed", 123)))
    set_probe_seed(seed)
    data = load_probe_dataset(_dataset_path(cfg))
    target_name = _target_name(cfg)
    if target_name not in data.examples.get("targets", {}):
        raise ValueError(
            f"Target {target_name!r} not found in probe dataset. "
            f"Available: {sorted(data.examples.get('targets', {}))}"
        )
    task_type = str(cfg.get("target", {}).get("task_type", data.target_types[target_name]))
    feature_source = _feature_source(cfg)
    random_seed = int(cfg.get("features", {}).get("random_seed", seed))
    datasets = _build_datasets(data, target_name, feature_source, random_seed=random_seed)
    if len(datasets["train"]) == 0 or len(datasets["val"]) == 0:
        raise ValueError(
            f"Target {target_name!r} has empty train or val split after mask filtering."
        )
    batch_size = int(train_cfg.get("batch_size", 256))
    num_workers = int(train_cfg.get("num_workers", 0))
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
        )
        for split, dataset in datasets.items()
    }
    input_dim = int(datasets["train"].features.shape[1])
    output_dim = _num_classes(data, target_name) if task_type == "classification" else 1
    model_cfg = cfg.get("model", {})
    model = create_probe_model(
        _probe_type(cfg),
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )
    device = resolve_device(train_cfg.get("device", "auto"))
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    run_root = Path(train_cfg.get("run_root", "runs"))
    run_dir = (
        run_root
        / "probes"
        / target_name
        / feature_source
        / _probe_type(cfg)
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, run_dir / "config.yaml")
    if task_type == "classification" and target_name in data.label_maps:
        save_json(data.label_maps[target_name], run_dir / "label_map.json")

    best_metric_name = _best_metric_name(task_type, cfg)
    best_metric = -float("inf") if task_type == "classification" else float("inf")
    latest_path = run_dir / "latest.pt"
    best_path = run_dir / "best.pt"
    max_epochs = int(train_cfg.get("max_epochs", train_cfg.get("epochs", 50)))
    patience = int(train_cfg.get("patience", 8))
    max_train_batches = train_cfg.get("max_train_batches")
    max_train_batches = int(max_train_batches) if max_train_batches is not None else None
    stale_epochs = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses: list[float] = []
        for batch_idx, batch in enumerate(loaders["train"], start=1):
            batch = _batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch["x"])
            if task_type == "classification":
                loss = torch.nn.functional.cross_entropy(output, batch["y"].long())
            else:
                loss = torch.nn.functional.mse_loss(output.view(-1), batch["y"].float().view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.item()))
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break

        train_metrics = evaluate_probe_model(
            model,
            loaders["train"],
            task_type,
            device,
            num_classes=output_dim if task_type == "classification" else None,
        )
        train_metrics["loss"] = float(np.mean(losses)) if losses else math.nan
        val_metrics = evaluate_probe_model(
            model,
            loaders["val"],
            task_type,
            device,
            num_classes=output_dim if task_type == "classification" else None,
        )
        append_jsonl(run_dir / "metrics_train.jsonl", {"epoch": epoch, **train_metrics})
        append_jsonl(run_dir / "metrics_val.jsonl", {"epoch": epoch, **val_metrics})

        current = float(val_metrics.get(best_metric_name, math.nan))
        _save_checkpoint(
            latest_path,
            model,
            optimizer,
            cfg,
            data,
            run_dir,
            epoch,
            best_metric,
            input_dim,
            output_dim,
            task_type,
        )
        if _is_better(task_type, current, best_metric):
            best_metric = current
            stale_epochs = 0
            _save_checkpoint(
                best_path,
                model,
                optimizer,
                cfg,
                data,
                run_dir,
                epoch,
                best_metric,
                input_dim,
                output_dim,
                task_type,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    if not best_path.exists():
        _save_checkpoint(
            best_path,
            model,
            optimizer,
            cfg,
            data,
            run_dir,
            max_epochs,
            best_metric,
            input_dim,
            output_dim,
            task_type,
        )
    test_metrics = evaluate_probe_checkpoint(best_path, split="test")
    _save_prediction_sample(run_dir / "predictions_sample.pt", model, loaders["test"], task_type, device)
    return {
        "run_dir": run_dir,
        "latest_checkpoint": latest_path,
        "best_checkpoint": best_path,
        "best_metric": best_metric,
        "test_metrics": test_metrics["metrics"],
    }


def evaluate_probe_checkpoint(
    checkpoint: str | Path,
    split: str = "test",
    data_path: str | Path | None = None,
    device: str | None = "auto",
) -> dict[str, Any]:
    """Load a saved probe checkpoint and evaluate one split."""

    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    cfg = dict(payload["config"])
    if data_path is not None:
        cfg.setdefault("data", {})["probe_dataset"] = str(data_path)
    data = load_probe_dataset(_dataset_path(cfg))
    target_name = str(payload["target_name"])
    task_type = str(payload["task_type"])
    feature_source = str(payload["feature_source"])
    random_seed = int(cfg.get("features", {}).get("random_seed", cfg.get("training", {}).get("seed", 123)))
    dataset = ProbeDataset(
        data,
        target_name=target_name,
        feature_source=feature_source,
        split=split,
        random_seed=random_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("training", {}).get("batch_size", 256)),
        shuffle=False,
        num_workers=int(cfg.get("training", {}).get("num_workers", 0)),
    )
    model = create_probe_model(
        str(payload["probe_type"]),
        input_dim=int(payload["input_dim"]),
        output_dim=int(payload["output_dim"]),
        hidden_dim=int(cfg.get("model", {}).get("hidden_dim", 128)),
        dropout=float(cfg.get("model", {}).get("dropout", 0.1)),
    )
    torch_device = resolve_device(device)
    model.load_state_dict(payload["model_state_dict"])
    model = model.to(torch_device)
    metrics = evaluate_probe_model(
        model,
        loader,
        task_type,
        torch_device,
        num_classes=int(payload["output_dim"]) if task_type == "classification" else None,
    )
    metrics["feature_source"] = feature_source
    metrics["target_name"] = target_name
    metrics["probe_type"] = str(payload["probe_type"])
    metrics["split"] = split
    metrics["num_train_examples"] = len(
        ProbeDataset(data, target_name, feature_source, split="train", random_seed=random_seed)
    )
    metrics["num_val_examples"] = len(
        ProbeDataset(data, target_name, feature_source, split="val", random_seed=random_seed)
    )
    metrics["num_test_examples"] = len(
        ProbeDataset(data, target_name, feature_source, split="test", random_seed=random_seed)
    )
    metrics["split_match_ids"] = {
        key: value
        for key, value in data.splits.items()
        if key.endswith("_match_ids")
    }
    run_dir = Path(payload.get("run_dir", Path(checkpoint).parent))
    save_json(metrics, run_dir / f"eval_{split}.json")
    return {"metrics": metrics, "run_dir": run_dir}
