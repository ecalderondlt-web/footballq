"""Training loop for Soccer TD-JEPA pretraining."""

from __future__ import annotations

import json
import math
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - minimal environments

    def tqdm(iterable: Any, *_args: Any, **_kwargs: Any) -> Any:
        return iterable


from footballq.data.td_jepa_dataset import TDJEPAData, TDJEPADataset, load_td_jepa_data
from footballq.models.td_jepa import SoccerTDJEPA
from footballq.repro.manifest import build_run_manifest, write_run_manifest
from footballq.repro.splits import split_indices_from_manifest
from footballq.training.ema import update_ema
from footballq.training.td_jepa_losses import td_jepa_loss
from footballq.training.train import resolve_device, split_indices_by_match


def load_td_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    with Path(config).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_td_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_td_jepa_model(config: dict[str, Any], data: TDJEPAData) -> SoccerTDJEPA:
    model_cfg = config.get("model", {})
    return SoccerTDJEPA(
        context_steps=data.context_steps,
        delta_steps=data.delta_steps,
        n_entities=int(data.state_t.shape[2]),
        n_features=data.n_features,
        z_dim=int(model_cfg.get("z_dim", 128)),
        d_model=int(model_cfg.get("d_model", 128)),
        n_heads=int(model_cfg.get("n_heads", 4)),
        n_layers=int(model_cfg.get("n_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        motion_hidden_dim=int(model_cfg.get("motion_hidden_dim", 256)),
        pooling=str(model_cfg.get("pooling", "mean")),
        state_decoder_hidden_dim=(
            int(model_cfg["state_decoder_hidden_dim"])
            if model_cfg.get("state_decoder_hidden_dim") is not None
            else None
        ),
    )


def td_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _metric_row(metrics: dict[str, float]) -> dict[str, float | int | None]:
    row: dict[str, float | int | None] = {}
    for key, value in metrics.items():
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            row[key] = None
        else:
            row[key] = value
    return row


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")


def _loss_from_outputs(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    loss_cfg: dict[str, Any],
) -> dict[str, torch.Tensor]:
    return td_jepa_loss(
        outputs["z_pred"],
        outputs["z_target"],
        outputs["z_t"],
        variance_weight=float(loss_cfg.get("variance_weight", 0.1)),
        variance_threshold=float(loss_cfg.get("variance_threshold", 1.0)),
        state_reconstruction=outputs.get("state_reconstruction"),
        state_target=batch.get("state_t_plus_delta"),
        state_mask=batch.get("mask_t_plus_delta"),
        slot_reconstruction_weight=float(loss_cfg.get("slot_reconstruction_weight", 0.0)),
        no_motion_margin_weight=float(loss_cfg.get("no_motion_margin_weight", 0.0)),
        no_motion_margin=float(loss_cfg.get("no_motion_margin", 0.01)),
    )


def evaluate_td_model(
    model: SoccerTDJEPA,
    loader: DataLoader,
    device: torch.device,
    loss_cfg: dict[str, Any],
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    num_examples = 0
    with torch.no_grad():
        for batch in loader:
            batch = td_batch_to_device(batch, device)
            outputs = model(batch)
            losses = _loss_from_outputs(outputs, batch, loss_cfg)
            batch_size = int(batch["state_t"].shape[0])
            num_examples += batch_size
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu().item()) * batch_size
    if num_examples == 0:
        return {"num_examples": 0}
    return {key: value / num_examples for key, value in totals.items()} | {
        "num_examples": num_examples
    }


def _save_checkpoint(
    path: Path,
    model: SoccerTDJEPA,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    data: TDJEPAData,
    split_indices: dict[str, list[int]],
    run_dir: Path,
    epoch: int,
    step: int,
    best_metric: float,
) -> None:
    torch.save(
        {
            "online_encoder": model.online_encoder.state_dict(),
            "target_encoder": model.target_encoder.state_dict(),
            "motion_encoder": model.motion_encoder.state_dict(),
            "state_decoder": (
                model.state_decoder.state_dict() if model.state_decoder is not None else None
            ),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "epoch": epoch,
            "step": step,
            "best_validation_metric": best_metric,
            "split_indices": split_indices,
            "run_dir": str(run_dir),
            "data_meta": {
                "feature_names": data.feature_names,
                "feature_view": data.feature_view,
                "objective_mode": data.objective_mode,
                "prediction_gap_frames": data.prediction_gap_frames,
                "fps": data.fps,
                "context_seconds": data.context_seconds,
                "delta_seconds": data.delta_seconds,
                "delta_frames": data.delta_frames,
                "repro_metadata": data.metadata or {},
            },
        },
        path,
    )


def _save_embedding_sample(
    path: Path,
    model: SoccerTDJEPA,
    loader: DataLoader,
    device: torch.device,
) -> None:
    try:
        batch = next(iter(loader))
    except StopIteration:
        return
    model.eval()
    with torch.no_grad():
        device_batch = td_batch_to_device(batch, device)
        z = model.online_encoder(device_batch["state_t"], device_batch["mask_t"]).detach().cpu()
    torch.save(
        {
            "z": z,
            "match_id": list(batch["match_id"]),
            "period": [int(value) for value in batch["period"]],
            "frame_t": [int(value) for value in batch["frame_t"]],
            "sample_id": list(batch["sample_id"]),
            "delta_frames": [int(value) for value in batch["delta_frames"]],
        },
        path,
    )


def train_td_jepa_from_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    cfg = load_td_config(config)
    seed = int(cfg.get("seed", cfg.get("training", {}).get("seed", 7)))
    set_td_seed(seed)
    data_path = Path(cfg.get("data", {}).get("path", cfg.get("data", {}).get("td_jepa", "")))
    if not data_path.exists():
        raise FileNotFoundError(
            f"TD-JEPA data file not found: {data_path}. Run scripts/prepare_td_jepa_data.py first."
        )
    data = load_td_jepa_data(data_path)
    if len(data.match_id) == 0:
        raise ValueError("TD-JEPA data file contains zero examples.")

    split_cfg = cfg.get("split", {})
    split_manifest = split_cfg.get("manifest_path") or split_cfg.get("manifest")
    if split_manifest:
        split_indices = split_indices_from_manifest(data.match_id, split_manifest)
    else:
        split_indices = split_indices_by_match(
            data.match_id,
            val_fraction=float(split_cfg.get("val_fraction", 0.2)),
            test_fraction=float(split_cfg.get("test_fraction", 0.2)),
            seed=seed,
        )
    train_cfg = cfg.get("training", {})
    batch_size = int(train_cfg.get("batch_size", cfg.get("data", {}).get("batch_size", 64)))
    num_workers = int(train_cfg.get("num_workers", cfg.get("data", {}).get("num_workers", 0)))
    loaders = {
        split: DataLoader(
            TDJEPADataset(data, indices=indices),
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
        )
        for split, indices in split_indices.items()
    }

    device = resolve_device(train_cfg.get("device", "auto"))
    model = create_td_jepa_model(cfg, data).to(device)
    trainable_parameters = list(model.online_encoder.parameters()) + list(
        model.motion_encoder.parameters()
    )
    if model.state_decoder is not None:
        trainable_parameters += list(model.state_decoder.parameters())
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    loss_cfg = cfg.get("loss", {})
    ema_momentum = float(cfg.get("ema", {}).get("momentum", train_cfg.get("ema_momentum", 0.996)))
    run_root = Path(train_cfg.get("run_root", "runs"))
    run_dir = run_root / "td_jepa" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = run_dir / "config.yaml"
    with run_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    latest_path = run_dir / "latest.pt"
    best_path = run_dir / "best.pt"
    best_metric = float("inf")
    step = 0
    epochs = int(train_cfg.get("max_epochs", train_cfg.get("epochs", 1)))
    max_train_batches = train_cfg.get("max_train_batches")
    max_train_batches = int(max_train_batches) if max_train_batches is not None else None
    metric_name = str(train_cfg.get("best_metric", "total_loss"))

    for epoch in range(1, epochs + 1):
        model.train()
        train_totals: dict[str, float] = {}
        train_examples = 0
        iterator = tqdm(loaders["train"], desc=f"td-jepa epoch {epoch}", leave=False)
        for batch_idx, batch in enumerate(iterator, start=1):
            batch = td_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            losses = _loss_from_outputs(outputs, batch, loss_cfg)
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=float(train_cfg.get("grad_clip_norm", 1.0)),
            )
            optimizer.step()
            update_ema(model.target_encoder, model.online_encoder, momentum=ema_momentum)
            step += 1

            batch_size_now = int(batch["state_t"].shape[0])
            train_examples += batch_size_now
            for key, value in losses.items():
                train_totals[key] = (
                    train_totals.get(key, 0.0) + float(value.detach().cpu().item()) * batch_size_now
                )
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(total_loss=f"{losses['total_loss'].item():.4f}")
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break

        train_metrics = {
            key: value / max(train_examples, 1) for key, value in train_totals.items()
        } | {"num_examples": train_examples}
        val_metrics = evaluate_td_model(model, loaders["val"], device, loss_cfg)
        _append_jsonl(
            run_dir / "metrics_train.jsonl",
            {"epoch": epoch, **_metric_row(train_metrics)},
        )
        _append_jsonl(
            run_dir / "metrics_val.jsonl",
            {"epoch": epoch, **_metric_row(val_metrics)},
        )

        current = float(val_metrics.get(metric_name, float("inf")))
        _save_checkpoint(
            latest_path,
            model,
            optimizer,
            cfg,
            data,
            split_indices,
            run_dir,
            epoch,
            step,
            best_metric,
        )
        if current < best_metric:
            best_metric = current
            _save_checkpoint(
                best_path,
                model,
                optimizer,
                cfg,
                data,
                split_indices,
                run_dir,
                epoch,
                step,
                best_metric,
            )

    _save_embedding_sample(run_dir / "embeddings_sample.pt", model, loaders["test"], device)
    if split_manifest:
        manifest_path = run_dir / "run_manifest.json"
        manifest = build_run_manifest(
            command=sys.argv,
            config_path=run_config_path,
            split_manifest_path=split_manifest,
            evaluation_protocol=str(split_cfg.get("protocol", "inductive")),
            feature_view=data.feature_view,
            objective_mode=data.objective_mode,
            dataset_paths={"td_jepa": data_path},
            output_paths={
                "run_dir": run_dir,
                "latest_checkpoint": latest_path,
                "best_checkpoint": best_path,
                "embeddings_sample": run_dir / "embeddings_sample.pt",
                "run_manifest": manifest_path,
            },
            warnings=list((data.metadata or {}).get("warnings", [])),
        )
        write_run_manifest(manifest_path, manifest)
    model_root = run_root / "td_jepa"
    shutil.copy2(latest_path, model_root / "latest.pt")
    shutil.copy2(best_path, model_root / "best.pt")
    shutil.copy2(run_dir / "embeddings_sample.pt", model_root / "embeddings_sample.pt")
    return {
        "run_dir": run_dir,
        "latest_checkpoint": latest_path,
        "best_checkpoint": best_path,
        "best_metric": best_metric,
    }
