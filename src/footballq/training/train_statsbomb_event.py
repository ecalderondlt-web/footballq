"""Training loop for causal StatsBomb event encoders."""

from __future__ import annotations

import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader

from footballq.data.statsbomb_event_dataset import (
    ShardedStatsBombEventDataset,
    StatsBombShardGroupedSampler,
)
from footballq.models.statsbomb_event_encoder import (
    StatsBombEventEncoder,
    statsbomb_event_loss,
)
from footballq.repro.manifest import build_run_manifest, file_sha256, write_run_manifest
from footballq.training.train import resolve_device


def load_statsbomb_event_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    path = Path(config)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["_config_path"] = str(path)
    return payload


def set_statsbomb_event_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_statsbomb_event_model(
    config: dict[str, Any],
    manifest: dict[str, Any],
) -> StatsBombEventEncoder:
    model_cfg = config.get("model", {})
    feature_view = str(model_cfg.get("feature_view", "event_only"))
    if feature_view not in {"event_only", "event_plus_360"}:
        raise ValueError("StatsBomb feature_view must be 'event_only' or 'event_plus_360'.")
    names = manifest["categorical_feature_names"]
    vocabularies = manifest["categorical_vocabularies"]
    sizes = [int(vocabularies[name]["size"]) for name in names]
    return StatsBombEventEncoder(
        sizes,
        len(manifest["continuous_feature_names"]),
        len(manifest["freeze_frame_feature_names"]),
        event_type_feature_index=names.index("event_type"),
        use_360=feature_view == "event_plus_360",
        categorical_dim=int(model_cfg.get("categorical_dim", 24)),
        d_model=int(model_cfg.get("d_model", 128)),
        n_heads=int(model_cfg.get("n_heads", 4)),
        n_layers=int(model_cfg.get("n_layers", 3)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        max_sequence_length=int(model_cfg.get("max_sequence_length", 64)),
    )


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")


@torch.no_grad()
def evaluate_statsbomb_event_model(
    model: StatsBombEventEncoder,
    loader: DataLoader,
    device: torch.device,
    *,
    location_weight: float,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    totals = {
        "event_type_loss_sum": 0.0,
        "event_type_correct": 0.0,
        "event_targets": 0.0,
        "location_loss_sum": 0.0,
        "location_mae_sum": 0.0,
        "location_targets": 0.0,
        "windows": 0.0,
        "events_with_360": 0.0,
        "anchored_event_type_loss_sum": 0.0,
        "anchored_event_type_correct": 0.0,
        "anchored_event_targets": 0.0,
        "anchored_location_loss_sum": 0.0,
        "anchored_location_mae_sum": 0.0,
        "anchored_location_targets": 0.0,
    }
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = _move_batch(batch, device)
        outputs = model(batch)
        losses = statsbomb_event_loss(outputs, batch, location_weight=location_weight)
        event_targets = float(batch["event_mask"].sum())
        location_targets = float(batch["target_location_mask"].sum())
        totals["event_type_loss_sum"] += float(losses["event_type_loss"]) * event_targets
        totals["event_type_correct"] += float(losses["event_type_accuracy"]) * event_targets
        totals["event_targets"] += event_targets
        totals["location_loss_sum"] += float(losses["location_loss"]) * location_targets
        totals["location_mae_sum"] += float(losses["location_mae"]) * location_targets
        totals["location_targets"] += location_targets
        totals["windows"] += int(batch["categorical"].shape[0])
        totals["events_with_360"] += float(batch["has_360"].sum())
        anchored = batch["event_mask"].bool() & batch["has_360"].bool()
        anchored_count = float(anchored.sum())
        if anchored_count:
            logits = outputs["next_event_type_logits"][anchored]
            target_type = batch["target_event_type"][anchored]
            totals["anchored_event_type_loss_sum"] += float(
                F.cross_entropy(logits, target_type, reduction="sum")
            )
            totals["anchored_event_type_correct"] += float(
                (logits.argmax(dim=-1) == target_type).sum()
            )
            totals["anchored_event_targets"] += anchored_count
        anchored_location = anchored & batch["target_location_mask"].bool()
        anchored_location_count = float(anchored_location.sum())
        if anchored_location_count:
            predicted = outputs["next_location"][anchored_location]
            target = batch["target_location"][anchored_location]
            totals["anchored_location_loss_sum"] += float(
                F.smooth_l1_loss(predicted, target, reduction="sum") / 2.0
            )
            totals["anchored_location_mae_sum"] += float(
                F.l1_loss(predicted, target, reduction="sum") / 2.0
            )
            totals["anchored_location_targets"] += anchored_location_count
    event_count = max(totals["event_targets"], 1.0)
    location_count = max(totals["location_targets"], 1.0)
    anchored_event_count = max(totals["anchored_event_targets"], 1.0)
    anchored_location_count = max(totals["anchored_location_targets"], 1.0)
    event_type_loss = totals["event_type_loss_sum"] / event_count
    location_loss = totals["location_loss_sum"] / location_count
    return {
        "total_loss": event_type_loss + location_weight * location_loss,
        "event_type_loss": event_type_loss,
        "event_type_accuracy": totals["event_type_correct"] / event_count,
        "location_loss": location_loss,
        "location_mae": totals["location_mae_sum"] / location_count,
        "event_targets": totals["event_targets"],
        "location_targets": totals["location_targets"],
        "windows": totals["windows"],
        "events_with_360": totals["events_with_360"],
        "anchored_event_type_loss": (
            totals["anchored_event_type_loss_sum"] / anchored_event_count
        ),
        "anchored_event_type_accuracy": (
            totals["anchored_event_type_correct"] / anchored_event_count
        ),
        "anchored_location_loss": (
            totals["anchored_location_loss_sum"] / anchored_location_count
        ),
        "anchored_location_mae": (
            totals["anchored_location_mae_sum"] / anchored_location_count
        ),
        "anchored_event_targets": totals["anchored_event_targets"],
        "anchored_location_targets": totals["anchored_location_targets"],
    }


def train_statsbomb_event_from_config(
    config: str | Path | dict[str, Any],
) -> dict[str, Any]:
    cfg = load_statsbomb_event_config(config)
    data_cfg = cfg["data"]
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    manifest_path = Path(data_cfg["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("loaded_splits") != ["train", "val"] or manifest.get("test_loaded"):
        raise ValueError("StatsBomb training requires a train/val-only processed manifest.")
    split_path = Path(data_cfg["split_manifest"])
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    seed = int(train_cfg.get("seed", cfg.get("seed", 7)))
    set_statsbomb_event_seed(seed)
    device = resolve_device(train_cfg.get("device", "auto"))
    train_data = ShardedStatsBombEventDataset(
        manifest_path,
        "train",
        cache_size=int(data_cfg.get("cache_size", 2)),
    )
    val_data = ShardedStatsBombEventDataset(
        manifest_path,
        "val",
        cache_size=int(data_cfg.get("cache_size", 2)),
    )
    train_sampler = StatsBombShardGroupedSampler(train_data, shuffle=True, seed=seed)
    val_sampler = StatsBombShardGroupedSampler(val_data, shuffle=False, seed=seed)
    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(train_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model = create_statsbomb_event_model(cfg, manifest).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    location_weight = float(train_cfg.get("location_weight", 1.0))
    max_updates = int(train_cfg.get("max_train_updates", 0)) or None
    max_epochs = int(train_cfg.get("max_epochs", 1))
    val_max_batches = train_cfg.get("val_max_batches")
    val_max_batches = int(val_max_batches) if val_max_batches is not None else None
    validation_curve_steps = {
        int(value) for value in train_cfg.get("validation_curve_steps", [])
    }
    validation_curve_max_batches = train_cfg.get("validation_curve_max_batches")
    validation_curve_max_batches = (
        int(validation_curve_max_batches)
        if validation_curve_max_batches is not None
        else None
    )
    log_every = max(1, int(train_cfg.get("log_every_updates", 100)))
    run_root = Path(cfg.get("run", {}).get("root", "runs/statsbomb_event"))
    run_dir = run_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    config_path = cfg.get("_config_path")
    if config_path is not None:
        shutil.copy2(config_path, run_dir / "config.yaml")

    step = 0
    best_metric = None
    last_epoch = -1
    for epoch in range(max_epochs):
        last_epoch = epoch
        model.train()
        running_loss = 0.0
        running_batches = 0
        for batch in train_loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            losses = statsbomb_event_loss(outputs, batch, location_weight=location_weight)
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(train_cfg.get("grad_clip", 1.0)),
            )
            optimizer.step()
            step += 1
            running_loss += float(losses["total_loss"].detach())
            running_batches += 1
            if step % log_every == 0:
                print(f"step={step} train_total_loss={running_loss / running_batches:.6f}")
                running_loss = 0.0
                running_batches = 0
            if step in validation_curve_steps:
                curve_metrics = evaluate_statsbomb_event_model(
                    model,
                    val_loader,
                    device,
                    location_weight=location_weight,
                    max_batches=validation_curve_max_batches,
                )
                _append_jsonl(
                    run_dir / "metrics_val_curve.jsonl",
                    {"epoch": epoch, "step": step, "split": "val", **curve_metrics},
                )
                model.train()
            if max_updates is not None and step >= max_updates:
                break

        val_metrics = evaluate_statsbomb_event_model(
            model,
            val_loader,
            device,
            location_weight=location_weight,
            max_batches=val_max_batches,
        )
        row = {"epoch": epoch, "step": step, "split": "val", **val_metrics}
        _append_jsonl(run_dir / "metrics_val.jsonl", row)
        checkpoint = {
            "version": 1,
            "epoch": epoch,
            "step": step,
            "config": {key: value for key, value in cfg.items() if key != "_config_path"},
            "feature_view": model_cfg.get("feature_view", "event_only"),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "data_manifest_path": str(manifest_path),
            "data_manifest_sha256": file_sha256(manifest_path),
            "data_manifest_payload_sha256": manifest["manifest_payload_sha256"],
            "split_manifest_sha256": manifest["split_manifest_sha256"],
            "validation": val_metrics,
        }
        torch.save(checkpoint, run_dir / "latest.pt")
        if best_metric is None or val_metrics["total_loss"] < best_metric:
            best_metric = val_metrics["total_loss"]
            shutil.copy2(run_dir / "latest.pt", run_dir / "best.pt")
        if max_updates is not None and step >= max_updates:
            break

    run_manifest = build_run_manifest(
        command="train_statsbomb_event",
        config_path=config_path,
        split_manifest_path=split_path,
        evaluation_protocol="match_inductive_validation_only",
        feature_view=str(model_cfg.get("feature_view", "event_only")),
        objective_mode="causal_next_event_type_and_location",
        dataset_paths={"event_manifest": manifest_path},
        output_paths={"run_dir": run_dir, "latest_checkpoint": run_dir / "latest.pt"},
        warnings=(
            ["Smoke run for pipeline validation only; not scientific evidence."]
            if max_updates is not None and max_updates < 100
            else []
        ),
    )
    run_manifest["data_access"] = {
        "loaded_splits": ["train", "val"],
        "test_loaded": False,
        "embedding_export_split": None,
    }
    run_manifest["event_data"] = {
        "manifest_sha256": file_sha256(manifest_path),
        "manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "source_manifest_payload_sha256": manifest["source_manifest_payload_sha256"],
        "schema_audit_payload_sha256": manifest["schema_audit_payload_sha256"],
        "vocabulary_payload_sha256": manifest["vocabulary_payload_sha256"],
        "source_commit": manifest["source_commit"],
    }
    run_manifest["training"] = {
        "seed": seed,
        "device": str(device),
        "final_step": step,
        "final_epoch": last_epoch,
        "best_validation_total_loss": best_metric,
    }
    write_run_manifest(run_dir / "run_manifest.json", run_manifest)
    return {
        "run_dir": run_dir,
        "latest_checkpoint": run_dir / "latest.pt",
        "best_checkpoint": run_dir / "best.pt",
        "best_metric": best_metric,
        "step": step,
    }
