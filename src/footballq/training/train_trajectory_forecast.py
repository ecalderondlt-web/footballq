"""Validation-only training for matched PFF multi-horizon trajectory forecasters."""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from footballq.constants import PITCH_CENTER_X_M, PITCH_CENTER_Y_M
from footballq.data.pff_forecasting import PFFForecastDataset
from footballq.data.sharded_td_dataset import ShardGroupedSampler
from footballq.data.windows import ENTITY_BALL, ENTITY_PLAYER
from footballq.models.trajectory_forecaster import (
    FORECAST_FAMILIES,
    MultiHorizonTrajectoryForecaster,
    predict_constant_velocity,
    predict_last_position,
)
from footballq.repro.manifest import build_run_manifest, file_sha256, write_run_manifest
from footballq.training.train import resolve_device
from footballq.training.train_td_jepa import create_td_jepa_model


def _set_seed(seed: int) -> None:
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


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def masked_displacement_loss_m(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean endpoint displacement in meters over valid entity-horizon pairs."""

    scale = torch.tensor(
        [PITCH_CENTER_X_M, PITCH_CENTER_Y_M],
        dtype=prediction.dtype,
        device=prediction.device,
    )
    distance = torch.linalg.vector_norm((prediction - target) * scale, dim=-1)
    if not bool(mask.any()):
        return prediction.sum() * 0.0
    return distance[mask].mean()


def _horizon_label(seconds: float) -> str:
    return str(float(seconds)).replace(".", "p")


def evaluate_trajectory_predictor(
    predictor: Callable[[dict[str, Any]], torch.Tensor],
    loader: DataLoader,
    device: torch.device,
    *,
    horizons_seconds: tuple[float, ...],
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Stream displacement metrics without retaining validation predictions."""

    sums = {
        kind: torch.zeros(len(horizons_seconds), dtype=torch.float64)
        for kind in ("all_entity", "player", "ball")
    }
    counts = {
        kind: torch.zeros(len(horizons_seconds), dtype=torch.long)
        for kind in ("all_entity", "player", "ball")
    }
    sample_digest = hashlib.sha256()
    example_count = 0
    scale = torch.tensor([PITCH_CENTER_X_M, PITCH_CENTER_Y_M], device=device)
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            for sample_id in batch["sample_id"]:
                sample_digest.update(str(sample_id).encode("utf-8") + b"\n")
            example_count += int(batch["state_t"].shape[0])
            batch = _batch_to_device(batch, device)
            prediction = predictor(batch)
            distance = torch.linalg.vector_norm(
                (prediction - batch["future_xy"]) * scale.to(prediction.dtype), dim=-1
            )
            selectors = {
                "all_entity": torch.ones_like(batch["entity_type"], dtype=torch.bool),
                "player": batch["entity_type"] == ENTITY_PLAYER,
                "ball": batch["entity_type"] == ENTITY_BALL,
            }
            for kind, selector in selectors.items():
                valid = batch["future_mask"] & selector.unsqueeze(1)
                sums[kind] += (distance * valid).sum(dim=(0, 2)).double().cpu()
                counts[kind] += valid.sum(dim=(0, 2)).long().cpu()
    if example_count == 0:
        raise ValueError("Forecast evaluation received zero examples.")
    metrics: dict[str, Any] = {
        "num_examples": example_count,
        "sample_id_sha256": sample_digest.hexdigest(),
        "horizons_seconds": list(horizons_seconds),
    }
    for kind in sums:
        per_horizon = sums[kind] / counts[kind].clamp_min(1)
        metrics[f"{kind}_ADE_m"] = float(sums[kind].sum() / counts[kind].sum().clamp_min(1))
        metrics[f"{kind}_FDE_m"] = float(per_horizon[-1])
        metrics[f"{kind}_valid_endpoint_count"] = int(counts[kind].sum())
        for index, horizon in enumerate(horizons_seconds):
            label = _horizon_label(horizon)
            metrics[f"{kind}_error_h{label}s_m"] = float(per_horizon[index])
            metrics[f"{kind}_count_h{label}s"] = int(counts[kind][index])
    finite = [
        math.isfinite(value)
        for key, value in metrics.items()
        if isinstance(value, float) and not key.endswith("seconds")
    ]
    if not all(finite):
        raise ValueError("Forecast evaluation produced non-finite metrics.")
    return metrics


def _create_forecaster(
    *,
    family: str,
    seed: int,
    checkpoint_path: Path,
    dataset: PFFForecastDataset,
    config: dict[str, Any],
) -> tuple[MultiHorizonTrajectoryForecaster, dict[str, Any]]:
    if family not in FORECAST_FAMILIES:
        raise ValueError(f"Unknown forecast family {family!r}.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    torch.manual_seed(seed + 10_000)
    source_model = create_td_jepa_model(checkpoint["config"], dataset.prototype)
    encoder = source_model.online_encoder
    weights_loaded = family in {"frozen", "finetuned"}
    if weights_loaded:
        encoder.load_state_dict(checkpoint["online_encoder"], strict=True)
    model_cfg = config.get("model", {})
    source_model_cfg = checkpoint["config"].get("model", {})
    horizons = tuple(float(value) for value in dataset.manifest["horizons_seconds"])
    torch.manual_seed(seed + 20_000)
    model = MultiHorizonTrajectoryForecaster(
        encoder,
        family=family,
        z_dim=int(source_model_cfg.get("z_dim", 128)),
        n_entities=int(dataset.prototype.state_t.shape[2]),
        horizons_seconds=horizons,
        fps=float(dataset.prototype.fps),
        hidden_dim=int(model_cfg.get("hidden_dim", 512)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        representation_mode=str(model_cfg.get("representation_mode", "global")),
        token_dim=int(source_model_cfg.get("d_model", 128)),
        decoder_mode=str(model_cfg.get("decoder_mode", "shared")),
    )
    source = {
        "tracking_checkpoint": str(checkpoint_path),
        "tracking_checkpoint_sha256": file_sha256(checkpoint_path),
        "tracking_checkpoint_step": int(checkpoint["step"]),
        "tracking_checkpoint_family": "scratch",
        "online_encoder_weights_loaded": weights_loaded,
        "encoder_trainable": family in {"raw", "finetuned"},
        "encoder_initialization_seed": seed + 10_000,
        "decoder_initialization_seed": seed + 20_000,
        "representation_mode": str(model_cfg.get("representation_mode", "global")),
        "decoder_mode": str(model_cfg.get("decoder_mode", "shared")),
    }
    return model, source


def evaluate_forecast_baseline(
    config: str | Path | dict[str, Any],
    *,
    baseline: str,
    split: str = "val",
) -> dict[str, Any]:
    """Evaluate last-position or constant-velocity on the frozen validation subset."""

    cfg = (
        yaml.safe_load(Path(config).read_text(encoding="utf-8"))
        if not isinstance(config, dict)
        else dict(config)
    )
    if baseline not in {"last_position", "constant_velocity"}:
        raise ValueError("Unknown forecast baseline.")
    dataset = PFFForecastDataset(cfg["data"]["manifest"], split)
    train_cfg = cfg.get("training", {})
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 128)),
        sampler=ShardGroupedSampler(dataset, shuffle=False, seed=0),
        num_workers=int(train_cfg.get("num_workers", 0)),
    )
    device = resolve_device(train_cfg.get("device", "auto"))
    horizons = tuple(float(value) for value in dataset.manifest["horizons_seconds"])
    prediction_function = (
        predict_last_position if baseline == "last_position" else predict_constant_velocity
    )

    def predictor(batch: dict[str, Any]) -> torch.Tensor:
        return prediction_function(
            batch["state_t"],
            batch["mask_t"],
            horizons,
            fps=float(dataset.prototype.fps),
        )

    return evaluate_trajectory_predictor(
        predictor,
        loader,
        device,
        horizons_seconds=horizons,
        max_batches=train_cfg.get("max_val_batches"),
    )


def train_trajectory_forecast_from_config(
    config: str | Path | dict[str, Any],
    *,
    family: str | None = None,
    seed: int | None = None,
    tracking_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Train one matched forecast family and evaluate validation only."""

    config_path = Path(config) if not isinstance(config, dict) else None
    cfg = (
        yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config_path is not None
        else dict(config)
    )
    train_cfg = cfg.get("training", {})
    family = str(family or cfg.get("family", "raw"))
    seed = int(seed if seed is not None else train_cfg.get("seed", 7))
    checkpoint_path = Path(tracking_checkpoint or cfg["sources"]["tracking_checkpoint"])
    validation_split = str(train_cfg.get("validation_split", "val"))
    if validation_split not in {"train", "val"}:
        raise ValueError("Forecast training permits only train or validation evaluation.")
    cfg["family"] = family
    cfg.setdefault("training", {})["seed"] = seed
    cfg.setdefault("sources", {})["tracking_checkpoint"] = str(checkpoint_path)
    _set_seed(seed)

    train_dataset = PFFForecastDataset(cfg["data"]["manifest"], "train")
    validation_dataset = (
        train_dataset
        if validation_split == "train"
        else PFFForecastDataset(cfg["data"]["manifest"], validation_split)
    )
    batch_size = int(train_cfg.get("batch_size", 128))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=ShardGroupedSampler(train_dataset, shuffle=True, seed=seed),
        num_workers=int(train_cfg.get("num_workers", 0)),
    )
    val_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        sampler=ShardGroupedSampler(validation_dataset, shuffle=False, seed=0),
        num_workers=int(train_cfg.get("num_workers", 0)),
    )
    model, source = _create_forecaster(
        family=family,
        seed=seed,
        checkpoint_path=checkpoint_path,
        dataset=train_dataset,
        config=cfg,
    )
    device = resolve_device(train_cfg.get("device", "auto"))
    model = model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(train_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    max_updates = int(train_cfg["max_train_updates"])
    curve_steps = sorted({int(value) for value in train_cfg.get("validation_curve_steps", [])})
    curve_max_batches = train_cfg.get("validation_curve_max_batches")
    max_val_batches = train_cfg.get("max_val_batches")
    max_epochs = int(train_cfg.get("max_epochs", 3))
    run_root = Path(train_cfg.get("run_root", "runs/pff_trajectory_forecast_v1/models"))
    run_dir = run_root / family / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    run_config_path = run_dir / "config.yaml"
    run_config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    horizons = tuple(float(value) for value in train_dataset.manifest["horizons_seconds"])

    step = 0
    epoch = -1
    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            batch = _batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch["state_t"], batch["mask_t"])
            loss = masked_displacement_loss_m(
                prediction, batch["future_xy"], batch["future_mask"]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable, float(train_cfg.get("grad_clip", 1.0))
            )
            optimizer.step()
            step += 1
            if step in curve_steps:
                model.eval()
                curve = evaluate_trajectory_predictor(
                    lambda value: model(value["state_t"], value["mask_t"]),
                    val_loader,
                    device,
                    horizons_seconds=horizons,
                    max_batches=curve_max_batches,
                )
                _append_jsonl(
                    run_dir / "metrics_val_curve.jsonl",
                    {"epoch": epoch, "step": step, "split": validation_split, **curve},
                )
                model.train()
            if step >= max_updates:
                break
        if step >= max_updates:
            break
    if step != max_updates:
        raise ValueError(f"Forecast run stopped at {step}, expected {max_updates} updates.")

    model.eval()
    final = evaluate_trajectory_predictor(
        lambda value: model(value["state_t"], value["mask_t"]),
        val_loader,
        device,
        horizons_seconds=horizons,
        max_batches=max_val_batches,
    )
    final_row = {"epoch": epoch, "step": step, "split": validation_split, **final}
    _append_jsonl(run_dir / "metrics_val.jsonl", final_row)
    checkpoint = {
        "version": 1,
        "family": family,
        "seed": seed,
        "step": step,
        "config": cfg,
        "encoder": model.encoder.state_dict(),
        "decoder": model.decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "source": source,
        "validation": final_row,
    }
    checkpoint_path_out = run_dir / "latest.pt"
    torch.save(checkpoint, checkpoint_path_out)

    loaded_splits = ["train"] if validation_split == "train" else ["train", "val"]
    representation_mode = str(cfg.get("model", {}).get("representation_mode", "global"))
    decoder_mode = str(cfg.get("model", {}).get("decoder_mode", "shared"))
    evaluation_protocol = {
        "player_ball": "matched_player_ball_head_forecast_validation_only_v1",
        "player_global_ball": "matched_hybrid_context_ball_forecast_validation_only_v1",
    }.get(
        decoder_mode,
        (
            "matched_entity_token_forecast_validation_only_v1"
            if representation_mode == "entity_tokens"
            else "matched_multihorizon_forecast_validation_only_v1"
        ),
    )
    manifest = build_run_manifest(
        command=sys.argv,
        config_path=run_config_path,
        split_manifest_path=cfg["data"]["split_manifest"],
        evaluation_protocol=evaluation_protocol,
        feature_view="position_only_observed_tracking",
        objective_mode=(
            f"{representation_mode}_{decoder_mode}_multi_horizon_endpoint_displacement"
        ),
        dataset_paths={
            "forecast_manifest": cfg["data"]["manifest"],
            "tracking_checkpoint": checkpoint_path,
        },
        output_paths={
            "run_dir": run_dir,
            "latest_checkpoint": checkpoint_path_out,
            "validation_metrics": run_dir / "metrics_val.jsonl",
        },
        warnings=[
            "PFF test tracking is sealed and was not loaded.",
            "Validation curves are diagnostic; the frozen gate uses only the final update.",
        ],
    )
    manifest.update(
        {
            "loaded_splits": loaded_splits,
            "test_loaded": False,
            "embedding_exported": False,
            "family": family,
            "seed": seed,
            "step": step,
            "source": source,
            "validation_sample_id_sha256": final["sample_id_sha256"],
        }
    )
    write_run_manifest(run_dir / "run_manifest.json", manifest)
    return {
        "run_dir": run_dir,
        "latest_checkpoint": checkpoint_path_out,
        "metrics": final,
        "run_manifest": run_dir / "run_manifest.json",
    }
