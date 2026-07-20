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


from footballq.data.sharded_td_dataset import (
    ShardedTDJEPADataset,
    ShardGroupedSampler,
    ShardTemperatureSampler,
)
from footballq.data.td_jepa_dataset import TDJEPAData, TDJEPADataset, load_td_jepa_data
from footballq.models.td_jepa import SoccerTDJEPA
from footballq.repro.manifest import build_run_manifest, file_sha256, write_run_manifest
from footballq.repro.splits import split_indices_from_manifest
from footballq.training.ema import update_ema
from footballq.training.td_jepa_losses import (
    match_mean_invariance_loss,
    td_jepa_loss,
    temporal_motion_reconstruction_loss,
)
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
        temporal_motion_head_hidden_dim=(
            int(model_cfg["temporal_motion_head_hidden_dim"])
            if model_cfg.get("temporal_motion_head_hidden_dim") is not None
            else None
        ),
        transition_decoder_hidden_dim=(
            int(model_cfg["transition_decoder_hidden_dim"])
            if model_cfg.get("transition_decoder_hidden_dim") is not None
            else None
        ),
    )


def _transfer_signature(config: dict[str, Any], data: TDJEPAData) -> dict[str, Any]:
    model_cfg = config.get("model", {})
    return {
        "context_steps": data.context_steps,
        "delta_steps": data.delta_steps,
        "n_entities": int(data.state_t.shape[2]),
        "n_features": data.n_features,
        "z_dim": int(model_cfg.get("z_dim", 128)),
        "d_model": int(model_cfg.get("d_model", 128)),
        "n_heads": int(model_cfg.get("n_heads", 4)),
        "n_layers": int(model_cfg.get("n_layers", 2)),
        "motion_hidden_dim": int(model_cfg.get("motion_hidden_dim", 256)),
        "pooling": str(model_cfg.get("pooling", "mean")),
        "temporal_motion_head_hidden_dim": (
            int(model_cfg["temporal_motion_head_hidden_dim"])
            if model_cfg.get("temporal_motion_head_hidden_dim") is not None
            else None
        ),
        "transition_decoder_hidden_dim": (
            int(model_cfg["transition_decoder_hidden_dim"])
            if model_cfg.get("transition_decoder_hidden_dim") is not None
            else None
        ),
    }


def initialize_td_jepa_from_checkpoint(
    model: SoccerTDJEPA,
    config: dict[str, Any],
    data: TDJEPAData,
    checkpoint: str | Path,
) -> dict[str, Any]:
    """Initialize a fresh TD-JEPA run from a compatible pretrained checkpoint."""

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"TD-JEPA initialization checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_config = payload.get("config")
    source_meta = payload.get("data_meta")
    if not isinstance(source_config, dict) or not isinstance(source_meta, dict):
        raise ValueError(
            "TD-JEPA initialization checkpoint is missing config or data_meta provenance."
        )

    mismatches: list[str] = []
    for field, target_value in {
        "feature_names": list(data.feature_names),
        "feature_view": data.feature_view,
        "objective_mode": data.objective_mode,
    }.items():
        source_value = source_meta.get(field)
        if source_value != target_value:
            mismatches.append(f"{field}: source={source_value!r}, target={target_value!r}")

    source_signature = {
        "context_steps": source_meta.get("context_steps"),
        "delta_steps": source_meta.get("delta_steps"),
        "n_entities": source_meta.get("n_entities"),
        "n_features": source_meta.get("n_features"),
        **{
            key: value
            for key, value in _transfer_signature(source_config, data).items()
            if key not in {"context_steps", "delta_steps", "n_entities", "n_features"}
        },
    }
    target_signature = _transfer_signature(config, data)
    for field, target_value in target_signature.items():
        source_value = source_signature.get(field)
        if source_value != target_value:
            mismatches.append(f"{field}: source={source_value!r}, target={target_value!r}")
    if mismatches:
        raise ValueError(
            "TD-JEPA initialization checkpoint is incompatible with the target run: "
            + "; ".join(mismatches)
        )

    loaded_components = ["online_encoder", "target_encoder", "motion_encoder"]
    try:
        for component in loaded_components:
            getattr(model, component).load_state_dict(payload[component], strict=True)
    except (KeyError, RuntimeError) as exc:
        raise ValueError(
            "TD-JEPA initialization checkpoint component shapes are incompatible with the "
            "target model."
        ) from exc

    skipped_components: list[str] = []
    if model.state_decoder is not None and payload.get("state_decoder") is not None:
        try:
            model.state_decoder.load_state_dict(payload["state_decoder"], strict=True)
            loaded_components.append("state_decoder")
        except RuntimeError as exc:
            raise ValueError(
                "TD-JEPA initialization checkpoint state decoder is incompatible with the "
                "target model."
            ) from exc
    elif model.state_decoder is not None or payload.get("state_decoder") is not None:
        skipped_components.append("state_decoder")

    if model.temporal_motion_head is not None and payload.get("temporal_motion_head") is not None:
        try:
            model.temporal_motion_head.load_state_dict(
                payload["temporal_motion_head"], strict=True
            )
            loaded_components.append("temporal_motion_head")
        except RuntimeError as exc:
            raise ValueError(
                "TD-JEPA initialization checkpoint temporal-motion head is incompatible with "
                "the target model."
            ) from exc
    elif model.temporal_motion_head is not None or payload.get("temporal_motion_head") is not None:
        skipped_components.append("temporal_motion_head")

    if model.transition_decoder is not None and payload.get("transition_decoder") is not None:
        try:
            model.transition_decoder.load_state_dict(
                payload["transition_decoder"], strict=True
            )
            loaded_components.append("transition_decoder")
        except RuntimeError as exc:
            raise ValueError(
                "TD-JEPA initialization checkpoint transition decoder is incompatible with "
                "the target model."
            ) from exc
    elif model.transition_decoder is not None or payload.get("transition_decoder") is not None:
        skipped_components.append("transition_decoder")

    return {
        "mode": "pretrained_weights_fresh_optimizer",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "source_run_dir": payload.get("run_dir"),
        "source_experiment": source_config.get("experiment"),
        "source_dataset": source_config.get("data", {}).get("source"),
        "source_data_meta": source_meta,
        "loaded_components": loaded_components,
        "skipped_components": skipped_components,
    }


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
        context_reconstruction=outputs.get("context_reconstruction"),
        context_target=batch.get("state_t"),
        context_mask=batch.get("mask_t"),
        context_reconstruction_weight=float(
            loss_cfg.get("context_reconstruction_weight", 0.0)
        ),
        no_motion_margin_weight=float(loss_cfg.get("no_motion_margin_weight", 0.0)),
        no_motion_margin=float(loss_cfg.get("no_motion_margin", 0.01)),
        transition_reconstruction=outputs.get("transition_reconstruction"),
        transition_target=(
            batch["state_t_plus_delta"][..., :2] - batch["state_t"][..., :2]
        ),
        transition_mask=batch["mask_t_plus_delta"] & batch["mask_t"],
        transition_reconstruction_weight=float(
            loss_cfg.get("transition_reconstruction_weight", 0.0)
        ),
    )


def _loss_from_model(
    model: SoccerTDJEPA,
    batch: dict[str, Any],
    loss_cfg: dict[str, Any],
) -> dict[str, torch.Tensor]:
    outputs = model(batch)
    losses = _loss_from_outputs(outputs, batch, loss_cfg)
    losses["base_total_loss"] = losses["total_loss"]

    temporal_weight = float(loss_cfg.get("temporal_motion_weight", 0.0))
    temporal_loss = outputs["z_pred"].new_tensor(0.0)
    temporal_cosine = outputs["z_pred"].new_tensor(0.0)
    if temporal_weight > 0.0:
        if model.temporal_motion_head is None:
            raise ValueError(
                "temporal_motion_weight requires model.temporal_motion_head_hidden_dim."
            )
        reversed_state = torch.flip(batch["state_t"], dims=[1])
        reversed_mask = torch.flip(batch["mask_t"], dims=[1])
        z_reversed = model.online_encoder(reversed_state, reversed_mask)
        displacement = batch["state_t"][:, -1, :, :2] - batch["state_t"][:, 0, :, :2]
        endpoint_mask = batch["mask_t"][:, -1, :] & batch["mask_t"][:, 0, :]
        temporal_loss, temporal_cosine = temporal_motion_reconstruction_loss(
            model.predict_temporal_motion(outputs["z_t"]),
            model.predict_temporal_motion(z_reversed),
            displacement,
            endpoint_mask,
        )

    match_weight = float(loss_cfg.get("match_invariance_weight", 0.0))
    match_loss = outputs["z_t"].new_tensor(0.0)
    match_groups = 0
    if match_weight > 0.0:
        match_loss, match_groups = match_mean_invariance_loss(
            outputs["z_t"],
            batch.get("match_id", []),
        )

    losses["temporal_motion_loss"] = temporal_loss
    losses["temporal_motion_cosine_similarity"] = temporal_cosine
    losses["match_invariance_loss"] = match_loss
    losses["match_groups_in_batch"] = outputs["z_t"].new_tensor(float(match_groups))
    losses["total_loss"] = (
        losses["total_loss"]
        + temporal_weight * temporal_loss
        + match_weight * match_loss
    )
    return losses


def evaluate_td_model(
    model: SoccerTDJEPA,
    loader: DataLoader,
    device: torch.device,
    loss_cfg: dict[str, Any],
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    num_examples = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = td_batch_to_device(batch, device)
            losses = _loss_from_model(model, batch, loss_cfg)
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
    split_indices: dict[str, Any],
    run_dir: Path,
    epoch: int,
    step: int,
    best_metric: float,
    initialization: dict[str, Any] | None,
) -> None:
    torch.save(
        {
            "online_encoder": model.online_encoder.state_dict(),
            "target_encoder": model.target_encoder.state_dict(),
            "motion_encoder": model.motion_encoder.state_dict(),
            "state_decoder": (
                model.state_decoder.state_dict() if model.state_decoder is not None else None
            ),
            "temporal_motion_head": (
                model.temporal_motion_head.state_dict()
                if model.temporal_motion_head is not None
                else None
            ),
            "transition_decoder": (
                model.transition_decoder.state_dict()
                if model.transition_decoder is not None
                else None
            ),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "epoch": epoch,
            "step": step,
            "best_validation_metric": best_metric,
            "initialization": initialization,
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
                "context_steps": data.context_steps,
                "delta_steps": data.delta_steps,
                "n_entities": int(data.state_t.shape[2]),
                "n_features": data.n_features,
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


def _configured_optional_split(
    train_cfg: dict[str, Any],
    key: str,
    *,
    default: str | None,
) -> str | None:
    value = train_cfg[key] if key in train_cfg else default
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "null", "disabled"}:
        return None
    if normalized not in {"train", "val", "test"}:
        raise ValueError(f"training.{key} must be train, val, test, or null.")
    return normalized


def train_td_jepa_from_config(
    config: str | Path | dict[str, Any],
    *,
    init_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    cfg = load_td_config(config)
    configured_init = cfg.get("initialization", {}).get("checkpoint")
    if init_checkpoint is not None and configured_init is not None:
        if Path(init_checkpoint) != Path(configured_init):
            raise ValueError(
                "Initialization checkpoint was supplied both in config and as an argument "
                "with different paths."
            )
    selected_init = init_checkpoint if init_checkpoint is not None else configured_init
    if selected_init is not None:
        cfg.setdefault("initialization", {})["checkpoint"] = str(selected_init)
    seed = int(cfg.get("seed", cfg.get("training", {}).get("seed", 7)))
    set_td_seed(seed)
    data_path = Path(cfg.get("data", {}).get("path", cfg.get("data", {}).get("td_jepa", "")))
    if not data_path.exists():
        raise FileNotFoundError(
            f"TD-JEPA data file not found: {data_path}. Run scripts/prepare_td_jepa_data.py first."
        )
    split_cfg = cfg.get("split", {})
    split_manifest = split_cfg.get("manifest_path") or split_cfg.get("manifest")
    train_cfg = cfg.get("training", {})
    batch_size = int(train_cfg.get("batch_size", cfg.get("data", {}).get("batch_size", 64)))
    num_workers = int(train_cfg.get("num_workers", cfg.get("data", {}).get("num_workers", 0)))
    drop_last_train = bool(train_cfg.get("drop_last_train", False))
    validation_split = _configured_optional_split(
        train_cfg, "validation_split", default="val"
    )
    embedding_sample_split = _configured_optional_split(
        train_cfg, "embedding_sample_split", default=None
    )
    if data_path.suffix.lower() == ".json":
        dataset_manifest = json.loads(data_path.read_text(encoding="utf-8"))
        available_splits = {
            str(shard["split"]) for shard in dataset_manifest.get("shards", [])
        }
        requested_splits = {"train"}
        if validation_split is not None:
            requested_splits.add(validation_split)
        if embedding_sample_split is not None:
            requested_splits.add(embedding_sample_split)
        missing_splits = requested_splits - available_splits
        if missing_splits:
            raise ValueError(
                "Sharded TD-JEPA manifest is missing configured tensor splits: "
                f"{sorted(missing_splits)}"
            )
        sharded = {
            split_name: ShardedTDJEPADataset(data_path, split_name)
            for split_name in sorted(requested_splits)
        }
        data = sharded["train"].prototype
        manifest_split_counts = {
            split_name: sum(
                int(shard["example_count"])
                for shard in dataset_manifest["shards"]
                if str(shard["split"]) == split_name
            )
            for split_name in sorted(available_splits)
        }
        split_indices: dict[str, Any] = {
            split_name: {
                "mode": "sharded_manifest",
                "num_examples": example_count,
                "loaded_during_training": split_name in sharded,
            }
            for split_name, example_count in manifest_split_counts.items()
        }
        train_sampler_mode = str(train_cfg.get("sharded_train_sampler", "grouped"))
        if train_sampler_mode not in {"grouped", "shard_temperature"}:
            raise ValueError(
                "training.sharded_train_sampler must be 'grouped' or 'shard_temperature'."
            )
        loaders = {}
        for split_name, dataset in sharded.items():
            if split_name == "train" and train_sampler_mode == "shard_temperature":
                sampler = ShardTemperatureSampler(
                    dataset,
                    num_samples=int(train_cfg["sampler_num_samples"]),
                    temperature=float(train_cfg.get("sampler_temperature", 0.5)),
                    seed=seed,
                )
                split_indices["train"]["sampler"] = {
                    "mode": "shard_temperature",
                    "temperature": sampler.temperature,
                    "num_samples": len(sampler),
                    "allocations": sampler.allocations,
                }
            else:
                sampler = ShardGroupedSampler(
                    dataset,
                    shuffle=(split_name == "train"),
                    seed=seed,
                )
            loaders[split_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                drop_last=drop_last_train and split_name == "train",
            )
    else:
        data = load_td_jepa_data(data_path)
        if len(data.match_id) == 0:
            raise ValueError("TD-JEPA data file contains zero examples.")
        if split_manifest:
            split_indices = split_indices_from_manifest(data.match_id, split_manifest)
        else:
            split_indices = split_indices_by_match(
                data.match_id,
                val_fraction=float(split_cfg.get("val_fraction", 0.2)),
                test_fraction=float(split_cfg.get("test_fraction", 0.2)),
                seed=seed,
            )
        loaders = {
            split_name: DataLoader(
                TDJEPADataset(data, indices=indices),
                batch_size=batch_size,
                shuffle=(split_name == "train"),
                num_workers=num_workers,
                drop_last=drop_last_train and split_name == "train",
            )
            for split_name, indices in split_indices.items()
        }

    for purpose, split_name in {
        "validation": validation_split,
        "embedding sample": embedding_sample_split,
    }.items():
        if split_name is not None and split_name not in loaders:
            raise ValueError(f"Configured {purpose} split {split_name!r} is unavailable.")

    device = resolve_device(train_cfg.get("device", "auto"))
    model = create_td_jepa_model(cfg, data).to(device)
    initialization = (
        initialize_td_jepa_from_checkpoint(model, cfg, data, selected_init)
        if selected_init is not None
        else None
    )
    trainable_parameters = list(model.online_encoder.parameters()) + list(
        model.motion_encoder.parameters()
    )
    if model.state_decoder is not None:
        trainable_parameters += list(model.state_decoder.parameters())
    if model.temporal_motion_head is not None:
        trainable_parameters += list(model.temporal_motion_head.parameters())
    if model.transition_decoder is not None:
        trainable_parameters += list(model.transition_decoder.parameters())
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
    max_train_updates = train_cfg.get("max_train_updates")
    max_train_updates = int(max_train_updates) if max_train_updates is not None else None
    if max_train_updates is not None and max_train_updates <= 0:
        raise ValueError("training.max_train_updates must be positive when provided.")
    validation_curve_steps = sorted(
        {int(value) for value in train_cfg.get("validation_curve_steps", [])}
    )
    if any(value <= 0 for value in validation_curve_steps):
        raise ValueError("training.validation_curve_steps must contain positive updates.")
    if validation_curve_steps and validation_split is None:
        raise ValueError("training.validation_curve_steps requires a validation split.")
    validation_curve_max_batches = train_cfg.get("validation_curve_max_batches")
    validation_curve_max_batches = (
        int(validation_curve_max_batches)
        if validation_curve_max_batches is not None
        else train_cfg.get("max_val_batches")
    )
    metric_name = str(train_cfg.get("best_metric", "total_loss"))

    for epoch in range(1, epochs + 1):
        model.train()
        train_totals: dict[str, float] = {}
        train_examples = 0
        iterator = tqdm(loaders["train"], desc=f"td-jepa epoch {epoch}", leave=False)
        for batch_idx, batch in enumerate(iterator, start=1):
            if max_train_updates is not None and step >= max_train_updates:
                break
            batch = td_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            losses = _loss_from_model(model, batch, loss_cfg)
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
            if step in validation_curve_steps:
                curve_metrics = evaluate_td_model(
                    model,
                    loaders[validation_split],
                    device,
                    loss_cfg,
                    max_batches=validation_curve_max_batches,
                )
                _append_jsonl(
                    run_dir / "metrics_val_curve.jsonl",
                    {
                        "step": step,
                        "split": validation_split,
                        **_metric_row(curve_metrics),
                    },
                )
                model.train()
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break

        train_metrics = {
            key: value / max(train_examples, 1) for key, value in train_totals.items()
        } | {"num_examples": train_examples}
        _append_jsonl(
            run_dir / "metrics_train.jsonl",
            {"epoch": epoch, **_metric_row(train_metrics)},
        )
        if validation_split is not None:
            val_metrics = evaluate_td_model(
                model,
                loaders[validation_split],
                device,
                loss_cfg,
                max_batches=train_cfg.get("max_val_batches"),
            )
            _append_jsonl(
                run_dir / "metrics_val.jsonl",
                {
                    "epoch": epoch,
                    "step": step,
                    "split": validation_split,
                    **_metric_row(val_metrics),
                },
            )
            current: float | None = float(val_metrics.get(metric_name, float("inf")))
        else:
            current = None
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
            initialization,
        )
        if current is not None and current < best_metric:
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
                initialization,
            )
        if max_train_updates is not None and step >= max_train_updates:
            break

    if not best_path.exists():
        shutil.copy2(latest_path, best_path)
    embedding_path: Path | None = None
    if embedding_sample_split is not None:
        embedding_path = run_dir / "embeddings_sample.pt"
        _save_embedding_sample(
            embedding_path,
            model,
            loaders[embedding_sample_split],
            device,
        )
    if split_manifest:
        manifest_path = run_dir / "run_manifest.json"
        dataset_paths: dict[str, str | Path] = {"td_jepa": data_path}
        if initialization is not None:
            dataset_paths["initialization_checkpoint"] = initialization["checkpoint"]
        output_paths: dict[str, str | Path] = {
            "run_dir": run_dir,
            "latest_checkpoint": latest_path,
            "best_checkpoint": best_path,
            "run_manifest": manifest_path,
        }
        if embedding_path is not None and embedding_path.exists():
            output_paths["embeddings_sample"] = embedding_path
        manifest = build_run_manifest(
            command=sys.argv,
            config_path=run_config_path,
            split_manifest_path=split_manifest,
            evaluation_protocol=str(split_cfg.get("protocol", "inductive")),
            feature_view=data.feature_view,
            objective_mode=data.objective_mode,
            dataset_paths=dataset_paths,
            output_paths=output_paths,
            warnings=list((data.metadata or {}).get("warnings", [])),
        )
        manifest["initialization"] = initialization
        manifest["data_access"] = {
            "loaded_tensor_splits": sorted(loaders),
            "validation_split": validation_split,
            "embedding_sample_split": embedding_sample_split,
        }
        write_run_manifest(manifest_path, manifest)
    model_root = run_root / "td_jepa"
    shutil.copy2(latest_path, model_root / "latest.pt")
    shutil.copy2(best_path, model_root / "best.pt")
    if embedding_path is not None and embedding_path.exists():
        shutil.copy2(embedding_path, model_root / "embeddings_sample.pt")
    return {
        "run_dir": run_dir,
        "latest_checkpoint": latest_path,
        "best_checkpoint": best_path,
        "best_metric": best_metric if math.isfinite(best_metric) else None,
    }
