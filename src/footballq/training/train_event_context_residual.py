"""Train matched residual heads over frozen tracking and event representations."""

from __future__ import annotations

import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from footballq.data.pff_event_context import PFFTrackingEventContextDataset
from footballq.data.sharded_td_dataset import ShardGroupedSampler
from footballq.models.event_context_residual import (
    EVENT_CONTEXT_FAMILIES,
    FrozenTrackingEventResidual,
    event_context_residual_loss,
)
from footballq.models.statsbomb_event_encoder import StatsBombEventEncoder
from footballq.repro.manifest import build_run_manifest, file_sha256, write_run_manifest
from footballq.training.train import resolve_device
from footballq.training.train_td_jepa import create_td_jepa_model, td_batch_to_device


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _load_tracking_model(
    checkpoint_path: Path,
    dataset: PFFTrackingEventContextDataset,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = create_td_jepa_model(payload["config"], dataset.prototype)
    for name in ("online_encoder", "target_encoder", "motion_encoder"):
        getattr(model, name).load_state_dict(payload[name], strict=True)
    return model, payload


def _event_encoder_from_checkpoint(
    checkpoint_path: Path,
    event_manifest: dict[str, Any],
    *,
    load_pretrained: bool,
    random_seed: int,
) -> tuple[StatsBombEventEncoder, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = payload["config"]["model"]
    vocab_sizes = [
        int(event_manifest["categorical_vocabularies"][name]["size"])
        for name in event_manifest["categorical_feature_names"]
    ]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(random_seed))
        model = StatsBombEventEncoder(
            vocab_sizes,
            len(event_manifest["continuous_feature_names"]),
            len(event_manifest["freeze_frame_feature_names"]),
            use_360=False,
            categorical_dim=int(model_cfg["categorical_dim"]),
            d_model=int(model_cfg["d_model"]),
            n_heads=int(model_cfg["n_heads"]),
            n_layers=int(model_cfg["n_layers"]),
            dropout=float(model_cfg["dropout"]),
            max_sequence_length=int(model_cfg["max_sequence_length"]),
        )
    if load_pretrained:
        model.load_state_dict(payload["model"], strict=True)
    return model, payload


def create_event_context_residual_model(
    *,
    family: str,
    seed: int,
    z_dim: int,
    hidden_dim: int,
    tracking_checkpoint: Path,
    event_checkpoint: Path,
    dataset: PFFTrackingEventContextDataset,
    statsbomb_manifest: dict[str, Any],
) -> tuple[FrozenTrackingEventResidual, dict[str, Any]]:
    """Create matched heads with deterministic random and pretrained controls."""

    if family not in EVENT_CONTEXT_FAMILIES:
        raise ValueError(f"Unknown event-context family {family!r}.")
    tracking_model, tracking_payload = _load_tracking_model(tracking_checkpoint, dataset)
    event_encoder = None
    event_payload = None
    if family in {"random", "pretrained"}:
        event_encoder, event_payload = _event_encoder_from_checkpoint(
            event_checkpoint,
            statsbomb_manifest,
            load_pretrained=family == "pretrained",
            random_seed=seed + 10_000,
        )
    torch.manual_seed(seed + 20_000)
    model = FrozenTrackingEventResidual(
        tracking_model,
        family=family,
        z_dim=z_dim,
        event_encoder=event_encoder,
        hidden_dim=hidden_dim,
    )
    source = {
        "tracking_checkpoint": str(tracking_checkpoint),
        "tracking_checkpoint_sha256": file_sha256(tracking_checkpoint),
        "tracking_step": int(tracking_payload["step"]),
        "event_checkpoint": str(event_checkpoint),
        "event_checkpoint_sha256": file_sha256(event_checkpoint),
        "event_checkpoint_step": int(
            torch.load(event_checkpoint, map_location="cpu", weights_only=False)["step"]
        ),
        "event_weights_loaded": family == "pretrained",
        "random_event_encoder_seed": seed + 10_000 if family == "random" else None,
        "event_source_payload_available": event_payload is not None,
    }
    return model, source


def evaluate_event_context_residual(
    model: FrozenTrackingEventResidual,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int | None = None,
    ablate_event: bool = False,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    examples = 0
    event_examples = 0
    no_event_examples = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = td_batch_to_device(batch, device)
            outputs = model(batch, ablate_event=ablate_event)
            losses = event_context_residual_loss(outputs, batch["event_mask"])
            batch_size = int(batch["state_t"].shape[0])
            current_event = int(losses["event_history_examples"].item())
            current_no_event = int(losses["no_event_history_examples"].item())
            examples += batch_size
            event_examples += current_event
            no_event_examples += current_no_event
            for name in (
                "td_loss",
                "base_td_loss",
                "cosine_similarity",
                "correction_norm",
            ):
                totals[name] = totals.get(name, 0.0) + float(losses[name].item()) * batch_size
            totals["event_history_td_loss"] = totals.get(
                "event_history_td_loss", 0.0
            ) + float(losses["event_history_td_loss"].item()) * current_event
            totals["no_event_history_td_loss"] = totals.get(
                "no_event_history_td_loss", 0.0
            ) + float(losses["no_event_history_td_loss"].item()) * current_no_event
    if examples == 0:
        raise ValueError("Event-context evaluation received zero examples.")
    return {
        "td_loss": totals["td_loss"] / examples,
        "base_td_loss": totals["base_td_loss"] / examples,
        "event_history_td_loss": totals["event_history_td_loss"]
        / max(event_examples, 1),
        "no_event_history_td_loss": totals["no_event_history_td_loss"]
        / max(no_event_examples, 1),
        "cosine_similarity": totals["cosine_similarity"] / examples,
        "correction_norm": totals["correction_norm"] / examples,
        "num_examples": float(examples),
        "event_history_examples": float(event_examples),
        "no_event_history_examples": float(no_event_examples),
    }


def train_event_context_residual_from_config(
    config: str | Path | dict[str, Any],
    *,
    family: str | None = None,
    seed: int | None = None,
    tracking_checkpoint: str | Path | None = None,
    event_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    config_path = Path(config) if not isinstance(config, dict) else None
    cfg = (
        yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config_path is not None
        else dict(config)
    )
    family = str(family or cfg.get("family", "tracking"))
    if family not in EVENT_CONTEXT_FAMILIES:
        raise ValueError(f"Unknown event-context family {family!r}.")
    train_cfg = cfg.get("training", {})
    seed = int(seed if seed is not None else train_cfg.get("seed", cfg.get("seed", 7)))
    tracking_checkpoint = Path(
        tracking_checkpoint or cfg["sources"]["tracking_checkpoint"]
    )
    event_checkpoint = Path(event_checkpoint or cfg["sources"]["event_checkpoint"])
    cfg["family"] = family
    cfg.setdefault("training", {})["seed"] = seed
    cfg.setdefault("sources", {})["tracking_checkpoint"] = str(tracking_checkpoint)
    cfg.setdefault("sources", {})["event_checkpoint"] = str(event_checkpoint)
    _set_seed(seed)

    data_cfg = cfg["data"]
    validation_split = str(train_cfg.get("validation_split", "val"))
    train_dataset = PFFTrackingEventContextDataset(
        data_cfg["tracking_manifest"],
        data_cfg["event_manifest"],
        "train",
        sequence_length=int(data_cfg.get("event_history_length", 32)),
        raw_context_dim=int(cfg.get("model", {}).get("z_dim", 128)),
    )
    validation_dataset = (
        train_dataset
        if validation_split == "train"
        else PFFTrackingEventContextDataset(
            data_cfg["tracking_manifest"],
            data_cfg["event_manifest"],
            validation_split,
            sequence_length=int(data_cfg.get("event_history_length", 32)),
            raw_context_dim=int(cfg.get("model", {}).get("z_dim", 128)),
        )
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
        sampler=ShardGroupedSampler(validation_dataset, shuffle=False, seed=seed),
        num_workers=int(train_cfg.get("num_workers", 0)),
    )
    statsbomb_manifest_path = Path(data_cfg["statsbomb_manifest"])
    statsbomb_manifest = json.loads(statsbomb_manifest_path.read_text(encoding="utf-8"))
    model_cfg = cfg.get("model", {})
    model, source = create_event_context_residual_model(
        family=family,
        seed=seed,
        z_dim=int(model_cfg.get("z_dim", 128)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        tracking_checkpoint=tracking_checkpoint,
        event_checkpoint=event_checkpoint,
        dataset=train_dataset,
        statsbomb_manifest=statsbomb_manifest,
    )
    device = resolve_device(train_cfg.get("device", "auto"))
    model = model.to(device)
    trainable = list(model.correction_head.parameters())
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    max_updates = int(train_cfg["max_train_updates"])
    curve_steps = sorted({int(value) for value in train_cfg.get("validation_curve_steps", [])})
    max_epochs = int(train_cfg.get("max_epochs", 3))
    curve_max_batches = train_cfg.get("validation_curve_max_batches")
    final_max_batches = train_cfg.get("max_val_batches")
    run_root = Path(train_cfg.get("run_root", "runs/pff_statsbomb_context_residual_v1"))
    run_dir = run_root / family / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    run_config_path = run_dir / "config.yaml"
    run_config_path.write_text(
        yaml.safe_dump(cfg, sort_keys=False),
        encoding="utf-8",
    )

    step = 0
    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            batch = td_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            losses = event_context_residual_loss(outputs, batch["event_mask"])
            losses["td_loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                trainable,
                float(train_cfg.get("grad_clip", 1.0)),
            )
            optimizer.step()
            step += 1
            if step in curve_steps:
                curve = evaluate_event_context_residual(
                    model,
                    val_loader,
                    device,
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
        raise ValueError(f"Event-context run stopped at {step}, expected {max_updates} updates.")

    final = evaluate_event_context_residual(
        model,
        val_loader,
        device,
        max_batches=final_max_batches,
    )
    ablated = evaluate_event_context_residual(
        model,
        val_loader,
        device,
        max_batches=final_max_batches,
        ablate_event=True,
    )
    final_row = {
        "epoch": epoch,
        "step": step,
        "split": validation_split,
        **final,
        "ablated_event_td_loss": ablated["td_loss"],
        "ablated_event_history_td_loss": ablated["event_history_td_loss"],
    }
    finite_values = (
        math.isfinite(float(value))
        for value in final_row.values()
        if isinstance(value, float)
    )
    if not all(finite_values):
        raise ValueError("Event-context final metrics contain non-finite values.")
    _append_jsonl(run_dir / "metrics_val.jsonl", final_row)
    checkpoint = {
        "version": 1,
        "family": family,
        "seed": seed,
        "step": step,
        "config": cfg,
        "correction_head": model.correction_head.state_dict(),
        "event_encoder": (
            model.event_encoder.state_dict() if model.event_encoder is not None else None
        ),
        "optimizer": optimizer.state_dict(),
        "source": source,
        "validation": final_row,
    }
    torch.save(checkpoint, run_dir / "latest.pt")

    loaded_splits = ["train"] if validation_split == "train" else ["train", validation_split]
    manifest = build_run_manifest(
        command=sys.argv,
        config_path=run_config_path,
        split_manifest_path=data_cfg["split_manifest"],
        evaluation_protocol="frozen_backbones_period_aware_event_history",
        feature_view=f"tracking_plus_{family}_event_context",
        objective_mode="frozen_tracking_latent_residual_prediction",
        dataset_paths={
            "tracking_manifest": data_cfg["tracking_manifest"],
            "event_manifest": data_cfg["event_manifest"],
            "statsbomb_manifest": statsbomb_manifest_path,
            "tracking_checkpoint": tracking_checkpoint,
            "event_checkpoint": event_checkpoint,
        },
        output_paths={
            "run_dir": run_dir,
            "latest_checkpoint": run_dir / "latest.pt",
            "metrics": run_dir / "metrics_val.jsonl",
        },
        warnings=(
            ["Train-only architecture preflight; not result-bearing evidence."]
            if validation_split == "train"
            else []
        ),
    )
    manifest["data_access"] = {
        "loaded_tensor_splits": loaded_splits,
        "test_loaded": False,
        "embedding_export_split": None,
    }
    manifest["frozen_sources"] = source
    manifest["training"] = {
        "family": family,
        "seed": seed,
        "final_step": step,
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "tracking_parameters_frozen": True,
        "event_encoder_parameters_frozen": True,
    }
    write_run_manifest(run_dir / "run_manifest.json", manifest)
    return {
        "run_dir": run_dir,
        "latest_checkpoint": run_dir / "latest.pt",
        "metrics_path": run_dir / "metrics_val.jsonl",
        "run_manifest_path": run_dir / "run_manifest.json",
        "final": final_row,
    }
