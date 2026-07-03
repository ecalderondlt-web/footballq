"""Experiment 4C coordinate-decoder comparison suite."""

from __future__ import annotations

import csv
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from footballq.data.windows import FEATURE_NAMES
from footballq.decoding.dataset import DecoderDataset, DecoderDatasetData, load_decoder_dataset
from footballq.decoding.metrics import decoder_metrics
from footballq.decoding.models import create_coordinate_decoder
from footballq.decoding.train import train_coordinate_decoder_from_config
from footballq.latent_flow.io import save_json
from footballq.models.constant_velocity import predict_constant_velocity
from footballq.training.train import resolve_device

SUITE_FIELDS = [
    "model",
    "input_type",
    "decoder_type",
    "split",
    "player_ADE_m",
    "player_FDE_m",
    "ball_ADE_m",
    "ball_FDE_m",
    "all_entity_ADE_m",
    "all_entity_FDE_m",
    "team_centroid_error_m",
    "team_width_error_m",
    "team_length_error_m",
    "stretch_index_error_m",
    "checkpoint",
]


def _finite(value: Any) -> float:
    value = float(value)
    if not math.isfinite(value):
        return float("nan")
    return value


def _row(
    *,
    model: str,
    input_type: str,
    decoder_type: str,
    split: str,
    metrics: dict[str, Any],
    checkpoint: str = "",
) -> dict[str, Any]:
    return {
        "model": model,
        "input_type": input_type,
        "decoder_type": decoder_type,
        "split": split,
        "player_ADE_m": metrics.get("player_ADE_m"),
        "player_FDE_m": metrics.get("player_FDE_m"),
        "ball_ADE_m": metrics.get("ball_ADE_m"),
        "ball_FDE_m": metrics.get("ball_FDE_m"),
        "all_entity_ADE_m": metrics.get("all_entity_ADE_m"),
        "all_entity_FDE_m": metrics.get("all_entity_FDE_m"),
        "team_centroid_error_m": metrics.get("team_centroid_error_m"),
        "team_width_error_m": metrics.get("team_width_error_m"),
        "team_length_error_m": metrics.get("team_length_error_m"),
        "stretch_index_error_m": metrics.get(
            "stretch_index_error_m",
            metrics.get("team_stretch_index_error_m"),
        ),
        "checkpoint": checkpoint,
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUITE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUITE_FIELDS})


@torch.no_grad()
def evaluate_constant_velocity_baseline(
    data: DecoderDatasetData,
    split: str = "test",
    batch_size: int = 256,
    device: str | None = "auto",
) -> dict[str, float]:
    torch_device = resolve_device(device)
    loader = DataLoader(
        DecoderDataset(data, mode="future_from_z", split=split),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    entity_types: list[torch.Tensor] = []
    team_ids: list[torch.Tensor] = []
    feature_names = list(data.metadata.get("feature_names", FEATURE_NAMES))
    for batch in loader:
        past = batch["past"].to(torch_device)
        past_mask = batch["past_mask"].to(torch_device)
        pred = predict_constant_velocity(
            past,
            past_mask,
            horizon_steps=data.horizon_steps,
            dt=1.0 / float(data.metadata.get("fps", 10.0)),
            feature_names=feature_names,
        )
        preds.append(pred.detach().cpu())
        targets.append(batch["target_xy"].detach().cpu())
        masks.append(batch["target_mask"].detach().cpu())
        entity_types.append(batch["entity_type"].detach().cpu())
        team_ids.append(batch["team_id"].detach().cpu())
    if not preds:
        return {"all_entity_ADE_m": math.nan}
    return decoder_metrics(
        torch.cat(preds, dim=0),
        torch.cat(targets, dim=0),
        torch.cat(masks, dim=0),
        torch.cat(entity_types, dim=0),
        torch.cat(team_ids, dim=0),
        mode="future_from_z",
    )


@torch.no_grad()
def _evaluate_rollout_override(
    checkpoint: str | Path,
    data: DecoderDatasetData,
    split: str,
    input_name: str,
    transform: Callable[[dict[str, Any]], torch.Tensor],
    device: str | None = "auto",
    batch_size: int = 256,
) -> dict[str, float]:
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    cfg = dict(payload["config"])
    torch_device = resolve_device(device)
    model = create_coordinate_decoder(cfg, data)
    model.load_state_dict(payload["model_state_dict"])
    model = model.to(torch_device)
    model.eval()
    loader = DataLoader(
        DecoderDataset(data, mode="rollout_from_latents", split=split),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    entity_types: list[torch.Tensor] = []
    team_ids: list[torch.Tensor] = []
    for batch in loader:
        x = transform(batch).to(torch_device)
        pred = model(x)
        preds.append(pred.detach().cpu())
        targets.append(batch["target_xy"].detach().cpu())
        masks.append(batch["target_mask"].detach().cpu())
        entity_types.append(batch["entity_type"].detach().cpu())
        team_ids.append(batch["team_id"].detach().cpu())
    if not preds:
        return {"all_entity_ADE_m": math.nan}
    metrics = decoder_metrics(
        torch.cat(preds, dim=0),
        torch.cat(targets, dim=0),
        torch.cat(masks, dim=0),
        torch.cat(entity_types, dim=0),
        torch.cat(team_ids, dim=0),
        mode="rollout_from_latents",
    )
    metrics["input_name"] = input_name  # type: ignore[assignment]
    return metrics


def _latent_last_rollout(batch: dict[str, Any]) -> torch.Tensor:
    steps = int(batch["z_rollout"].shape[1])
    return batch["z"].unsqueeze(1).expand(-1, steps, -1).float()


def _latent_constant_velocity_rollout(batch: dict[str, Any]) -> torch.Tensor:
    steps = int(batch["z_rollout"].shape[1])
    z = batch["z"].float()
    context = batch["z_context"].float()
    previous = context[:, -2] if context.shape[1] > 1 else z
    velocity = z - previous
    step_values = torch.arange(1, steps + 1, dtype=z.dtype).view(1, steps, 1)
    return z.unsqueeze(1) + step_values * velocity.unsqueeze(1)


def _best_row(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get(key) not in {"", None}]
    candidates = [row for row in candidates if math.isfinite(float(row[key]))]
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row[key]))


def run_decoder_suite(
    dataset: str | Path,
    out: str | Path,
    *,
    split: str = "test",
    device: str | None = "auto",
    epochs: int = 1,
    max_train_batches: int | None = 20,
    max_eval_batches: int | None = 20,
    batch_size: int = 256,
    run_root: str | Path = "runs",
) -> dict[str, Any]:
    """Train/evaluate a compact Experiment 4C decoder comparison suite."""

    data_path = Path(dataset)
    data = load_decoder_dataset(data_path)
    out_path = Path(out)
    rows: list[dict[str, Any]] = []

    baseline_metrics = evaluate_constant_velocity_baseline(
        data,
        split=split,
        batch_size=batch_size,
        device=device,
    )
    rows.append(
        _row(
            model="coordinate_constant_velocity",
            input_type="past_coordinates",
            decoder_type="analytic",
            split=split,
            metrics=baseline_metrics,
            checkpoint=str(data_path),
        )
    )

    train_specs = [
        ("linear_future_from_z", "future_from_z", "linear", []),
        ("mlp_future_from_z", "future_from_z", "mlp", [128]),
        ("context_mlp_future", "future_from_context", "context_mlp", [128]),
        ("rollout_mlp_true_latents", "rollout_from_latents", "rollout_mlp", [128]),
    ]
    trained: dict[str, dict[str, Any]] = {}
    for model_label, mode, model_name, hidden_sizes in train_specs:
        cfg = {
            "experiment": f"decoder_suite_{model_label}",
            "seed": 123,
            "data": {
                "decoder_dataset": str(data_path),
                "batch_size": batch_size,
                "num_workers": 0,
            },
            "target": {"mode": mode},
            "model": {
                "name": model_name,
                "hidden_sizes": hidden_sizes,
                "dropout": 0.0,
                "pooling": "mean" if mode == "future_from_context" else "flatten",
            },
            "training": {
                "epochs": epochs,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "device": device or "auto",
                "run_root": str(run_root),
                "max_train_batches": max_train_batches,
                "max_eval_batches": max_eval_batches,
                "best_metric": "all_entity_ADE_m",
            },
        }
        result = train_coordinate_decoder_from_config(cfg)
        trained[model_label] = result
        rows.append(
            _row(
                model=model_label,
                input_type=mode,
                decoder_type=model_name,
                split=split,
                metrics=result["test_metrics"],
                checkpoint=str(result["best_checkpoint"]),
            )
        )

    rollout_checkpoint = trained["rollout_mlp_true_latents"]["best_checkpoint"]
    for model_label, transform in [
        ("last_latent_rollout_decoded", _latent_last_rollout),
        ("constant_latent_velocity_rollout_decoded", _latent_constant_velocity_rollout),
    ]:
        metrics = _evaluate_rollout_override(
            rollout_checkpoint,
            data,
            split,
            model_label,
            transform,
            device=device,
            batch_size=batch_size,
        )
        rows.append(
            _row(
                model=model_label,
                input_type="latent_rollout",
                decoder_type="rollout_mlp",
                split=split,
                metrics=metrics,
                checkpoint=str(rollout_checkpoint),
            )
        )

    out_path.mkdir(parents=True, exist_ok=True)
    results_csv = out_path / "results.csv"
    summary_json = out_path / "summary.json"
    _write_csv(rows, results_csv)
    summary = {
        "results_csv": str(results_csv),
        "summary_json": str(summary_json),
        "best_by_all_entity_ADE_m": _best_row(rows, "all_entity_ADE_m"),
        "best_by_ball_ADE_m": _best_row(rows, "ball_ADE_m"),
        "num_rows": len(rows),
        "split": split,
    }
    save_json({"rows": rows, "summary": summary}, out_path / "results.json")
    save_json(summary, summary_json)
    return {
        "rows": rows,
        "summary": summary,
        "results_csv": results_csv,
        "summary_json": summary_json,
    }
