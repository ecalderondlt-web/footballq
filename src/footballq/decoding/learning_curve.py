"""Learning-curve runners for decoder diagnostics and stress-slice evaluation."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from footballq.decoding.dataset import (
    DecoderDataset,
    DecoderDatasetData,
    decoder_split_diagnostics,
    load_decoder_dataset,
    save_decoder_dataset,
    subset_decoder_dataset,
)
from footballq.decoding.metrics import decoder_metrics
from footballq.decoding.models import create_coordinate_decoder
from footballq.decoding.stress import STRESS_SLICE_NAMES, compute_stress_slices
from footballq.decoding.train import (
    prediction_from_decoder_output,
    train_coordinate_decoder_from_config,
)
from footballq.latent_flow.io import save_json
from footballq.training.train import resolve_device

LEARNING_CURVE_FIELDS = [
    "num_matches",
    "num_examples",
    "num_train_examples",
    "num_val_examples",
    "num_test_examples",
    "num_slice_examples",
    "dataset",
    "horizon_steps",
    "horizon_seconds",
    "train_match_ids",
    "val_match_ids",
    "test_match_ids",
    "match_ids_train",
    "match_ids_val",
    "match_ids_test",
    "disjoint_match_split",
    "smoke_split",
    "model",
    "model_name",
    "split",
    "slice_name",
    "feature_source",
    "decoder_type",
    "target_type",
    "all_entity_ADE_m",
    "player_ADE_m",
    "ball_ADE_m",
    "all_entity_FDE_m",
    "player_FDE_m",
    "ball_FDE_m",
    "team_centroid_error_m",
    "team_width_error_m",
    "team_length_error_m",
    "stretch_index_error_m",
    "current_all_entity_error_m",
    "current_player_error_m",
    "current_ball_error_m",
    "current_team_centroid_error_m",
    "current_team_width_error_m",
    "current_team_length_error_m",
    "current_stretch_index_error_m",
    "best_checkpoint",
    "finite_metrics",
]

STRESS_RESULTS_FIELDS = [
    "dataset",
    "num_matches",
    "horizon_steps",
    "horizon_seconds",
    "split",
    "slice_name",
    "num_examples",
    "fraction_of_test_set",
    "coordinate_constant_velocity_ADE_m",
    "coordinate_constant_velocity_FDE_m",
    "residual_context_only_ADE_m",
    "residual_context_only_FDE_m",
    "residual_z_plus_context_ADE_m",
    "residual_z_plus_context_FDE_m",
    "z_plus_context_minus_context_only_ADE_m",
    "z_plus_context_minus_context_only_FDE_m",
    "z_plus_context_minus_coordinate_cv_ADE_m",
    "z_plus_context_minus_coordinate_cv_FDE_m",
    "finite_metrics",
]

TRAIN_SPECS: dict[str, tuple[str, str, str, str, str, str]] = {
    "mlp_reconstruct_current_from_z": (
        "mlp_reconstruct_current_from_z",
        "td_jepa_z",
        "mlp",
        "reconstruct_current",
        "current",
        "current_all_entity_error_m",
    ),
    "context_mlp_reconstruct_current": (
        "context_mlp_reconstruct_current",
        "raw_past_context",
        "raw_context_mlp",
        "reconstruct_current_from_context",
        "current",
        "current_all_entity_error_m",
    ),
    "z_plus_context_reconstruct_current": (
        "z_plus_context_reconstruct_current",
        "z_plus_context",
        "z_context_mlp",
        "reconstruct_current_from_z_context",
        "current",
        "current_all_entity_error_m",
    ),
    "z_only_decoder": (
        "z_only_decoder",
        "td_jepa_z",
        "mlp",
        "future_from_z",
        "future",
        "all_entity_ADE_m",
    ),
    "context_only_decoder": (
        "context_only_decoder",
        "raw_past_context",
        "raw_context_mlp",
        "future_from_past_context",
        "future",
        "all_entity_ADE_m",
    ),
    "z_plus_context_decoder": (
        "z_plus_context_decoder",
        "z_plus_context",
        "z_context_mlp",
        "future_from_z_past_context",
        "future",
        "all_entity_ADE_m",
    ),
    "residual_context_only_decoder": (
        "residual_context_only_decoder",
        "raw_past_context",
        "residual_context_mlp",
        "residual_future_from_past_context",
        "future",
        "all_entity_ADE_m",
    ),
    "residual_z_plus_context_decoder": (
        "residual_z_plus_context_decoder",
        "z_plus_context",
        "residual_context_mlp",
        "residual_future_from_z_past_context",
        "future",
        "all_entity_ADE_m",
    ),
}

MODEL_ALIASES = {
    "coordinate_cv": "coordinate_constant_velocity",
    "constant_velocity": "coordinate_constant_velocity",
    "last_position": "last_coordinate_position",
    "raw_past_summary_mlp": "context_only_decoder",
    "context_only": "context_only_decoder",
    "z_plus_context": "z_plus_context_decoder",
    "z_plus_context_direct_decoder": "z_plus_context_decoder",
    "residual_context_only": "residual_context_only_decoder",
    "residual_z_plus_context": "residual_z_plus_context_decoder",
}

DEFAULT_MODELS = [
    "coordinate_constant_velocity",
    "last_coordinate_position",
    "mlp_reconstruct_current_from_z",
    "context_mlp_reconstruct_current",
    "z_plus_context_reconstruct_current",
    "z_only_decoder",
    "context_only_decoder",
    "z_plus_context_decoder",
    "residual_context_only_decoder",
    "residual_z_plus_context_decoder",
]


def _join(values: list[str]) -> str:
    return ";".join(str(value) for value in values)


def _finite_metrics(row: dict[str, Any]) -> bool:
    metric_keys = [
        "all_entity_ADE_m",
        "player_ADE_m",
        "ball_ADE_m",
        "all_entity_FDE_m",
        "player_FDE_m",
        "ball_FDE_m",
        "team_centroid_error_m",
        "team_width_error_m",
        "team_length_error_m",
        "stretch_index_error_m",
        "current_all_entity_error_m",
        "current_player_error_m",
        "current_ball_error_m",
        "current_team_centroid_error_m",
        "current_team_width_error_m",
        "current_team_length_error_m",
        "current_stretch_index_error_m",
    ]
    values = [row.get(key) for key in metric_keys if row.get(key) not in {"", None}]
    return bool(values) and all(math.isfinite(float(value)) for value in values)


def _base_row(
    diagnostics: dict[str, Any],
    *,
    dataset: str,
    horizon_steps: int,
    horizon_seconds: float,
    model: str,
    split: str,
    slice_name: str,
    num_slice_examples: int,
    feature_source: str,
    decoder_type: str,
    target_type: str,
    metrics: dict[str, Any],
    checkpoint: str = "",
) -> dict[str, Any]:
    row = {
        "num_matches": diagnostics["num_matches"],
        "num_examples": diagnostics["num_examples"],
        "num_train_examples": diagnostics["num_train_examples"],
        "num_val_examples": diagnostics["num_val_examples"],
        "num_test_examples": diagnostics["num_test_examples"],
        "num_slice_examples": int(num_slice_examples),
        "dataset": dataset,
        "horizon_steps": int(horizon_steps),
        "horizon_seconds": float(horizon_seconds),
        "train_match_ids": _join(diagnostics["train_match_ids"]),
        "val_match_ids": _join(diagnostics["val_match_ids"]),
        "test_match_ids": _join(diagnostics["test_match_ids"]),
        "match_ids_train": _join(diagnostics["train_match_ids"]),
        "match_ids_val": _join(diagnostics["val_match_ids"]),
        "match_ids_test": _join(diagnostics["test_match_ids"]),
        "disjoint_match_split": bool(diagnostics["disjoint_match_split"]),
        "smoke_split": bool(diagnostics["smoke_split"]),
        "model": model,
        "model_name": model,
        "split": split,
        "slice_name": slice_name,
        "feature_source": feature_source,
        "decoder_type": decoder_type,
        "target_type": target_type,
        "all_entity_ADE_m": metrics.get("all_entity_ADE_m", ""),
        "player_ADE_m": metrics.get("player_ADE_m", ""),
        "ball_ADE_m": metrics.get("ball_ADE_m", ""),
        "all_entity_FDE_m": metrics.get("all_entity_FDE_m", ""),
        "player_FDE_m": metrics.get("player_FDE_m", ""),
        "ball_FDE_m": metrics.get("ball_FDE_m", ""),
        "team_centroid_error_m": metrics.get("team_centroid_error_m", ""),
        "team_width_error_m": metrics.get("team_width_error_m", ""),
        "team_length_error_m": metrics.get("team_length_error_m", ""),
        "stretch_index_error_m": metrics.get(
            "stretch_index_error_m",
            metrics.get("team_stretch_index_error_m", ""),
        ),
        "current_all_entity_error_m": metrics.get("current_all_entity_error_m", ""),
        "current_player_error_m": metrics.get("current_player_error_m", ""),
        "current_ball_error_m": metrics.get("current_ball_error_m", ""),
        "current_team_centroid_error_m": metrics.get("current_team_centroid_error_m", ""),
        "current_team_width_error_m": metrics.get("current_team_width_error_m", ""),
        "current_team_length_error_m": metrics.get("current_team_length_error_m", ""),
        "current_stretch_index_error_m": metrics.get("current_stretch_index_error_m", ""),
        "best_checkpoint": checkpoint,
    }
    row["finite_metrics"] = _finite_metrics(row)
    return row


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEARNING_CURVE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in LEARNING_CURVE_FIELDS})


def _write_stress_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STRESS_RESULTS_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in STRESS_RESULTS_FIELDS})


def _indices_for_matches(data: DecoderDatasetData, match_ids: list[str]) -> list[int]:
    selected = set(str(value) for value in match_ids)
    return [
        idx
        for idx, match_id in enumerate(data.examples["match_id"])
        if str(match_id) in selected
    ]


def _subset_specs(
    data: DecoderDatasetData,
    match_counts: list[str | int],
) -> list[tuple[str, list[str]]]:
    unique = sorted(set(str(value) for value in data.examples["match_id"]))
    specs: list[tuple[str, list[str]]] = []
    seen: set[tuple[str, ...]] = set()
    for value in match_counts:
        if str(value).lower() == "all":
            selected = unique
            label = "all"
        else:
            count = max(1, int(value))
            selected = unique[: min(count, len(unique))]
            label = str(min(count, len(unique)))
        key = tuple(selected)
        if key not in seen:
            seen.add(key)
            specs.append((label, selected))
    return specs


def _last_position_metrics(data: DecoderDatasetData, split: str) -> dict[str, float]:
    indices = [int(value) for value in data.splits.get(f"{split}_indices", [])]
    if not indices:
        return {"all_entity_ADE_m": math.nan}
    pred = data.examples["last_position_xy"][indices]
    target = data.examples["future_xy"][indices]
    mask = data.examples["future_mask"][indices]
    entity_type = data.examples["entity_type"][indices]
    team_id = data.examples["team_id"][indices]
    return decoder_metrics(pred, target, mask, entity_type, team_id, mode="future_from_z")


def _future_metrics_from_predictions(
    data: DecoderDatasetData,
    split_indices: list[int],
    pred: torch.Tensor,
    slice_mask: torch.Tensor,
) -> tuple[dict[str, float], int]:
    local = slice_mask[torch.as_tensor(split_indices, dtype=torch.long)].bool()
    count = int(local.sum().item())
    if count == 0:
        return {"all_entity_ADE_m": math.nan}, 0
    target = data.examples["future_xy"][split_indices][local]
    mask = data.examples["future_mask"][split_indices][local]
    entity_type = data.examples["entity_type"][split_indices][local]
    team_id = data.examples["team_id"][split_indices][local]
    metrics = decoder_metrics(
        pred[local],
        target,
        mask,
        entity_type,
        team_id,
        mode="future_from_z",
    )
    return metrics, count


def _append_future_slice_rows(
    rows: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    *,
    data: DecoderDatasetData,
    dataset: str,
    split: str,
    model: str,
    feature_source: str,
    decoder_type: str,
    checkpoint: str,
    pred: torch.Tensor,
    slice_masks: dict[str, torch.Tensor],
) -> None:
    split_indices = [int(value) for value in data.splits.get(f"{split}_indices", [])]
    horizon_steps = data.horizon_steps
    horizon_seconds = float(
        data.metadata.get(
            "horizon_seconds",
            horizon_steps / data.metadata.get("fps", 10.0),
        )
    )
    for slice_name in STRESS_SLICE_NAMES:
        if slice_name not in slice_masks:
            continue
        metrics, count = _future_metrics_from_predictions(
            data,
            split_indices,
            pred,
            slice_masks[slice_name],
        )
        if count == 0:
            continue
        rows.append(
            _base_row(
                diagnostics,
                dataset=dataset,
                horizon_steps=horizon_steps,
                horizon_seconds=horizon_seconds,
                model=model,
                split=split,
                slice_name=slice_name,
                num_slice_examples=count,
                feature_source=feature_source,
                decoder_type=decoder_type,
                target_type="future",
                metrics=metrics,
                checkpoint=checkpoint,
            )
        )


@torch.no_grad()
def _checkpoint_predictions(
    checkpoint: str | Path,
    data: DecoderDatasetData,
    *,
    mode: str,
    split: str,
    device: str | None,
    batch_size: int,
) -> torch.Tensor:
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    cfg = dict(payload["config"])
    torch_device = resolve_device(device)
    model = create_coordinate_decoder(cfg, data)
    model.load_state_dict(payload["model_state_dict"])
    model = model.to(torch_device)
    model.eval()
    loader = DataLoader(
        DecoderDataset(data, mode=mode, split=split),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    preds: list[torch.Tensor] = []
    for batch in loader:
        device_batch = {
            key: value.to(torch_device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        raw = model(device_batch["x"])
        pred = prediction_from_decoder_output(raw, device_batch, mode)
        preds.append(pred.detach().cpu())
    if not preds:
        return torch.empty((0, data.horizon_steps, data.n_entities, 2), dtype=torch.float32)
    return torch.cat(preds, dim=0)


def _best_row(
    rows: list[dict[str, Any]],
    key: str,
    target_type: str | None = None,
    include_smoke: bool = True,
    exclude_models: set[str] | None = None,
) -> dict[str, Any] | None:
    exclude_models = exclude_models or set()
    candidates = [
        row
        for row in rows
        if row.get(key) not in {"", None}
        and (target_type is None or row.get("target_type") == target_type)
        and (include_smoke or not bool(row.get("smoke_split")))
        and row.get("slice_name", "all_windows") == "all_windows"
        and str(row.get("model")) not in exclude_models
        and math.isfinite(float(row[key]))
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row[key]))


def _best_rows_by(
    rows: list[dict[str, Any]],
    group_key: str,
    *,
    slice_name: str | None = None,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("target_type") != "future" or row.get("all_entity_ADE_m") in {"", None}:
            continue
        if slice_name is not None and row.get("slice_name") != slice_name:
            continue
        if not math.isfinite(float(row["all_entity_ADE_m"])):
            continue
        if bool(row.get("smoke_split")):
            continue
        grouped.setdefault(str(row.get(group_key)), []).append(row)
    return {
        key: min(value, key=lambda row: float(row["all_entity_ADE_m"]))
        for key, value in grouped.items()
    }


def _comparison_rows(
    rows: list[dict[str, Any]],
    model_a: str,
    model_b: str,
) -> list[dict[str, Any]]:
    by_key = {
        (
            row.get("dataset"),
            row.get("num_matches"),
            row.get("horizon_seconds"),
            row.get("slice_name"),
        ): row
        for row in rows
        if row.get("model") == model_b and row.get("all_entity_ADE_m") not in {"", None}
    }
    comparisons: list[dict[str, Any]] = []
    for row in rows:
        if row.get("model") != model_a or row.get("all_entity_ADE_m") in {"", None}:
            continue
        key = (
            row.get("dataset"),
            row.get("num_matches"),
            row.get("horizon_seconds"),
            row.get("slice_name"),
        )
        other = by_key.get(key)
        if other is None or other.get("all_entity_ADE_m") in {"", None}:
            continue
        a = float(row["all_entity_ADE_m"])
        b = float(other["all_entity_ADE_m"])
        comparisons.append(
            {
                "dataset": row.get("dataset"),
                "num_matches": row.get("num_matches"),
                "horizon_seconds": row.get("horizon_seconds"),
                "slice_name": row.get("slice_name"),
                f"{model_a}_all_entity_ADE_m": a,
                f"{model_b}_all_entity_ADE_m": b,
                "improvement_m": b - a,
                "beats_reference": a < b,
                "smoke_split": row.get("smoke_split"),
            }
        )
    return comparisons


def _stress_result_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted(
        {
            (
                row.get("dataset"),
                row.get("num_matches"),
                row.get("horizon_steps"),
                row.get("horizon_seconds"),
                row.get("split"),
                row.get("slice_name"),
            )
            for row in rows
            if row.get("target_type") == "future" and row.get("slice_name") != "all_windows"
        },
        key=lambda item: tuple(str(value) for value in item),
    )
    out: list[dict[str, Any]] = []
    for key in keys:
        dataset, num_matches, horizon_steps, horizon_seconds, split, slice_name = key
        selected = [
            row
            for row in rows
            if (
                row.get("dataset"),
                row.get("num_matches"),
                row.get("horizon_steps"),
                row.get("horizon_seconds"),
                row.get("split"),
                row.get("slice_name"),
            )
            == key
        ]
        by_model = {row["model"]: row for row in selected}
        cv = by_model.get("coordinate_constant_velocity")
        context = by_model.get("residual_context_only_decoder")
        z_context = by_model.get("residual_z_plus_context_decoder")
        if cv is None or context is None or z_context is None:
            continue

        def _value(row: dict[str, Any], field: str) -> float | str:
            value = row.get(field)
            return "" if value in {"", None} else float(value)

        z_ade = _value(z_context, "all_entity_ADE_m")
        z_fde = _value(z_context, "all_entity_FDE_m")
        context_ade = _value(context, "all_entity_ADE_m")
        context_fde = _value(context, "all_entity_FDE_m")
        cv_ade = _value(cv, "all_entity_ADE_m")
        cv_fde = _value(cv, "all_entity_FDE_m")
        test_examples = max(int(z_context.get("num_test_examples", 0)), 1)
        num_examples = int(z_context.get("num_slice_examples", 0))
        row = {
            "dataset": dataset,
            "num_matches": num_matches,
            "horizon_steps": horizon_steps,
            "horizon_seconds": horizon_seconds,
            "split": split,
            "slice_name": slice_name,
            "num_examples": num_examples,
            "fraction_of_test_set": float(num_examples / test_examples),
            "coordinate_constant_velocity_ADE_m": cv_ade,
            "coordinate_constant_velocity_FDE_m": cv_fde,
            "residual_context_only_ADE_m": context_ade,
            "residual_context_only_FDE_m": context_fde,
            "residual_z_plus_context_ADE_m": z_ade,
            "residual_z_plus_context_FDE_m": z_fde,
            "z_plus_context_minus_context_only_ADE_m": (
                "" if "" in {z_ade, context_ade} else float(z_ade) - float(context_ade)
            ),
            "z_plus_context_minus_context_only_FDE_m": (
                "" if "" in {z_fde, context_fde} else float(z_fde) - float(context_fde)
            ),
            "z_plus_context_minus_coordinate_cv_ADE_m": (
                "" if "" in {z_ade, cv_ade} else float(z_ade) - float(cv_ade)
            ),
            "z_plus_context_minus_coordinate_cv_FDE_m": (
                "" if "" in {z_fde, cv_fde} else float(z_fde) - float(cv_fde)
            ),
        }
        metric_values = [
            value
            for value in [
                cv_ade,
                cv_fde,
                context_ade,
                context_fde,
                z_ade,
                z_fde,
            ]
            if value != ""
        ]
        row["finite_metrics"] = bool(metric_values) and all(
            math.isfinite(float(value)) for value in metric_values
        )
        out.append(row)
    return out


def _primary_real_split(diagnostics: list[dict[str, Any]]) -> dict[str, Any] | None:
    real = [item for item in diagnostics if not bool(item.get("smoke_split"))]
    if not real:
        return None
    return max(
        real,
        key=lambda item: (
            int(item.get("num_matches", 0)),
            int(item.get("num_examples", 0)),
        ),
    )


def _main_limitation(
    dataset_summaries: list[dict[str, Any]],
    expected_horizons: list[float],
) -> str:
    max_matches = max((int(item["num_matches"]) for item in dataset_summaries), default=0)
    completed = {float(item["horizon_seconds"]) for item in dataset_summaries}
    missing = [value for value in expected_horizons if float(value) not in completed]
    parts: list[str] = []
    if max_matches <= 3:
        parts.append("only three or fewer local matches are available")
    if missing:
        parts.append(
            "missing completed decoder datasets for horizons: " + ", ".join(map(str, missing))
        )
    return "; ".join(parts) if parts else "no major local scale limitation detected"


def _normalize_model_names(models: list[str] | None) -> tuple[list[str], set[str]]:
    requested = models or DEFAULT_MODELS
    canonical: list[str] = []
    original = {str(value) for value in requested}
    for value in requested:
        key = str(value)
        mapped = MODEL_ALIASES.get(key, key)
        if mapped not in canonical:
            canonical.append(mapped)
    return canonical, original


def run_decoder_learning_curve(
    dataset: str | Path | list[str | Path],
    out: str | Path,
    *,
    datasets: list[str | Path] | None = None,
    match_counts: list[str | int] | None = None,
    models: list[str] | None = None,
    stress_percentile: float = 0.75,
    require_real_split: bool = False,
    expected_horizons: list[float] | None = None,
    split: str = "test",
    device: str | None = "auto",
    epochs: int = 1,
    max_train_batches: int | None = 20,
    max_eval_batches: int | None = 20,
    batch_size: int = 256,
    seed: int = 123,
    run_root: str | Path = "runs",
) -> dict[str, Any]:
    """Run decoder variants over match-count subsets, horizons, and stress slices."""

    dataset_paths = datasets or (dataset if isinstance(dataset, list) else [dataset])
    dataset_paths = [Path(value) for value in dataset_paths]
    match_counts = match_counts or [1, 3, "all"]
    expected_horizons = expected_horizons or [2.0, 4.0, 6.0]
    model_names, original_model_names = _normalize_model_names(models)
    out_path = Path(out)
    subset_root = out_path / "subsets"
    rows: list[dict[str, Any]] = []
    subset_diagnostics: list[dict[str, Any]] = []
    dataset_summaries: list[dict[str, Any]] = []
    stress_thresholds: dict[str, Any] = {}
    for dataset_path in dataset_paths:
        data = load_decoder_dataset(dataset_path)
        dataset_label = dataset_path.stem
        available_match_count = len(set(str(value) for value in data.examples["match_id"]))
        if require_real_split and available_match_count < 3:
            raise ValueError(
                f"Dataset {dataset_label} has only {available_match_count} match IDs and cannot "
                "support a real disjoint train/val/test split."
            )
        dataset_summaries.append(
            {
                "dataset": str(dataset_path),
                "dataset_label": dataset_label,
                "num_examples": data.num_examples,
                "num_matches": available_match_count,
                "match_ids": sorted(set(str(value) for value in data.examples["match_id"])),
                "horizon_steps": data.horizon_steps,
                "horizon_seconds": float(
                    data.metadata.get(
                        "horizon_seconds",
                        data.horizon_steps / float(data.metadata.get("fps", 10.0)),
                    )
                ),
            }
        )
        for label, match_ids in _subset_specs(data, match_counts):
            indices = _indices_for_matches(data, match_ids)
            subset = subset_decoder_dataset(data, indices, seed=seed)
            diagnostics = decoder_split_diagnostics(subset)
            slice_masks, thresholds = compute_stress_slices(subset, percentile=stress_percentile)
            threshold_key = f"{dataset_label}:{label}"
            stress_thresholds[threshold_key] = thresholds
            subset_diagnostics.append(
                {
                    "dataset": str(dataset_path),
                    "dataset_label": dataset_label,
                    "subset_label": label,
                    "selected_match_ids": list(match_ids),
                    **diagnostics,
                    "warnings": list(subset.metadata.get("subset_warnings", [])),
                    "stress_slice_counts": thresholds.get("slice_counts_all_examples", {}),
                }
            )
            subset_path = subset_root / dataset_label / f"decoder_dataset_{label}_matches.pt"
            save_decoder_dataset(subset, subset_path)
            split_indices = [int(value) for value in subset.splits.get(f"{split}_indices", [])]
            if "coordinate_constant_velocity" in model_names:
                _append_future_slice_rows(
                    rows,
                    diagnostics,
                    data=subset,
                    dataset=str(dataset_path),
                    split=split,
                    model="coordinate_constant_velocity",
                    feature_source="past_coordinates",
                    decoder_type="analytic",
                    checkpoint=str(subset_path),
                    pred=subset.examples["coordinate_baseline_xy"][split_indices],
                    slice_masks=slice_masks,
                )
            if "last_coordinate_position" in model_names:
                _append_future_slice_rows(
                    rows,
                    diagnostics,
                    data=subset,
                    dataset=str(dataset_path),
                    split=split,
                    model="last_coordinate_position",
                    feature_source="past_coordinates",
                    decoder_type="analytic",
                    checkpoint=str(subset_path),
                    pred=subset.examples["last_position_xy"][split_indices],
                    slice_masks=slice_masks,
                )
            for model_key in model_names:
                if model_key in {"coordinate_constant_velocity", "last_coordinate_position"}:
                    continue
                if model_key not in TRAIN_SPECS:
                    raise ValueError(f"Unknown decoder learning-curve model: {model_key}")
                (
                    model_label,
                    feature_source,
                    decoder_type,
                    mode,
                    target_type,
                    best_metric,
                ) = TRAIN_SPECS[model_key]
                cfg = {
                    "experiment": f"decoder_learning_curve_{model_label}_{dataset_label}_{label}",
                    "seed": seed,
                    "data": {
                        "decoder_dataset": str(subset_path),
                        "batch_size": batch_size,
                        "num_workers": 0,
                    },
                    "target": {"mode": mode},
                    "model": {
                        "name": decoder_type,
                        "hidden_sizes": [128],
                        "dropout": 0.0,
                        "pooling": "flatten",
                    },
                    "training": {
                        "epochs": epochs,
                        "learning_rate": 0.001,
                        "weight_decay": 0.0,
                        "device": device or "auto",
                        "run_root": str(run_root),
                        "max_train_batches": max_train_batches,
                        "max_eval_batches": max_eval_batches,
                        "best_metric": best_metric,
                    },
                }
                result = train_coordinate_decoder_from_config(cfg)
                if target_type == "future":
                    pred = _checkpoint_predictions(
                        result["best_checkpoint"],
                        subset,
                        mode=mode,
                        split=split,
                        device=device,
                        batch_size=batch_size,
                    )
                    before = len(rows)
                    _append_future_slice_rows(
                        rows,
                        diagnostics,
                        data=subset,
                        dataset=str(dataset_path),
                        split=split,
                        model=model_label,
                        feature_source=feature_source,
                        decoder_type=decoder_type,
                        checkpoint=str(result["best_checkpoint"]),
                        pred=pred,
                        slice_masks=slice_masks,
                    )
                    if (
                        model_label == "context_only_decoder"
                        and (
                            "raw_past_summary_mlp" in original_model_names
                            or models is None
                            or "context_only_decoder" in original_model_names
                            or "context_only" in original_model_names
                        )
                    ):
                        for row in rows[before:]:
                            rows.append(
                                {
                                    **row,
                                    "model": "raw_past_summary_mlp",
                                    "model_name": "raw_past_summary_mlp",
                                }
                            )
                else:
                    metrics = result["test_metrics"]
                    rows.append(
                        _base_row(
                            diagnostics,
                            dataset=str(dataset_path),
                            horizon_steps=subset.horizon_steps,
                            horizon_seconds=float(
                                subset.metadata.get(
                                    "horizon_seconds",
                                    subset.horizon_steps / float(subset.metadata.get("fps", 10.0)),
                                )
                            ),
                            model=model_label,
                            split=split,
                            slice_name="all_windows",
                            num_slice_examples=diagnostics[f"num_{split}_examples"],
                            feature_source=feature_source,
                            decoder_type=decoder_type,
                            target_type=target_type,
                            metrics=metrics,
                            checkpoint=str(result["best_checkpoint"]),
                        )
                    )
    results_csv = out_path / "results.csv"
    stress_results_csv = out_path / "stress_results.csv"
    summary_json = out_path / "summary.json"
    _write_csv(rows, results_csv)
    stress_rows = _stress_result_rows(rows)
    _write_stress_csv(stress_rows, stress_results_csv)
    all_matches = max(int(row["num_matches"]) for row in rows) if rows else 0
    all_rows = [
        row
        for row in rows
        if int(row["num_matches"]) == all_matches and row.get("slice_name") == "all_windows"
    ]
    context_only = next((row for row in all_rows if row["model"] == "context_only_decoder"), None)
    z_plus_context = next(
        (row for row in all_rows if row["model"] == "z_plus_context_decoder"),
        None,
    )
    direct_comparisons = _comparison_rows(rows, "z_plus_context_decoder", "context_only_decoder")
    residual_comparisons = _comparison_rows(
        rows,
        "residual_z_plus_context_decoder",
        "residual_context_only_decoder",
    )
    cv_comparisons = _comparison_rows(
        rows,
        "residual_z_plus_context_decoder",
        "coordinate_constant_velocity",
    )
    overall_residual_comparisons = [
        item
        for item in residual_comparisons
        if item.get("slice_name") == "all_windows" and not bool(item.get("smoke_split"))
    ]
    stress_residual_comparisons = [
        item
        for item in residual_comparisons
        if item.get("slice_name") != "all_windows" and not bool(item.get("smoke_split"))
    ]
    stress_cv_comparisons = [
        item
        for item in cv_comparisons
        if item.get("slice_name") != "all_windows" and not bool(item.get("smoke_split"))
    ]
    primary_split = _primary_real_split(subset_diagnostics)
    prepared_match_count_by_horizon = {
        str(item["horizon_seconds"]): item["num_matches"] for item in dataset_summaries
    }
    decoder_example_count_by_horizon = {
        str(item["horizon_seconds"]): item["num_examples"] for item in dataset_summaries
    }
    horizon_completion = {
        str(value): any(
            math.isclose(float(item["horizon_seconds"]), float(value))
            for item in dataset_summaries
        )
        for value in expected_horizons
    }
    summary = {
        "results_csv": str(results_csv),
        "stress_results_csv": str(stress_results_csv),
        "summary_json": str(summary_json),
        "dataset_summaries": dataset_summaries,
        "num_available_matches": max(
            (item["num_matches"] for item in dataset_summaries),
            default=0,
        ),
        "raw_match_count": max((item["num_matches"] for item in dataset_summaries), default=0),
        "raw_match_count_source": "decoder_datasets",
        "prepared_match_count_by_horizon": prepared_match_count_by_horizon,
        "decoder_example_count_by_horizon": decoder_example_count_by_horizon,
        "train_match_ids": primary_split.get("train_match_ids", []) if primary_split else [],
        "val_match_ids": primary_split.get("val_match_ids", []) if primary_split else [],
        "test_match_ids": primary_split.get("test_match_ids", []) if primary_split else [],
        "split_disjoint": (
            bool(primary_split.get("disjoint_match_split")) if primary_split else False
        ),
        "horizon_completion": horizon_completion,
        "six_second_completed": bool(horizon_completion.get("6.0", False)),
        "best_current_reconstruction": _best_row(rows, "current_all_entity_error_m", "current"),
        "best_current_reconstruction_real_split": _best_row(
            rows,
            "current_all_entity_error_m",
            "current",
            include_smoke=False,
        ),
        "best_future_decoder": _best_row(rows, "all_entity_ADE_m", "future"),
        "best_future_decoder_real_split": _best_row(
            rows,
            "all_entity_ADE_m",
            "future",
            include_smoke=False,
        ),
        "best_learned_future_decoder_real_split": _best_row(
            rows,
            "all_entity_ADE_m",
            "future",
            include_smoke=False,
            exclude_models={"coordinate_constant_velocity", "last_coordinate_position"},
        ),
        "best_residual_future_decoder_real_split": _best_row(
            [
                row
                for row in rows
                if str(row.get("model", "")).startswith("residual_")
            ],
            "all_entity_ADE_m",
            "future",
            include_smoke=False,
        ),
        "context_only_all_entity_ADE_m": (
            context_only.get("all_entity_ADE_m") if context_only else None
        ),
        "z_plus_context_all_entity_ADE_m": (
            z_plus_context.get("all_entity_ADE_m") if z_plus_context else None
        ),
        "td_jepa_adds_value_over_context": (
            bool(
                context_only
                and z_plus_context
                and float(z_plus_context["all_entity_ADE_m"])
                < float(context_only["all_entity_ADE_m"])
            )
        ),
        "best_model_overall": _best_row(rows, "all_entity_ADE_m", "future", include_smoke=False),
        "best_model_per_horizon": _best_rows_by(rows, "horizon_seconds", slice_name="all_windows"),
        "best_model_per_stress_slice": _best_rows_by(rows, "slice_name"),
        "direct_z_plus_context_vs_context_only": direct_comparisons,
        "residual_z_plus_context_vs_context_only": residual_comparisons,
        "residual_z_plus_context_minus_context_only_overall": overall_residual_comparisons,
        "residual_z_plus_context_minus_coordinate_cv_overall": [
            item
            for item in cv_comparisons
            if item.get("slice_name") == "all_windows" and not bool(item.get("smoke_split"))
        ],
        "residual_z_plus_context_beats_context_only_overall": any(
            bool(item["beats_reference"]) for item in overall_residual_comparisons
        ),
        "residual_z_plus_context_beats_context_only_all_overall_rows": bool(
            overall_residual_comparisons
        )
        and all(bool(item["beats_reference"]) for item in overall_residual_comparisons),
        "residual_z_plus_context_beats_context_only_any_stress_slice": any(
            bool(item["beats_reference"]) for item in stress_residual_comparisons
        ),
        "residual_z_plus_context_beats_coordinate_constant_velocity_any_stress_slice": any(
            bool(item["beats_reference"]) for item in stress_cv_comparisons
        ),
        "coordinate_constant_velocity_approach_threshold_m": 0.5,
        "residual_z_plus_context_approaches_coordinate_constant_velocity_any_stress_slice": any(
            float(item["improvement_m"]) >= -0.5 for item in stress_cv_comparisons
        ),
        "stress_thresholds": stress_thresholds,
        "limited_to_three_matches": max(
            (item["num_matches"] for item in dataset_summaries),
            default=0,
        )
        <= 3,
        "subset_diagnostics": subset_diagnostics,
        "stress_row_count": len(stress_rows),
        "main_limitation": _main_limitation(dataset_summaries, expected_horizons),
        "num_rows": len(rows),
    }
    save_json({"rows": rows, "summary": summary}, out_path / "results.json")
    save_json(summary, summary_json)
    return {
        "rows": rows,
        "stress_rows": stress_rows,
        "summary": summary,
        "results_csv": results_csv,
        "stress_results_csv": stress_results_csv,
        "summary_json": summary_json,
    }
