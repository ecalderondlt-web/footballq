"""Learning-curve runner for Experiment 4C.1 decoder diagnostics."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import torch

from footballq.decoding.dataset import (
    DecoderDatasetData,
    decoder_split_diagnostics,
    load_decoder_dataset,
    save_decoder_dataset,
    subset_decoder_dataset,
)
from footballq.decoding.metrics import decoder_metrics
from footballq.decoding.suite import evaluate_constant_velocity_baseline
from footballq.decoding.train import train_coordinate_decoder_from_config
from footballq.latent_flow.io import save_json

LEARNING_CURVE_FIELDS = [
    "num_matches",
    "num_examples",
    "num_train_examples",
    "num_val_examples",
    "num_test_examples",
    "train_match_ids",
    "val_match_ids",
    "test_match_ids",
    "disjoint_match_split",
    "smoke_split",
    "model",
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
    model: str,
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
        "train_match_ids": _join(diagnostics["train_match_ids"]),
        "val_match_ids": _join(diagnostics["val_match_ids"]),
        "test_match_ids": _join(diagnostics["test_match_ids"]),
        "disjoint_match_split": bool(diagnostics["disjoint_match_split"]),
        "smoke_split": bool(diagnostics["smoke_split"]),
        "model": model,
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


def _indices_for_matches(data: DecoderDatasetData, match_ids: list[str]) -> list[int]:
    selected = set(str(value) for value in match_ids)
    return [
        idx
        for idx, match_id in enumerate(data.examples["match_id"])
        if str(match_id) in selected
    ]


def _subset_specs(data: DecoderDatasetData, match_counts: list[str | int]) -> list[tuple[str, list[str]]]:
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
        and str(row.get("model")) not in exclude_models
        and math.isfinite(float(row[key]))
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row[key]))


def run_decoder_learning_curve(
    dataset: str | Path,
    out: str | Path,
    *,
    match_counts: list[str | int] | None = None,
    split: str = "test",
    device: str | None = "auto",
    epochs: int = 1,
    max_train_batches: int | None = 20,
    max_eval_batches: int | None = 20,
    batch_size: int = 256,
    seed: int = 123,
    run_root: str | Path = "runs",
) -> dict[str, Any]:
    """Run decoder variants over match-count subsets and write reports."""

    data = load_decoder_dataset(dataset)
    match_counts = match_counts or [1, 3, "all"]
    out_path = Path(out)
    subset_root = out_path / "subsets"
    rows: list[dict[str, Any]] = []
    subset_diagnostics: list[dict[str, Any]] = []
    for label, match_ids in _subset_specs(data, match_counts):
        indices = _indices_for_matches(data, match_ids)
        subset = subset_decoder_dataset(data, indices, seed=seed)
        diagnostics = decoder_split_diagnostics(subset)
        subset_diagnostics.append(
            {
                "subset_label": label,
                "selected_match_ids": list(match_ids),
                **diagnostics,
                "warnings": list(subset.metadata.get("subset_warnings", [])),
            }
        )
        subset_path = subset_root / f"decoder_dataset_{label}_matches.pt"
        save_decoder_dataset(subset, subset_path)
        rows.append(
            _base_row(
                diagnostics,
                model="coordinate_constant_velocity",
                feature_source="past_coordinates",
                decoder_type="analytic",
                target_type="future",
                metrics=evaluate_constant_velocity_baseline(
                    subset,
                    split=split,
                    batch_size=batch_size,
                    device=device,
                ),
                checkpoint=str(subset_path),
            )
        )
        rows.append(
            _base_row(
                diagnostics,
                model="last_coordinate_position",
                feature_source="past_coordinates",
                decoder_type="analytic",
                target_type="future",
                metrics=_last_position_metrics(subset, split),
                checkpoint=str(subset_path),
            )
        )
        train_specs = [
            (
                "mlp_reconstruct_current_from_z",
                "td_jepa_z",
                "mlp",
                "reconstruct_current",
                "current",
                "current_all_entity_error_m",
            ),
            (
                "context_mlp_reconstruct_current",
                "raw_past_context",
                "raw_context_mlp",
                "reconstruct_current_from_context",
                "current",
                "current_all_entity_error_m",
            ),
            (
                "z_plus_context_reconstruct_current",
                "z_plus_context",
                "z_context_mlp",
                "reconstruct_current_from_z_context",
                "current",
                "current_all_entity_error_m",
            ),
            (
                "z_only_decoder",
                "td_jepa_z",
                "mlp",
                "future_from_z",
                "future",
                "all_entity_ADE_m",
            ),
            (
                "context_only_decoder",
                "raw_past_context",
                "raw_context_mlp",
                "future_from_past_context",
                "future",
                "all_entity_ADE_m",
            ),
            (
                "z_plus_context_decoder",
                "z_plus_context",
                "z_context_mlp",
                "future_from_z_past_context",
                "future",
                "all_entity_ADE_m",
            ),
            (
                "residual_context_only_decoder",
                "raw_past_context",
                "residual_context_mlp",
                "residual_future_from_past_context",
                "future",
                "all_entity_ADE_m",
            ),
            (
                "residual_z_plus_context_decoder",
                "z_plus_context",
                "residual_context_mlp",
                "residual_future_from_z_past_context",
                "future",
                "all_entity_ADE_m",
            ),
        ]
        for model_label, feature_source, decoder_type, mode, target_type, best_metric in train_specs:
            cfg = {
                "experiment": f"decoder_learning_curve_{model_label}_{label}",
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
            row = _base_row(
                diagnostics,
                model=model_label,
                feature_source=feature_source,
                decoder_type=decoder_type,
                target_type=target_type,
                metrics=result["test_metrics"],
                checkpoint=str(result["best_checkpoint"]),
            )
            rows.append(row)
            if model_label == "context_only_decoder":
                rows.append({**row, "model": "raw_past_summary_mlp"})
    results_csv = out_path / "results.csv"
    summary_json = out_path / "summary.json"
    _write_csv(rows, results_csv)
    all_matches = max(int(row["num_matches"]) for row in rows) if rows else 0
    all_rows = [row for row in rows if int(row["num_matches"]) == all_matches]
    context_only = next((row for row in all_rows if row["model"] == "context_only_decoder"), None)
    z_plus_context = next((row for row in all_rows if row["model"] == "z_plus_context_decoder"), None)
    summary = {
        "results_csv": str(results_csv),
        "summary_json": str(summary_json),
        "num_available_matches": len(set(str(value) for value in data.examples["match_id"])),
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
        "context_only_all_entity_ADE_m": context_only.get("all_entity_ADE_m") if context_only else None,
        "z_plus_context_all_entity_ADE_m": z_plus_context.get("all_entity_ADE_m") if z_plus_context else None,
        "td_jepa_adds_value_over_context": (
            bool(
                context_only
                and z_plus_context
                and float(z_plus_context["all_entity_ADE_m"]) < float(context_only["all_entity_ADE_m"])
            )
        ),
        "limited_to_three_matches": len(set(str(value) for value in data.examples["match_id"])) <= 3,
        "subset_diagnostics": subset_diagnostics,
        "num_rows": len(rows),
    }
    save_json({"rows": rows, "summary": summary}, out_path / "results.json")
    save_json(summary, summary_json)
    return {"rows": rows, "summary": summary, "results_csv": results_csv, "summary_json": summary_json}
