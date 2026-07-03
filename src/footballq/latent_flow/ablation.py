"""Stochastic latent-flow ablation utilities."""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from footballq.latent_flow.eval import evaluate_latent_baseline, evaluate_latent_checkpoint
from footballq.latent_flow.io import load_yaml, save_json
from footballq.latent_flow.train import train_latent_flow_from_config

ABLATION_FIELDS = [
    "model",
    "baseline_mode",
    "noise_scale",
    "num_sampling_steps",
    "num_samples",
    "latent_ADE",
    "latent_FDE",
    "minADE",
    "minFDE",
    "cosine_similarity",
    "diversity_mean_pairwise_distance",
    "sample_std_mean",
    "checkpoint",
    "split",
]


def _load_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    return load_yaml(config)


def _dataset_path(cfg: dict[str, Any]) -> Path:
    value = cfg.get("data", {}).get("rollout_dataset", cfg.get("data", {}).get("path", ""))
    if not value:
        raise ValueError("Latent-flow ablation requires data.rollout_dataset.")
    return Path(value)


def _finite_metric(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Metric {key!r} is not finite: {value!r}")
    return value


def _checkpoint_model_name(metrics: dict[str, Any], residual_mode: str) -> str:
    model_name = str(metrics.get("model", "residual_flow"))
    if model_name == "residual_latent_flow_mlp":
        if residual_mode == "constant_latent_velocity":
            return "residual_flow_cv"
        if residual_mode == "last_latent":
            return "residual_flow_last"
    return model_name


def _row_from_metrics(
    metrics: dict[str, Any],
    *,
    model: str,
    baseline_mode: str,
    noise_scale: float,
    num_sampling_steps: int,
    num_samples: int,
    checkpoint: str,
    split: str,
) -> dict[str, Any]:
    row = {
        "model": model,
        "baseline_mode": baseline_mode,
        "noise_scale": float(noise_scale),
        "num_sampling_steps": int(num_sampling_steps),
        "num_samples": int(num_samples),
        "latent_ADE": _finite_metric(metrics, "latent_ADE"),
        "latent_FDE": _finite_metric(metrics, "latent_FDE"),
        "minADE": _finite_metric(metrics, "minADE", _finite_metric(metrics, "latent_ADE")),
        "minFDE": _finite_metric(metrics, "minFDE", _finite_metric(metrics, "latent_FDE")),
        "cosine_similarity": _finite_metric(
            metrics,
            "cosine_similarity",
            _finite_metric(metrics, "latent_cosine_similarity", 0.0),
        ),
        "diversity_mean_pairwise_distance": _finite_metric(
            metrics,
            "diversity_mean_pairwise_distance",
            0.0,
        ),
        "sample_std_mean": _finite_metric(metrics, "sample_std_mean", 0.0),
        "checkpoint": checkpoint,
        "split": split,
    }
    return row


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ABLATION_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in ABLATION_FIELDS})


def _best_row(rows: Iterable[dict[str, Any]], key: str) -> dict[str, Any] | None:
    candidates = list(rows)
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row[key]))


def _compact_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {field: row.get(field) for field in ABLATION_FIELDS}


def summarize_ablation(
    rows: list[dict[str, Any]],
    *,
    max_mean_ade_multiplier: float = 2.0,
) -> dict[str, Any]:
    baseline_rows = [
        row for row in rows if row["model"] in {"last_latent", "constant_latent_velocity"}
    ]
    residual_rows = [row for row in rows if str(row["model"]).startswith("residual_flow")]
    stochastic_rows = [row for row in residual_rows if float(row["noise_scale"]) > 0.0]
    cv_row = next(
        (row for row in baseline_rows if row["model"] == "constant_latent_velocity"),
        None,
    )
    last_row = next((row for row in baseline_rows if row["model"] == "last_latent"), None)
    cv_ade = float(cv_row["latent_ADE"]) if cv_row else math.inf
    last_fde = float(last_row["latent_FDE"]) if last_row else math.inf
    ade_threshold = cv_ade * float(max_mean_ade_multiplier)
    best_stochastic_by_minade = _best_row(stochastic_rows, "minADE")
    improves_minade = any(float(row["minADE"]) < cv_ade for row in stochastic_rows)
    improves_minfde = any(float(row["minFDE"]) < last_fde for row in stochastic_rows)
    unacceptable = bool(
        best_stochastic_by_minade is not None
        and float(best_stochastic_by_minade["latent_ADE"]) > ade_threshold
    )
    if (improves_minade or improves_minfde) and not unacceptable:
        decision = "stochastic_residual_flow_useful"
    elif stochastic_rows:
        decision = "stochastic_residual_flow_not_useful_yet"
    else:
        decision = "more_training_or_ablation_needed"
    return {
        "best_deterministic_baseline_by_latent_ADE": _compact_row(
            _best_row(baseline_rows, "latent_ADE")
        ),
        "best_deterministic_baseline_by_latent_FDE": _compact_row(
            _best_row(baseline_rows, "latent_FDE")
        ),
        "best_residual_flow_config_by_minADE": _compact_row(
            _best_row(residual_rows, "minADE")
        ),
        "best_residual_flow_config_by_minFDE": _compact_row(
            _best_row(residual_rows, "minFDE")
        ),
        "best_stochastic_residual_flow_by_minADE": _compact_row(best_stochastic_by_minade),
        "improves_minADE_over_constant_latent_velocity_ADE": bool(improves_minade),
        "improves_minFDE_over_last_latent_FDE": bool(improves_minfde),
        "max_mean_ADE_multiplier": float(max_mean_ade_multiplier),
        "mean_ADE_degradation_threshold": ade_threshold,
        "unacceptable_mean_ADE_degradation": unacceptable,
        "any_stochastic_mean_ADE_over_threshold": any(
            float(row["latent_ADE"]) > ade_threshold for row in stochastic_rows
        ),
        "decision": decision,
        "num_rows": len(rows),
    }


def run_latent_flow_ablation(
    base_config: str | Path | dict[str, Any],
    out: str | Path,
    *,
    checkpoint: str | Path | None = None,
    noise_scales: Iterable[float] = (0.0, 0.01, 0.03, 0.05, 0.1),
    num_steps: Iterable[int] = (5, 10, 20),
    num_samples: Iterable[int] = (4, 8, 16),
    split: str = "test",
    device: str | None = "auto",
    max_mean_ade_multiplier: float = 2.0,
) -> dict[str, Any]:
    """Run stochastic residual-flow evaluation and write CSV/JSON reports."""

    cfg = _load_config(base_config)
    dataset = _dataset_path(cfg)
    out_path = Path(out)
    residual_mode = str(
        cfg.get("flow", {}).get("residual_mode", cfg.get("model", {}).get("residual_mode", ""))
    )
    if checkpoint is None:
        train_result = train_latent_flow_from_config(cfg)
        checkpoint_path = Path(train_result["best_checkpoint"])
    else:
        train_result = None
        checkpoint_path = Path(checkpoint)

    rows: list[dict[str, Any]] = []
    for baseline in ["last_latent", "constant_latent_velocity"]:
        result = evaluate_latent_baseline(dataset, baseline=baseline, split=split, device=device)
        rows.append(
            _row_from_metrics(
                result["metrics"],
                model=baseline,
                baseline_mode=baseline,
                noise_scale=0.0,
                num_sampling_steps=0,
                num_samples=1,
                checkpoint=str(dataset),
                split=split,
            )
        )

    for noise_scale in noise_scales:
        for steps in num_steps:
            for samples in num_samples:
                result = evaluate_latent_checkpoint(
                    checkpoint_path,
                    dataset=dataset,
                    split=split,
                    device=device,
                    num_samples=int(samples),
                    num_steps=int(steps),
                    noise_scale=float(noise_scale),
                )
                metrics = result["metrics"]
                row_model = _checkpoint_model_name(metrics, residual_mode)
                rows.append(
                    _row_from_metrics(
                        metrics,
                        model=row_model,
                        baseline_mode=str(metrics.get("residual_mode", residual_mode)),
                        noise_scale=float(noise_scale),
                        num_sampling_steps=int(steps),
                        num_samples=int(samples),
                        checkpoint=str(checkpoint_path),
                        split=split,
                    )
                )

    out_path.mkdir(parents=True, exist_ok=True)
    results_csv = out_path / "results.csv"
    summary_json = out_path / "summary.json"
    _write_csv(rows, results_csv)
    summary = summarize_ablation(rows, max_mean_ade_multiplier=max_mean_ade_multiplier)
    summary.update(
        {
            "results_csv": str(results_csv),
            "summary_json": str(summary_json),
            "checkpoint": str(checkpoint_path),
            "trained_run_dir": str(train_result["run_dir"]) if train_result else None,
        }
    )
    save_json(summary, summary_json)
    save_json({"results": rows, "summary": summary}, out_path / "results.json")
    return {
        "rows": rows,
        "summary": summary,
        "results_csv": results_csv,
        "summary_json": summary_json,
    }
