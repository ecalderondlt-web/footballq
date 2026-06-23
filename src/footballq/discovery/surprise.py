"""Tactical surprise metrics for latent transitions."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from footballq.discovery.transitions import STRESS_FIELDS, TransitionDatasetData, load_transition_dataset


def _mask_for_delta(data: TransitionDatasetData, delta_seconds: float | None) -> torch.Tensor:
    if delta_seconds is None:
        return torch.ones(data.num_examples, dtype=torch.bool)
    values = torch.as_tensor(data.examples["delta_seconds"]).float()
    return torch.isclose(values, torch.tensor(float(delta_seconds)), atol=1e-6)


def _finite_stats(values: torch.Tensor, prefix: str) -> dict[str, float]:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return {f"{prefix}_{name}": float("nan") for name in ["mean", "std", "p50", "p90", "p95", "p99"]}
    return {
        f"{prefix}_mean": float(finite.mean().item()),
        f"{prefix}_std": float(finite.std().item()) if finite.numel() > 1 else 0.0,
        f"{prefix}_p50": float(torch.quantile(finite, 0.50).item()),
        f"{prefix}_p90": float(torch.quantile(finite, 0.90).item()),
        f"{prefix}_p95": float(torch.quantile(finite, 0.95).item()),
        f"{prefix}_p99": float(torch.quantile(finite, 0.99).item()),
    }


def compute_surprise(data: TransitionDatasetData) -> dict[str, torch.Tensor]:
    z_next = torch.as_tensor(data.examples["z_next"]).float()
    z_t = torch.as_tensor(data.examples["z_t"]).float()
    z_prev = torch.as_tensor(data.examples["z_prev"]).float()
    has_prev = torch.as_tensor(data.examples["has_prev"]).bool()
    surprise_last = torch.linalg.vector_norm(z_next - z_t, dim=1)
    z_pred_cv = z_t + (z_t - z_prev)
    surprise_cv = torch.linalg.vector_norm(z_next - z_pred_cv, dim=1)
    surprise_cv = surprise_cv.masked_fill(~has_prev, float("nan"))
    return {
        "surprise_last": surprise_last,
        "surprise_cv": surprise_cv,
    }


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    mask = torch.isfinite(a) & torch.isfinite(b)
    if int(mask.sum().item()) < 3:
        return float("nan")
    x = a[mask].float()
    y = b[mask].float()
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denom.item()) <= 1e-12:
        return float("nan")
    return float(((x * y).sum() / denom).item())


def _row_value(values: Any, idx: int) -> Any:
    if isinstance(values, torch.Tensor):
        item = values[idx]
        return item.item() if item.numel() == 1 else item.tolist()
    return values[idx]


def _example_rows(
    data: TransitionDatasetData,
    scores: torch.Tensor,
    indices: list[int],
    score_name: str,
    cluster_ids: dict[int, int] | None = None,
) -> list[dict[str, Any]]:
    meta = data.examples.get("metadata", {})
    rows = []
    for idx in indices:
        row = {
            "rank_source": score_name,
            "match_id": data.examples["match_id"][idx],
            "period": data.examples["period"][idx],
            "frame_t": int(data.examples["frame_t"][idx].item()),
            "frame_next": int(data.examples["frame_next"][idx].item()),
            "delta_seconds": float(data.examples["delta_seconds"][idx].item()),
            "actual_delta_seconds": float(data.examples["actual_delta_seconds"][idx].item()),
            "surprise_score": float(scores[idx].item()),
            "cluster_id": "" if cluster_ids is None or idx not in cluster_ids else cluster_ids[idx],
        }
        for field in [
            "future_ball_displacement_m",
            "team_shape_change_m",
            "ball_acceleration_mps2",
            "ball_direction_change_rad",
            "event_type",
            "phase",
            *STRESS_FIELDS,
        ]:
            values = meta.get(field)
            row[field] = "" if values is None else _row_value(values, idx)
        rows.append(row)
    return rows


def _load_cluster_ids(assignments_path: str | Path | None) -> dict[int, int] | None:
    if assignments_path is None:
        return None
    payload = torch.load(Path(assignments_path), map_location="cpu", weights_only=False)
    global_indices = [int(value) for value in payload["global_indices"]]
    assignments = [int(value) for value in payload["assignments"].tolist()]
    return dict(zip(global_indices, assignments, strict=True))


def analyze_surprise(
    data: TransitionDatasetData,
    delta_seconds: float | None = None,
    assignments_path: str | Path | None = None,
    top_n: int = 100,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    surprise = compute_surprise(data)
    mask = _mask_for_delta(data, delta_seconds)
    selected = mask.nonzero(as_tuple=False).flatten()
    if selected.numel() == 0:
        raise ValueError(f"No transitions found for delta_seconds={delta_seconds}")
    score_name = "surprise_cv"
    score_all = surprise["surprise_cv"]
    if not bool(torch.isfinite(score_all[selected]).any()):
        score_name = "surprise_last"
        score_all = surprise["surprise_last"]
    selected_scores = score_all[selected]
    finite_selected = selected[torch.isfinite(selected_scores)]
    finite_scores = score_all[finite_selected]
    order_desc = torch.argsort(finite_scores, descending=True)
    order_asc = torch.argsort(finite_scores, descending=False)
    top_indices = [int(finite_selected[idx].item()) for idx in order_desc[:top_n]]
    bottom_indices = [int(finite_selected[idx].item()) for idx in order_asc[:top_n]]
    cluster_ids = _load_cluster_ids(assignments_path)
    rows = _example_rows(data, score_all, top_indices, score_name, cluster_ids)
    for row in _example_rows(data, score_all, bottom_indices, score_name, cluster_ids):
        row["rank_source"] = f"low_{score_name}"
        rows.append(row)

    meta = data.examples.get("metadata", {})
    threshold = float(torch.quantile(finite_scores.float(), 0.95).item())
    high_mask_global = torch.zeros(data.num_examples, dtype=torch.bool)
    high_mask_global[finite_selected] = score_all[finite_selected] >= threshold
    high_selected = high_mask_global[selected]

    by_match: dict[str, list[float]] = defaultdict(list)
    match_ids = data.examples["match_id"]
    for idx in finite_selected.tolist():
        by_match[str(match_ids[int(idx)])].append(float(score_all[int(idx)].item()))
    surprise_by_match = {
        match_id: {
            "count": len(values),
            "mean": float(sum(values) / max(1, len(values))),
            "max": float(max(values)),
        }
        for match_id, values in sorted(by_match.items())
    }

    correlations = {}
    field_map = {
        "future_ball_displacement_m": "future_ball_displacement_corr",
        "ball_acceleration_mps2": "ball_acceleration_corr",
        "team_shape_change_m": "team_shape_change_corr",
    }
    for field, name in field_map.items():
        if field in meta:
            values = torch.tensor([float(_row_value(meta[field], idx)) for idx in range(data.num_examples)])
            correlations[name] = _corr(score_all, values)

    stress_enrichment = {}
    for field in STRESS_FIELDS:
        if field not in meta:
            continue
        values = torch.tensor([bool(_row_value(meta[field], idx)) for idx in range(data.num_examples)])
        global_rate = float(values[selected].float().mean().item()) if selected.numel() else 0.0
        high_rate = float(values[selected][high_selected].float().mean().item()) if bool(high_selected.any()) else 0.0
        stress_enrichment[field] = {
            "global_rate": global_rate,
            "high_surprise_rate": high_rate,
            "enrichment_ratio": high_rate / max(global_rate, 1e-12),
        }

    summary = {
        "delta_seconds": delta_seconds,
        "score_name": score_name,
        "num_examples": int(selected.numel()),
        "num_finite_scores": int(finite_selected.numel()),
        "high_surprise_threshold": threshold,
        **_finite_stats(surprise["surprise_last"][selected], "surprise_last"),
        **_finite_stats(surprise["surprise_cv"][selected], "surprise_cv"),
        "surprise_by_match": surprise_by_match,
        "correlations": correlations,
        "stress_enrichment": stress_enrichment,
    }
    if cluster_ids is not None:
        cluster_scores: dict[int, list[float]] = defaultdict(list)
        for idx in finite_selected.tolist():
            cluster_id = cluster_ids.get(int(idx))
            if cluster_id is not None:
                cluster_scores[int(cluster_id)].append(float(score_all[int(idx)].item()))
        summary["surprise_by_cluster"] = {
            str(cluster_id): {
                "count": len(values),
                "mean": sum(values) / max(1, len(values)),
                "max": max(values),
            }
            for cluster_id, values in sorted(cluster_scores.items())
        }
    return rows, summary


def write_surprise_outputs(
    dataset: str | Path,
    out: str | Path,
    delta_seconds: float | None = None,
    assignments: str | Path | None = None,
    top_n: int = 100,
) -> dict[str, Any]:
    data = load_transition_dataset(dataset)
    rows, summary = analyze_surprise(
        data,
        delta_seconds=delta_seconds,
        assignments_path=assignments,
        top_n=top_n,
    )
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    examples_path = out_dir / "surprise_examples.csv"
    with examples_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = out_dir / "surprise_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return {
        "surprise_examples": str(examples_path),
        "surprise_summary": str(summary_path),
        "summary": summary,
    }
