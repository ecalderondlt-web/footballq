"""Cluster-label enrichment for latent transition discovery."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from footballq.discovery.transitions import (
    STRESS_FIELDS,
    TransitionDatasetData,
    load_transition_dataset,
)

CATEGORICAL_LABELS = [
    "future_ball_global_x_bucket",
    "future_ball_displacement_bucket",
    "possession_team_id",
    "possession_available",
    "team_shape_change_bucket",
    *STRESS_FIELDS,
    "high_width_change",
    "high_length_change",
    "event_type",
    "phase",
]

CONTINUOUS_LABELS = [
    "future_ball_displacement_m",
    "future_ball_dx_global_m",
    "team_shape_change_m",
    "team_width_change_m",
    "team_length_change_m",
    "stretch_index_change_m",
    "ball_acceleration_mps2",
    "ball_direction_change_rad",
]


def _load_assignment_payload(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def _subset_values(values: Any, indices: list[int]) -> list[Any]:
    if isinstance(values, torch.Tensor):
        out = []
        for idx in indices:
            item = values[idx]
            out.append(item.item() if item.numel() == 1 else item.tolist())
        return out
    return [values[idx] for idx in indices]


def _categorical_rows(
    data: TransitionDatasetData,
    assignments: torch.Tensor,
    global_indices: list[int],
    labels: list[str],
) -> list[dict[str, Any]]:
    rows = []
    meta = data.examples.get("metadata", {})
    k = int(assignments.max().item()) + 1 if assignments.numel() else 0
    n = len(global_indices)
    for label in labels:
        if label not in meta:
            continue
        global_values = [str(value) for value in _subset_values(meta[label], global_indices)]
        global_counts = Counter(global_values)
        for cluster_id in range(k):
            local_indices = [
                pos
                for pos, assignment in enumerate(assignments.tolist())
                if int(assignment) == cluster_id
            ]
            if not local_indices:
                continue
            cluster_values = [global_values[pos] for pos in local_indices]
            cluster_counts = Counter(cluster_values)
            for value, support in cluster_counts.items():
                global_p = global_counts[value] / max(1, n)
                cluster_p = support / max(1, len(local_indices))
                ratio = cluster_p / max(global_p, 1e-12)
                rows.append(
                    {
                        "kind": "categorical",
                        "label": label,
                        "value": value,
                        "cluster_id": cluster_id,
                        "support": int(support),
                        "cluster_size": len(local_indices),
                        "global_count": int(global_counts[value]),
                        "cluster_probability": cluster_p,
                        "global_probability": global_p,
                        "enrichment_ratio": ratio,
                        "log_enrichment": math.log(ratio + 1e-12),
                        "score": math.log(ratio + 1e-12) * math.sqrt(float(support)),
                    }
                )
    return rows


def _continuous_rows(
    data: TransitionDatasetData,
    assignments: torch.Tensor,
    global_indices: list[int],
    labels: list[str],
) -> list[dict[str, Any]]:
    rows = []
    meta = data.examples.get("metadata", {})
    k = int(assignments.max().item()) + 1 if assignments.numel() else 0
    for label in labels:
        if label not in meta:
            continue
        values = torch.tensor(
            [float(value) for value in _subset_values(meta[label], global_indices)]
        )
        finite = values[torch.isfinite(values)]
        if finite.numel() == 0:
            continue
        global_mean = float(finite.mean().item())
        global_std = float(finite.std().clamp_min(1e-6).item())
        for cluster_id in range(k):
            local_mask = assignments == cluster_id
            cluster_values = values[local_mask]
            cluster_finite = cluster_values[torch.isfinite(cluster_values)]
            if cluster_finite.numel() == 0:
                continue
            cluster_mean = float(cluster_finite.mean().item())
            cluster_std = float(cluster_finite.std().item()) if cluster_finite.numel() > 1 else 0.0
            z_score = (cluster_mean - global_mean) / global_std
            rows.append(
                {
                    "kind": "continuous",
                    "label": label,
                    "value": "__mean__",
                    "cluster_id": cluster_id,
                    "support": int(cluster_finite.numel()),
                    "cluster_size": int(local_mask.sum().item()),
                    "global_count": int(finite.numel()),
                    "cluster_probability": "",
                    "global_probability": "",
                    "enrichment_ratio": "",
                    "log_enrichment": "",
                    "score": z_score,
                    "cluster_mean": cluster_mean,
                    "cluster_std": cluster_std,
                    "global_mean": global_mean,
                    "global_std": global_std,
                }
            )
    return rows


def compute_enrichment(
    data: TransitionDatasetData,
    assignments_payload: dict[str, Any],
    categorical_labels: list[str] | None = None,
    continuous_labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    assignments = torch.as_tensor(assignments_payload["assignments"]).long()
    global_indices = [int(value) for value in assignments_payload["global_indices"]]
    categorical = categorical_labels or CATEGORICAL_LABELS
    continuous = continuous_labels or CONTINUOUS_LABELS
    rows = _categorical_rows(data, assignments, global_indices, categorical)
    rows.extend(_continuous_rows(data, assignments, global_indices, continuous))
    return sorted(rows, key=lambda row: abs(float(row.get("score") or 0.0)), reverse=True)


def write_enrichment_outputs(
    dataset: str | Path,
    assignments: str | Path,
    out_csv: str | Path,
    summary_out: str | Path | None = None,
) -> dict[str, Any]:
    data = load_transition_dataset(dataset)
    payload = _load_assignment_payload(assignments)
    rows = compute_enrichment(data, payload)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "kind",
        "label",
        "value",
        "cluster_id",
        "support",
        "cluster_size",
        "global_count",
        "cluster_probability",
        "global_probability",
        "enrichment_ratio",
        "log_enrichment",
        "score",
        "cluster_mean",
        "cluster_std",
        "global_mean",
        "global_std",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    top = rows[:20]
    summary = {
        "dataset": str(dataset),
        "assignments": str(assignments),
        "enrichment_csv": str(out_path),
        "num_rows": len(rows),
        "top_associations": top,
    }
    summary_path = (
        Path(summary_out)
        if summary_out is not None
        else out_path.with_name("enrichment_summary.json")
    )
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
