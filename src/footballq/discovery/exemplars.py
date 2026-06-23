"""Representative latent transition exemplars for cluster inspection."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from footballq.discovery.surprise import compute_surprise
from footballq.discovery.transitions import STRESS_FIELDS, TransitionDatasetData, load_transition_dataset


def _value(values: Any, idx: int) -> Any:
    if isinstance(values, torch.Tensor):
        item = values[idx]
        return item.item() if item.numel() == 1 else item.tolist()
    return values[idx]


def export_exemplars(
    data: TransitionDatasetData,
    assignments_payload: dict[str, Any],
    seed: int = 123,
) -> list[dict[str, Any]]:
    global_indices = [int(value) for value in assignments_payload["global_indices"]]
    assignments = [int(value) for value in assignments_payload["assignments"].tolist()]
    distances = [float(value) for value in assignments_payload["distances"].tolist()]
    surprise = compute_surprise(data)
    surprise_score = surprise["surprise_cv"]
    if not bool(torch.isfinite(surprise_score).any()):
        surprise_score = surprise["surprise_last"]
    meta = data.examples.get("metadata", {})
    ball_disp = meta.get("future_ball_displacement_m")
    by_cluster: dict[int, list[int]] = defaultdict(list)
    for local_idx, cluster_id in enumerate(assignments):
        by_cluster[int(cluster_id)].append(local_idx)
    rng = random.Random(int(seed))
    rows = []

    def add_row(cluster_id: int, exemplar_type: str, local_idx: int) -> None:
        global_idx = global_indices[local_idx]
        row = {
            "cluster_id": cluster_id,
            "exemplar_type": exemplar_type,
            "match_id": data.examples["match_id"][global_idx],
            "period": data.examples["period"][global_idx],
            "frame_t": int(data.examples["frame_t"][global_idx].item()),
            "frame_next": int(data.examples["frame_next"][global_idx].item()),
            "delta_seconds": float(data.examples["delta_seconds"][global_idx].item()),
            "actual_delta_seconds": float(data.examples["actual_delta_seconds"][global_idx].item()),
            "distance_to_centroid": distances[local_idx],
            "surprise_score": float(surprise_score[global_idx].item()),
        }
        for field in [
            "future_ball_displacement_m",
            "team_shape_change_m",
            "event_type",
            "phase",
            *STRESS_FIELDS,
        ]:
            values = meta.get(field)
            row[field] = "" if values is None else _value(values, global_idx)
        rows.append(row)

    for cluster_id in sorted(by_cluster):
        local_indices = by_cluster[cluster_id]
        centroid_idx = min(local_indices, key=lambda idx: distances[idx])
        add_row(cluster_id, "centroid", centroid_idx)
        high_surprise_idx = max(
            local_indices,
            key=lambda idx: float(surprise_score[global_indices[idx]].item())
            if torch.isfinite(surprise_score[global_indices[idx]])
            else -float("inf"),
        )
        add_row(cluster_id, "high_surprise", high_surprise_idx)
        if ball_disp is not None:
            high_ball_idx = max(
                local_indices,
                key=lambda idx: float(_value(ball_disp, global_indices[idx]))
                if torch.isfinite(torch.tensor(float(_value(ball_disp, global_indices[idx]))))
                else -float("inf"),
            )
            add_row(cluster_id, "high_future_ball_displacement", high_ball_idx)
        add_row(cluster_id, "random", rng.choice(local_indices))
    return rows


def write_exemplars(
    dataset: str | Path,
    assignments: str | Path,
    out_csv: str | Path,
    seed: int = 123,
) -> Path:
    data = load_transition_dataset(dataset)
    payload = torch.load(Path(assignments), map_location="cpu", weights_only=False)
    rows = export_exemplars(data, payload, seed=seed)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out_path
