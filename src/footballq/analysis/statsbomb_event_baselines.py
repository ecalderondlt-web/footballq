"""Train-only raw baselines for StatsBomb causal event prediction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from footballq.data.statsbomb_events import file_sha256


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _event_transition_weights(shard: dict[str, Any]) -> torch.Tensor:
    event_count = int(shard["categorical"].shape[0])
    sequence_length = int(shard["sequence_length"])
    difference = torch.zeros(event_count + 1, dtype=torch.int64)
    starts = shard["window_starts"]
    difference.scatter_add_(0, starts, torch.ones_like(starts))
    ends = starts + sequence_length
    difference.scatter_add_(0, ends, -torch.ones_like(ends))
    return difference.cumsum(dim=0)[:event_count]


def _iter_shards(manifest_path: Path, manifest: dict[str, Any], split: str):
    for row in manifest["shards"]:
        if row["split"] == split:
            yield torch.load(
                manifest_path.parent / row["path"],
                map_location="cpu",
                weights_only=False,
            )


def compute_statsbomb_event_baselines(
    manifest_path: str | Path,
    *,
    laplace_alpha: float = 1.0,
) -> dict[str, Any]:
    """Fit frequency/Markov controls on train and score causal validation windows."""

    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("loaded_splits") != ["train", "val"] or manifest.get("test_loaded"):
        raise ValueError("StatsBomb baselines require a train/val-only event manifest.")
    event_type_size = int(manifest["categorical_vocabularies"]["event_type"]["size"])
    frequency_counts = torch.full((event_type_size,), float(laplace_alpha), dtype=torch.float64)
    transition_counts = torch.full(
        (event_type_size, event_type_size),
        float(laplace_alpha),
        dtype=torch.float64,
    )
    train_weight = 0
    for shard in _iter_shards(path, manifest, "train"):
        weights = _event_transition_weights(shard)[:-1]
        current = shard["categorical"][:-1, 0]
        target = shard["categorical"][1:, 0]
        valid = weights > 0
        weights_valid = weights[valid].to(torch.float64)
        frequency_counts.scatter_add_(0, target[valid], weights_valid)
        flat = current[valid] * event_type_size + target[valid]
        transition_counts.view(-1).scatter_add_(0, flat, weights_valid)
        train_weight += int(weights_valid.sum())
    frequency_log_prob = (frequency_counts / frequency_counts.sum()).log()
    transition_log_prob = (
        transition_counts / transition_counts.sum(dim=1, keepdim=True)
    ).log()

    totals = {
        "target_weight": 0.0,
        "anchored_target_weight": 0.0,
        "frequency_nll": 0.0,
        "markov_nll": 0.0,
        "anchored_frequency_nll": 0.0,
        "anchored_markov_nll": 0.0,
        "location_weight": 0.0,
        "location_absolute_error": 0.0,
        "anchored_location_weight": 0.0,
        "anchored_location_absolute_error": 0.0,
    }
    for shard in _iter_shards(path, manifest, "val"):
        weights = _event_transition_weights(shard)[:-1].to(torch.float64)
        current = shard["categorical"][:-1, 0]
        target = shard["categorical"][1:, 0]
        valid = weights > 0
        anchored = valid & (shard["event_to_freeze"][:-1] >= 0)
        totals["target_weight"] += float(weights[valid].sum())
        totals["anchored_target_weight"] += float(weights[anchored].sum())
        totals["frequency_nll"] += float(
            (-frequency_log_prob[target[valid]] * weights[valid]).sum()
        )
        totals["markov_nll"] += float(
            (-transition_log_prob[current[valid], target[valid]] * weights[valid]).sum()
        )
        totals["anchored_frequency_nll"] += float(
            (-frequency_log_prob[target[anchored]] * weights[anchored]).sum()
        )
        totals["anchored_markov_nll"] += float(
            (-transition_log_prob[current[anchored], target[anchored]] * weights[anchored]).sum()
        )

        continuous = shard["continuous"]
        location_valid = (
            valid
            & (continuous[:-1, 4] > 0.5)
            & (continuous[:-1, 5] > 0.5)
            & (continuous[1:, 4] > 0.5)
            & (continuous[1:, 5] > 0.5)
        )
        location_error = (continuous[:-1, :2] - continuous[1:, :2]).abs().mean(dim=-1)
        location_weights = weights[location_valid]
        totals["location_weight"] += float(location_weights.sum())
        totals["location_absolute_error"] += float(
            (location_error[location_valid].to(torch.float64) * location_weights).sum()
        )
        anchored_location = location_valid & anchored
        anchored_location_weights = weights[anchored_location]
        totals["anchored_location_weight"] += float(anchored_location_weights.sum())
        totals["anchored_location_absolute_error"] += float(
            (
                location_error[anchored_location].to(torch.float64)
                * anchored_location_weights
            ).sum()
        )

    target_weight = max(totals["target_weight"], 1.0)
    anchored_target_weight = max(totals["anchored_target_weight"], 1.0)
    location_weight = max(totals["location_weight"], 1.0)
    anchored_location_weight = max(totals["anchored_location_weight"], 1.0)
    report = {
        "version": 1,
        "dataset": "statsbomb_open_data",
        "scope": "train_fit_validation_score",
        "loaded_splits": ["train", "val"],
        "test_loaded": False,
        "manifest_path": str(path),
        "manifest_sha256": file_sha256(path),
        "manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "laplace_alpha": float(laplace_alpha),
        "train_transition_weight": train_weight,
        "validation_target_weight": totals["target_weight"],
        "anchored_validation_target_weight": totals["anchored_target_weight"],
        "global_frequency_event_type_nll": totals["frequency_nll"] / target_weight,
        "first_order_markov_event_type_nll": totals["markov_nll"] / target_weight,
        "anchored_global_frequency_event_type_nll": (
            totals["anchored_frequency_nll"] / anchored_target_weight
        ),
        "anchored_first_order_markov_event_type_nll": (
            totals["anchored_markov_nll"] / anchored_target_weight
        ),
        "copy_current_location_mae": totals["location_absolute_error"] / location_weight,
        "anchored_copy_current_location_mae": (
            totals["anchored_location_absolute_error"] / anchored_location_weight
        ),
        "validation_location_weight": totals["location_weight"],
        "anchored_validation_location_weight": totals["anchored_location_weight"],
    }
    report["report_payload_sha256"] = _stable_hash(report)
    return report
