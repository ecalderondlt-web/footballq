"""Held-out discovery control summaries."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import torch

from footballq.discovery.transitions import TransitionDatasetData


def match_concentration_by_cluster(
    assignments: torch.Tensor,
    match_ids: list[str],
) -> dict[int, dict[str, Any]]:
    """Report whether clusters are dominated by one match."""

    out: dict[int, dict[str, Any]] = {}
    for cluster_id in sorted(set(int(value) for value in assignments.tolist())):
        indices = [
            idx for idx, value in enumerate(assignments.tolist()) if int(value) == cluster_id
        ]
        counts = Counter(str(match_ids[idx]) for idx in indices)
        total = sum(counts.values())
        top_match, top_count = counts.most_common(1)[0]
        out[cluster_id] = {
            "cluster_size": total,
            "top_match_id": top_match,
            "top_match_fraction": top_count / max(1, total),
            "match_id_counts": dict(sorted(counts.items())),
        }
    return out


def split_counts_by_cluster(
    assignments: torch.Tensor,
    source_split: list[str],
) -> dict[int, dict[str, int]]:
    """Count train/val/test assignments for each cluster."""

    rows: dict[int, Counter[str]] = defaultdict(Counter)
    for assignment, split in zip(assignments.tolist(), source_split, strict=True):
        rows[int(assignment)][str(split)] += 1
    return {cluster_id: dict(counter) for cluster_id, counter in sorted(rows.items())}


def discovery_control_summary(
    data: TransitionDatasetData,
    assignments: torch.Tensor,
) -> dict[str, Any]:
    """Return held-out assignment diagnostics for a transition dataset."""

    match_ids = [str(value) for value in data.examples["match_id"]]
    source_split = [
        str(value) for value in data.examples.get("source_split", ["unknown"] * len(match_ids))
    ]
    return {
        "match_concentration": match_concentration_by_cluster(assignments, match_ids),
        "split_counts": split_counts_by_cluster(assignments, source_split),
        "assignment_protocol": "fit_train_assign_heldout_when_available",
    }
