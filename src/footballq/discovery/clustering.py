"""Deterministic k-means clustering for latent transitions."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from footballq.discovery.transitions import TransitionDatasetData, load_transition_dataset


def _mask_for_delta(data: TransitionDatasetData, delta_seconds: float | None) -> torch.Tensor:
    n = data.num_examples
    if delta_seconds is None:
        return torch.ones(n, dtype=torch.bool)
    values = torch.as_tensor(data.examples["delta_seconds"]).float()
    return torch.isclose(values, torch.tensor(float(delta_seconds)), atol=1e-6)


def transition_feature_matrix(
    data: TransitionDatasetData,
    feature: str = "normalized_delta_z",
    delta_seconds: float | None = None,
) -> tuple[torch.Tensor, list[int]]:
    mask = _mask_for_delta(data, delta_seconds)
    indices = mask.nonzero(as_tuple=False).flatten().tolist()
    if not indices:
        raise ValueError(f"No transitions found for delta_seconds={delta_seconds}")
    idx = torch.tensor(indices, dtype=torch.long)
    if feature == "raw_delta_z":
        x = torch.as_tensor(data.examples["delta_z"]).float()[idx]
    elif feature == "normalized_delta_z":
        x = torch.as_tensor(data.features["normalized_delta_z"]).float()[idx]
    elif feature == "z_t_delta_z":
        x = torch.cat(
            [
                torch.as_tensor(data.examples["z_t"]).float()[idx],
                torch.as_tensor(data.features["normalized_delta_z"]).float()[idx],
            ],
            dim=1,
        )
    else:
        raise ValueError(
            f"Unknown transition feature {feature!r}. Expected raw_delta_z, "
            "normalized_delta_z, or z_t_delta_z."
        )
    if not bool(torch.isfinite(x).all()):
        raise ValueError(f"Transition feature matrix {feature!r} contains non-finite values.")
    return x, indices


def _squared_distances(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    return torch.cdist(x.float(), centroids.float(), p=2).pow(2)


def _assign(
    x: torch.Tensor, centroids: torch.Tensor, chunk_size: int = 8192
) -> tuple[torch.Tensor, torch.Tensor]:
    assignments = []
    distances = []
    for start in range(0, x.shape[0], chunk_size):
        chunk = x[start : start + chunk_size]
        dist = _squared_distances(chunk, centroids)
        values, labels = dist.min(dim=1)
        assignments.append(labels.cpu())
        distances.append(values.cpu())
    return torch.cat(assignments), torch.cat(distances)


def kmeans(
    x: torch.Tensor,
    k: int,
    seed: int = 123,
    max_iter: int = 30,
    fit_sample_size: int = 50000,
    chunk_size: int = 8192,
) -> dict[str, Any]:
    """Fit k-means centroids on a deterministic sample, then assign all rows."""

    if k <= 0:
        raise ValueError("k must be positive.")
    if x.shape[0] < k:
        raise ValueError(f"Cannot fit k={k} clusters with only {x.shape[0]} examples.")
    generator = torch.Generator().manual_seed(int(seed))
    if x.shape[0] > fit_sample_size:
        fit_idx = torch.randperm(x.shape[0], generator=generator)[:fit_sample_size]
        fit_x = x[fit_idx].contiguous()
    else:
        fit_idx = torch.arange(x.shape[0])
        fit_x = x.contiguous()

    init_idx = torch.randperm(fit_x.shape[0], generator=generator)[:k]
    centroids = fit_x[init_idx].clone()
    for _ in range(int(max_iter)):
        labels, _ = _assign(fit_x, centroids, chunk_size=chunk_size)
        new_centroids = torch.zeros_like(centroids)
        counts = torch.bincount(labels, minlength=k).float()
        for cluster_id in range(k):
            selected = labels == cluster_id
            if bool(selected.any()):
                new_centroids[cluster_id] = fit_x[selected].mean(dim=0)
            else:
                repl = torch.randint(0, fit_x.shape[0], (1,), generator=generator).item()
                new_centroids[cluster_id] = fit_x[repl]
        shift = torch.linalg.vector_norm(centroids - new_centroids, dim=1).max().item()
        centroids = new_centroids
        if shift < 1e-5:
            break

    assignments, distances_sq = _assign(x, centroids, chunk_size=chunk_size)
    distances = torch.sqrt(distances_sq.clamp_min(0.0))
    counts = torch.bincount(assignments, minlength=k)
    return {
        "centroids": centroids.cpu(),
        "assignments": assignments.cpu(),
        "distances": distances.cpu(),
        "distances_sq": distances_sq.cpu(),
        "fit_indices": fit_idx.cpu(),
        "cluster_counts": counts.cpu(),
    }


def _entropy(counts: torch.Tensor) -> float:
    probs = counts.float() / counts.sum().clamp_min(1)
    selected = probs[probs > 0]
    if selected.numel() == 0:
        return 0.0
    return float((-(selected * torch.log(selected)).sum() / math.log(max(len(counts), 2))).item())


def _centroid_margin_proxy(
    x: torch.Tensor,
    labels: torch.Tensor,
    centroids: torch.Tensor,
    distances: torch.Tensor,
    seed: int = 123,
) -> float:
    if x.shape[0] == 0 or centroids.shape[0] <= 1:
        return float("nan")
    generator = torch.Generator().manual_seed(int(seed))
    sample_size = min(5000, x.shape[0])
    sample_idx = torch.randperm(x.shape[0], generator=generator)[:sample_size]
    dist = torch.cdist(x[sample_idx].float(), centroids.float(), p=2)
    own = distances[sample_idx].float()
    row_labels = labels[sample_idx]
    dist[torch.arange(sample_size), row_labels] = float("inf")
    nearest_other = dist.min(dim=1).values
    denom = torch.maximum(own, nearest_other).clamp_min(1e-6)
    score = (nearest_other - own) / denom
    return float(score[torch.isfinite(score)].mean().item())


def cluster_quality(
    x: torch.Tensor,
    result: dict[str, Any],
    seed: int = 123,
) -> dict[str, Any]:
    centroids = torch.as_tensor(result["centroids"]).float()
    assignments = torch.as_tensor(result["assignments"]).long()
    distances = torch.as_tensor(result["distances"]).float()
    distances_sq = torch.as_tensor(result["distances_sq"]).float()
    counts = torch.as_tensor(result["cluster_counts"]).long()
    if centroids.shape[0] > 1:
        centroid_dist = torch.cdist(centroids, centroids, p=2)
        centroid_dist = centroid_dist.masked_fill(
            torch.eye(centroids.shape[0], dtype=torch.bool),
            float("inf"),
        )
        min_sep = float(centroid_dist.min().item())
        mean_sep = float(centroid_dist[torch.isfinite(centroid_dist)].mean().item())
    else:
        min_sep = float("nan")
        mean_sep = float("nan")
    return {
        "k": int(centroids.shape[0]),
        "num_examples": int(x.shape[0]),
        "inertia": float(distances_sq.sum().item()),
        "average_squared_distance": float(distances_sq.mean().item()),
        "average_within_cluster_distance": float(distances.mean().item()),
        "centroid_separation_min": min_sep,
        "centroid_separation_mean": mean_sep,
        "centroid_margin_proxy": _centroid_margin_proxy(
            x,
            assignments,
            centroids,
            distances,
            seed=seed,
        ),
        "silhouette_proxy": _centroid_margin_proxy(
            x,
            assignments,
            centroids,
            distances,
            seed=seed,
        ),
        "cluster_size_entropy": _entropy(counts),
        "min_cluster_size": int(counts.min().item()),
        "max_cluster_size": int(counts.max().item()),
        "empty_cluster_count": int((counts == 0).sum().item()),
    }


def _value_at(values: Any, row: int) -> Any:
    if isinstance(values, torch.Tensor):
        item = values[row]
        if item.numel() == 1:
            return item.item()
        return item.tolist()
    return values[row]


def _cluster_rows(
    data: TransitionDatasetData,
    global_indices: list[int],
    result: dict[str, Any],
    quality: dict[str, Any],
) -> list[dict[str, Any]]:
    assignments = torch.as_tensor(result["assignments"]).long()
    distances = torch.as_tensor(result["distances"]).float()
    centroids = torch.as_tensor(result["centroids"]).float()
    counts = torch.bincount(assignments, minlength=centroids.shape[0])
    delta_norm = torch.as_tensor(data.features["delta_norm"]).float()
    meta = data.examples.get("metadata", {})
    source_split = data.examples.get("source_split", ["unknown"] * data.num_examples)
    match_ids = data.examples.get("match_id", [])
    rows = []
    for cluster_id in range(centroids.shape[0]):
        local = (assignments == cluster_id).nonzero(as_tuple=False).flatten().tolist()
        global_rows = [global_indices[idx] for idx in local]
        cluster_distances = distances[local] if local else torch.empty(0)
        cluster_delta_norm = delta_norm[global_rows] if global_rows else torch.empty(0)
        split_counts = Counter(str(source_split[idx]) for idx in global_rows)
        match_counts = Counter(str(match_ids[idx]) for idx in global_rows)
        row = {
            "cluster_id": cluster_id,
            "n_examples": int(counts[cluster_id].item()),
            "fraction": float(counts[cluster_id].item() / max(1, len(global_indices))),
            "mean_delta_norm": float(cluster_delta_norm.mean().item()) if local else float("nan"),
            "median_delta_norm": float(cluster_delta_norm.median().item())
            if local
            else float("nan"),
            "max_delta_norm": float(cluster_delta_norm.max().item()) if local else float("nan"),
            "delta_norm_top_fraction": (
                float(
                    (cluster_delta_norm >= torch.quantile(delta_norm, 0.95)).float().mean().item()
                )
                if local and int(delta_norm.numel()) >= 1
                else float("nan")
            ),
            "centroid_norm": float(torch.linalg.vector_norm(centroids[cluster_id]).item()),
            "within_cluster_distance_mean": (
                float(cluster_distances.mean().item()) if local else float("nan")
            ),
            "within_cluster_distance_std": (
                float(cluster_distances.std().item()) if len(local) > 1 else 0.0
            ),
            "train_count": int(split_counts.get("train", 0)),
            "val_count": int(split_counts.get("val", 0)),
            "test_count": int(split_counts.get("test", 0)),
            "match_id_counts": json.dumps(dict(sorted(match_counts.items()))),
        }
        for field in ["future_ball_displacement_m", "team_shape_change_m"]:
            values = meta.get(field)
            if values is not None and global_rows:
                tensor = torch.tensor([float(_value_at(values, idx)) for idx in global_rows])
                finite = tensor[torch.isfinite(tensor)]
                row[f"mean_{field}"] = (
                    float(finite.mean().item()) if finite.numel() else float("nan")
                )
            else:
                row[f"mean_{field}"] = float("nan")
        rows.append(row)
    return rows


def write_cluster_outputs(
    data: TransitionDatasetData,
    x: torch.Tensor,
    global_indices: list[int],
    result: dict[str, Any],
    out_dir: Path,
    k: int,
    feature: str,
    delta_seconds: float | None,
    seed: int = 123,
    assignment_protocol: str = "fit_all_assign_all",
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    quality = cluster_quality(x, result, seed=seed)
    rows = _cluster_rows(data, global_indices, result, quality)
    clusters_path = out_dir / f"clusters_k{k}.csv"
    with clusters_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    assignments_path = out_dir / f"assignments_k{k}.pt"
    torch.save(
        {
            "k": int(k),
            "feature": feature,
            "delta_seconds": delta_seconds,
            "global_indices": global_indices,
            "assignments": result["assignments"],
            "distances": result["distances"],
            "centroids": result["centroids"],
            "quality": quality,
            "assignment_protocol": assignment_protocol,
        },
        assignments_path,
    )
    return {
        "k": int(k),
        "clusters_csv": str(clusters_path),
        "assignments": str(assignments_path),
        "quality": quality,
        "assignment_protocol": assignment_protocol,
    }


def _fit_train_assign_all(
    data: TransitionDatasetData,
    x: torch.Tensor,
    global_indices: list[int],
    k: int,
    seed: int,
    max_iter: int,
    fit_sample_size: int,
) -> tuple[dict[str, Any], str]:
    source_split = data.examples.get("source_split", ["unknown"] * data.num_examples)
    train_local = [
        local_idx
        for local_idx, global_idx in enumerate(global_indices)
        if str(source_split[global_idx]).lower() == "train"
    ]
    if len(train_local) < int(k):
        if data.metadata.get("scientific_mode"):
            raise ValueError(
                "Scientific discovery clustering requires at least k train examples for "
                "train-fit/held-out assignment."
            )
        return (
            kmeans(
                x,
                int(k),
                seed=seed,
                max_iter=max_iter,
                fit_sample_size=fit_sample_size,
            ),
            "fit_all_assign_all_no_train_split",
        )
    train_idx = torch.tensor(train_local, dtype=torch.long)
    train_result = kmeans(
        x[train_idx],
        int(k),
        seed=seed,
        max_iter=max_iter,
        fit_sample_size=fit_sample_size,
    )
    centroids = torch.as_tensor(train_result["centroids"]).float()
    assignments, distances_sq = _assign(x, centroids)
    distances = torch.sqrt(distances_sq.clamp_min(0.0))
    return (
        {
            "centroids": centroids.cpu(),
            "assignments": assignments.cpu(),
            "distances": distances.cpu(),
            "distances_sq": distances_sq.cpu(),
            "fit_indices": train_idx.cpu(),
            "cluster_counts": torch.bincount(assignments, minlength=int(k)).cpu(),
        },
        "fit_train_assign_all",
    )


def cluster_transition_file(
    dataset: str | Path,
    out: str | Path,
    k_values: list[int],
    delta_seconds: float | None = None,
    feature: str = "normalized_delta_z",
    seed: int = 123,
    max_iter: int = 30,
    fit_sample_size: int = 50000,
) -> dict[str, Any]:
    data = load_transition_dataset(dataset)
    x, global_indices = transition_feature_matrix(
        data, feature=feature, delta_seconds=delta_seconds
    )
    out_dir = Path(out)
    outputs = []
    for k in k_values:
        result, assignment_protocol = _fit_train_assign_all(
            data,
            x,
            global_indices,
            int(k),
            seed,
            max_iter,
            fit_sample_size,
        )
        outputs.append(
            write_cluster_outputs(
                data,
                x,
                global_indices,
                result,
                out_dir,
                int(k),
                feature=feature,
                delta_seconds=delta_seconds,
                seed=seed,
                assignment_protocol=assignment_protocol,
            )
        )
    summary = {
        "dataset": str(dataset),
        "out": str(out_dir),
        "feature": feature,
        "delta_seconds": delta_seconds,
        "num_examples": int(x.shape[0]),
        "k_values": [int(value) for value in k_values],
        "clusters": outputs,
        "assignment_default": "fit_train_assign_all_when_train_split_available",
        "seed": int(seed),
        "seed_stability": {
            "status": "single_seed_result",
            "required_for_validation": True,
        },
        "scientific_mode": bool(data.metadata.get("scientific_mode", False)),
        "split_manifest_sha256": data.metadata.get("split_manifest_sha256"),
    }
    with (out_dir / "cluster_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
