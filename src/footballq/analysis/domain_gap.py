"""Train-only kinematic and geometric domain-gap summaries."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from footballq.data.normalize import XY_SCALE_M
from footballq.data.sharded_td_dataset import temperature_shard_allocations
from footballq.data.td_jepa_dataset import TDJEPAData, load_td_jepa_data

METRIC_UNITS = {
    "visible_player_count": "players",
    "ball_visible_indicator": "rate",
    "player_speed_mps": "m/s",
    "player_stationary_indicator": "rate",
    "player_high_speed_indicator": "rate",
    "player_acceleration_mps2": "m/s^2",
    "player_high_acceleration_indicator": "rate",
    "player_turn_deg": "degrees",
    "ball_speed_mps": "m/s",
    "ball_high_speed_indicator": "rate",
    "ball_acceleration_mps2": "m/s^2",
    "ball_turn_deg": "degrees",
    "nearest_player_distance_m": "m",
    "player_ball_distance_m": "m",
    "visible_team_x_span_m": "m",
    "visible_team_y_span_m": "m",
    "visible_team_centroid_distance_m": "m",
}


def _stable_seed(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def deterministic_indices(length: int, count: int, *, key: str, seed: int) -> np.ndarray:
    """Select deterministic, unique tensor rows without reading outcome values."""

    length = int(length)
    count = min(int(count), length)
    if length <= 0 or count <= 0:
        return np.empty(0, dtype=np.int64)
    if count == length:
        return np.arange(length, dtype=np.int64)
    rng = np.random.default_rng(_stable_seed(key, seed))
    return np.sort(rng.choice(length, size=count, replace=False).astype(np.int64))


def select_train_shards_by_match(
    shards: list[dict[str, Any]],
    *,
    max_shards_per_match: int,
) -> list[dict[str, Any]]:
    """Select evenly spaced train shards for every match, ignoring non-train entries."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for shard in shards:
        if shard.get("split") != "train":
            continue
        match_id = str(shard.get("match_id") or shard.get("match_ids", ["unknown"])[0])
        grouped[match_id].append(shard)
    selected = []
    for match_id in sorted(grouped):
        match_shards = grouped[match_id]
        count = min(max(1, int(max_shards_per_match)), len(match_shards))
        indices = np.linspace(0, len(match_shards) - 1, num=count, dtype=int)
        selected.extend(match_shards[int(index)] for index in np.unique(indices))
    return selected


def _feature_indices(feature_names: list[str]) -> tuple[int, int, int, int]:
    required = ("x_norm", "y_norm", "vx_norm", "vy_norm")
    missing = [name for name in required if name not in feature_names]
    if missing:
        raise ValueError(f"Domain-gap audit requires geometry features: {missing}")
    return tuple(feature_names.index(name) for name in required)  # type: ignore[return-value]


def _masked_values(values: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    return values[mask].detach().cpu().numpy().astype(np.float64, copy=False)


def extract_geometry_metrics(
    state_t: torch.Tensor,
    mask_t: torch.Tensor,
    *,
    fps: float,
    feature_names: list[str],
) -> dict[str, np.ndarray]:
    """Extract observable endpoint kinematics and geometry from context tensors."""

    if state_t.ndim != 4 or mask_t.shape != state_t.shape[:3]:
        raise ValueError("Expected state [N,T,E,F] and aligned mask [N,T,E].")
    if state_t.shape[1] < 2 or state_t.shape[2] != 23:
        raise ValueError("Domain-gap audit requires at least two frames and 23 entity slots.")
    x_idx, y_idx, vx_idx, vy_idx = _feature_indices(feature_names)
    scale = torch.as_tensor(XY_SCALE_M, dtype=state_t.dtype, device=state_t.device)
    xy_m = state_t[..., [x_idx, y_idx]] * scale + scale
    velocity_mps = state_t[..., [vx_idx, vy_idx]] * scale
    endpoint_xy = xy_m[:, -1]
    endpoint_velocity = velocity_mps[:, -1]
    previous_velocity = velocity_mps[:, -2]
    endpoint_mask = mask_t[:, -1].bool()
    previous_mask = mask_t[:, -2].bool()
    player_mask = endpoint_mask[:, 1:]
    transition_player_mask = player_mask & previous_mask[:, 1:]

    player_speed = torch.linalg.vector_norm(endpoint_velocity[:, 1:], dim=-1)
    previous_player_speed = torch.linalg.vector_norm(previous_velocity[:, 1:], dim=-1)
    player_acceleration = torch.linalg.vector_norm(
        endpoint_velocity[:, 1:] - previous_velocity[:, 1:], dim=-1
    ) * float(fps)
    moving_player_mask = transition_player_mask & (player_speed >= 0.5) & (
        previous_player_speed >= 0.5
    )
    player_dot = (endpoint_velocity[:, 1:] * previous_velocity[:, 1:]).sum(dim=-1)
    player_cosine = player_dot / (player_speed * previous_player_speed).clamp_min(1e-8)
    player_turn = torch.rad2deg(torch.acos(player_cosine.clamp(-1.0, 1.0)))

    ball_mask = endpoint_mask[:, 0]
    transition_ball_mask = ball_mask & previous_mask[:, 0]
    ball_speed = torch.linalg.vector_norm(endpoint_velocity[:, 0], dim=-1)
    previous_ball_speed = torch.linalg.vector_norm(previous_velocity[:, 0], dim=-1)
    ball_acceleration = torch.linalg.vector_norm(
        endpoint_velocity[:, 0] - previous_velocity[:, 0], dim=-1
    ) * float(fps)
    moving_ball_mask = transition_ball_mask & (ball_speed >= 0.5) & (previous_ball_speed >= 0.5)
    ball_dot = (endpoint_velocity[:, 0] * previous_velocity[:, 0]).sum(dim=-1)
    ball_cosine = ball_dot / (ball_speed * previous_ball_speed).clamp_min(1e-8)
    ball_turn = torch.rad2deg(torch.acos(ball_cosine.clamp(-1.0, 1.0)))

    player_xy = endpoint_xy[:, 1:]
    pairwise = torch.cdist(player_xy, player_xy)
    valid_pairs = player_mask.unsqueeze(2) & player_mask.unsqueeze(1)
    diagonal = torch.eye(22, dtype=torch.bool, device=state_t.device).unsqueeze(0)
    pairwise = pairwise.masked_fill(~valid_pairs | diagonal, float("inf"))
    nearest = pairwise.min(dim=-1).values
    nearest_mask = player_mask & torch.isfinite(nearest)

    player_ball_distance = torch.linalg.vector_norm(
        player_xy - endpoint_xy[:, 0].unsqueeze(1), dim=-1
    )
    player_ball_mask = player_mask & ball_mask.unsqueeze(1)

    team_x_spans = []
    team_y_spans = []
    centroid_distances = []
    for row in range(state_t.shape[0]):
        centroids = []
        for start, end in ((0, 11), (11, 22)):
            visible = player_mask[row, start:end]
            points = player_xy[row, start:end][visible]
            if len(points) >= 2:
                spans = points.max(dim=0).values - points.min(dim=0).values
                team_x_spans.append(float(spans[0]))
                team_y_spans.append(float(spans[1]))
            if len(points) >= 1:
                centroids.append(points.mean(dim=0))
        if len(centroids) == 2:
            centroid_distances.append(float(torch.linalg.vector_norm(centroids[0] - centroids[1])))

    visible_player_speed = _masked_values(player_speed, player_mask)
    visible_player_acceleration = _masked_values(player_acceleration, transition_player_mask)
    return {
        "visible_player_count": player_mask.sum(dim=1).cpu().numpy().astype(np.float64),
        "ball_visible_indicator": ball_mask.cpu().numpy().astype(np.float64),
        "player_speed_mps": visible_player_speed,
        "player_stationary_indicator": (visible_player_speed < 0.5).astype(np.float64),
        "player_high_speed_indicator": (visible_player_speed > 7.0).astype(np.float64),
        "player_acceleration_mps2": visible_player_acceleration,
        "player_high_acceleration_indicator": (
            visible_player_acceleration > 5.0
        ).astype(np.float64),
        "player_turn_deg": _masked_values(player_turn, moving_player_mask),
        "ball_speed_mps": _masked_values(ball_speed, ball_mask),
        "ball_high_speed_indicator": (_masked_values(ball_speed, ball_mask) > 20.0).astype(
            np.float64
        ),
        "ball_acceleration_mps2": _masked_values(ball_acceleration, transition_ball_mask),
        "ball_turn_deg": _masked_values(ball_turn, moving_ball_mask),
        "nearest_player_distance_m": _masked_values(nearest, nearest_mask),
        "player_ball_distance_m": _masked_values(player_ball_distance, player_ball_mask),
        "visible_team_x_span_m": np.asarray(team_x_spans, dtype=np.float64),
        "visible_team_y_span_m": np.asarray(team_y_spans, dtype=np.float64),
        "visible_team_centroid_distance_m": np.asarray(centroid_distances, dtype=np.float64),
    }


def _merge_metric_parts(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        return {name: np.empty(0, dtype=np.float64) for name in METRIC_UNITS}
    return {
        name: np.concatenate([part[name] for part in parts if len(part[name])])
        if any(len(part[name]) for part in parts)
        else np.empty(0, dtype=np.float64)
        for name in METRIC_UNITS
    }


def summarize_metric_values(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"count": 0}
    quantiles = np.quantile(finite, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "count": int(len(finite)),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "p01": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "p25": float(quantiles[2]),
        "p50": float(quantiles[3]),
        "p75": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "p99": float(quantiles[6]),
    }


def compare_metric_samples(
    real_metrics: dict[str, np.ndarray],
    synthetic_metrics: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Rank empirical distribution gaps by quantile-Wasserstein distance over pooled IQR."""

    rows = []
    quantile_grid = np.linspace(0.01, 0.99, 99)
    for name in METRIC_UNITS:
        real = real_metrics[name][np.isfinite(real_metrics[name])]
        synthetic = synthetic_metrics[name][np.isfinite(synthetic_metrics[name])]
        if not len(real) or not len(synthetic):
            continue
        real_quantiles = np.quantile(real, quantile_grid)
        synthetic_quantiles = np.quantile(synthetic, quantile_grid)
        wasserstein = float(np.mean(np.abs(real_quantiles - synthetic_quantiles)))
        real_iqr = float(np.quantile(real, 0.75) - np.quantile(real, 0.25))
        synthetic_iqr = float(np.quantile(synthetic, 0.75) - np.quantile(synthetic, 0.25))
        real_mad = float(np.median(np.abs(real - np.median(real))))
        synthetic_mad = float(np.median(np.abs(synthetic - np.median(synthetic))))
        if METRIC_UNITS[name] == "rate":
            pooled_scale = 1.0
            scale_method = "fixed_probability_range"
        else:
            pooled_scale = (real_iqr + synthetic_iqr) / 2.0
            if pooled_scale <= 1e-8:
                pooled_scale = (real_mad + synthetic_mad) * 1.4826 / 2.0
            scale_method = "pooled_iqr_with_mad_fallback"
        score = 0.0 if wasserstein == 0.0 else wasserstein / max(pooled_scale, 1e-8)
        rows.append(
            {
                "metric": name,
                "unit": METRIC_UNITS[name],
                "gap_score": float(score),
                "quantile_wasserstein": wasserstein,
                "gap_scale": float(pooled_scale),
                "gap_scale_method": scale_method,
                "real": summarize_metric_values(real),
                "synthetic": summarize_metric_values(synthetic),
                "mean_difference_synthetic_minus_real": float(synthetic.mean() - real.mean()),
            }
        )
    return sorted(rows, key=lambda row: (-row["gap_score"], row["metric"]))


def _validate_manifest_pair(real: dict[str, Any], synthetic: dict[str, Any]) -> None:
    keys = (
        "fps_out",
        "context_seconds",
        "delta_seconds",
        "stride_seconds",
        "objective_mode",
        "prediction_gap_seconds",
        "feature_view",
    )
    mismatches = [key for key in keys if real["config"].get(key) != synthetic["config"].get(key)]
    if mismatches:
        raise ValueError(f"Domain-gap manifests have incompatible configs: {mismatches}")


def _manifest_root(path: Path) -> Path:
    return path.parent.parent


def _sample_entries(
    manifest_path: Path,
    entries: list[dict[str, Any]],
    allocations: list[int],
    *,
    seed: int,
    scenario_cap: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    global_parts = []
    scenario_parts: dict[str, list[dict[str, np.ndarray]]] = defaultdict(list)
    sampled_shards = []
    for entry, allocation in zip(entries, allocations, strict=True):
        path = _manifest_root(manifest_path) / entry["path"]
        data: TDJEPAData = load_td_jepa_data(path)
        indices = deterministic_indices(
            len(data.match_id), allocation, key=str(entry["path"]), seed=seed
        )
        index_tensor = torch.from_numpy(indices)
        global_parts.append(
            extract_geometry_metrics(
                data.state_t[index_tensor],
                data.mask_t[index_tensor],
                fps=data.fps,
                feature_names=data.feature_names,
            )
        )
        scenario = entry.get("scenario")
        if scenario and scenario_cap:
            scenario_indices = deterministic_indices(
                len(data.match_id), scenario_cap, key=f"scenario:{entry['path']}", seed=seed
            )
            scenario_tensor = torch.from_numpy(scenario_indices)
            scenario_parts[str(scenario)].append(
                extract_geometry_metrics(
                    data.state_t[scenario_tensor],
                    data.mask_t[scenario_tensor],
                    fps=data.fps,
                    feature_names=data.feature_names,
                )
            )
        sampled_shards.append(
            {
                "path": str(entry["path"]),
                "source_examples": int(entry["example_count"]),
                "sampled_examples": int(len(indices)),
            }
        )
    return (
        _merge_metric_parts(global_parts),
        {name: _merge_metric_parts(parts) for name, parts in scenario_parts.items()},
        {"shards": sampled_shards, "sampled_examples": sum(allocations)},
    )


def run_train_domain_gap_audit(
    real_manifest_path: str | Path,
    synthetic_manifest_path: str | Path,
    *,
    sample_examples: int = 24576,
    real_shards_per_match: int = 4,
    scenario_examples: int = 5000,
    seed: int = 20260713,
) -> dict[str, Any]:
    """Compare train-only real and synthetic tensors under a fixed sampling protocol."""

    real_path = Path(real_manifest_path)
    synthetic_path = Path(synthetic_manifest_path)
    real_manifest = json.loads(real_path.read_text(encoding="utf-8"))
    synthetic_manifest = json.loads(synthetic_path.read_text(encoding="utf-8"))
    _validate_manifest_pair(real_manifest, synthetic_manifest)

    real_entries = select_train_shards_by_match(
        real_manifest["shards"], max_shards_per_match=real_shards_per_match
    )
    synthetic_entries = [
        entry for entry in synthetic_manifest["shards"] if entry.get("split") == "train"
    ]
    if not real_entries or not synthetic_entries:
        raise ValueError("Domain-gap audit requires train shards from both manifests.")
    real_budget = min(
        int(sample_examples),
        sum(int(entry["example_count"]) for entry in real_entries),
    )
    synthetic_budget = min(
        int(sample_examples), sum(int(entry["example_count"]) for entry in synthetic_entries)
    )
    shared_budget = min(real_budget, synthetic_budget)
    real_allocations = temperature_shard_allocations(
        [int(entry["example_count"]) for entry in real_entries],
        num_samples=shared_budget,
        temperature=1.0,
    )
    synthetic_allocations = temperature_shard_allocations(
        [int(entry["example_count"]) for entry in synthetic_entries],
        num_samples=shared_budget,
        temperature=1.0,
    )
    real_metrics, _, real_sampling = _sample_entries(
        real_path, real_entries, real_allocations, seed=seed
    )
    synthetic_metrics, scenario_metrics, synthetic_sampling = _sample_entries(
        synthetic_path,
        synthetic_entries,
        synthetic_allocations,
        seed=seed,
        scenario_cap=scenario_examples,
    )
    return {
        "status": "complete",
        "scope": "train_only",
        "seed": int(seed),
        "sampling": {
            "shared_context_examples": shared_budget,
            "real_shards_per_match": int(real_shards_per_match),
            "synthetic_scenario_context_cap_per_shard": int(scenario_examples),
            "real": real_sampling,
            "synthetic": synthetic_sampling,
        },
        "real": {
            "manifest_path": str(real_path),
            "manifest_payload_sha256": real_manifest["manifest_payload_sha256"],
            "train_match_count": len(
                {str(entry.get("match_id")) for entry in real_entries if entry.get("match_id")}
            ),
            "metrics": {
                name: summarize_metric_values(values) for name, values in real_metrics.items()
            },
        },
        "synthetic": {
            "manifest_path": str(synthetic_path),
            "manifest_payload_sha256": synthetic_manifest["manifest_payload_sha256"],
            "metrics": {
                name: summarize_metric_values(values) for name, values in synthetic_metrics.items()
            },
        },
        "global_gap_ranking": compare_metric_samples(real_metrics, synthetic_metrics),
        "scenario_gap_rankings": {
            scenario: compare_metric_samples(real_metrics, metrics)
            for scenario, metrics in sorted(scenario_metrics.items())
        },
    }
