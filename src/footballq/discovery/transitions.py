"""Transition dataset construction from TD-JEPA latent embeddings."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from footballq.data.normalize import denormalize_xy_to_meters
from footballq.data.windows import (
    BALL_INDEX,
    TEAM_AWAY,
    TEAM_HOME,
    TrackingWindowTensorData,
    load_windows_pt,
)
from footballq.repro.identity import (
    ensure_unique_sample_ids,
    payload_periods,
    payload_sample_ids,
)
from footballq.repro.splits import assert_split_hash_compatible, split_manifest_metadata

UNKNOWN = "unknown"
STRESS_FIELDS = [
    "high_future_ball_displacement",
    "high_ball_acceleration",
    "high_ball_direction_change",
    "high_team_shape_change",
    "high_team_width_change",
    "high_team_length_change",
    "high_stretch_index_change",
]


@dataclass
class TransitionDatasetData:
    """Tensor payload for latent transition discovery."""

    examples: dict[str, Any]
    features: dict[str, Any]
    metadata: dict[str, Any]

    @property
    def num_examples(self) -> int:
        return int(torch.as_tensor(self.examples["z_t"]).shape[0])

    @property
    def latent_dim(self) -> int:
        return int(torch.as_tensor(self.examples["z_t"]).shape[-1])

    @property
    def delta_seconds_values(self) -> list[float]:
        values = torch.as_tensor(self.examples["delta_seconds"]).float()
        return sorted(float(value) for value in values.unique().tolist())

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TransitionDatasetData:
        return cls(
            examples=dict(payload["examples"]),
            features=dict(payload.get("features", {})),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "examples": self.examples,
            "features": self.features,
            "metadata": self.metadata,
        }


def save_transition_dataset(data: TransitionDatasetData, out: str | Path) -> Path:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data.to_dict(), out_path)
    return out_path


def load_transition_dataset(path: str | Path) -> TransitionDatasetData:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return TransitionDatasetData.from_dict(payload)


def _as_list(value: Any, n: int, default: str = UNKNOWN) -> list[str]:
    if isinstance(value, str):
        return [value] * n
    if isinstance(value, (list, tuple)) and len(value) == n:
        return [str(item) for item in value]
    return [default] * n


def _safe_tensor(values: list[float], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype)


def _bucket_by_quantiles(
    values: torch.Tensor, low_name: str = "low", high_name: str = "high"
) -> list[str]:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return [UNKNOWN] * int(values.numel())
    q1 = torch.quantile(finite.float(), 1.0 / 3.0)
    q2 = torch.quantile(finite.float(), 2.0 / 3.0)
    out = []
    for value in values.tolist():
        if not math.isfinite(float(value)):
            out.append(UNKNOWN)
        elif float(value) <= float(q1):
            out.append(low_name)
        elif float(value) >= float(q2):
            out.append(high_name)
        else:
            out.append("medium")
    return out


def _progression_bucket(values: torch.Tensor) -> list[str]:
    out = []
    for value in values.tolist():
        if not math.isfinite(float(value)):
            out.append(UNKNOWN)
        elif float(value) >= 5.0:
            out.append("forward")
        elif float(value) <= -5.0:
            out.append("backward")
        else:
            out.append("neutral")
    return out


def _nan_quantile(values: torch.Tensor, q: float) -> float:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return float("nan")
    return float(torch.quantile(finite.float(), q).item())


def _high_flag(values: torch.Tensor, q: float = 0.75) -> torch.Tensor:
    threshold = _nan_quantile(values, q)
    if not math.isfinite(threshold):
        return torch.zeros_like(values, dtype=torch.bool)
    return torch.isfinite(values) & (values >= threshold)


def _masked_norm(a: torch.Tensor, b: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    values = torch.linalg.vector_norm(a - b, dim=-1)
    return values.masked_fill(~valid, float("nan"))


def _team_selector(windows: TrackingWindowTensorData, team_code: int) -> torch.Tensor:
    return windows.team_id.long() == int(team_code)


def _team_stat(
    xy_m: torch.Tensor, mask: torch.Tensor, selector: torch.Tensor, stat: str
) -> torch.Tensor:
    valid = mask & selector
    valid_f = valid.unsqueeze(-1).float()
    count = valid_f.sum(dim=1).clamp_min(1.0)
    centroid = (xy_m * valid_f).sum(dim=1) / count
    if stat == "centroid":
        return centroid
    if stat == "width":
        values = xy_m[..., 1]
    elif stat == "length":
        values = xy_m[..., 0]
    elif stat == "stretch":
        distance = torch.linalg.vector_norm(xy_m - centroid.unsqueeze(1), dim=-1)
        out = (distance * valid.float()).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return out.masked_fill(valid.sum(dim=1) == 0, float("nan"))
    else:
        raise ValueError(f"Unknown team stat: {stat}")
    max_values = values.masked_fill(~valid, float("-inf")).max(dim=1).values
    min_values = values.masked_fill(~valid, float("inf")).min(dim=1).values
    out = max_values - min_values
    return out.masked_fill(valid.sum(dim=1) == 0, float("nan"))


def _window_metric_payload(windows: TrackingWindowTensorData) -> dict[str, Any]:
    current = denormalize_xy_to_meters(windows.past.float()[:, -1, :, :2])
    future = denormalize_xy_to_meters(windows.future_xy.float())
    current_mask = windows.past_mask.bool()[:, -1]
    future_mask = windows.future_mask.bool()

    ball_current = current[:, BALL_INDEX]
    ball_future = future[:, :, BALL_INDEX]
    ball_current_mask = current_mask[:, BALL_INDEX]
    ball_future_mask = future_mask[:, :, BALL_INDEX]
    final_ball_valid = ball_current_mask & ball_future_mask[:, -1]
    ball_displacement = _masked_norm(ball_future[:, -1], ball_current, final_ball_valid)
    ball_progression = (ball_future[:, -1, 0] - ball_current[:, 0]).masked_fill(
        ~final_ball_valid,
        float("nan"),
    )

    if future.shape[1] >= 3:
        fps = float(windows.fps)
        velocities = (ball_future[:, 1:] - ball_future[:, :-1]) * fps
        velocity_mask = ball_future_mask[:, 1:] & ball_future_mask[:, :-1]
        acceleration = (velocities[:, 1:] - velocities[:, :-1]) * fps
        acceleration_mask = velocity_mask[:, 1:] & velocity_mask[:, :-1]
        acceleration_mag = torch.linalg.vector_norm(acceleration, dim=-1)
        acceleration_mag = acceleration_mag.masked_fill(~acceleration_mask, float("nan"))
        ball_acceleration = (
            torch.nan_to_num(
                acceleration_mag,
                nan=-float("inf"),
            )
            .max(dim=1)
            .values
        )
        ball_acceleration = ball_acceleration.masked_fill(
            ~torch.isfinite(ball_acceleration),
            float("nan"),
        )

        v1 = velocities[:, :-1]
        v2 = velocities[:, 1:]
        direction_mask = velocity_mask[:, :-1] & velocity_mask[:, 1:]
        denom = (
            torch.linalg.vector_norm(v1, dim=-1) * torch.linalg.vector_norm(v2, dim=-1)
        ).clamp_min(1e-6)
        cos = ((v1 * v2).sum(dim=-1) / denom).clamp(-1.0, 1.0)
        angles = torch.acos(cos).masked_fill(~direction_mask, float("nan"))
        ball_direction_change = torch.nan_to_num(angles, nan=-float("inf")).max(dim=1).values
        ball_direction_change = ball_direction_change.masked_fill(
            ~torch.isfinite(ball_direction_change),
            float("nan"),
        )
    else:
        n = len(windows.match_id)
        ball_acceleration = torch.full((n,), float("nan"))
        ball_direction_change = torch.full((n,), float("nan"))

    future_last = future[:, -1]
    future_last_mask = future_mask[:, -1]
    centroid_parts: list[torch.Tensor] = []
    width_parts: list[torch.Tensor] = []
    length_parts: list[torch.Tensor] = []
    stretch_parts: list[torch.Tensor] = []
    for team_code in [TEAM_HOME, TEAM_AWAY]:
        selector = _team_selector(windows, team_code)
        if not bool(selector.any()):
            continue
        current_centroid = _team_stat(current, current_mask, selector, "centroid")
        future_centroid = _team_stat(future_last, future_last_mask, selector, "centroid")
        centroid_parts.append(torch.linalg.vector_norm(future_centroid - current_centroid, dim=-1))
        for stat, bucket in [
            ("width", width_parts),
            ("length", length_parts),
            ("stretch", stretch_parts),
        ]:
            current_stat = _team_stat(current, current_mask, selector, stat)
            future_stat = _team_stat(future_last, future_last_mask, selector, stat)
            bucket.append((future_stat - current_stat).abs())

    def _mean(parts: list[torch.Tensor]) -> torch.Tensor:
        if not parts:
            return torch.full((len(windows.match_id),), float("nan"))
        return torch.nanmean(torch.stack(parts, dim=1), dim=1)

    width_change = _mean(width_parts)
    length_change = _mean(length_parts)
    stretch_change = _mean(stretch_parts)
    shape_change = _mean(centroid_parts) + width_change + length_change + stretch_change

    metrics = {
        "future_ball_displacement_m": ball_displacement,
        "future_ball_dx_global_m": ball_progression,
        "future_ball_progression_m": ball_progression,
        "ball_acceleration_mps2": ball_acceleration,
        "ball_direction_change_rad": ball_direction_change,
        "team_shape_change_m": shape_change,
        "team_width_change_m": width_change,
        "team_length_change_m": length_change,
        "stretch_index_change_m": stretch_change,
    }
    categorical = {
        "future_ball_global_x_bucket": _progression_bucket(ball_progression),
        "future_ball_progression_bucket": _progression_bucket(ball_progression),
        "future_ball_displacement_bucket": _bucket_by_quantiles(ball_displacement),
        "team_shape_change_bucket": _bucket_by_quantiles(shape_change),
    }
    stress = {
        "high_future_ball_displacement": _high_flag(ball_displacement),
        "high_ball_acceleration": _high_flag(ball_acceleration),
        "high_ball_direction_change": _high_flag(ball_direction_change),
        "high_team_shape_change": _high_flag(shape_change),
        "high_team_width_change": _high_flag(width_change),
        "high_team_length_change": _high_flag(length_change),
        "high_stretch_index_change": _high_flag(stretch_change),
    }
    return {**metrics, **categorical, **stress}


def _window_lookup(
    windows_path: Path | None,
) -> tuple[dict[str, int], dict[str, Any], dict[str, Any]]:
    if windows_path is None:
        return {}, {}, {"windows_available": False, "missing_metadata_fields": ["windows"]}
    windows = load_windows_pt(windows_path)
    ensure_unique_sample_ids(list(windows.sample_id), context="transition window rows")
    lookup: dict[str, int] = {}
    for idx, sample_id in enumerate(windows.sample_id):
        lookup.setdefault(str(sample_id), idx)
    metrics = _window_metric_payload(windows)
    values: dict[str, Any] = {
        "label_frame": torch.tensor(windows.label_frame, dtype=torch.long),
        "period": [int(value) for value in windows.period],
        "phase": list(windows.phase or [UNKNOWN] * len(windows.match_id)),
        "event_type": list(windows.event_type or [UNKNOWN] * len(windows.match_id)),
        "possession_team_id": list(windows.possession_team_id or [UNKNOWN] * len(windows.match_id)),
        "possession_available": [bool(value) for value in windows.possession_available],
        **metrics,
    }
    diagnostics = {
        "windows_available": True,
        "num_window_rows": len(windows.match_id),
        "window_match_count": len(set(str(value) for value in windows.match_id)),
        "window_match_ids": sorted(set(str(value) for value in windows.match_id)),
        "missing_metadata_fields": [],
    }
    return lookup, values, diagnostics


def _source_splits(payload: dict[str, Any], n: int) -> list[str]:
    values = payload.get("source_split", "unknown")
    if isinstance(values, str):
        return [values] * n
    if isinstance(values, (list, tuple)) and len(values) == n:
        return [str(value) for value in values]
    return ["unknown"] * n


def _ordered_by_match_period(
    match_ids: list[str],
    periods: list[int],
    frame_t: list[int],
) -> dict[tuple[str, int], list[int]]:
    out: dict[tuple[str, int], list[int]] = defaultdict(list)
    for idx, (match_id, period) in enumerate(zip(match_ids, periods, strict=True)):
        out[(str(match_id), int(period))].append(idx)
    for key in list(out):
        out[key].sort(key=lambda idx: (int(frame_t[idx]), idx))
    return out


def _median_frame_stride(indices: list[int], frame_t: list[int]) -> int:
    diffs = [
        int(frame_t[b]) - int(frame_t[a])
        for a, b in zip(indices, indices[1:], strict=False)
        if int(frame_t[b]) > int(frame_t[a])
    ]
    if not diffs:
        return 1
    diffs_sorted = sorted(diffs)
    return max(1, int(diffs_sorted[len(diffs_sorted) // 2]))


def _pair_indices_for_delta(
    by_match: dict[tuple[str, int], list[int]],
    frame_t: list[int],
    delta_step: int,
) -> tuple[list[tuple[int, int, int | None]], dict[str, Any]]:
    exact_pairs: list[tuple[int, int, int | None]] = []
    missing_by_match: dict[str, int] = {}
    for (match_id, period), indices in by_match.items():
        lookup = {int(frame_t[idx]): idx for idx in indices}
        missing = 0
        for idx in indices:
            next_idx = lookup.get(int(frame_t[idx]) + int(delta_step))
            if next_idx is None:
                missing += 1
                continue
            prev_idx = lookup.get(int(frame_t[idx]) - int(delta_step))
            exact_pairs.append((idx, next_idx, prev_idx))
        missing_by_match[f"{match_id}:period{period}"] = missing
    if exact_pairs:
        return exact_pairs, {
            "pairing_mode": "exact_frame_delta",
            "requested_delta_steps": int(delta_step),
            "num_pairs": len(exact_pairs),
            "missing_start_rows": int(sum(missing_by_match.values())),
            "missing_start_rows_by_match": missing_by_match,
        }

    fallback_pairs: list[tuple[int, int, int | None]] = []
    offsets: dict[str, int] = {}
    actual_deltas: list[int] = []
    for (match_id, period), indices in by_match.items():
        stride = _median_frame_stride(indices, frame_t)
        offset = max(1, int(math.floor(float(delta_step) / float(stride) + 0.5)))
        offsets[f"{match_id}:period{period}"] = offset
        for pos, idx in enumerate(indices):
            next_pos = pos + offset
            if next_pos >= len(indices):
                continue
            prev_pos = pos - offset
            prev_idx = indices[prev_pos] if prev_pos >= 0 else None
            next_idx = indices[next_pos]
            actual_delta = int(frame_t[next_idx]) - int(frame_t[idx])
            if actual_delta <= 0 or actual_delta > int(delta_step) + stride:
                continue
            fallback_pairs.append((idx, next_idx, prev_idx))
            actual_deltas.append(actual_delta)
    return fallback_pairs, {
        "pairing_mode": "sequence_offset_fallback",
        "requested_delta_steps": int(delta_step),
        "num_pairs": len(fallback_pairs),
        "offsets_by_match": offsets,
        "median_actual_delta_steps": (
            int(sorted(actual_deltas)[len(actual_deltas) // 2]) if actual_deltas else None
        ),
        "warning": (
            f"no exact frame_t+{int(delta_step)} pairs were available; "
            "used sequence-offset fallback and recorded actual_delta_seconds"
        ),
    }


def _append_metadata(
    metadata_acc: dict[str, list[Any]],
    window_idx: int | None,
    window_values: dict[str, Any],
) -> None:
    fields = [
        "period",
        "label_frame",
        "phase",
        "event_type",
        "possession_team_id",
        "possession_available",
        "future_ball_global_x_bucket",
        "future_ball_progression_bucket",
        "future_ball_displacement_bucket",
        "team_shape_change_bucket",
    ]
    continuous = [
        "future_ball_displacement_m",
        "future_ball_dx_global_m",
        "future_ball_progression_m",
        "team_shape_change_m",
        "team_width_change_m",
        "team_length_change_m",
        "stretch_index_change_m",
        "ball_acceleration_mps2",
        "ball_direction_change_rad",
    ]
    for field in fields:
        values = window_values.get(field)
        if window_idx is None or values is None:
            if field == "possession_available":
                metadata_acc[field].append(False)
            elif field in {"period", "label_frame"}:
                metadata_acc[field].append(None)
            else:
                metadata_acc[field].append(UNKNOWN)
        elif isinstance(values, torch.Tensor):
            metadata_acc[field].append(values[window_idx].item())
        else:
            metadata_acc[field].append(values[window_idx])
    for field in continuous:
        values = window_values.get(field)
        value = (
            float("nan")
            if window_idx is None or values is None
            else float(values[window_idx].item())
        )
        metadata_acc[field].append(value)
    for field in STRESS_FIELDS:
        values = window_values.get(field)
        value = False if window_idx is None or values is None else bool(values[window_idx].item())
        metadata_acc[field].append(value)
    metadata_acc["high_width_change"].append(metadata_acc["high_team_width_change"][-1])
    metadata_acc["high_length_change"].append(metadata_acc["high_team_length_change"][-1])


def _feature_payload(
    z_t: torch.Tensor,
    z_next: torch.Tensor,
    delta_z: torch.Tensor,
    delta_seconds: torch.Tensor,
    source_split: list[str],
    scientific_mode: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    delta_norm = torch.linalg.vector_norm(delta_z, dim=1)
    z_t_norm = torch.linalg.vector_norm(z_t, dim=1)
    z_next_norm = torch.linalg.vector_norm(z_next, dim=1)
    cosine = torch.nn.functional.cosine_similarity(z_t, z_next, dim=1, eps=1e-8)
    safe_delta_seconds = delta_seconds.float().clamp_min(1e-6).unsqueeze(1)
    latent_velocity = delta_z / safe_delta_seconds
    latent_velocity_norm = torch.linalg.vector_norm(latent_velocity, dim=1)
    source_train_indices = [
        idx for idx, split in enumerate(source_split) if str(split).lower() == "train"
    ]
    if not source_train_indices and scientific_mode:
        raise ValueError(
            "Scientific transition feature normalization requires at least one train row."
        )
    train_indices = source_train_indices or list(range(delta_z.shape[0]))
    train_delta = delta_z[train_indices]
    train_mean = train_delta.mean(dim=0)
    train_std = train_delta.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized_delta_z = (delta_z - train_mean) / train_std
    pca_delta_z, pca_diag = _pca_projection(normalized_delta_z, train_indices)
    random_delta_z = _random_projection(normalized_delta_z, output_dim=normalized_delta_z.shape[1])
    features = {
        "delta_norm": delta_norm,
        "z_t_norm": z_t_norm,
        "z_next_norm": z_next_norm,
        "cosine_z_t_z_next": cosine,
        "latent_velocity": latent_velocity,
        "latent_velocity_norm": latent_velocity_norm,
        "normalized_delta_z": normalized_delta_z,
        "pca_delta_z": pca_delta_z,
        "random_encoder_delta_z": random_delta_z,
        "delta_z_train_mean": train_mean,
        "delta_z_train_std": train_std,
    }
    diagnostics = {
        "normalization_train_rows": len(train_indices),
        "normalization_source": "source_split=train" if source_train_indices else "all_rows",
        **pca_diag,
    }
    return features, diagnostics


def _train_indices(
    source_split: list[str],
    n: int,
    *,
    scientific_mode: bool = False,
    context: str = "feature fit",
) -> list[int]:
    train_indices = [idx for idx, split in enumerate(source_split) if str(split).lower() == "train"]
    if not train_indices and scientific_mode:
        raise ValueError(f"Scientific transition {context} requires at least one train row.")
    return train_indices or list(range(n))


def _standardize_feature_matrix(
    matrix: torch.Tensor,
    source_split: list[str],
    scientific_mode: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    train_indices = _train_indices(
        source_split,
        int(matrix.shape[0]),
        scientific_mode=scientific_mode,
        context="standardization",
    )
    train_values = matrix[train_indices]
    train_mean = train_values.mean(dim=0, keepdim=True)
    train_std = train_values.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (matrix - train_mean) / train_std, {"standardization_train_rows": len(train_indices)}


def _fill_nan_with_train_mean(
    matrix: torch.Tensor,
    source_split: list[str],
    scientific_mode: bool = False,
) -> torch.Tensor:
    train_indices = _train_indices(
        source_split,
        int(matrix.shape[0]),
        scientific_mode=scientific_mode,
        context="NaN imputation",
    )
    train_values = matrix[train_indices]
    finite = torch.isfinite(train_values)
    filled = torch.where(finite, train_values, torch.zeros_like(train_values))
    counts = finite.float().sum(dim=0).clamp_min(1.0)
    train_mean = filled.sum(dim=0) / counts
    return torch.where(torch.isfinite(matrix), matrix, train_mean.unsqueeze(0))


def _pca_projection(
    standardized: torch.Tensor,
    train_indices: list[int],
    max_components: int = 8,
) -> tuple[torch.Tensor, dict[str, Any]]:
    train_values = standardized[train_indices]
    n_components = max(
        1,
        min(int(max_components), int(train_values.shape[0]), int(train_values.shape[1])),
    )
    try:
        _u, _s, vh = torch.linalg.svd(train_values, full_matrices=False)
        components = vh[:n_components].T.contiguous()
        projected = standardized @ components
    except RuntimeError:
        projected = standardized[:, :n_components].contiguous()
    return projected, {"pca_delta_z_components": n_components}


def _random_projection(
    matrix: torch.Tensor,
    output_dim: int,
    seed: int = 123,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    projection = torch.randn(
        matrix.shape[1],
        int(output_dim),
        generator=generator,
        dtype=matrix.dtype,
    ) / math.sqrt(max(1, int(matrix.shape[1])))
    return matrix @ projection


def _handcrafted_feature_payload(
    metadata_acc: dict[str, list[Any]],
    source_split: list[str],
    scientific_mode: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fields = [
        "future_ball_displacement_m",
        "future_ball_dx_global_m",
        "team_shape_change_m",
        "team_width_change_m",
        "team_length_change_m",
        "stretch_index_change_m",
        "ball_acceleration_mps2",
        "ball_direction_change_rad",
    ]
    columns = [
        torch.tensor([float(value) for value in metadata_acc.get(field, [])], dtype=torch.float32)
        for field in fields
    ]
    if not columns:
        return {}, {"handcrafted_feature_fields": []}
    matrix = torch.stack(columns, dim=1)
    filled = _fill_nan_with_train_mean(matrix, source_split, scientific_mode=scientific_mode)
    standardized, diag = _standardize_feature_matrix(
        filled,
        source_split,
        scientific_mode=scientific_mode,
    )
    train_indices = _train_indices(
        source_split,
        int(standardized.shape[0]),
        scientific_mode=scientific_mode,
        context="PCA fit",
    )
    pca_features, pca_diag = _pca_projection(standardized, train_indices, max_components=4)
    return (
        {
            "handcrafted_structure_metrics": standardized,
            "pca_handcrafted_structure_metrics": pca_features,
        },
        {
            **diag,
            **pca_diag,
            "handcrafted_feature_fields": fields,
        },
    )


def build_transition_dataset(
    embeddings_path: str | Path,
    windows_path: str | Path | None = None,
    out: str | Path | None = None,
    delta_steps: list[int] | tuple[int, ...] = (2,),
    fps: float = 10.0,
    split_manifest_path: str | Path | None = None,
    scientific_mode: bool = False,
) -> TransitionDatasetData:
    """Build latent transition examples for one or more requested frame deltas."""

    embeddings_path = Path(embeddings_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(f"TD-JEPA embeddings not found: {embeddings_path}")
    windows_path_obj = Path(windows_path) if windows_path is not None else None
    if windows_path_obj is not None and not windows_path_obj.exists():
        raise FileNotFoundError(f"Tracking windows not found: {windows_path_obj}")

    payload = torch.load(embeddings_path, map_location="cpu", weights_only=False)
    z = torch.as_tensor(payload["z"]).float()
    if z.ndim != 2:
        raise ValueError(f"Expected rank-2 embeddings, got shape {tuple(z.shape)}")
    n = int(z.shape[0])
    match_ids = _as_list(payload.get("match_id"), n)
    periods = payload_periods(payload, n, default=None if scientific_mode else 1)
    frame_t = [int(value) for value in payload.get("frame_t", list(range(n)))]
    sample_ids = payload_sample_ids(payload, match_ids, periods, frame_t)
    ensure_unique_sample_ids(sample_ids, context="transition embedding rows")
    source_splits = _source_splits(payload, n)
    by_match = _ordered_by_match_period(match_ids, periods, frame_t)
    window_key_to_idx, window_values, window_diag = _window_lookup(windows_path_obj)

    embedding_matches = set(str(value) for value in match_ids)
    if window_diag.get("windows_available"):
        window_matches = set(window_diag.get("window_match_ids", []))
        missing_window_matches = sorted(embedding_matches - window_matches)
        if missing_window_matches:
            raise ValueError(
                "Tracking windows do not cover all embedding matches. Missing match IDs: "
                f"{', '.join(missing_window_matches)}"
            )

    z_t_rows: list[torch.Tensor] = []
    z_next_rows: list[torch.Tensor] = []
    z_prev_rows: list[torch.Tensor] = []
    has_prev: list[bool] = []
    match_rows: list[str] = []
    period_rows: list[int] = []
    sample_id_rows: list[str] = []
    frame_rows: list[int] = []
    next_frame_rows: list[int] = []
    requested_delta_rows: list[int] = []
    actual_delta_rows: list[int] = []
    source_split_rows: list[str] = []
    metadata_acc: dict[str, list[Any]] = defaultdict(list)
    pairing_diagnostics: list[dict[str, Any]] = []
    unmatched_metadata = 0

    for delta in [int(value) for value in delta_steps]:
        pairs, diagnostics = _pair_indices_for_delta(by_match, frame_t, delta)
        if not pairs:
            raise ValueError(f"No transition pairs were produced for delta_steps={delta}")
        pairing_diagnostics.append(diagnostics)
        pair_matches = {match_ids[idx] for idx, _, _ in pairs}
        missing_transition_matches = sorted(embedding_matches - set(pair_matches))
        if missing_transition_matches:
            raise ValueError(
                f"Transition delta_steps={delta} has no examples for match IDs: "
                f"{', '.join(missing_transition_matches)}"
            )
        for idx, next_idx, prev_idx in pairs:
            z_t_rows.append(z[idx])
            z_next_rows.append(z[next_idx])
            if prev_idx is None:
                z_prev_rows.append(torch.full_like(z[idx], float("nan")))
                has_prev.append(False)
            else:
                z_prev_rows.append(z[prev_idx])
                has_prev.append(True)
            match_rows.append(str(match_ids[idx]))
            period_rows.append(int(periods[idx]))
            sample_id_rows.append(str(sample_ids[idx]))
            frame_rows.append(int(frame_t[idx]))
            next_frame_rows.append(int(frame_t[next_idx]))
            requested_delta_rows.append(delta)
            actual_delta = int(frame_t[next_idx]) - int(frame_t[idx])
            actual_delta_rows.append(actual_delta)
            source_split_rows.append(str(source_splits[idx]))
            window_idx = window_key_to_idx.get(str(sample_ids[idx]))
            if window_idx is None:
                unmatched_metadata += 1
            _append_metadata(metadata_acc, window_idx, window_values)

    z_t = torch.stack(z_t_rows).float()
    z_next = torch.stack(z_next_rows).float()
    z_prev = torch.stack(z_prev_rows).float()
    delta_z = z_next - z_t
    requested_delta_tensor = torch.tensor(requested_delta_rows, dtype=torch.long)
    actual_delta_tensor = torch.tensor(actual_delta_rows, dtype=torch.long)
    requested_seconds = requested_delta_tensor.float() / float(fps)
    actual_seconds = actual_delta_tensor.float() / float(fps)
    features, feature_diag = _feature_payload(
        z_t,
        z_next,
        delta_z,
        actual_seconds,
        source_split_rows,
        scientific_mode=scientific_mode,
    )
    handcrafted_features, handcrafted_diag = _handcrafted_feature_payload(
        metadata_acc,
        source_split_rows,
        scientific_mode=scientific_mode,
    )
    features.update(handcrafted_features)
    feature_diag.update(handcrafted_diag)
    repro_metadata = split_manifest_metadata(split_manifest_path, scientific_mode=scientific_mode)
    assert_split_hash_compatible(
        payload,
        repro_metadata,
        source_name="transition embedding payload",
        require_source_hash=scientific_mode,
    )

    examples: dict[str, Any] = {
        "z_t": z_t,
        "z_next": z_next,
        "z_prev": z_prev,
        "has_prev": torch.tensor(has_prev, dtype=torch.bool),
        "delta_z": delta_z,
        "delta_steps": requested_delta_tensor,
        "actual_delta_steps": actual_delta_tensor,
        "delta_seconds": requested_seconds,
        "actual_delta_seconds": actual_seconds,
        "match_id": match_rows,
        "period": period_rows,
        "sample_id": sample_id_rows,
        "frame_t": torch.tensor(frame_rows, dtype=torch.long),
        "frame_next": torch.tensor(next_frame_rows, dtype=torch.long),
        "source_split": source_split_rows,
        "metadata": dict(metadata_acc),
    }
    metadata = {
        "created_by": "build_transition_dataset.py",
        "source_embeddings": str(embeddings_path),
        "source_windows": str(windows_path_obj) if windows_path_obj is not None else None,
        "num_embedding_rows": n,
        "num_examples": int(z_t.shape[0]),
        "latent_dim": int(z_t.shape[1]),
        "num_matches": len(embedding_matches),
        "match_ids": sorted(embedding_matches),
        "requested_delta_steps": sorted(set(int(value) for value in requested_delta_rows)),
        "requested_delta_seconds": sorted(
            set(float(value) for value in requested_seconds.tolist())
        ),
        "actual_delta_seconds": sorted(set(float(value) for value in actual_seconds.tolist())),
        "fps": float(fps),
        "pairing_diagnostics": pairing_diagnostics,
        "unmatched_metadata_rows": int(unmatched_metadata),
        "missing_metadata_fields": window_diag.get("missing_metadata_fields", []),
        "match_counts": dict(Counter(match_rows)),
        "feature_view": str(
            payload.get(
                "feature_view",
                payload.get("data_meta", {}).get("feature_view", "unknown"),
            )
        ),
        "objective_mode": str(
            payload.get(
                "objective_mode",
                payload.get("data_meta", {}).get("objective_mode", "unknown"),
            )
        ),
        "legacy_alignment_allowed": False,
        **window_diag,
        **feature_diag,
        **repro_metadata,
    }
    data = TransitionDatasetData(examples=examples, features=features, metadata=metadata)
    if out is not None:
        save_transition_dataset(data, out)
    return data


def transition_summary(data: TransitionDatasetData) -> dict[str, Any]:
    """Return a compact JSON-safe summary for reports."""

    return {
        "num_examples": data.num_examples,
        "latent_dim": data.latent_dim,
        "num_matches": data.metadata.get("num_matches"),
        "match_ids": data.metadata.get("match_ids", []),
        "requested_delta_steps": data.metadata.get("requested_delta_steps", []),
        "requested_delta_seconds": data.metadata.get("requested_delta_seconds", []),
        "actual_delta_seconds": data.metadata.get("actual_delta_seconds", []),
        "feature_view": data.metadata.get("feature_view", "unknown"),
        "objective_mode": data.metadata.get("objective_mode", "unknown"),
        "split_manifest_sha256": data.metadata.get("split_manifest_sha256"),
        "missing_metadata_fields": data.metadata.get("missing_metadata_fields", []),
        "unmatched_metadata_rows": data.metadata.get("unmatched_metadata_rows", 0),
        "pairing_diagnostics": data.metadata.get("pairing_diagnostics", []),
    }
