"""Latent rollout dataset construction from exported TD-JEPA embeddings."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from footballq.latent_flow.baselines import residual_future
from footballq.repro.identity import (
    ensure_unique_sample_ids,
    payload_periods,
    payload_sample_ids,
)
from footballq.repro.splits import (
    assert_split_hash_compatible,
    named_split_indices_from_manifest,
    split_manifest_metadata,
)


@dataclass
class LatentRolloutData:
    """Inspect-friendly tensor payload for latent rollout experiments."""

    metadata: dict[str, Any]
    examples: dict[str, Any]
    splits: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LatentRolloutData:
        return cls(
            metadata=dict(payload["metadata"]),
            examples=dict(payload["examples"]),
            splits=dict(payload.get("splits", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "examples": self.examples,
            "splits": self.splits,
        }

    @property
    def num_examples(self) -> int:
        return int(torch.as_tensor(self.examples["past_z"]).shape[0])

    @property
    def latent_dim(self) -> int:
        return int(torch.as_tensor(self.examples["past_z"]).shape[-1])

    @property
    def context_steps(self) -> int:
        return int(torch.as_tensor(self.examples["past_z"]).shape[1])

    @property
    def horizon_steps(self) -> int:
        return int(torch.as_tensor(self.examples["future_z"]).shape[1])


class LatentRolloutDataset(Dataset):
    """PyTorch dataset for latent rollout examples."""

    def __init__(
        self,
        data: LatentRolloutData,
        split: str | None = None,
        indices: list[int] | None = None,
    ) -> None:
        self.data = data
        if indices is not None:
            self.indices = [int(value) for value in indices]
        elif split is not None:
            self.indices = [int(value) for value in data.splits.get(f"{split}_indices", [])]
        else:
            self.indices = list(range(data.num_examples))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.indices[index]
        return {
            "past_z": self.data.examples["past_z"][row].float(),
            "future_z": self.data.examples["future_z"][row].float(),
            "future_mask": self.data.examples["future_mask"][row].bool(),
            "baseline_future_z": self.data.examples.get(
                "baseline_future_z", self.data.examples["future_z"]
            )[row].float(),
            "residual_future_z": self.data.examples.get(
                "residual_future_z",
                torch.zeros_like(self.data.examples["future_z"]),
            )[row].float(),
            "match_id": self.data.examples["match_id"][row],
            "period": int(self.data.examples["period"][row]),
            "sample_id": self.data.examples["sample_id"][row],
            "start_index": int(self.data.examples["start_index"][row]),
            "frame_t": int(self.data.examples["frame_t"][row]),
            "source_split": self.data.examples["source_split"][row],
        }


def save_latent_rollout_dataset(data: LatentRolloutData, out: str | Path) -> Path:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data.to_dict(), out_path)
    return out_path


def load_latent_rollout_dataset(path: str | Path) -> LatentRolloutData:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return LatentRolloutData.from_dict(payload)


def _source_split_values(source_split: object, n: int) -> list[str]:
    if isinstance(source_split, str):
        return [source_split for _ in range(n)]
    if isinstance(source_split, (list, tuple)) and len(source_split) == n:
        return [str(value) for value in source_split]
    return ["unknown" for _ in range(n)]


def split_latent_indices_by_match(
    match_ids: list[str],
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 123,
) -> tuple[dict[str, Any], list[str]]:
    """Create deterministic match-level splits for rollout examples."""

    unique = sorted(set(str(value) for value in match_ids))
    rng = random.Random(seed)
    rng.shuffle(unique)
    warnings: list[str] = []
    if len(unique) >= 3:
        n_test = max(1, int(round(len(unique) * test_fraction)))
        n_val = max(1, int(round(len(unique) * val_fraction)))
        if n_test + n_val >= len(unique):
            n_test = 1
            n_val = 1
        test_matches = set(unique[:n_test])
        val_matches = set(unique[n_test : n_test + n_val])
        train_matches = set(unique[n_test + n_val :])
    elif len(unique) == 2:
        train_matches = {unique[0]}
        val_matches = {unique[1]}
        test_matches = {unique[1]}
        warnings.append(
            "only two match IDs are available; val and test reuse one match for smoke evaluation"
        )
    elif len(unique) == 1:
        train_matches = val_matches = test_matches = {unique[0]}
        warnings.append(
            "only one match ID is available; train/val/test reuse it for smoke evaluation"
        )
    else:
        raise ValueError("Cannot split an empty latent rollout dataset.")

    return (
        {
            "train_indices": [
                idx for idx, match_id in enumerate(match_ids) if str(match_id) in train_matches
            ],
            "val_indices": [
                idx for idx, match_id in enumerate(match_ids) if str(match_id) in val_matches
            ],
            "test_indices": [
                idx for idx, match_id in enumerate(match_ids) if str(match_id) in test_matches
            ],
            "train_match_ids": sorted(train_matches),
            "val_match_ids": sorted(val_matches),
            "test_match_ids": sorted(test_matches),
        },
        warnings,
    )


def _masked_train_stats(
    values: torch.Tensor,
    mask: torch.Tensor,
    train_indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    train_values = values[train_indices]
    train_mask = mask[train_indices].bool().unsqueeze(-1).expand_as(train_values)
    flat = train_values[train_mask].view(-1, values.shape[-1])
    mean = flat.mean(dim=0)
    std = flat.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean.float(), std.float()


def add_residual_targets(
    data: LatentRolloutData,
    residual_mode: str,
) -> LatentRolloutData:
    """Attach baseline, residual targets, and train-only normalization stats."""

    if residual_mode in {"none", "", None}:  # type: ignore[comparison-overlap]
        return data
    baseline, residual = residual_future(
        data.examples["past_z"].float(),
        data.examples["future_z"].float(),
        residual_mode,
    )
    train_indices = [int(value) for value in data.splits.get("train_indices", [])]
    if not train_indices:
        raise ValueError("Cannot compute residual normalization without train_indices.")
    residual_mean, residual_std = _masked_train_stats(
        residual,
        data.examples["future_mask"],
        train_indices,
    )
    latent_mean, latent_std = _masked_train_stats(
        data.examples["future_z"].float(),
        data.examples["future_mask"],
        train_indices,
    )
    data.examples["baseline_future_z"] = baseline.float()
    data.examples["residual_future_z"] = residual.float()
    data.metadata["residual_mode"] = residual_mode
    data.metadata["normalization"] = {
        "residual_mode": residual_mode,
        "residual_mean": residual_mean,
        "residual_std": residual_std,
        "latent_mean": latent_mean,
        "latent_std": latent_std,
        "stats_source": "train_split_only",
    }
    return data


def ensure_residual_targets(data: LatentRolloutData, residual_mode: str) -> LatentRolloutData:
    """Ensure a rollout dataset has residual fields for the requested mode."""

    current = str(data.metadata.get("residual_mode", ""))
    if (
        current == residual_mode
        and "baseline_future_z" in data.examples
        and "residual_future_z" in data.examples
        and data.metadata.get("normalization", {}).get("residual_mode") == residual_mode
    ):
        return data
    return add_residual_targets(data, residual_mode)


def residual_normalization_stats(data: LatentRolloutData) -> tuple[torch.Tensor, torch.Tensor]:
    """Return residual mean/std tensors from dataset metadata."""

    normalization = data.metadata.get("normalization", {})
    if "residual_mean" not in normalization or "residual_std" not in normalization:
        raise ValueError("Residual normalization stats are missing from the rollout dataset.")
    return normalization["residual_mean"].float(), normalization["residual_std"].float()


def build_latent_rollout_dataset(
    embeddings_path: str | Path,
    out: str | Path | None = None,
    context_steps: int = 5,
    horizon_steps: int = 5,
    stride_steps: int = 1,
    seed: int = 123,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    residual_mode: str | None = "constant_latent_velocity",
    split_manifest_path: str | Path | None = None,
    scientific_mode: bool = False,
) -> LatentRolloutData:
    """Build latent rollout examples without crossing match boundaries."""

    payload = torch.load(Path(embeddings_path), map_location="cpu", weights_only=False)
    z = torch.as_tensor(payload["z"]).float()
    if z.ndim != 2:
        raise ValueError(f"Expected rank-2 embeddings z, got shape {tuple(z.shape)}")
    n = int(z.shape[0])
    if n == 0:
        raise ValueError("Embedding payload contains zero rows.")
    context_steps = int(context_steps)
    horizon_steps = int(horizon_steps)
    stride_steps = max(1, int(stride_steps))
    if context_steps < 1 or horizon_steps < 1:
        raise ValueError("context_steps and horizon_steps must both be positive.")

    match_ids = [str(value) for value in payload["match_id"]]
    periods = payload_periods(payload, n, default=None if scientific_mode else 1)
    frame_ts = [int(value) for value in payload.get("frame_t", list(range(n)))]
    sample_ids = payload_sample_ids(payload, match_ids, periods, frame_ts)
    ensure_unique_sample_ids(sample_ids, context="latent rollout embedding rows")
    source_splits = _source_split_values(payload.get("source_split", "unknown"), n)
    by_group: dict[tuple[str, int], list[int]] = defaultdict(list)
    for idx, (match_id, period) in enumerate(zip(match_ids, periods, strict=True)):
        by_group[(match_id, int(period))].append(idx)

    past_parts: list[torch.Tensor] = []
    future_parts: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    example_match_ids: list[str] = []
    example_periods: list[int] = []
    example_sample_ids: list[str] = []
    start_indices: list[int] = []
    example_frame_ts: list[int] = []
    example_source_splits: list[str] = []
    dropped_short_matches: list[str] = []
    total_steps = context_steps + horizon_steps
    for (match_id, period), rows in sorted(by_group.items()):
        ordered = sorted(rows, key=lambda idx: (frame_ts[idx], idx))
        if len(ordered) < total_steps:
            dropped_short_matches.append(f"{match_id}:period{period}")
            continue
        for start in range(0, len(ordered) - total_steps + 1, stride_steps):
            past_idx = ordered[start : start + context_steps]
            future_idx = ordered[start + context_steps : start + total_steps]
            past_parts.append(z[past_idx])
            future_parts.append(z[future_idx])
            masks.append(torch.ones(horizon_steps, dtype=torch.bool))
            example_match_ids.append(match_id)
            example_periods.append(int(period))
            example_sample_ids.append(sample_ids[past_idx[0]])
            start_indices.append(int(past_idx[0]))
            example_frame_ts.append(int(frame_ts[past_idx[0]]))
            example_source_splits.append(str(source_splits[past_idx[0]]))

    if not past_parts:
        raise ValueError(
            "No latent rollout examples were produced. Check that each match has at least "
            f"context_steps+horizon_steps={total_steps} embedding rows."
        )

    past_z = torch.stack(past_parts).float()
    future_z = torch.stack(future_parts).float()
    future_mask = torch.stack(masks).bool()
    if split_manifest_path is not None:
        splits = named_split_indices_from_manifest(example_match_ids, split_manifest_path)
        split_warnings: list[str] = []
    else:
        splits, split_warnings = split_latent_indices_by_match(
            example_match_ids,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )
    warnings = list(split_warnings)
    if dropped_short_matches:
        warnings.append(
            "dropped matches shorter than the requested rollout length: "
            + ", ".join(dropped_short_matches)
        )
    repro_metadata = split_manifest_metadata(split_manifest_path, scientific_mode=scientific_mode)
    assert_split_hash_compatible(
        payload,
        repro_metadata,
        source_name="latent rollout embedding payload",
        require_source_hash=scientific_mode,
    )
    data = LatentRolloutData(
        metadata={
            "created_by": "build_latent_rollout_dataset.py",
            "source_embeddings": str(embeddings_path),
            "num_embedding_rows": n,
            "num_examples": int(past_z.shape[0]),
            "latent_dim": int(z.shape[1]),
            "context_steps": context_steps,
            "horizon_steps": horizon_steps,
            "stride_steps": stride_steps,
            "warnings": warnings,
            "encoder_frozen": True,
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
            **repro_metadata,
        },
        examples={
            "past_z": past_z,
            "future_z": future_z,
            "future_mask": future_mask,
            "match_id": example_match_ids,
            "period": torch.tensor(example_periods, dtype=torch.long),
            "sample_id": example_sample_ids,
            "start_index": torch.tensor(start_indices, dtype=torch.long),
            "frame_t": torch.tensor(example_frame_ts, dtype=torch.long),
            "source_split": example_source_splits,
        },
        splits=splits,
    )
    if residual_mode and residual_mode != "none":
        data = add_residual_targets(data, residual_mode)
    if out is not None:
        save_latent_rollout_dataset(data, out)
    return data
