"""Probe dataset construction and loading."""

from __future__ import annotations

import random
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from footballq.data.windows import TrackingWindowTensorData, load_windows_pt
from footballq.probes.features import probe_feature_matrix
from footballq.probes.labels import derive_probe_targets, raw_state_summary_features


@dataclass
class ProbeDatasetData:
    """Inspect-friendly tensor payload for Experiment 3 probes."""

    metadata: dict[str, Any]
    examples: dict[str, Any]
    label_maps: dict[str, dict[str, int]]
    splits: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProbeDatasetData":
        return cls(
            metadata=dict(payload["metadata"]),
            examples=dict(payload["examples"]),
            label_maps=dict(payload.get("label_maps", {})),
            splits=dict(payload.get("splits", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "examples": self.examples,
            "label_maps": self.label_maps,
            "splits": self.splits,
        }

    @property
    def num_examples(self) -> int:
        return int(torch.as_tensor(self.examples["z"]).shape[0])

    @property
    def target_types(self) -> dict[str, str]:
        return dict(self.metadata.get("target_types", {}))


class ProbeDataset(Dataset):
    """PyTorch dataset for one feature source and one target."""

    def __init__(
        self,
        data: ProbeDatasetData,
        target_name: str,
        feature_source: str = "td_jepa",
        split: str | None = None,
        indices: list[int] | None = None,
        random_seed: int = 123,
    ) -> None:
        if target_name not in data.examples.get("targets", {}):
            available = sorted(data.examples.get("targets", {}))
            raise ValueError(f"Target {target_name!r} not found. Available: {available}")
        self.data = data
        self.target_name = target_name
        self.feature_source = feature_source
        self.features = probe_feature_matrix(data.examples, feature_source, seed=random_seed)
        target = torch.as_tensor(data.examples["targets"][target_name])
        mask = torch.as_tensor(data.examples["target_masks"][target_name]).bool()
        if indices is None:
            if split is None:
                indices = list(range(data.num_examples))
            else:
                split_key = f"{split}_indices"
                indices = [int(value) for value in data.splits.get(split_key, [])]
        self.indices = [int(idx) for idx in indices if bool(mask[int(idx)].item())]
        self.target = target
        self.task_type = data.target_types[target_name]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.indices[index]
        y = self.target[row]
        if self.task_type == "classification":
            y = y.long()
        else:
            y = y.float().view(1)
        return {
            "x": self.features[row].float(),
            "y": y,
            "index": torch.tensor(row, dtype=torch.long),
        }


def save_probe_dataset(data: ProbeDatasetData, out: str | Path) -> Path:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data.to_dict(), out_path)
    return out_path


def load_probe_dataset(path: str | Path) -> ProbeDatasetData:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return ProbeDatasetData.from_dict(payload)


def _source_split_values(source_split: object, n: int) -> list[str]:
    if isinstance(source_split, str):
        return [source_split for _ in range(n)]
    if isinstance(source_split, (list, tuple)) and len(source_split) == n:
        return [str(value) for value in source_split]
    return ["unknown" for _ in range(n)]


def _align_embeddings_to_windows(
    embeddings: dict[str, Any],
    windows: TrackingWindowTensorData,
) -> tuple[list[int], list[int], str, list[str]]:
    match_ids = [str(value) for value in embeddings["match_id"]]
    frame_ts = [int(value) for value in embeddings["frame_t"]]
    window_lookup: dict[tuple[str, int], deque[int]] = defaultdict(deque)
    for idx, (match_id, start_frame) in enumerate(zip(windows.match_id, windows.start_frame, strict=True)):
        window_lookup[(str(match_id), int(start_frame))].append(idx)

    embedding_indices: list[int] = []
    window_indices: list[int] = []
    for idx, key in enumerate(zip(match_ids, frame_ts, strict=True)):
        candidates = window_lookup.get(key)
        if candidates:
            embedding_indices.append(idx)
            window_indices.append(candidates.popleft())

    warnings: list[str] = []
    if embedding_indices:
        missing = len(match_ids) - len(embedding_indices)
        if missing:
            warnings.append(
                f"dropped {missing} embedding rows without matching window keys"
            )
        return embedding_indices, window_indices, "match_id_frame_t", warnings

    n = min(len(match_ids), len(windows.match_id))
    if n == 0:
        raise ValueError("No embedding rows or window rows are available for probes.")
    warnings.append(
        "no exact match_id/frame_t alignment was possible; falling back to index order"
    )
    return list(range(n)), list(range(n)), "index_order", warnings


def _subset_windows(windows: TrackingWindowTensorData, indices: list[int]) -> TrackingWindowTensorData:
    return TrackingWindowTensorData(
        past=windows.past[indices],
        future_xy=windows.future_xy[indices],
        past_mask=windows.past_mask[indices],
        future_mask=windows.future_mask[indices],
        entity_type=windows.entity_type[indices],
        team_id=windows.team_id[indices],
        match_id=[windows.match_id[idx] for idx in indices],
        start_frame=[windows.start_frame[idx] for idx in indices],
        label_frame=[windows.label_frame[idx] for idx in indices],
        phase=[windows.phase[idx] for idx in indices],
        event_type=[windows.event_type[idx] for idx in indices],
        possession_team_id=[windows.possession_team_id[idx] for idx in indices],
        possession_available=[windows.possession_available[idx] for idx in indices],
        feature_names=list(windows.feature_names),
        fps=windows.fps,
        context_seconds=windows.context_seconds,
        horizon_seconds=windows.horizon_seconds,
        stride_seconds=windows.stride_seconds,
        coordinate_mode=windows.coordinate_mode,
    )


def split_probe_indices_by_match(
    match_ids: list[str],
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 123,
) -> tuple[dict[str, Any], list[str]]:
    """Create match-based probe splits, disjoint when at least three matches exist."""

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
        raise ValueError("Cannot split an empty probe dataset.")

    split_indices = {
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
    }
    return split_indices, warnings


def build_probe_dataset(
    embeddings_path: str | Path,
    windows_path: str | Path,
    target_names: list[str],
    out: str | Path | None = None,
    seed: int = 123,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> ProbeDatasetData:
    """Build a saved probe dataset from TD-JEPA embeddings and window tensors."""

    embeddings = torch.load(Path(embeddings_path), map_location="cpu", weights_only=False)
    windows = load_windows_pt(windows_path)
    z = torch.as_tensor(embeddings["z"]).float()
    if z.ndim != 2:
        raise ValueError(f"Expected embeddings z to be rank-2, got shape {tuple(z.shape)}")
    embedding_indices, window_indices, alignment, alignment_warnings = _align_embeddings_to_windows(
        embeddings,
        windows,
    )
    aligned_windows = _subset_windows(windows, window_indices)
    z_aligned = z[embedding_indices]
    raw_features, raw_feature_names = raw_state_summary_features(aligned_windows)
    derived = derive_probe_targets(aligned_windows, target_names)

    match_ids = [str(aligned_windows.match_id[idx]) for idx in range(len(aligned_windows.match_id))]
    frame_t = [int(aligned_windows.start_frame[idx]) for idx in range(len(aligned_windows.start_frame))]
    split_payload, split_warnings = split_probe_indices_by_match(
        match_ids,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    source_splits_all = _source_split_values(
        embeddings.get("source_split", "unknown"),
        int(z.shape[0]),
    )
    source_splits = [source_splits_all[idx] for idx in embedding_indices]
    warnings = alignment_warnings + split_warnings + derived.warnings
    examples = {
        "z": z_aligned,
        "raw_state_summary": raw_features,
        "match_id": match_ids,
        "frame_t": torch.tensor(frame_t, dtype=torch.long),
        "source_split": source_splits,
        "targets": derived.targets,
        "target_masks": derived.masks,
    }
    metadata = {
        "created_by": "build_probe_dataset.py",
        "source_embeddings": str(embeddings_path),
        "source_windows": str(windows_path),
        "alignment": alignment,
        "num_embedding_rows": int(z.shape[0]),
        "num_window_rows": int(len(windows.match_id)),
        "num_examples": int(z_aligned.shape[0]),
        "feature_dim": int(z_aligned.shape[1]),
        "raw_state_summary_dim": int(raw_features.shape[1]),
        "raw_state_summary_names": raw_feature_names,
        "targets": sorted(derived.targets),
        "requested_targets": list(target_names),
        "skipped_targets": [
            name for name in target_names if name not in derived.targets
        ],
        "target_types": derived.target_types,
        "warnings": warnings,
        "encoder_frozen": True,
    }
    data = ProbeDatasetData(
        metadata=metadata,
        examples=examples,
        label_maps=derived.label_maps,
        splits=split_payload,
    )
    if out is not None:
        save_probe_dataset(data, out)
    return data
