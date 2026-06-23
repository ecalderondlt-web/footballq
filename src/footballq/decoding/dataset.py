"""Decoder dataset construction from TD-JEPA embeddings and tracking windows."""

from __future__ import annotations

import random
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from footballq.data.windows import TrackingWindowTensorData, load_windows_pt
from footballq.models.constant_velocity import predict_constant_velocity

DECODER_MODES = {
    "reconstruct_current",
    "reconstruct_current_from_context",
    "reconstruct_current_from_z_context",
    "future_from_z",
    "future_from_context",
    "future_from_past_context",
    "future_from_z_past_context",
    "residual_future_from_z",
    "residual_future_from_past_context",
    "residual_future_from_z_past_context",
    "rollout_from_latents",
}


@dataclass
class DecoderDatasetData:
    """Tensor payload for Experiment 4C coordinate decoding."""

    metadata: dict[str, Any]
    examples: dict[str, Any]
    splits: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DecoderDatasetData":
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
        return int(torch.as_tensor(self.examples["z"]).shape[0])

    @property
    def latent_dim(self) -> int:
        return int(torch.as_tensor(self.examples["z"]).shape[-1])

    @property
    def n_entities(self) -> int:
        return int(torch.as_tensor(self.examples["entity_type"]).shape[-1])

    @property
    def horizon_steps(self) -> int:
        return int(torch.as_tensor(self.examples["future_xy"]).shape[1])

    @property
    def context_z_steps(self) -> int:
        return int(torch.as_tensor(self.examples["z_context"]).shape[1])

    @property
    def rollout_steps(self) -> int:
        return int(torch.as_tensor(self.examples["z_rollout"]).shape[1])

    @property
    def past_context_dim(self) -> int:
        return int(torch.as_tensor(self.examples["past_context"]).shape[-1])


class DecoderDataset(Dataset):
    """PyTorch dataset for one decoder input/target mode."""

    def __init__(
        self,
        data: DecoderDatasetData,
        mode: str,
        split: str | None = None,
        indices: list[int] | None = None,
    ) -> None:
        if mode not in DECODER_MODES:
            raise ValueError(f"Unknown decoder mode {mode!r}. Expected one of {sorted(DECODER_MODES)}")
        self.data = data
        self.mode = mode
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
        examples = self.data.examples
        if self.mode == "reconstruct_current":
            x = examples["z"][row]
            target_xy = examples["current_xy"][row]
            target_mask = examples["current_mask"][row]
        elif self.mode == "reconstruct_current_from_context":
            x = examples["past_context"][row]
            target_xy = examples["current_xy"][row]
            target_mask = examples["current_mask"][row]
        elif self.mode == "reconstruct_current_from_z_context":
            x = examples["z_past_context"][row]
            target_xy = examples["current_xy"][row]
            target_mask = examples["current_mask"][row]
        elif self.mode == "future_from_z":
            x = examples["z"][row]
            target_xy = examples["future_xy"][row]
            target_mask = examples["future_mask"][row]
        elif self.mode == "future_from_context":
            x = examples["z_context"][row]
            target_xy = examples["future_xy"][row]
            target_mask = examples["future_mask"][row]
        elif self.mode == "future_from_past_context":
            x = examples["past_context"][row]
            target_xy = examples["future_xy"][row]
            target_mask = examples["future_mask"][row]
        elif self.mode == "future_from_z_past_context":
            x = examples["z_past_context"][row]
            target_xy = examples["future_xy"][row]
            target_mask = examples["future_mask"][row]
        elif self.mode in {
            "residual_future_from_z",
            "residual_future_from_past_context",
            "residual_future_from_z_past_context",
        }:
            if self.mode == "residual_future_from_z":
                x = examples["z"][row]
            elif self.mode == "residual_future_from_past_context":
                x = examples["past_context"][row]
            else:
                x = examples["z_past_context"][row]
            target_xy = examples["future_xy"][row]
            target_mask = examples["future_mask"][row]
        else:
            steps = self.data.rollout_steps
            x = examples["z_rollout"][row]
            target_xy = examples["future_xy"][row, :steps]
            target_mask = examples["future_mask"][row, :steps]
        return {
            "x": x.float(),
            "target_xy": target_xy.float(),
            "target_mask": target_mask.bool(),
            "current_xy": examples["current_xy"][row].float(),
            "current_mask": examples["current_mask"][row].bool(),
            "future_xy": examples["future_xy"][row].float(),
            "future_mask": examples["future_mask"][row].bool(),
            "z": examples["z"][row].float(),
            "z_context": examples["z_context"][row].float(),
            "z_context_mask": examples["z_context_mask"][row].bool(),
            "past_context": examples["past_context"][row].float(),
            "z_past_context": examples["z_past_context"][row].float(),
            "z_rollout": examples["z_rollout"][row].float(),
            "z_rollout_mask": examples["z_rollout_mask"][row].bool(),
            "coordinate_baseline_xy": examples["coordinate_baseline_xy"][row].float(),
            "last_position_xy": examples["last_position_xy"][row].float(),
            "past": examples["past"][row].float(),
            "past_mask": examples["past_mask"][row].bool(),
            "entity_type": examples["entity_type"][row].long(),
            "team_id": examples["team_id"][row].long(),
            "match_id": examples["match_id"][row],
            "frame_t": examples["frame_t"][row],
            "label_frame": examples["label_frame"][row],
            "start_frame": examples["start_frame"][row],
            "source_split": examples["source_split"][row],
            "phase": examples["phase"][row],
            "possession_team_id": examples["possession_team_id"][row],
            "possession_available": bool(examples["possession_available"][row]),
        }


def save_decoder_dataset(data: DecoderDatasetData, out: str | Path) -> Path:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data.to_dict(), out_path)
    return out_path


def load_decoder_dataset(path: str | Path) -> DecoderDatasetData:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return DecoderDatasetData.from_dict(payload)


def subset_decoder_dataset(
    data: DecoderDatasetData,
    indices: list[int],
    seed: int = 123,
) -> DecoderDatasetData:
    """Return a re-indexed decoder dataset subset with fresh match-level splits."""

    indices = [int(value) for value in indices]
    examples: dict[str, Any] = {}
    for key, value in data.examples.items():
        if isinstance(value, torch.Tensor):
            examples[key] = value[indices]
        elif isinstance(value, list):
            examples[key] = [value[idx] for idx in indices]
        else:
            examples[key] = value
    match_ids = [str(value) for value in examples["match_id"]]
    splits, split_warnings = split_decoder_indices_by_match(match_ids, seed=seed)
    metadata = dict(data.metadata)
    metadata["num_examples"] = len(indices)
    metadata["num_matches"] = len(set(match_ids))
    metadata["subset_warnings"] = split_warnings
    metadata["warnings"] = list(metadata.get("warnings", [])) + split_warnings
    return DecoderDatasetData(metadata=metadata, examples=examples, splits=splits)


def decoder_split_diagnostics(data: DecoderDatasetData) -> dict[str, Any]:
    """Return split sizes, match IDs, and disjointness diagnostics."""

    train = set(str(value) for value in data.splits.get("train_match_ids", []))
    val = set(str(value) for value in data.splits.get("val_match_ids", []))
    test = set(str(value) for value in data.splits.get("test_match_ids", []))
    disjoint = train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    num_matches = len(set(str(value) for value in data.examples.get("match_id", [])))
    return {
        "num_matches": num_matches,
        "num_examples": data.num_examples,
        "num_train_examples": len(data.splits.get("train_indices", [])),
        "num_val_examples": len(data.splits.get("val_indices", [])),
        "num_test_examples": len(data.splits.get("test_indices", [])),
        "train_match_ids": sorted(train),
        "val_match_ids": sorted(val),
        "test_match_ids": sorted(test),
        "disjoint_match_split": bool(disjoint),
        "smoke_split": bool(num_matches < 3 or not disjoint),
    }


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
    for idx, (match_id, start_frame) in enumerate(
        zip(windows.match_id, windows.start_frame, strict=True)
    ):
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
            warnings.append(f"dropped {missing} embedding rows without matching window keys")
        return embedding_indices, window_indices, "match_id_frame_t", warnings

    n = min(len(match_ids), len(windows.match_id))
    if n == 0:
        raise ValueError("No embedding rows or window rows are available for decoder building.")
    warnings.append("no exact match_id/frame_t alignment was possible; falling back to index order")
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


def split_decoder_indices_by_match(
    match_ids: list[str],
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 123,
) -> tuple[dict[str, Any], list[str]]:
    """Create match-level decoder splits, disjoint when at least three matches exist."""

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
        raise ValueError("Cannot split an empty decoder dataset.")

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


def _sequence_context(
    z: torch.Tensor,
    match_ids: list[str],
    frame_t: list[int],
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.zeros((len(match_ids), steps, z.shape[1]), dtype=z.dtype)
    mask = torch.zeros((len(match_ids), steps), dtype=torch.bool)
    by_match: dict[str, list[int]] = defaultdict(list)
    for idx, match_id in enumerate(match_ids):
        by_match[str(match_id)].append(idx)
    for indices in by_match.values():
        ordered = sorted(indices, key=lambda idx: (int(frame_t[idx]), idx))
        first = ordered[0]
        for pos, idx in enumerate(ordered):
            selected = ordered[max(0, pos - steps + 1) : pos + 1]
            pad = [first] * (steps - len(selected))
            full = pad + selected
            out[idx] = z[full]
            mask[idx, steps - len(selected) :] = True
    return out, mask


def _sequence_rollout(
    z: torch.Tensor,
    match_ids: list[str],
    frame_t: list[int],
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.zeros((len(match_ids), steps, z.shape[1]), dtype=z.dtype)
    mask = torch.zeros((len(match_ids), steps), dtype=torch.bool)
    by_match: dict[str, list[int]] = defaultdict(list)
    for idx, match_id in enumerate(match_ids):
        by_match[str(match_id)].append(idx)
    for indices in by_match.values():
        ordered = sorted(indices, key=lambda idx: (int(frame_t[idx]), idx))
        for pos, idx in enumerate(ordered):
            selected = ordered[pos + 1 : pos + 1 + steps]
            if not selected:
                selected = [idx]
            pad_value = selected[-1]
            full = selected + [pad_value] * (steps - len(selected))
            out[idx] = z[full]
            mask[idx, : len(selected)] = True
    return out, mask


def build_decoder_dataset(
    embeddings_path: str | Path,
    windows_path: str | Path,
    out: str | Path | None = None,
    horizon_steps: int | None = None,
    context_z_steps: int = 5,
    rollout_steps: int | None = None,
    seed: int = 123,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> DecoderDatasetData:
    """Build a saved coordinate-decoder dataset from embeddings and windows."""

    embeddings_path = Path(embeddings_path)
    windows_path = Path(windows_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"TD-JEPA embeddings not found: {embeddings_path}. "
            "Run scripts/export_td_embeddings.py with --split all first."
        )
    if not windows_path.exists():
        raise FileNotFoundError(
            f"Tracking windows not found: {windows_path}. "
            "Run scripts/prepare_tracking_data.py first."
        )
    embeddings = torch.load(embeddings_path, map_location="cpu", weights_only=False)
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
    horizon = int(horizon_steps or aligned_windows.horizon_steps)
    horizon = min(horizon, aligned_windows.horizon_steps)
    rollout = int(rollout_steps or min(5, horizon))
    rollout = max(1, min(rollout, horizon))
    context_z_steps = max(1, int(context_z_steps))

    match_ids = [str(value) for value in aligned_windows.match_id]
    frame_t = [int(value) for value in aligned_windows.start_frame]
    z_context, z_context_mask = _sequence_context(
        z_aligned,
        match_ids,
        frame_t,
        context_z_steps,
    )
    z_rollout, z_rollout_mask = _sequence_rollout(
        z_aligned,
        match_ids,
        frame_t,
        rollout,
    )
    past_context = torch.cat(
        [
            aligned_windows.past.flatten(start_dim=1),
            aligned_windows.past_mask.float().flatten(start_dim=1),
        ],
        dim=1,
    ).float()
    z_past_context = torch.cat([z_aligned.float(), past_context], dim=1)
    coordinate_baseline_xy = predict_constant_velocity(
        aligned_windows.past.float(),
        aligned_windows.past_mask.bool(),
        horizon_steps=horizon,
        dt=1.0 / float(aligned_windows.fps),
        feature_names=list(aligned_windows.feature_names),
    ).float()
    last_position_xy = aligned_windows.past[:, -1:, :, :2].expand(-1, horizon, -1, -1).float()
    split_payload, split_warnings = split_decoder_indices_by_match(
        match_ids,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    source_splits_all = _source_split_values(embeddings.get("source_split", "unknown"), int(z.shape[0]))
    source_splits = [source_splits_all[idx] for idx in embedding_indices]
    warnings = alignment_warnings + split_warnings
    examples = {
        "z": z_aligned.float(),
        "z_context": z_context.float(),
        "z_context_mask": z_context_mask,
        "past_context": past_context,
        "z_past_context": z_past_context,
        "z_rollout": z_rollout.float(),
        "z_rollout_mask": z_rollout_mask,
        "current_xy": aligned_windows.past[:, -1, :, :2].float(),
        "current_mask": aligned_windows.past_mask[:, -1].bool(),
        "future_xy": aligned_windows.future_xy[:, :horizon].float(),
        "future_mask": aligned_windows.future_mask[:, :horizon].bool(),
        "coordinate_baseline_xy": coordinate_baseline_xy,
        "last_position_xy": last_position_xy,
        "past": aligned_windows.past.float(),
        "past_mask": aligned_windows.past_mask.bool(),
        "entity_type": aligned_windows.entity_type.long(),
        "team_id": aligned_windows.team_id.long(),
        "match_id": match_ids,
        "frame_t": torch.tensor(frame_t, dtype=torch.long),
        "start_frame": torch.tensor(aligned_windows.start_frame, dtype=torch.long),
        "label_frame": torch.tensor(aligned_windows.label_frame, dtype=torch.long),
        "source_split": source_splits,
        "phase": [str(value) for value in aligned_windows.phase],
        "event_type": [str(value) for value in aligned_windows.event_type],
        "possession_team_id": [str(value) for value in aligned_windows.possession_team_id],
        "possession_available": [bool(value) for value in aligned_windows.possession_available],
    }
    metadata = {
        "created_by": "build_decoder_dataset.py",
        "source_embeddings": str(embeddings_path),
        "source_windows": str(windows_path),
        "alignment": alignment,
        "num_embedding_rows": int(z.shape[0]),
        "num_window_rows": int(len(windows.match_id)),
        "num_examples": int(z_aligned.shape[0]),
        "latent_dim": int(z_aligned.shape[1]),
        "past_context_dim": int(past_context.shape[1]),
        "z_past_context_dim": int(z_past_context.shape[1]),
        "n_entities": int(aligned_windows.n_entities),
        "horizon_steps": horizon,
        "context_z_steps": context_z_steps,
        "rollout_steps": rollout,
        "history_steps": int(aligned_windows.history_steps),
        "fps": float(aligned_windows.fps),
        "feature_names": list(aligned_windows.feature_names),
        "coordinate_mode": str(aligned_windows.coordinate_mode),
        "warnings": warnings,
        "encoder_frozen": True,
    }
    data = DecoderDatasetData(metadata=metadata, examples=examples, splits=split_payload)
    if out is not None:
        save_decoder_dataset(data, out)
    return data
