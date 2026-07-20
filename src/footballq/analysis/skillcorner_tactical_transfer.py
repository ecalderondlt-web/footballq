"""Leakage-controlled SkillCorner tactical transfer benchmark."""

from __future__ import annotations

import bisect
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from footballq.models.soccer_state_encoder import SoccerStateEncoder
from footballq.repro.manifest import file_sha256
from footballq.repro.splits import load_split_manifest

TARGET_TURNOVER = "turnover_within_5s"
TARGET_PENALTY_ENTRY = "penalty_area_entry_within_5s"
TARGETS = (TARGET_TURNOVER, TARGET_PENALTY_ENTRY)

PHASE_COLUMNS = [
    "index",
    "match_id",
    "frame_start",
    "frame_end",
    "duration",
    "period",
    "team_possession_loss_in_phase",
    "penalty_area_start",
]
EVENT_COLUMNS = [
    "phase_index",
    "period",
    "frame_start",
    "penalty_area_start",
    "penalty_area_end",
]


@dataclass
class TacticalExamples:
    """Small causal subset aligned from phase labels to tracking contexts."""

    state: torch.Tensor
    mask: torch.Tensor
    raw_flat: torch.Tensor
    labels: dict[str, torch.Tensor]
    label_masks: dict[str, torch.Tensor]
    match_id: list[str]
    period: list[int]
    phase_index: list[int]
    phase_start_frame: list[int]
    context_end_frame: list[int]
    source_sample_index: list[int]
    split_indices: dict[str, list[int]]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "mask": self.mask,
            "raw_flat": self.raw_flat,
            "labels": self.labels,
            "label_masks": self.label_masks,
            "match_id": self.match_id,
            "period": self.period,
            "phase_index": self.phase_index,
            "phase_start_frame": self.phase_start_frame,
            "context_end_frame": self.context_end_frame,
            "source_sample_index": self.source_sample_index,
            "split_indices": self.split_indices,
            "metadata": self.metadata,
        }


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return dict(yaml.safe_load(handle))


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _phase_paths(raw_root: Path, match_id: str) -> tuple[Path, Path]:
    match_dir = raw_root / str(match_id)
    return (
        match_dir / f"{match_id}_phases_of_play.csv",
        match_dir / f"{match_id}_dynamic_events.csv",
    )


def source_file_inventory(raw_root: Path, match_ids: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match_id in match_ids:
        for kind, path in zip(
            ("phases_of_play", "dynamic_events"),
            _phase_paths(raw_root, match_id),
            strict=True,
        ):
            if not path.exists():
                raise FileNotFoundError(f"Missing SkillCorner {kind} file: {path}")
            rows.append(
                {
                    "match_id": str(match_id),
                    "kind": kind,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    return rows


def _penalty_entry_frames(events: pd.DataFrame) -> dict[tuple[int, int], list[int]]:
    period = pd.to_numeric(events["period"], errors="coerce")
    phase_index = pd.to_numeric(events["phase_index"], errors="coerce")
    frame_start = pd.to_numeric(events["frame_start"], errors="coerce")
    penalty = events["penalty_area_start"].map(_as_bool) | events[
        "penalty_area_end"
    ].map(_as_bool)
    valid = period.notna() & phase_index.notna() & frame_start.notna() & penalty
    out: dict[tuple[int, int], list[int]] = defaultdict(list)
    for per, phase, frame in zip(
        period[valid], phase_index[valid], frame_start[valid], strict=True
    ):
        key = (int(per), int(phase))
        out[key].append(int(frame))
    return {key: sorted(set(frames)) for key, frames in out.items()}


def _split_map(split: Any) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name, values in (
        ("train", split.train_match_ids),
        ("val", split.val_match_ids),
        ("test", split.test_match_ids),
    ):
        for value in values:
            mapping[str(value)] = name
    return mapping


def _td_lookup(
    payload: dict[str, Any], included_match_ids: set[str]
) -> dict[tuple[str, int], tuple[list[int], list[int]]]:
    grouped: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    context_frames = torch.as_tensor(payload["context_frame_indices"])
    for idx, (match_id, period) in enumerate(
        zip(payload["match_id"], payload["period"], strict=True)
    ):
        match_text = str(match_id)
        if match_text not in included_match_ids:
            continue
        context_end = int(context_frames[idx, -1].item())
        grouped[(match_text, int(period))].append((context_end, idx))
    lookup: dict[tuple[str, int], tuple[list[int], list[int]]] = {}
    for key, rows in grouped.items():
        rows.sort()
        lookup[key] = ([row[0] for row in rows], [row[1] for row in rows])
    return lookup


def align_phase_start(
    ends: list[int], indices: list[int], phase_start: int, max_gap_frames: int
) -> tuple[int, int] | None:
    """Return the nearest context ending strictly before a phase starts."""

    position = bisect.bisect_left(ends, int(phase_start)) - 1
    if position < 0:
        return None
    context_end = int(ends[position])
    gap = int(phase_start) - context_end
    if gap < 1 or gap > int(max_gap_frames):
        return None
    return int(indices[position]), context_end


def _raw_flat_features(state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = torch.nan_to_num(state) * mask.unsqueeze(-1).to(state.dtype)
    return torch.cat(
        [masked.flatten(start_dim=1), mask.to(state.dtype).flatten(start_dim=1)], dim=1
    ).float()


def build_tactical_examples(
    config: dict[str, Any],
    *,
    workspace_root: str | Path,
    include_splits: tuple[str, ...] = ("train", "val", "test"),
) -> TacticalExamples:
    """Build phase-start examples without exposing phase or possession fields to the encoder."""

    root = Path(workspace_root)
    data_cfg = dict(config["data"])
    raw_root = _resolve_path(root, data_cfg["raw_root"])
    td_path = _resolve_path(root, data_cfg["td_path"])
    split_path = _resolve_path(root, data_cfg["split_manifest"])
    split = load_split_manifest(split_path)
    split_for_match = _split_map(split)
    included_match_ids = {
        match_id
        for match_id, split_name in split_for_match.items()
        if split_name in set(include_splits)
    }

    payload = torch.load(td_path, map_location="cpu", weights_only=False, mmap=True)
    td_meta = dict(payload.get("metadata", {}))
    if td_meta.get("split_manifest_sha256") != split.sha256:
        raise ValueError("TD tensor and requested SkillCorner split manifest do not match.")
    if str(payload.get("objective_mode")) != "future_nonoverlap_context_only":
        raise ValueError("Tactical transfer requires a non-overlapping TD source tensor.")

    source_feature_names = [str(value) for value in payload["feature_names"]]
    input_feature_names = [str(value) for value in data_cfg["input_feature_names"]]
    missing = [name for name in input_feature_names if name not in source_feature_names]
    if missing:
        raise ValueError(f"TD tensor is missing required position-only features: {missing}")
    feature_indices = [source_feature_names.index(name) for name in input_feature_names]
    if input_feature_names != ["x_norm", "y_norm", "is_ball", "is_home", "is_away"]:
        raise ValueError("This benchmark is frozen to the five-feature position-only view.")

    lookup = _td_lookup(payload, included_match_ids)
    max_gap = int(data_cfg["max_alignment_gap_frames"])
    min_visible = int(data_cfg["min_visible_entities_at_anchor"])
    require_ball = bool(data_cfg.get("require_ball_visible_at_anchor", True))
    fps = float(payload["fps"])
    horizon_seconds = float(data_cfg["prediction_horizon_seconds"])
    horizon_frames = int(round(fps * horizon_seconds))

    rows: list[dict[str, Any]] = []
    audit_by_match: dict[str, dict[str, int]] = {}
    used_source_indices: set[int] = set()
    mask_source = torch.as_tensor(payload["mask_t"])
    ordered_ids = [value for value in split.all_match_ids if value in included_match_ids]
    for match_id in ordered_ids:
        phases_path, events_path = _phase_paths(raw_root, match_id)
        if not phases_path.exists() or not events_path.exists():
            raise FileNotFoundError(f"Missing tactical labels for SkillCorner match {match_id}.")
        phases = pd.read_csv(phases_path, usecols=PHASE_COLUMNS)
        events = pd.read_csv(events_path, usecols=EVENT_COLUMNS, low_memory=False)
        penalty_frames = _penalty_entry_frames(events)
        audit = {
            "phase_rows": int(len(phases)),
            "aligned_rows": 0,
            "alignment_drops": 0,
            "duplicate_drops": 0,
            "visibility_drops": 0,
        }
        for phase in phases.sort_values(["period", "frame_start"]).itertuples(index=False):
            period = int(phase.period)
            phase_start = int(phase.frame_start)
            key = (str(match_id), period)
            candidate = lookup.get(key)
            if candidate is None:
                audit["alignment_drops"] += 1
                continue
            aligned = align_phase_start(candidate[0], candidate[1], phase_start, max_gap)
            if aligned is None:
                audit["alignment_drops"] += 1
                continue
            source_idx, context_end = aligned
            if source_idx in used_source_indices:
                audit["duplicate_drops"] += 1
                continue
            anchor_mask = mask_source[source_idx, -1]
            visible_count = int(anchor_mask.sum().item())
            if visible_count < min_visible or (require_ball and not bool(anchor_mask[0].item())):
                audit["visibility_drops"] += 1
                continue

            phase_index = int(phase.index)
            phase_end = int(phase.frame_end)
            turnover = _as_bool(phase.team_possession_loss_in_phase) and (
                phase_end <= phase_start + horizon_frames
            )
            starts_in_penalty_area = _as_bool(phase.penalty_area_start)
            candidate_frames = penalty_frames.get((period, phase_index), [])
            penalty_position = bisect.bisect_left(candidate_frames, phase_start)
            penalty_frame = (
                candidate_frames[penalty_position]
                if penalty_position < len(candidate_frames)
                else None
            )
            penalty_entry = bool(
                not starts_in_penalty_area
                and penalty_frame is not None
                and phase_start <= penalty_frame <= phase_start + horizon_frames
            )
            used_source_indices.add(source_idx)
            rows.append(
                {
                    "match_id": str(match_id),
                    "split": split_for_match[str(match_id)],
                    "period": period,
                    "phase_index": phase_index,
                    "phase_start": phase_start,
                    "context_end": context_end,
                    "source_idx": source_idx,
                    TARGET_TURNOVER: int(turnover),
                    TARGET_PENALTY_ENTRY: int(penalty_entry),
                    f"{TARGET_PENALTY_ENTRY}_valid": int(not starts_in_penalty_area),
                }
            )
            audit["aligned_rows"] += 1
        audit_by_match[str(match_id)] = audit

    if not rows:
        raise ValueError("No SkillCorner phases aligned to causal tracking contexts.")
    source_indices = [int(row["source_idx"]) for row in rows]
    state = torch.as_tensor(payload["state_t"])[source_indices][..., feature_indices].contiguous()
    mask = mask_source[source_indices].contiguous().bool()
    raw_flat = _raw_flat_features(state, mask)
    match_ids = [str(row["match_id"]) for row in rows]
    split_indices = {
        name: [idx for idx, row in enumerate(rows) if row["split"] == name]
        for name in ("train", "val", "test")
    }
    labels = {
        target: torch.tensor([int(row[target]) for row in rows], dtype=torch.long)
        for target in TARGETS
    }
    label_masks = {
        TARGET_TURNOVER: torch.ones(len(rows), dtype=torch.bool),
        TARGET_PENALTY_ENTRY: torch.tensor(
            [bool(row[f"{TARGET_PENALTY_ENTRY}_valid"]) for row in rows],
            dtype=torch.bool,
        ),
    }
    metadata = {
        "split_manifest_path": str(split_path),
        "split_manifest_sha256": split.sha256,
        "td_path": str(td_path),
        "td_feature_names": source_feature_names,
        "input_feature_names": input_feature_names,
        "feature_view": "position_only",
        "objective_mode": str(payload["objective_mode"]),
        "context_rule": "context_end_frame < phase_start_frame",
        "max_alignment_gap_frames": max_gap,
        "prediction_horizon_seconds": horizon_seconds,
        "included_splits": list(include_splits),
        "audit_by_match": audit_by_match,
    }
    return TacticalExamples(
        state=state,
        mask=mask,
        raw_flat=raw_flat,
        labels=labels,
        label_masks=label_masks,
        match_id=match_ids,
        period=[int(row["period"]) for row in rows],
        phase_index=[int(row["phase_index"]) for row in rows],
        phase_start_frame=[int(row["phase_start"]) for row in rows],
        context_end_frame=[int(row["context_end"]) for row in rows],
        source_sample_index=source_indices,
        split_indices=split_indices,
        metadata=metadata,
    )


def support_summary(examples: TacticalExamples) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split_name, indices in examples.split_indices.items():
        split_out: dict[str, Any] = {"examples": len(indices)}
        index_tensor = torch.tensor(indices, dtype=torch.long)
        for target in TARGETS:
            if not indices:
                split_out[target] = {"valid": 0, "positive": 0, "negative": 0}
                continue
            valid = examples.label_masks[target][index_tensor]
            values = examples.labels[target][index_tensor][valid]
            positives = int(values.sum().item())
            split_out[target] = {
                "valid": int(values.numel()),
                "positive": positives,
                "negative": int(values.numel()) - positives,
            }
        out[split_name] = split_out
    return out


def validate_preflight(examples: TacticalExamples, config: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    gates = dict(config["gates"])
    support = support_summary(examples)
    minimum_examples = int(gates["minimum_examples_per_split"])
    minimum_positives = int(gates["minimum_positive_examples_per_target_split"])
    for split_name in examples.metadata["included_splits"]:
        if support[split_name]["examples"] < minimum_examples:
            failures.append(f"{split_name} has fewer than {minimum_examples} examples")
        for target in TARGETS:
            target_support = support[split_name][target]
            if target_support["positive"] < minimum_positives:
                failures.append(
                    f"{split_name}/{target} has fewer than {minimum_positives} positives"
                )
            if target_support["negative"] < minimum_positives:
                failures.append(
                    f"{split_name}/{target} has fewer than {minimum_positives} negatives"
                )
    if any(
        end >= start
        for end, start in zip(
            examples.context_end_frame, examples.phase_start_frame, strict=True
        )
    ):
        failures.append("at least one context reaches or crosses its phase start")
    if len(examples.source_sample_index) != len(set(examples.source_sample_index)):
        failures.append("aligned phase examples reuse a TD source sample")
    return failures


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def encode_checkpoint(
    checkpoint_path: Path,
    expected_sha256: str,
    examples: TacticalExamples,
    *,
    device: str = "auto",
    batch_size: int = 256,
) -> tuple[torch.Tensor, dict[str, Any]]:
    actual_sha256 = file_sha256(checkpoint_path)
    if actual_sha256 != str(expected_sha256):
        raise ValueError(
            f"Checkpoint hash mismatch for {checkpoint_path}: {actual_sha256}"
        )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    data_meta = dict(payload.get("data_meta", {}))
    expected_features = examples.metadata["input_feature_names"]
    if list(data_meta.get("feature_names", [])) != expected_features:
        raise ValueError(f"Checkpoint {checkpoint_path} is not position-only compatible.")
    if int(data_meta.get("context_steps", -1)) != int(examples.state.shape[1]):
        raise ValueError(f"Checkpoint {checkpoint_path} has a different context length.")
    if int(data_meta.get("n_entities", -1)) != int(examples.state.shape[2]):
        raise ValueError(f"Checkpoint {checkpoint_path} has a different entity count.")

    model_cfg = dict(payload["config"].get("model", {}))
    encoder = SoccerStateEncoder(
        context_steps=int(examples.state.shape[1]),
        n_entities=int(examples.state.shape[2]),
        n_features=int(examples.state.shape[3]),
        z_dim=int(model_cfg.get("z_dim", 128)),
        d_model=int(model_cfg.get("d_model", 128)),
        n_heads=int(model_cfg.get("n_heads", 4)),
        n_layers=int(model_cfg.get("n_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        pooling=str(model_cfg.get("pooling", "mean")),
    )
    encoder.load_state_dict(payload["online_encoder"], strict=True)
    torch_device = _device(device)
    encoder = encoder.to(torch_device).eval()
    parts: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(examples.match_id), int(batch_size)):
            stop = min(start + int(batch_size), len(examples.match_id))
            state = examples.state[start:stop].to(torch_device)
            mask = examples.mask[start:stop].to(torch_device)
            parts.append(encoder(state, mask).cpu())
    return torch.cat(parts, dim=0), {
        "path": str(checkpoint_path),
        "sha256": actual_sha256,
        "feature_names": data_meta.get("feature_names"),
        "z_dim": int(parts[0].shape[1]),
    }


def _train_standardize(
    features: torch.Tensor, train_indices: list[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train = features[train_indices].float()
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (features.float() - mean) / std, mean, std


def control_features(
    examples: TacticalExamples, config: dict[str, Any], *, device: str = "auto"
) -> dict[str, torch.Tensor]:
    controls = dict(config["controls"])
    train_indices = examples.split_indices["train"]
    standardized, _mean, _std = _train_standardize(examples.raw_flat, train_indices)
    seed = int(controls["random_seed"])
    pca_dim = min(
        int(controls["pca_dimensions"]),
        len(train_indices) - 1,
        int(standardized.shape[1]),
    )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    compute_device = _device(device)
    train = standardized[train_indices].to(compute_device)
    _u, _s, components = torch.pca_lowrank(
        train, q=pca_dim, center=False, niter=3
    )
    pca = standardized.to(compute_device).matmul(components).cpu()

    projection_dim = int(controls["random_projection_dimensions"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    projection = torch.randn(
        standardized.shape[1], projection_dim, generator=generator
    ) / math.sqrt(float(standardized.shape[1]))
    random_projection = standardized.matmul(projection)
    random_noise = torch.randn(
        len(examples.match_id), projection_dim, generator=generator
    )
    return {
        "raw_flat": examples.raw_flat,
        "raw_pca_128": pca,
        "random_projection_128": random_projection,
        "random_noise_128": random_noise,
    }


def binary_metrics(labels: torch.Tensor, probabilities: torch.Tensor) -> dict[str, Any]:
    labels = labels.long().cpu()
    probabilities = probabilities.float().cpu()
    prediction = (probabilities >= 0.5).long()
    positives = labels == 1
    negatives = labels == 0
    tp = int((prediction.bool() & positives).sum().item())
    tn = int((~prediction.bool() & negatives).sum().item())
    fp = int((prediction.bool() & negatives).sum().item())
    fn = int((~prediction.bool() & positives).sum().item())

    def ratio(numerator: float, denominator: float) -> float | None:
        return float(numerator / denominator) if denominator else None

    recall_positive = ratio(tp, tp + fn)
    recall_negative = ratio(tn, tn + fp)
    f1_positive = ratio(2 * tp, 2 * tp + fp + fn)
    f1_negative = ratio(2 * tn, 2 * tn + fp + fn)
    balanced_accuracy = (
        None
        if recall_positive is None or recall_negative is None
        else (recall_positive + recall_negative) / 2.0
    )
    macro_f1 = (
        None
        if f1_positive is None or f1_negative is None
        else (f1_positive + f1_negative) / 2.0
    )

    average_precision: float | None = None
    auroc: float | None = None
    n_positive = int(positives.sum().item())
    n_negative = int(negatives.sum().item())
    if n_positive and n_negative:
        order = torch.argsort(probabilities, descending=True)
        ordered = labels[order]
        tp_curve = torch.cumsum((ordered == 1).float(), dim=0)
        fp_curve = torch.cumsum((ordered == 0).float(), dim=0)
        precision = tp_curve / (tp_curve + fp_curve).clamp_min(1.0)
        average_precision = float(precision[ordered == 1].mean().item())
        tpr = torch.cat([torch.zeros(1), tp_curve / float(n_positive)])
        fpr = torch.cat([torch.zeros(1), fp_curve / float(n_negative)])
        auroc = float(torch.trapezoid(tpr, fpr).item())
    return {
        "num_examples": int(labels.numel()),
        "positive": n_positive,
        "negative": n_negative,
        "accuracy": ratio(tp + tn, labels.numel()),
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "average_precision": average_precision,
        "auroc": auroc,
        "brier": float(torch.mean((probabilities - labels.float()) ** 2).item()),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def fit_linear_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    label_mask: torch.Tensor,
    split_indices: dict[str, list[int]],
    probe_config: dict[str, Any],
    match_ids: list[str],
    *,
    device: str = "auto",
) -> dict[str, Any]:
    train_indices = [idx for idx in split_indices["train"] if bool(label_mask[idx])]
    standardized, _mean, _std = _train_standardize(features, train_indices)
    torch_device = _device(device)
    x_train = standardized[train_indices].to(torch_device)
    y_train = labels[train_indices].float().to(torch_device)
    positives = float(y_train.sum().item())
    negatives = float(y_train.numel() - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("Linear probe training requires both target classes.")
    pos_weight = torch.tensor(negatives / positives, device=torch_device)
    model = torch.nn.Linear(int(features.shape[1]), 1).to(torch_device)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        max_iter=int(probe_config["max_iterations"]),
        history_size=int(probe_config["history_size"]),
        line_search_fn="strong_wolfe",
    )
    l2_weight = float(probe_config["l2_weight"])

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train).view(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y_train, pos_weight=pos_weight
        )
        loss = loss + 0.5 * l2_weight * model.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    model.eval()
    with torch.inference_mode():
        probabilities = torch.sigmoid(
            model(standardized.to(torch_device)).view(-1)
        ).cpu()
    result: dict[str, Any] = {"splits": {}, "per_test_match": {}}
    for split_name, indices in split_indices.items():
        valid_indices = [idx for idx in indices if bool(label_mask[idx])]
        result["splits"][split_name] = binary_metrics(
            labels[valid_indices], probabilities[valid_indices]
        )
    for match_id in sorted({match_ids[idx] for idx in split_indices["test"]}):
        indices = [
            idx
            for idx in split_indices["test"]
            if match_ids[idx] == match_id and bool(label_mask[idx])
        ]
        result["per_test_match"][match_id] = binary_metrics(
            labels[indices], probabilities[indices]
        )
    return result


def _metric_mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def summarize_families(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["feature_source"], row["target"])].append(row)
    out: list[dict[str, Any]] = []
    for (family, feature_source, target), values in sorted(grouped.items()):
        out.append(
            {
                "family": family,
                "feature_source": feature_source,
                "target": target,
                "runs": len(values),
                "test_macro_f1_mean": _metric_mean(
                    [row["metrics"]["splits"]["test"]["macro_f1"] for row in values]
                ),
                "test_balanced_accuracy_mean": _metric_mean(
                    [
                        row["metrics"]["splits"]["test"]["balanced_accuracy"]
                        for row in values
                    ]
                ),
                "test_average_precision_mean": _metric_mean(
                    [
                        row["metrics"]["splits"]["test"]["average_precision"]
                        for row in values
                    ]
                ),
                "val_macro_f1_mean": _metric_mean(
                    [row["metrics"]["splits"]["val"]["macro_f1"] for row in values]
                ),
            }
        )
    return out


def evaluate_feature_source(
    rows: list[dict[str, Any]],
    *,
    family: str,
    feature_source: str,
    features: torch.Tensor,
    examples: TacticalExamples,
    probe_config: dict[str, Any],
    encoder_seed: int | None,
    device: str,
) -> None:
    for target in TARGETS:
        metrics = fit_linear_probe(
            features,
            examples.labels[target],
            examples.label_masks[target],
            examples.split_indices,
            probe_config,
            examples.match_id,
            device=device,
        )
        rows.append(
            {
                "family": family,
                "feature_source": feature_source,
                "encoder_seed": encoder_seed,
                "target": target,
                "feature_dim": int(features.shape[1]),
                "metrics": metrics,
            }
        )


def decision_gates(
    family_summary: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    lookup = {
        (row["family"], row["feature_source"], row["target"]): row
        for row in family_summary
    }
    material = float(config["gates"]["grf_material_macro_f1_gain"])
    incremental = float(config["gates"]["incremental_macro_f1_gain_over_raw"])
    raw = {
        target: lookup[("control", "raw_flat", target)]["test_macro_f1_mean"]
        for target in TARGETS
    }
    grf_families = ("grf_1x_pff", "grf_4x_pff", "grf_8x_pff")
    grf_gain: dict[str, Any] = {}
    incremental_gain: dict[str, Any] = {}
    for target in TARGETS:
        pff = lookup[("pff_only", "raw_plus_latent", target)]["test_macro_f1_mean"]
        candidates = {
            family: lookup[(family, "raw_plus_latent", target)]["test_macro_f1_mean"]
            for family in grf_families
        }
        best_family = max(candidates, key=lambda name: candidates[name])
        gain = float(candidates[best_family] - pff)
        grf_gain[target] = {
            "pff_only": pff,
            "best_grf_family": best_family,
            "best_grf": candidates[best_family],
            "gain": gain,
            "threshold": material,
            "passed": gain >= material,
        }
        incremental_gain[target] = {}
        for family in ("pff_only", *grf_families):
            value = lookup[(family, "raw_plus_latent", target)]["test_macro_f1_mean"]
            gain_over_raw = float(value - raw[target])
            incremental_gain[target][family] = {
                "gain": gain_over_raw,
                "threshold": incremental,
                "passed": gain_over_raw >= incremental,
            }
    return {
        "grf_material_gain_over_pff_only": grf_gain,
        "incremental_gain_over_raw_geometry": incremental_gain,
        "all_grf_targets_passed": all(value["passed"] for value in grf_gain.values()),
    }


def run_benchmark(
    examples: TacticalExamples,
    config: dict[str, Any],
    *,
    workspace_root: str | Path,
    device: str = "auto",
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    root = Path(workspace_root)
    rows: list[dict[str, Any]] = []
    controls = control_features(examples, config, device=device)
    for name, features in controls.items():
        evaluate_feature_source(
            rows,
            family="control",
            feature_source=name,
            features=features,
            examples=examples,
            probe_config=config["probe"],
            encoder_seed=None,
            device=device,
        )

    embeddings: dict[str, torch.Tensor] = {}
    checkpoint_audit: list[dict[str, Any]] = []
    for family, entries in config["checkpoints"].items():
        for entry in entries:
            seed = int(entry["seed"])
            checkpoint_path = _resolve_path(root, entry["path"])
            latent, audit = encode_checkpoint(
                checkpoint_path,
                entry["sha256"],
                examples,
                device=device,
            )
            key = f"{family}_seed{seed}"
            embeddings[key] = latent
            checkpoint_audit.append({"family": family, "seed": seed, **audit})
            evaluate_feature_source(
                rows,
                family=family,
                feature_source="latent_only",
                features=latent,
                examples=examples,
                probe_config=config["probe"],
                encoder_seed=seed,
                device=device,
            )
            evaluate_feature_source(
                rows,
                family=family,
                feature_source="raw_plus_latent",
                features=torch.cat([examples.raw_flat, latent], dim=1),
                examples=examples,
                probe_config=config["probe"],
                encoder_seed=seed,
                device=device,
            )
    family_summary = summarize_families(rows)
    result = {
        "experiment": config["experiment"],
        "support": support_summary(examples),
        "rows": rows,
        "family_summary": family_summary,
        "decision_gates": decision_gates(family_summary, config),
        "checkpoint_audit": checkpoint_audit,
        "interpretation_limit": (
            "Exploratory transfer evidence from two held-out matches; it cannot establish "
            "broad tactical understanding by itself."
        ),
    }
    return result, embeddings


def json_dump(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path
