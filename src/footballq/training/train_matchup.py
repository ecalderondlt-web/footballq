"""Matched training for the four RLCS identity conditions."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

from footballq.data.rlcs_replay import load_frozen_rlcs_split
from footballq.data.rlcs_touch_windows import (
    N_ENTITIES,
    N_FEATURES,
    N_PLAYERS,
    STATE_MASK_SIZE,
    STATE_SIZE,
    TIME_STEPS,
    load_identity_vocabulary,
    reflect_next_touch_zone,
    reflect_state_x,
)
from footballq.models.identity_matchup_transformer import (
    IDENTITY_CONDITIONS,
    IdentityCondition,
    IdentityMatchupTransformer,
    MatchupLossWeights,
    factorized_joint_nll,
    identity_matchup_loss,
)
from footballq.repro.manifest import build_run_manifest, file_sha256, write_run_manifest
from footballq.training.train import resolve_device


class TestSplitLockedError(PermissionError):
    """Raised whenever ordinary training code attempts to open sealed test data."""


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_preflight(config: Mapping[str, Any], manifest_path: Path) -> None:
    preflight = config.get("preflight", {})
    if not bool(preflight.get("required", False)):
        return
    manifest_hash = file_sha256(manifest_path)
    audit_path = Path(preflight["identity_audit"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not bool(audit.get("all_gates_pass", False)):
        raise RuntimeError("RLCS corpus or power preflight gate has not passed.")
    if audit.get("inputs", {}).get("dataset_manifest_sha256") != manifest_hash:
        raise RuntimeError("RLCS identity audit does not match the current dataset manifest.")
    overfit_path = Path(preflight["overfit_report"])
    overfit = json.loads(overfit_path.read_text(encoding="utf-8"))
    if overfit.get("status") != "passed":
        raise RuntimeError("RLCS 5,000-sample overfit preflight gate has not passed.")
    if overfit.get("dataset_manifest_sha256") != manifest_hash:
        raise RuntimeError("RLCS overfit report does not match the current dataset manifest.")


def _fixed_list_numpy(table: Any, name: str, width: int, dtype: Any) -> np.ndarray:
    array = table[name].combine_chunks()
    values = array.values.to_numpy(zero_copy_only=False)
    output = np.asarray(values, dtype=dtype).reshape(len(array), width)
    return output


class RLCSDecisionDataset(Dataset[dict[str, Any]]):
    """Memory-mapped-style Arrow loader with deterministic train augmentation."""

    def __init__(
        self,
        dataset_manifest_path: str | Path,
        split: str,
        *,
        allow_test: bool = False,
        reflection_probability: float = 0.0,
        augmentation_seed: int = 0,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split {split!r}.")
        if split == "test" and not allow_test:
            raise TestSplitLockedError("Training code may not load the sealed RLCS test split.")
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("RLCS training requires pyarrow.") from exc
        self.manifest_path = Path(dataset_manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        split_info = self.manifest["splits"][split]
        path = Path(split_info["path"])
        if file_sha256(path) != split_info["sha256"]:
            raise ValueError(f"Dataset hash mismatch for split {split!r}.")
        self.table = pq.read_table(path)
        if int(split_info["rows"]) != len(self.table):
            raise ValueError(f"Dataset row count mismatch for split {split!r}.")
        self.split = split
        self.states = _fixed_list_numpy(self.table, "state_flat", STATE_SIZE, np.float32).reshape(
            -1, TIME_STEPS, N_ENTITIES, N_FEATURES
        )
        self.state_masks = _fixed_list_numpy(
            self.table, "state_mask", STATE_MASK_SIZE, np.bool_
        ).reshape(-1, TIME_STEPS, N_ENTITIES)
        self.identity_indices = _fixed_list_numpy(
            self.table, "player_identity_idx", N_PLAYERS, np.int64
        )
        self.known_masks = _fixed_list_numpy(
            self.table, "player_known_mask", N_PLAYERS, np.bool_
        )
        self.seconds_remaining = self.table["seconds_remaining"].to_numpy().astype(np.float32)
        self.score_diff = self.table["score_diff_actor"].to_numpy().astype(np.int64)
        self.overtime = self.table["overtime"].to_numpy().astype(np.bool_)
        self.next_entity = self.table["next_touch_entity"].to_numpy().astype(np.int64)
        self.next_zone = self.table["next_touch_zone"].to_numpy().astype(np.int64)
        self.retained = self.table["retained_possession"].to_numpy().astype(np.bool_)
        self.goal = self.table["goal_for_within_8s"].to_numpy().astype(np.bool_)
        self.sample_ids = self.table["sample_id"].to_pylist()
        self.replay_ids = self.table["replay_id"].to_pylist()
        self.series_ids = self.table["series_id"].to_pylist()
        self.regions = self.table["region"].to_pylist()
        self.group_paths = self.table["group_path"].to_pylist()
        self.event_times = self.table["event_time_utc"].to_pylist()
        self.team_roster_hashes = self.table["team_roster_hash"].to_pylist()
        self.opponent_roster_hashes = self.table["opponent_roster_hash"].to_pylist()
        self.reflection_probability = float(reflection_probability) if split == "train" else 0.0
        self.augmentation_seed = int(augmentation_seed)
        self.epoch = 0
        self.feature_mean = (
            np.zeros(N_FEATURES, dtype=np.float32)
            if feature_mean is None
            else np.asarray(feature_mean, dtype=np.float32)
        )
        self.feature_std = (
            np.ones(N_FEATURES, dtype=np.float32)
            if feature_std is None
            else np.asarray(feature_std, dtype=np.float32)
        )
        if self.feature_mean.shape != (N_FEATURES,) or self.feature_std.shape != (N_FEATURES,):
            raise ValueError("Feature statistics must each have shape [27].")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.feature_mean = np.asarray(mean, dtype=np.float32)
        self.feature_std = np.asarray(std, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.states)

    def _reflect(self, index: int) -> bool:
        if self.reflection_probability <= 0:
            return False
        payload = f"{self.augmentation_seed}:{self.epoch}:{self.sample_ids[index]}".encode()
        value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64
        return value < self.reflection_probability

    def __getitem__(self, index: int) -> dict[str, Any]:
        state = self.states[index]
        reflected = self._reflect(index)
        if reflected:
            state = reflect_state_x(state)
        else:
            state = state.copy()
        state = (state - self.feature_mean) / self.feature_std
        state[~self.state_masks[index]] = 0.0
        return {
            "state": torch.from_numpy(state),
            "state_mask": torch.from_numpy(self.state_masks[index].copy()),
            "identity_indices": torch.from_numpy(self.identity_indices[index].copy()),
            "player_known_mask": torch.from_numpy(self.known_masks[index].copy()),
            "seconds_remaining": torch.tensor(self.seconds_remaining[index]),
            "score_diff_actor": torch.tensor(self.score_diff[index]),
            "overtime": torch.tensor(self.overtime[index]),
            "next_touch_entity": torch.tensor(self.next_entity[index]),
            "next_touch_zone": torch.tensor(
                reflect_next_touch_zone(int(self.next_zone[index]))
                if reflected
                else self.next_zone[index]
            ),
            "retained_possession": torch.tensor(self.retained[index]),
            "goal_for_within_8s": torch.tensor(self.goal[index]),
            "sample_id": str(self.sample_ids[index]),
            "replay_id": str(self.replay_ids[index]),
            "series_id": str(self.series_ids[index]),
            "region": str(self.regions[index]),
            "group_path": str(self.group_paths[index]),
            "event_time_utc": str(self.event_times[index]),
            "team_roster_hash": bytes(self.team_roster_hashes[index]).hex(),
            "opponent_roster_hash": bytes(self.opponent_roster_hashes[index]).hex(),
        }


def compute_train_feature_statistics(dataset: RLCSDecisionDataset) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalization statistics from the train split only."""

    if dataset.split != "train":
        raise ValueError("Feature statistics may be fitted only on the train split.")
    values = dataset.states.astype(np.float64, copy=False)
    valid = dataset.state_masks[..., None]
    count = valid.sum(axis=(0, 1, 2)).astype(np.float64)
    total = np.where(valid, values, 0.0).sum(axis=(0, 1, 2))
    mean = total / np.maximum(count, 1.0)
    centered = np.where(valid, values - mean, 0.0)
    variance = (centered * centered).sum(axis=(0, 1, 2)) / np.maximum(count, 1.0)
    std = np.sqrt(variance)
    std[std < 1e-5] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _model_inputs(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "state": batch["state"],
        "state_mask": batch["state_mask"],
        "identity_indices": batch["identity_indices"],
        "seconds_remaining": batch["seconds_remaining"],
        "score_diff_actor": batch["score_diff_actor"],
        "overtime": batch["overtime"],
    }


def _targets(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: batch[key]
        for key in (
            "next_touch_entity",
            "next_touch_zone",
            "retained_possession",
            "goal_for_within_8s",
        )
    }


def evaluate_matchup_loader(
    model: IdentityMatchupTransformer,
    loader: DataLoader,
    device: torch.device,
    *,
    condition: IdentityCondition,
    precision: str = "bf16",
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Evaluate without retaining predictions or accessing sealed data implicitly."""

    model.eval()
    totals = {
        "joint_nll": 0.0,
        "entity_nll": 0.0,
        "zone_nll": 0.0,
        "entity_correct": 0,
        "zone_correct": 0,
        "retained_correct": 0,
        "goal_correct": 0,
    }
    sample_count = 0
    digest = hashlib.sha256()
    autocast = precision == "bf16" and device.type == "cuda"
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            for sample_id in raw_batch["sample_id"]:
                digest.update(str(sample_id).encode("utf-8") + b"\n")
            batch = _to_device(raw_batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast):
                outputs = model(**_model_inputs(batch), condition=condition)
                targets = _targets(batch)
                joint = factorized_joint_nll(outputs, targets)
                entity = torch.nn.functional.cross_entropy(
                    outputs["next_touch_entity_logits"],
                    targets["next_touch_entity"].long(),
                    reduction="none",
                )
                zone = torch.nn.functional.cross_entropy(
                    outputs["next_touch_zone_logits"],
                    targets["next_touch_zone"].long(),
                    reduction="none",
                )
            batch_size = int(joint.numel())
            sample_count += batch_size
            totals["joint_nll"] += float(joint.float().sum())
            totals["entity_nll"] += float(entity.float().sum())
            totals["zone_nll"] += float(zone.float().sum())
            totals["entity_correct"] += int(
                (
                    outputs["next_touch_entity_logits"].argmax(-1)
                    == targets["next_touch_entity"]
                ).sum()
            )
            totals["zone_correct"] += int(
                (outputs["next_touch_zone_logits"].argmax(-1) == targets["next_touch_zone"]).sum()
            )
            totals["retained_correct"] += int(
                (
                    (outputs["retained_possession_logit"] >= 0)
                    == targets["retained_possession"]
                ).sum()
            )
            totals["goal_correct"] += int(
                ((outputs["goal_within_8s_logit"] >= 0) == targets["goal_for_within_8s"]).sum()
            )
    if sample_count == 0:
        raise ValueError("Evaluation received zero decision samples.")
    return {
        "num_examples": sample_count,
        "sample_id_sha256": digest.hexdigest(),
        "factorized_joint_nll": totals["joint_nll"] / sample_count,
        "next_touch_entity_nll": totals["entity_nll"] / sample_count,
        "next_touch_zone_nll": totals["zone_nll"] / sample_count,
        "next_touch_entity_accuracy": totals["entity_correct"] / sample_count,
        "next_touch_zone_accuracy": totals["zone_correct"] / sample_count,
        "retained_possession_accuracy": totals["retained_correct"] / sample_count,
        "goal_within_8s_accuracy": totals["goal_correct"] / sample_count,
    }


def _learning_rate(
    step: int, *, base: float, minimum: float, warmup_steps: int, maximum_steps: int
) -> float:
    if step <= warmup_steps:
        return base * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(maximum_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return minimum + (base - minimum) * cosine


def _model_from_config(
    config: Mapping[str, Any], vocabulary_size: int
) -> IdentityMatchupTransformer:
    model = config.get("model", {})
    return IdentityMatchupTransformer(
        num_player_identities=vocabulary_size,
        input_features=int(model.get("input_features", 27)),
        time_steps=int(model.get("time_steps", 20)),
        entities=int(model.get("entities", 7)),
        width=int(model.get("width", 192)),
        layers=int(model.get("layers", 3)),
        attention_heads=int(model.get("attention_heads", 6)),
        feed_forward_width=int(model.get("feed_forward_width", 768)),
        dropout=float(model.get("dropout", 0.10)),
        identity_embedding_dim=int(model.get("identity_embedding_dim", 48)),
    )


def _loss_weights(config: Mapping[str, Any]) -> MatchupLossWeights:
    loss = config.get("loss", {})
    return MatchupLossWeights(
        next_touch_entity=float(loss.get("next_touch_entity_weight", 1.0)),
        next_touch_zone=float(loss.get("next_touch_zone_weight", 1.0)),
        retained_possession=float(loss.get("retained_possession_weight", 0.25)),
        goal_within_8s=float(loss.get("goal_within_8s_weight", 0.10)),
        focal_gamma=float(loss.get("focal_gamma", 2.0)),
    )


def overfit_matchup_subset_from_config(
    config: str | Path | Mapping[str, Any],
    *,
    condition: IdentityCondition = "full",
    seed: int = 17,
    sample_count: int = 5_000,
    maximum_steps: int = 3_000,
    evaluation_interval: int = 100,
    target_joint_nll: float = 0.10,
    learning_rate: float = 1e-3,
    output_report: str | Path | None = None,
) -> dict[str, Any]:
    """Memorize a fixed train-only subset before scientific training is allowed."""

    if condition not in IDENTITY_CONDITIONS:
        raise ValueError(f"Unknown identity condition {condition!r}.")
    if sample_count <= 0 or maximum_steps <= 0 or evaluation_interval <= 0:
        raise ValueError("Overfit sample and step counts must be positive.")
    if target_joint_nll <= 0:
        raise ValueError("Overfit target NLL must be positive.")
    config_path = Path(config) if not isinstance(config, Mapping) else None
    cfg = (
        yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config_path is not None
        else copy.deepcopy(dict(config))
    )
    data_cfg = cfg["data"]
    training = cfg["training"]
    load_frozen_rlcs_split(data_cfg["split_manifest"])
    manifest_path = Path(data_cfg["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    vocabulary_path = Path(manifest["identity_vocabulary"]["path"])
    if file_sha256(vocabulary_path) != manifest["identity_vocabulary"]["sha256"]:
        raise ValueError("Identity vocabulary hash mismatch.")
    vocabulary = load_identity_vocabulary(vocabulary_path)
    _set_seed(int(seed))
    dataset = RLCSDecisionDataset(
        manifest_path,
        "train",
        reflection_probability=0.0,
        augmentation_seed=int(seed),
    )
    if sample_count > len(dataset):
        raise ValueError(
            f"Overfit subset requests {sample_count} samples from a {len(dataset)}-row train split."
        )
    feature_mean, feature_std = compute_train_feature_statistics(dataset)
    dataset.set_normalization(feature_mean, feature_std)
    rng = np.random.default_rng(int(seed))
    selected_indices = np.sort(
        rng.choice(len(dataset), size=int(sample_count), replace=False).astype(np.int64)
    )
    subset = Subset(dataset, selected_indices.tolist())
    batch_size = min(int(training.get("batch_size", 256)), int(sample_count))
    generator = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=int(training.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    evaluation_loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(training.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    smoke_cfg = copy.deepcopy(cfg)
    smoke_cfg.setdefault("model", {})["dropout"] = 0.0
    _set_seed(int(seed))
    model = _model_from_config(smoke_cfg, vocabulary.size)
    device = resolve_device(str(training.get("device", "auto")))
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        betas=tuple(float(value) for value in training.get("betas", [0.9, 0.95])),
        weight_decay=0.0,
    )
    weights = _loss_weights(cfg)
    precision = str(training.get("precision", "bf16"))
    autocast = precision == "bf16" and device.type == "cuda"
    report_path = Path(
        output_report
        or Path(training.get("run_root", "runs/rlcs_identity_matchup_v1"))
        / "preflight"
        / f"overfit_{sample_count}.json"
    )
    checkpoint_path = report_path.with_suffix(".pt")
    metrics_path = report_path.with_suffix(".metrics.jsonl")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("", encoding="utf-8")
    started = time.perf_counter()
    step = 0
    epoch = 0
    passed = False
    final_metrics: dict[str, Any] | None = None
    while step < int(maximum_steps) and not passed:
        dataset.set_epoch(epoch)
        model.train()
        for raw_batch in train_loader:
            step += 1
            batch = _to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast):
                outputs = model(**_model_inputs(batch), condition=condition)
                loss, _ = identity_matchup_loss(outputs, _targets(batch), weights=weights)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Overfit smoke produced a non-finite loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training.get("gradient_clip_norm", 1.0))
            )
            optimizer.step()
            if step % int(evaluation_interval) == 0 or step == int(maximum_steps):
                final_metrics = evaluate_matchup_loader(
                    model,
                    evaluation_loader,
                    device,
                    condition=condition,
                    precision=precision,
                )
                record = {
                    "step": step,
                    "epoch": epoch,
                    "train_loss": float(loss.detach()),
                    **final_metrics,
                }
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                passed = final_metrics["factorized_joint_nll"] <= float(target_joint_nll)
                model.train()
                if passed:
                    break
            if step >= int(maximum_steps):
                break
        epoch += 1
    if final_metrics is None:
        raise RuntimeError("Overfit smoke completed without an evaluation checkpoint.")
    checkpoint_temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(
        {
            "version": 1,
            "purpose": "train_only_5000_sample_overfit_preflight",
            "condition": condition,
            "seed": int(seed),
            "step": step,
            "model": model.state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "vocabulary_size": vocabulary.size,
            "dataset_manifest_sha256": file_sha256(manifest_path),
            "metrics": final_metrics,
        },
        checkpoint_temporary,
    )
    checkpoint_temporary.replace(checkpoint_path)
    report = {
        "version": 1,
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "status": "passed" if passed else "failed",
        "purpose": "train_only_5000_sample_overfit_preflight",
        "condition": condition,
        "seed": int(seed),
        "sample_count": int(sample_count),
        "selected_indices_sha256": hashlib.sha256(selected_indices.tobytes()).hexdigest(),
        "target_factorized_joint_nll": float(target_joint_nll),
        "achieved_factorized_joint_nll": float(final_metrics["factorized_joint_nll"]),
        "step": int(step),
        "maximum_steps": int(maximum_steps),
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "precision": precision,
        "dropout": 0.0,
        "reflection_probability": 0.0,
        "weight_decay": 0.0,
        "learning_rate": float(learning_rate),
        "test_loaded": False,
        "validation_loaded": False,
        "dataset_manifest_sha256": file_sha256(manifest_path),
        "identity_vocabulary_sha256": file_sha256(vocabulary_path),
        "dependency_lock_sha256": file_sha256("uv.lock"),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "metrics_path": str(metrics_path),
        "metrics": final_metrics,
    }
    _atomic_json(report_path, report)
    if not passed:
        raise RuntimeError(
            "RLCS overfit smoke failed: factorized joint NLL "
            f"{final_metrics['factorized_joint_nll']:.6f} exceeds {target_joint_nll:.6f}."
        )
    return report


def train_matchup_from_config(
    config: str | Path | Mapping[str, Any],
    *,
    condition: IdentityCondition,
    seed: int,
) -> dict[str, Any]:
    """Train one matched condition and evaluate validation only."""

    if condition not in IDENTITY_CONDITIONS:
        raise ValueError(f"Unknown identity condition {condition!r}.")
    config_path = Path(config) if not isinstance(config, Mapping) else None
    cfg = (
        yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config_path is not None
        else dict(config)
    )
    data_cfg = cfg["data"]
    training = cfg["training"]
    load_frozen_rlcs_split(data_cfg["split_manifest"])
    manifest_path = Path(data_cfg["manifest"])
    _validate_preflight(cfg, manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    vocabulary_path = Path(manifest["identity_vocabulary"]["path"])
    if file_sha256(vocabulary_path) != manifest["identity_vocabulary"]["sha256"]:
        raise ValueError("Identity vocabulary hash mismatch.")
    vocabulary = load_identity_vocabulary(vocabulary_path)
    _set_seed(int(seed))
    train_dataset = RLCSDecisionDataset(
        manifest_path,
        "train",
        reflection_probability=float(data_cfg.get("train_reflection_probability", 0.5)),
        augmentation_seed=int(seed),
    )
    feature_mean, feature_std = compute_train_feature_statistics(train_dataset)
    train_dataset.set_normalization(feature_mean, feature_std)
    val_dataset = RLCSDecisionDataset(
        manifest_path,
        "val",
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    batch_size = int(training.get("batch_size", 256))
    generator = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=int(training.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    if len(train_loader) == 0:
        raise ValueError("Train split is smaller than one configured full batch.")
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(training.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    _set_seed(int(seed))
    model = _model_from_config(cfg, vocabulary.size)
    device = resolve_device(str(training.get("device", "auto")))
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 3e-4)),
        betas=tuple(float(value) for value in training.get("betas", [0.9, 0.95])),
        weight_decay=float(training.get("weight_decay", 0.05)),
    )
    weights = _loss_weights(cfg)
    maximum_steps = int(training.get("maximum_steps", 8000))
    validation_interval = int(training.get("validation_interval_steps", 500))
    patience = int(training.get("early_stop_patience_validations", 4))
    precision = str(training.get("precision", "bf16"))
    base_lr = float(training.get("learning_rate", 3e-4))
    minimum_lr = float(training.get("minimum_learning_rate", 3e-5))
    warmup = int(training.get("warmup_steps", 500))
    run_root = Path(training.get("run_root", "runs/rlcs_identity_matchup_v1"))
    run_dir = (
        run_root
        / condition
        / f"seed_{seed}"
        / datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    run_config = run_dir / "config.yaml"
    run_config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    np.savez(run_dir / "train_feature_statistics.npz", mean=feature_mean, std=feature_std)
    autocast = precision == "bf16" and device.type == "cuda"
    best_nll = math.inf
    best_step = 0
    stale = 0
    metrics_path = run_dir / "metrics_val.jsonl"
    step = 0
    epoch = 0
    stop = False
    while step < maximum_steps and not stop:
        train_dataset.set_epoch(epoch)
        model.train()
        for raw_batch in train_loader:
            step += 1
            lr = _learning_rate(
                step,
                base=base_lr,
                minimum=minimum_lr,
                warmup_steps=warmup,
                maximum_steps=maximum_steps,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch = _to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast):
                outputs = model(**_model_inputs(batch), condition=condition)
                loss, _ = identity_matchup_loss(outputs, _targets(batch), weights=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training.get("gradient_clip_norm", 1.0))
            )
            optimizer.step()
            if step % validation_interval == 0 or step == maximum_steps:
                validation = evaluate_matchup_loader(
                    model,
                    val_loader,
                    device,
                    condition=condition,
                    precision=precision,
                    max_batches=training.get("max_val_batches"),
                )
                record = {
                    "step": step,
                    "epoch": epoch,
                    "learning_rate": lr,
                    "train_loss": float(loss.detach()),
                    **validation,
                }
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                if validation["factorized_joint_nll"] < best_nll:
                    best_nll = float(validation["factorized_joint_nll"])
                    best_step = step
                    stale = 0
                    torch.save(
                        {
                            "version": 1,
                            "condition": condition,
                            "seed": int(seed),
                            "step": step,
                            "config": cfg,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "feature_mean": feature_mean,
                            "feature_std": feature_std,
                            "vocabulary_size": vocabulary.size,
                            "dataset_manifest_sha256": file_sha256(manifest_path),
                            "validation": validation,
                        },
                        run_dir / "best.pt",
                    )
                else:
                    stale += 1
                model.train()
                if stale >= patience:
                    stop = True
                    break
            if step >= maximum_steps:
                break
        epoch += 1
    if best_step == 0:
        raise RuntimeError("Training completed without a validation checkpoint.")
    run_manifest = build_run_manifest(
        command=sys.argv,
        config_path=run_config,
        split_manifest_path=data_cfg["split_manifest"],
        evaluation_protocol="rlcs_validation_only_identity_matchup_v1",
        feature_view="actor_oriented_full_geometry_plus_conditioned_identity",
        objective_mode="factorized_next_touch_entity_zone_multitask",
        dataset_paths={
            "dataset_manifest": manifest_path,
            "train": manifest["splits"]["train"]["path"],
            "val": manifest["splits"]["val"]["path"],
            "identity_vocabulary": vocabulary_path,
        },
        output_paths={
            "run_dir": run_dir,
            "best_checkpoint": run_dir / "best.pt",
            "validation_metrics": metrics_path,
            "feature_statistics": run_dir / "train_feature_statistics.npz",
        },
        warnings=[
            "The sealed test split was not loaded.",
            "Condition selection uses validation factorized joint NLL only.",
        ],
    )
    run_manifest.update(
        {
            "condition": condition,
            "seed": int(seed),
            "best_step": best_step,
            "best_validation_factorized_joint_nll": best_nll,
            "test_loaded": False,
            "paired_batch_seed": int(seed),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "device": str(device),
            "dependency_lock_path": "uv.lock",
            "dependency_lock_sha256": file_sha256("uv.lock"),
        }
    )
    write_run_manifest(run_dir / "run_manifest.json", run_manifest)
    return {
        "run_dir": run_dir,
        "best_checkpoint": run_dir / "best.pt",
        "best_step": best_step,
        "best_validation_factorized_joint_nll": best_nll,
    }
