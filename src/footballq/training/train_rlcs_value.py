"""Matched training stack for RLCS Player-Matchup Critical Value V2."""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from footballq.data.rlcs_player_profiles import PROFILE_DIMENSION
from footballq.data.rlcs_touch_windows import (
    N_ENTITIES,
    N_FEATURES,
    N_PLAYERS,
    STATE_MASK_SIZE,
    STATE_SIZE,
    TIME_STEPS,
    reflect_state_x,
)
from footballq.data.rlcs_value_windows import PAIR_GEOMETRY_DIMENSION, TEAM_FORM_DIMENSION
from footballq.models.player_matchup_value import (
    VALUE_CONDITIONS,
    PlayerMatchupValueModel,
    ValueCondition,
    critical_value_loss,
)
from footballq.repro.manifest import file_sha256, git_metadata
from footballq.training.train import resolve_device


class V2TestSplitLockedError(PermissionError):
    """Raised whenever an ordinary V2 path attempts to open the sealed test."""


def _fixed_list_numpy(table: Any, name: str, width: int, dtype: Any) -> np.ndarray:
    array = table[name].combine_chunks()
    values = array.values.to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=dtype).reshape(len(array), width)


class RLCSValueDataset(Dataset[dict[str, Any]]):
    """Arrow-backed V2 loader that never exposes provenance IDs to the model."""

    def __init__(
        self,
        dataset_manifest_path: str | Path,
        stage: str,
        *,
        allow_test: bool = False,
        reflection_probability: float = 0.0,
        augmentation_seed: int = 0,
        normalization: Mapping[str, Any] | None = None,
        control: str | None = None,
        control_seed: int = 0,
    ) -> None:
        allowed = {"train", "internal_development", "validation", "test"}
        if stage not in allowed:
            raise ValueError(f"Unknown V2 dataset stage {stage!r}.")
        if stage == "test" and not allow_test:
            raise V2TestSplitLockedError("Ordinary V2 code may not load the sealed test split.")
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("RLCS V2 training requires pyarrow.") from exc
        self.manifest_path = Path(dataset_manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        descriptor = self.manifest["stages"][stage]
        path = Path(descriptor["path"])
        if file_sha256(path) != descriptor["sha256"]:
            raise ValueError(f"V2 dataset hash mismatch for {stage!r}.")
        self.table = pq.read_table(path)
        if len(self.table) != int(descriptor["rows"]):
            raise ValueError(f"V2 row count mismatch for {stage!r}.")
        self.stage = stage
        self.states = _fixed_list_numpy(self.table, "state_flat", STATE_SIZE, np.float32).reshape(
            -1, TIME_STEPS, N_ENTITIES, N_FEATURES
        )
        self.state_masks = _fixed_list_numpy(
            self.table, "state_mask", STATE_MASK_SIZE, np.bool_
        ).reshape(-1, TIME_STEPS, N_ENTITIES)
        self.profiles = _fixed_list_numpy(
            self.table, "profile_flat", N_PLAYERS * PROFILE_DIMENSION, np.float32
        ).reshape(-1, N_PLAYERS, PROFILE_DIMENSION)
        self.profile_uncertainty = _fixed_list_numpy(
            self.table,
            "profile_uncertainty_flat",
            N_PLAYERS * PROFILE_DIMENSION,
            np.float32,
        ).reshape(-1, N_PLAYERS, PROFILE_DIMENSION)
        self.profile_effective = _fixed_list_numpy(
            self.table, "profile_effective_sample_size", N_PLAYERS, np.float32
        )
        self.team_form = _fixed_list_numpy(
            self.table, "team_form", TEAM_FORM_DIMENSION, np.float32
        )
        self.pair_geometry = _fixed_list_numpy(
            self.table, "pair_geometry_flat", 3 * PAIR_GEOMETRY_DIMENSION, np.float32
        ).reshape(-1, 3, PAIR_GEOMETRY_DIMENSION)
        self.teammate_geometry = _fixed_list_numpy(
            self.table, "teammate_geometry_flat", 2 * PAIR_GEOMETRY_DIMENSION, np.float32
        ).reshape(-1, 2, PAIR_GEOMETRY_DIMENSION)
        self.seconds_remaining = self.table["seconds_remaining"].to_numpy().astype(np.float32)
        self.game_time = self.table["game_time_s"].to_numpy().astype(np.float32)
        self.score_diff = self.table["score_diff_actor"].to_numpy().astype(np.float32)
        self.overtime = self.table["overtime"].to_numpy().astype(np.bool_)
        self.labels = self.table["outcome_label"].to_numpy().astype(np.int64)
        self.sample_ids = [str(value) for value in self.table["sample_id"].to_pylist()]
        self.replay_ids = [str(value) for value in self.table["replay_id"].to_pylist()]
        self.series_ids = [str(value) for value in self.table["series_id"].to_pylist()]
        self.regions = [str(value) for value in self.table["region"].to_pylist()]
        self.event_times = [str(value) for value in self.table["event_time_utc"].to_pylist()]
        self.reflection_probability = float(reflection_probability) if stage == "train" else 0.0
        self.augmentation_seed = int(augmentation_seed)
        self.epoch = 0
        self.normalization = _identity_normalization() if normalization is None else {
            key: np.asarray(value, dtype=np.float32) for key, value in normalization.items()
        }
        self.control = control
        self.control_seed = int(control_seed)
        self.control_source = self._control_source_indices(control, self.control_seed)

    def _control_source_indices(self, control: str | None, seed: int) -> np.ndarray:
        indices = np.arange(len(self.table), dtype=np.int64)
        if control is None or control in {
            "population_mean_profiles",
            "opponent_profile_geometry_permutation",
        }:
            return indices
        rng = np.random.default_rng(int(seed))
        if control == "actor_strength_matched_shuffle":
            bins: dict[tuple[str, int], list[int]] = {}
            for index in indices:
                key = (self.regions[index], int(np.floor(self.team_form[index, 0] * 5)))
                bins.setdefault(key, []).append(int(index))
            source = indices.copy()
            for values in bins.values():
                if len(values) > 1:
                    source[values] = rng.permutation(values)
            return source
        if control == "matched_series_opponent_shuffle":
            strata: dict[tuple[str, int, int], list[int]] = {}
            for index in indices:
                timestamp = np.datetime64(self.event_times[index])
                date_band = int(timestamp.astype("datetime64[D]").astype(int) // 14)
                strength_band = int(np.floor(self.team_form[index, 1] * 5))
                strata.setdefault((self.regions[index], date_band, strength_band), []).append(
                    int(index)
                )
            source = indices.copy()
            for values in strata.values():
                by_series: dict[str, list[int]] = {}
                for index in values:
                    by_series.setdefault(self.series_ids[index], []).append(index)
                if len(by_series) < 2:
                    continue
                for index in values:
                    candidates = [
                        candidate
                        for series, members in by_series.items()
                        if series != self.series_ids[index]
                        for candidate in members
                    ]
                    source[index] = int(rng.choice(candidates))
            return source
        raise ValueError(f"Unknown V2 profile control {control!r}.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_normalization(self, normalization: Mapping[str, Any]) -> None:
        self.normalization = {
            key: np.asarray(value, dtype=np.float32) for key, value in normalization.items()
        }

    def __len__(self) -> int:
        return len(self.labels)

    def _reflect(self, index: int) -> bool:
        if self.reflection_probability <= 0:
            return False
        payload = f"{self.augmentation_seed}:{self.epoch}:{self.sample_ids[index]}".encode()
        value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64
        return value < self.reflection_probability

    def _controlled_profiles(
        self, index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        profiles = self.profiles[index].copy()
        uncertainty = self.profile_uncertainty[index].copy()
        effective = self.profile_effective[index].copy()
        if self.control == "population_mean_profiles":
            profiles[:] = self.normalization["profile_mean"]
        elif self.control == "actor_strength_matched_shuffle":
            source = self.control_source[index]
            profiles[0] = self.profiles[source, 0]
            uncertainty[0] = self.profile_uncertainty[source, 0]
            effective[0] = self.profile_effective[source, 0]
        elif self.control == "matched_series_opponent_shuffle":
            source = self.control_source[index]
            profiles[3:6] = self.profiles[source, 3:6]
            uncertainty[3:6] = self.profile_uncertainty[source, 3:6]
            effective[3:6] = self.profile_effective[source, 3:6]
        elif self.control == "opponent_profile_geometry_permutation":
            payload = f"opponents:{self.control_seed}:{self.sample_ids[index]}".encode()
            seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
            permutation = np.random.default_rng(seed).permutation(3)
            profiles[3:6] = profiles[3:6][permutation]
            uncertainty[3:6] = uncertainty[3:6][permutation]
            effective[3:6] = effective[3:6][permutation]
        return profiles, uncertainty, effective

    def __getitem__(self, index: int) -> dict[str, Any]:
        state = self.states[index].copy()
        pair = self.pair_geometry[index].copy()
        teammate = self.teammate_geometry[index].copy()
        if self._reflect(index):
            state = reflect_state_x(state)
            pair[:, (0, 3)] *= -1.0
            teammate[:, (0, 3)] *= -1.0
        state = (state - self.normalization["state_mean"]) / self.normalization["state_std"]
        state[~self.state_masks[index]] = 0.0
        profiles, uncertainty, effective = self._controlled_profiles(index)
        profiles = (profiles - self.normalization["profile_mean"]) / self.normalization[
            "profile_std"
        ]
        uncertainty = uncertainty / self.normalization["profile_std"]
        team_form = self.team_form[index] - self.normalization["team_form_mean"]
        team_form = team_form / self.normalization["team_form_std"]
        pair = (pair - self.normalization["pair_mean"]) / self.normalization["pair_std"]
        teammate = (teammate - self.normalization["pair_mean"]) / self.normalization[
            "pair_std"
        ]
        scalar = np.asarray(
            [
                min(float(self.seconds_remaining[index]), 300.0) / 300.0,
                float(self.score_diff[index]) / 5.0,
                float(self.overtime[index]),
                min(float(self.game_time[index]), 600.0) / 300.0,
            ],
            dtype=np.float32,
        )
        return {
            "state": torch.from_numpy(state),
            "state_mask": torch.from_numpy(self.state_masks[index].copy()),
            "scalar_context": torch.from_numpy(scalar),
            "team_form": torch.from_numpy(team_form),
            "profiles": torch.from_numpy(profiles),
            "profile_uncertainty": torch.from_numpy(uncertainty),
            "profile_effective_sample_size": torch.from_numpy(effective),
            "pair_geometry": torch.from_numpy(pair),
            "teammate_geometry": torch.from_numpy(teammate),
            "outcome_label": torch.tensor(self.labels[index]),
            "sample_id": self.sample_ids[index],
            "replay_id": self.replay_ids[index],
            "series_id": self.series_ids[index],
            "region": self.regions[index],
        }


def _identity_normalization() -> dict[str, np.ndarray]:
    return {
        "state_mean": np.zeros(N_FEATURES, dtype=np.float32),
        "state_std": np.ones(N_FEATURES, dtype=np.float32),
        "profile_mean": np.zeros(PROFILE_DIMENSION, dtype=np.float32),
        "profile_std": np.ones(PROFILE_DIMENSION, dtype=np.float32),
        "team_form_mean": np.zeros(TEAM_FORM_DIMENSION, dtype=np.float32),
        "team_form_std": np.ones(TEAM_FORM_DIMENSION, dtype=np.float32),
        "pair_mean": np.zeros(PAIR_GEOMETRY_DIMENSION, dtype=np.float32),
        "pair_std": np.ones(PAIR_GEOMETRY_DIMENSION, dtype=np.float32),
    }


def _mean_std(values: np.ndarray, axes: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    mean = values.astype(np.float64).mean(axis=axes)
    std = values.astype(np.float64).std(axis=axes)
    std[std < 1e-5] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def compute_train_normalization(dataset: RLCSValueDataset) -> dict[str, np.ndarray]:
    if dataset.stage != "train":
        raise ValueError("V2 normalization may be fitted only on the model-training stage.")
    valid = dataset.state_masks[..., None]
    values = dataset.states.astype(np.float64)
    count = valid.sum(axis=(0, 1, 2)).astype(np.float64)
    state_mean = np.where(valid, values, 0.0).sum(axis=(0, 1, 2)) / np.maximum(count, 1)
    centered = np.where(valid, values - state_mean, 0.0)
    state_std = np.sqrt((centered * centered).sum(axis=(0, 1, 2)) / np.maximum(count, 1))
    state_std[state_std < 1e-5] = 1.0
    profile_mean, profile_std = _mean_std(dataset.profiles, (0, 1))
    form_mean, form_std = _mean_std(dataset.team_form, (0,))
    pairs = np.concatenate([dataset.pair_geometry, dataset.teammate_geometry], axis=1)
    pair_mean, pair_std = _mean_std(pairs, (0, 1))
    return {
        "state_mean": state_mean.astype(np.float32),
        "state_std": state_std.astype(np.float32),
        "profile_mean": profile_mean,
        "profile_std": profile_std,
        "team_form_mean": form_mean,
        "team_form_std": form_std,
        "pair_mean": pair_mean,
        "pair_std": pair_std,
    }


def model_from_config(config: Mapping[str, Any]) -> PlayerMatchupValueModel:
    model = config.get("model", {})
    return PlayerMatchupValueModel(
        input_features=int(model.get("input_features", N_FEATURES)),
        time_steps=int(model.get("time_steps", TIME_STEPS)),
        entities=int(model.get("entities", N_ENTITIES)),
        width=int(model.get("width", 192)),
        layers=int(model.get("layers", 3)),
        attention_heads=int(model.get("attention_heads", 6)),
        feed_forward_width=int(model.get("feed_forward_width", 768)),
        dropout=float(model.get("dropout", 0.10)),
        profile_dimension=int(model.get("profile_dimension", PROFILE_DIMENSION)),
        profile_projection=int(model.get("profile_projection", 64)),
        pair_geometry_dimension=int(
            model.get("pair_geometry_dimension", PAIR_GEOMETRY_DIMENSION)
        ),
        pair_output_dimension=int(model.get("pair_output_dimension", 64)),
        team_form_dimension=int(model.get("team_form_dimension", TEAM_FORM_DIMENSION)),
        scalar_context_dimension=int(model.get("scalar_context_dimension", 4)),
        outcome_classes=int(model.get("outcome_classes", 3)),
    )


def model_inputs(batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    names = (
        "state",
        "state_mask",
        "scalar_context",
        "team_form",
        "profiles",
        "profile_uncertainty",
        "profile_effective_sample_size",
        "pair_geometry",
        "teammate_geometry",
    )
    return {name: batch[name] for name in names}


def to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def multiclass_metrics(
    probabilities: np.ndarray, labels: np.ndarray, *, ece_bins: int = 15
) -> dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    chosen = np.clip(probabilities[np.arange(len(labels)), labels], 1e-9, 1.0)
    log_loss = float(-np.log(chosen).mean())
    one_hot = np.eye(3, dtype=np.float64)[labels]
    brier = float(np.square(probabilities - one_hot).sum(axis=1).mean())
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    ece = 0.0
    boundaries = np.linspace(0.0, 1.0, int(ece_bins) + 1)
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            accuracy = float(correct[mask].mean())
            mean_confidence = float(confidence[mask].mean())
            ece += float(mask.mean()) * abs(accuracy - mean_confidence)
    score_ap = _average_precision(probabilities[:, 1], labels == 1)
    concede_ap = _average_precision(probabilities[:, 2], labels == 2)
    return {
        "three_class_log_loss": log_loss,
        "multiclass_brier": brier,
        "ece": float(ece),
        "score_average_precision": score_ap,
        "concede_average_precision": concede_ap,
    }


def _average_precision(probability: np.ndarray, target: np.ndarray) -> float:
    """Compute dependency-free binary average precision for the outcome heads."""

    target = np.asarray(target, dtype=np.bool_)
    positives = int(target.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-np.asarray(probability, dtype=np.float64), kind="stable")
    ranked = target[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].sum() / positives)


def evaluate_value_loader(
    model: PlayerMatchupValueModel,
    loader: DataLoader,
    device: torch.device,
    *,
    condition: ValueCondition,
    precision: str = "bf16",
) -> dict[str, Any]:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    sample_ids: list[str] = []
    autocast = precision == "bf16" and device.type == "cuda"
    with torch.no_grad():
        for raw_batch in loader:
            sample_ids.extend(str(value) for value in raw_batch["sample_id"])
            batch = to_device(raw_batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast):
                outputs = model(**model_inputs(batch), condition=condition)
            probabilities.append(outputs["outcome_probabilities"].float().cpu().numpy())
            labels.append(batch["outcome_label"].cpu().numpy())
    if not probabilities:
        raise ValueError("V2 evaluation received no samples.")
    probability = np.concatenate(probabilities)
    label = np.concatenate(labels)
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        digest.update(sample_id.encode("utf-8") + b"\n")
    return {
        "num_examples": int(len(label)),
        "sample_id_sha256": digest.hexdigest(),
        **multiclass_metrics(probability, label),
    }


def _learning_rate(
    step: int, *, base: float, minimum: float, warmup_steps: int, maximum_steps: int
) -> float:
    if step <= warmup_steps:
        return base * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(maximum_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return minimum + (base - minimum) * cosine


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_ready_normalization(values: Mapping[str, np.ndarray]) -> dict[str, list[float]]:
    return {key: np.asarray(value).tolist() for key, value in values.items()}


def train_value_from_config(
    config_path: str | Path,
    *,
    condition: ValueCondition,
    seed: int,
) -> dict[str, Any]:
    """Train one matched V2 condition after enforcing the preregistered stop gates."""

    if condition not in VALUE_CONDITIONS:
        raise ValueError(f"Unknown V2 condition {condition!r}.")
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    training_cfg = config["training"]
    evaluation_cfg = config["evaluation"]
    profile_audit_path = Path(data_cfg["profile_audit"])
    profile_audit = json.loads(profile_audit_path.read_text(encoding="utf-8"))
    if not bool(profile_audit.get("all_gates_pass")):
        raise RuntimeError("V2 profile stability gate failed; outcome training is forbidden.")

    manifest_path = Path(data_cfg["dataset_manifest"])
    train_dataset = RLCSValueDataset(
        manifest_path,
        "train",
        reflection_probability=float(data_cfg.get("train_reflection_probability", 0.5)),
        augmentation_seed=int(seed),
    )
    counts = np.bincount(train_dataset.labels, minlength=3)
    minimum = int(evaluation_cfg["minimum_training_score_rows"])
    if int(counts[1]) < minimum or int(counts[2]) < int(
        evaluation_cfg["minimum_training_concede_rows"]
    ):
        raise RuntimeError(
            f"V2 label-count gate failed: score={counts[1]}, concede={counts[2]}."
        )
    normalization = compute_train_normalization(train_dataset)
    train_dataset.set_normalization(normalization)
    development_dataset = RLCSValueDataset(
        manifest_path, "internal_development", normalization=normalization
    )
    batch_size = int(training_cfg.get("batch_size", 256))
    generator = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=int(training_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    development_loader = DataLoader(
        development_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(training_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    _set_seed(int(seed))
    device = resolve_device(str(training_cfg.get("device", "auto")))
    model = model_from_config(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        betas=tuple(float(value) for value in training_cfg.get("betas", (0.9, 0.95))),
        weight_decay=float(training_cfg.get("weight_decay", 0.05)),
    )
    precision = str(training_cfg.get("precision", "bf16"))
    autocast = precision == "bf16" and device.type == "cuda"
    maximum_steps = int(training_cfg["maximum_steps"])
    validation_interval = int(training_cfg["validation_interval_steps"])
    patience_limit = int(training_cfg["early_stop_patience_validations"])
    warmup_steps = int(training_cfg["warmup_steps"])
    minimum_lr = float(training_cfg["minimum_learning_rate"])
    base_lr = float(training_cfg["learning_rate"])
    run_dir = (
        Path(training_cfg["run_root"])
        / str(condition)
        / f"seed_{seed}"
        / datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = run_dir / "best.pt"
    history: list[dict[str, Any]] = []
    best = float("inf")
    best_step = 0
    patience = 0
    epoch = 0
    step = 0
    while step < maximum_steps and patience < patience_limit:
        train_dataset.set_epoch(epoch)
        for raw_batch in train_loader:
            step += 1
            learning_rate = _learning_rate(
                step,
                base=base_lr,
                minimum=minimum_lr,
                warmup_steps=warmup_steps,
                maximum_steps=maximum_steps,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            batch = to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast):
                outputs = model(**model_inputs(batch), condition=condition)
                loss = critical_value_loss(outputs, batch["outcome_label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training_cfg.get("gradient_clip_norm", 1.0))
            )
            optimizer.step()
            if step % validation_interval == 0 or step == maximum_steps:
                metrics = evaluate_value_loader(
                    model,
                    development_loader,
                    device,
                    condition=condition,
                    precision=precision,
                )
                record = {"step": step, "train_loss": float(loss), **metrics}
                history.append(record)
                current = float(metrics["three_class_log_loss"])
                if current < best:
                    best = current
                    best_step = step
                    patience = 0
                    torch.save(
                        {
                            "version": 2,
                            "experiment": "rlcs_player_matchup_value_v2",
                            "condition": condition,
                            "seed": int(seed),
                            "model": model.state_dict(),
                            "config": config,
                            "normalization": _json_ready_normalization(normalization),
                            "dataset_manifest_sha256": file_sha256(manifest_path),
                            "profile_audit_sha256": file_sha256(profile_audit_path),
                            "best_step": best_step,
                            "internal_development": metrics,
                            "test_loaded": False,
                            "split2_validation_loaded": False,
                        },
                        checkpoint_path,
                    )
                else:
                    patience += 1
            if step >= maximum_steps or patience >= patience_limit:
                break
        epoch += 1

    if not checkpoint_path.exists():
        raise RuntimeError("V2 training ended before producing a checkpoint.")
    run_manifest = {
        "version": 2,
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "command": " ".join(sys.argv),
        "git": git_metadata(),
        "condition": condition,
        "seed": int(seed),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "dataset_manifest_path": str(manifest_path),
        "dataset_manifest_sha256": file_sha256(manifest_path),
        "profile_audit_path": str(profile_audit_path),
        "profile_audit_sha256": file_sha256(profile_audit_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "best_step": best_step,
        "best_internal_development_log_loss": best,
        "label_counts": counts.tolist(),
        "test_loaded": False,
        "split2_validation_loaded": False,
    }
    (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "run_dir": run_dir,
        "checkpoint": checkpoint_path,
        "best_internal_development_log_loss": best,
        "best_step": best_step,
    }
