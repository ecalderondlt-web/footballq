"""Validation metrics, controls, gates, and sealed-test lock for RLCS V2."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from footballq.models.player_matchup_value import VALUE_CONDITIONS, PlayerMatchupValueModel
from footballq.repro.manifest import file_sha256
from footballq.training.eval_matchup import (
    bca_relative_lift_interval,
    sign_flip_pvalue,
)
from footballq.training.train import resolve_device
from footballq.training.train_rlcs_value import (
    RLCSValueDataset,
    model_from_config,
    model_inputs,
    multiclass_metrics,
    to_device,
)


class V2TestUnlockError(PermissionError):
    """Raised when a sealed V2 test unlock is absent, invalid, or consumed."""


def load_value_model(
    checkpoint_path: str | Path, *, device: torch.device
) -> tuple[PlayerMatchupValueModel, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = model_from_config(checkpoint["config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def predict_value_loader(
    model: PlayerMatchupValueModel,
    loader: DataLoader,
    device: torch.device,
    *,
    condition: str,
    precision: str,
) -> tuple[np.ndarray, np.ndarray]:
    probability: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    autocast = precision == "bf16" and device.type == "cuda"
    with torch.no_grad():
        for raw_batch in loader:
            batch = to_device(raw_batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast):
                outputs = model(**model_inputs(batch), condition=condition)
            probability.append(outputs["outcome_probabilities"].float().cpu().numpy())
            labels.append(batch["outcome_label"].cpu().numpy())
    if not probability:
        raise ValueError("V2 evaluation received no rows.")
    return np.concatenate(probability), np.concatenate(labels)


def _sample_log_loss(probability: np.ndarray, labels: np.ndarray) -> np.ndarray:
    selected = probability[np.arange(len(labels)), labels]
    return -np.log(np.clip(selected, 1e-9, 1.0))


def _normalization(checkpoint: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(value, dtype=np.float32)
        for key, value in checkpoint["normalization"].items()
    }


def _validate_bundle(
    bundle: Mapping[str, Any], *, dataset_manifest_sha256: str
) -> dict[str, np.ndarray]:
    if set(bundle.get("checkpoints", {})) != set(VALUE_CONDITIONS):
        raise ValueError("V2 bundle must freeze all five matched conditions.")
    reference: dict[str, np.ndarray] | None = None
    for condition, seeds in bundle["checkpoints"].items():
        if set(str(value) for value in seeds) != {"17", "23", "41"}:
            raise ValueError(f"V2 bundle has wrong seeds for {condition}.")
        for seed, descriptor in seeds.items():
            path = Path(descriptor["path"])
            if file_sha256(path) != descriptor["sha256"]:
                raise ValueError(f"Checkpoint hash mismatch for {condition}/{seed}.")
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if checkpoint.get("condition") != condition or int(checkpoint.get("seed")) != int(seed):
                raise ValueError(f"Checkpoint metadata mismatch for {condition}/{seed}.")
            if checkpoint.get("dataset_manifest_sha256") != dataset_manifest_sha256:
                raise ValueError(f"Checkpoint lineage mismatch for {condition}/{seed}.")
            current = _normalization(checkpoint)
            if reference is None:
                reference = current
            elif any(not np.array_equal(current[key], reference[key]) for key in reference):
                raise ValueError("Matched V2 checkpoints use different train normalization.")
    if reference is None:
        raise ValueError("V2 checkpoint bundle is empty.")
    return reference


def _series_comparison(
    frame: pd.DataFrame,
    *,
    comparison_column: str,
    full_column: str = "nll_full_matchup",
    resamples: int,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    grouped = frame.groupby("series_id", sort=True)[[comparison_column, full_column]].mean()
    comparison = grouped[comparison_column].to_numpy()
    full = grouped[full_column].to_numpy()
    baseline = float(comparison.mean())
    relative = (baseline - float(full.mean())) / baseline
    lower, upper = bca_relative_lift_interval(
        comparison, full, resamples=resamples, seed=seed
    )
    return {
        "series_count": int(len(grouped)),
        "comparison_log_loss": baseline,
        "full_log_loss": float(full.mean()),
        "relative_log_loss_reduction": relative,
        "bca_95pct": [lower, upper],
        "one_sided_sign_flip_p": sign_flip_pvalue(
            comparison - full, permutations=permutations, seed=seed
        ),
    }


def evaluate_value_bundle(
    config_path: str | Path,
    *,
    bundle_path: str | Path,
    stage: str,
    output_dir: str | Path,
    _allow_test: bool = False,
) -> dict[str, Path]:
    """Evaluate internal development or frozen validation; test is private/locked."""

    if stage == "test" and not _allow_test:
        raise V2TestUnlockError("The sealed V2 test requires a consumed one-time unlock.")
    if stage not in {"internal_development", "validation", "test"}:
        raise ValueError("V2 evaluation stage must be internal development, validation, or test.")
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    training_cfg = config["training"]
    evaluation_cfg = config["evaluation"]
    manifest_path = Path(data_cfg["dataset_manifest"])
    bundle_path = Path(bundle_path)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if stage in {"validation", "test"} and not bool(bundle.get("architecture_frozen")):
        raise ValueError("Split 2 evaluation requires an architecture-frozen V2 bundle.")
    normalization = _validate_bundle(
        bundle, dataset_manifest_sha256=file_sha256(manifest_path)
    )
    device = resolve_device(str(training_cfg.get("device", "auto")))
    batch_size = int(training_cfg.get("batch_size", 256))
    precision = str(training_cfg.get("precision", "bf16"))
    probabilities: dict[str, np.ndarray] = {}
    seed_probabilities: dict[str, dict[str, np.ndarray]] = {}
    labels: np.ndarray | None = None
    base_dataset = RLCSValueDataset(
        manifest_path, stage, allow_test=_allow_test, normalization=normalization
    )
    base_loader = DataLoader(
        base_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(training_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    checkpoint_evidence: dict[str, Any] = {}
    for condition in VALUE_CONDITIONS:
        values = []
        seed_probabilities[condition] = {}
        evidence = []
        for seed, descriptor in sorted(
            bundle["checkpoints"][condition].items(), key=lambda item: int(item[0])
        ):
            model, checkpoint = load_value_model(descriptor["path"], device=device)
            probability, current_labels = predict_value_loader(
                model,
                base_loader,
                device,
                condition=condition,
                precision=precision,
            )
            if labels is None:
                labels = current_labels
            elif not np.array_equal(labels, current_labels):
                raise ValueError("V2 matched checkpoints saw different labels/order.")
            values.append(probability)
            seed_probabilities[condition][str(seed)] = probability
            evidence.append(
                {
                    "seed": int(seed),
                    "path": str(descriptor["path"]),
                    "sha256": descriptor["sha256"],
                    "best_step": checkpoint.get("best_step"),
                    "internal_development": checkpoint.get("internal_development"),
                }
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        probabilities[condition] = np.mean(values, axis=0)
        checkpoint_evidence[condition] = evidence
    assert labels is not None

    control_probabilities: dict[str, np.ndarray] = {}
    for control in config["controls"]["names"]:
        control_seed_count = (
            1
            if control == "population_mean_profiles"
            else int(config["controls"].get("seeds", 20))
        )
        randomized_values = []
        for control_seed in range(control_seed_count):
            checkpoint_values = []
            control_dataset = RLCSValueDataset(
                manifest_path,
                stage,
                allow_test=_allow_test,
                normalization=normalization,
                control=str(control),
                control_seed=control_seed,
            )
            control_loader = DataLoader(
                control_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=int(training_cfg.get("num_workers", 0)),
            )
            for _, descriptor in sorted(
                bundle["checkpoints"]["full_matchup"].items(),
                key=lambda item: int(item[0]),
            ):
                model, _ = load_value_model(descriptor["path"], device=device)
                probability, current_labels = predict_value_loader(
                    model,
                    control_loader,
                    device,
                    condition="full_matchup",
                    precision=precision,
                )
                if not np.array_equal(labels, current_labels):
                    raise ValueError("A V2 control changed label order.")
                checkpoint_values.append(probability)
                del model
            randomized_values.append(np.mean(checkpoint_values, axis=0))
        control_probabilities[str(control)] = np.mean(randomized_values, axis=0)

    metrics = {
        name: multiclass_metrics(value, labels, ece_bins=int(evaluation_cfg["ece_bins"]))
        for name, value in {**probabilities, **control_probabilities}.items()
    }
    sample_frame = pd.DataFrame(
        {
            "sample_id": base_dataset.sample_ids,
            "replay_id": base_dataset.replay_ids,
            "series_id": base_dataset.series_ids,
            "region": base_dataset.regions,
            "outcome_label": labels,
            **{
                f"nll_{name}": _sample_log_loss(value, labels)
                for name, value in {**probabilities, **control_probabilities}.items()
            },
        }
    )
    comparisons: dict[str, Any] = {}
    comparison_names = (
        "team_form",
        "actor_profile",
        "additive_profiles",
        *tuple(control_probabilities),
    )
    for name in comparison_names:
        comparisons[f"full_vs_{name}"] = _series_comparison(
            sample_frame,
            comparison_column=f"nll_{name}",
            resamples=int(evaluation_cfg["bootstrap_resamples"]),
            permutations=int(evaluation_cfg["sign_flip_permutations"]),
            seed=73,
        )
    seed_lifts = {}
    for seed in ("17", "23", "41"):
        baseline = _sample_log_loss(seed_probabilities["team_form"][seed], labels).mean()
        full = _sample_log_loss(seed_probabilities["full_matchup"][seed], labels).mean()
        seed_lifts[seed] = float((baseline - full) / baseline)
    regional = {}
    for region in ("EU", "NA"):
        mask = np.asarray(base_dataset.regions) == region
        baseline = _sample_log_loss(probabilities["team_form"][mask], labels[mask]).mean()
        full = _sample_log_loss(probabilities["full_matchup"][mask], labels[mask]).mean()
        regional[region] = float((baseline - full) / baseline)

    gate_cfg = evaluation_cfg["gates"]
    main = comparisons["full_vs_team_form"]
    gates = {
        "full_vs_team_form_two_of_three_seeds": sum(
            value >= float(gate_cfg["full_vs_team_form_relative_log_loss_reduction"])
            for value in seed_lifts.values()
        )
        >= int(gate_cfg["required_passing_seeds"]),
        "series_bootstrap_lower_positive": main["bca_95pct"][0]
        > float(gate_cfg["series_bootstrap_lower"]),
        "full_vs_additive": comparisons["full_vs_additive_profiles"][
            "relative_log_loss_reduction"
        ]
        >= float(gate_cfg["full_vs_additive_relative_log_loss_reduction"]),
        "full_vs_actor": comparisons["full_vs_actor_profile"][
            "relative_log_loss_reduction"
        ]
        >= float(gate_cfg["full_vs_actor_relative_log_loss_reduction"]),
        "full_vs_each_shuffle": all(
            comparisons[f"full_vs_{name}"]["relative_log_loss_reduction"]
            >= float(gate_cfg["full_vs_each_shuffle_relative_log_loss_reduction"])
            for name in control_probabilities
        ),
        "calibration": metrics["full_matchup"]["ece"]
        <= metrics["state"]["ece"] + float(gate_cfg["maximum_ece_degradation_vs_state"]),
        "regional_robustness": all(value > 0 for value in regional.values()),
    }
    results = {
        "version": 2,
        "experiment": "rlcs_player_matchup_value_v2",
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "stage": stage,
        "test_loaded": stage == "test",
        "bundle_path": str(bundle_path),
        "bundle_sha256": file_sha256(bundle_path),
        "dataset_manifest_sha256": file_sha256(manifest_path),
        "metrics": metrics,
        "comparisons": comparisons,
        "full_vs_team_form_relative_lift_by_seed": seed_lifts,
        "regional_full_vs_team_form_relative_lift": regional,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "checkpoint_evidence": checkpoint_evidence,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    samples_path = output / f"{stage}_per_sample.parquet"
    sample_frame.to_parquet(samples_path, index=False, compression="zstd")
    results_path = output / f"{stage}_results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return {"results": results_path, "per_sample": samples_path}


def validate_v2_test_unlock(
    unlock_path: str | Path,
    *,
    config_path: str | Path,
    dataset_manifest_path: str | Path,
    split_manifest_path: str | Path,
    bundle_path: str | Path,
) -> dict[str, Any]:
    unlock = json.loads(Path(unlock_path).read_text(encoding="utf-8"))
    required = {
        "experiment": "rlcs_player_matchup_value_v2",
        "status": "approved_once",
        "config_sha256": file_sha256(config_path),
        "dataset_manifest_sha256": file_sha256(dataset_manifest_path),
        "split_manifest_sha256": file_sha256(split_manifest_path),
        "bundle_sha256": file_sha256(bundle_path),
    }
    for key, expected in required.items():
        if unlock.get(key) != expected:
            raise V2TestUnlockError(f"V2 test unlock mismatch for {key}.")
    validation_path = Path(unlock.get("validation_results_path", ""))
    if not validation_path.exists() or file_sha256(validation_path) != unlock.get(
        "validation_results_sha256"
    ):
        raise V2TestUnlockError("V2 test unlock validation evidence is missing or changed.")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("stage") != "validation" or not validation.get("all_gates_pass"):
        raise V2TestUnlockError("V2 frozen validation did not pass every preregistered gate.")
    return unlock


def consume_v2_test_unlock(
    unlock_path: str | Path, *, output_dir: str | Path, resume: bool = False
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    receipt = Path(unlock_path).with_suffix(".consumed.json")
    payload = {
        "consumed_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "unlock_sha256": file_sha256(unlock_path),
        "output_dir": str(output.resolve()),
    }
    try:
        with receipt.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
    except FileExistsError as exc:
        old = json.loads(receipt.read_text(encoding="utf-8"))
        compatible = (
            old.get("unlock_sha256") == payload["unlock_sha256"]
            and old.get("output_dir") == payload["output_dir"]
        )
        if not resume or not compatible:
            raise V2TestUnlockError(f"V2 test unlock was already consumed at {receipt}.") from exc
    return receipt


def evaluate_sealed_value_test(
    config_path: str | Path,
    *,
    bundle_path: str | Path,
    unlock_path: str | Path,
    output_dir: str | Path,
    resume: bool = False,
) -> dict[str, Path]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    validate_v2_test_unlock(
        unlock_path,
        config_path=config_path,
        dataset_manifest_path=config["data"]["dataset_manifest"],
        split_manifest_path=config["data"]["split_manifest"],
        bundle_path=bundle_path,
    )
    receipt = consume_v2_test_unlock(unlock_path, output_dir=output_dir, resume=resume)
    paths = evaluate_value_bundle(
        config_path,
        bundle_path=bundle_path,
        stage="test",
        output_dir=output_dir,
        _allow_test=True,
    )
    return {**paths, "receipt": receipt}
