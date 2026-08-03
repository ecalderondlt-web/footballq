from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from footballq.models.identity_matchup_transformer import IDENTITY_CONDITIONS
from footballq.repro.manifest import file_sha256
from footballq.training.eval_matchup import _load_model, _validate_checkpoint_bundle
from footballq.training.train import resolve_device
from footballq.training.train_matchup import (
    RLCSDecisionDataset,
    _model_inputs,
    _to_device,
)


def _average_precision(probability: np.ndarray, target: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.bool_)
    positives = int(target.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-np.asarray(probability), kind="stable")
    ranked = target[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].sum() / positives)


def _binary_metrics(probability: np.ndarray, target: np.ndarray) -> dict[str, float]:
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    target = np.asarray(target, dtype=np.bool_)
    log_loss = -np.mean(
        np.where(target, np.log(probability), np.log1p(-probability))
    )
    return {
        "prevalence": float(target.mean()),
        "log_loss": float(log_loss),
        "brier": float(np.mean(np.square(probability - target))),
        "average_precision": _average_precision(probability, target),
    }


def _calibration(
    probability: np.ndarray, target: np.ndarray, *, bins: int
) -> list[dict[str, float | int | None]]:
    boundaries = np.linspace(0.0, 1.0, int(bins) + 1)
    output = []
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        mask = (probability > lower) & (probability <= upper)
        output.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": int(mask.sum()),
                "mean_prediction": float(probability[mask].mean()) if mask.any() else None,
                "observed_rate": float(target[mask].mean()) if mask.any() else None,
            }
        )
    return output


def _predict_heads(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    condition: str,
    precision: str,
) -> tuple[np.ndarray, np.ndarray]:
    goal: list[np.ndarray] = []
    retained: list[np.ndarray] = []
    autocast = precision == "bf16" and device.type == "cuda"
    with torch.no_grad():
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast):
                output = model(**_model_inputs(batch), condition=condition)
            goal.append(torch.sigmoid(output["goal_within_8s_logit"].float()).cpu().numpy())
            retained.append(
                torch.sigmoid(output["retained_possession_logit"].float()).cpu().numpy()
            )
    return np.concatenate(goal), np.concatenate(retained)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-only post hoc audit of the frozen V1 outcome heads."
    )
    parser.add_argument("--config", default="configs/rlcs_player_matchup_value_v2.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    v2_config_path = Path(args.config)
    v2_config = yaml.safe_load(v2_config_path.read_text(encoding="utf-8"))
    diagnostic_cfg = v2_config["v1_diagnostic"]
    summary_path = Path(diagnostic_cfg["validation_summary"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    descriptors = summary["runs"]
    if set(descriptors) != set(IDENTITY_CONDITIONS):
        raise ValueError("V1 validation ledger does not contain all four frozen conditions.")
    v1_config_path = Path(summary["config_path"])
    v1_config = yaml.safe_load(v1_config_path.read_text(encoding="utf-8"))
    manifest_path = Path(v1_config["data"]["manifest"])
    if file_sha256(manifest_path) != summary["dataset_manifest_sha256"]:
        raise ValueError("V1 validation ledger no longer matches its dataset manifest.")
    mean, std = _validate_checkpoint_bundle(
        descriptors, dataset_manifest_sha256=file_sha256(manifest_path)
    )
    dataset = RLCSDecisionDataset(
        manifest_path,
        "val",
        feature_mean=mean,
        feature_std=std,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(v1_config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(v1_config["training"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    device = resolve_device(str(v1_config["training"].get("device", "auto")))
    precision = str(v1_config["training"].get("precision", "bf16"))
    critical = (np.abs(dataset.score_diff) <= 1) & (
        (dataset.seconds_remaining <= 120.0) | dataset.overtime
    )
    all_known = dataset.known_masks.all(axis=1)
    subsets = {
        "all_validation": np.ones(len(dataset), dtype=np.bool_),
        "critical_state": critical,
        "critical_all_identities_known": critical & all_known,
    }
    results: dict[str, dict[str, dict]] = {}
    for condition in IDENTITY_CONDITIONS:
        results[condition] = {}
        for seed, descriptor in sorted(
            descriptors[condition].items(), key=lambda item: int(item[0])
        ):
            if file_sha256(descriptor["path"]) != descriptor["sha256"]:
                raise ValueError(f"V1 checkpoint hash changed for {condition}/{seed}.")
            model, checkpoint = _load_model(descriptor["path"], device=device)
            if checkpoint.get("condition") != condition or int(checkpoint.get("seed")) != int(seed):
                raise ValueError(f"V1 checkpoint metadata mismatch for {condition}/{seed}.")
            goal_probability, retained_probability = _predict_heads(
                model,
                loader,
                device,
                condition=condition,
                precision=precision,
            )
            subset_results = {}
            for name, mask in subsets.items():
                goal_metrics = _binary_metrics(goal_probability[mask], dataset.goal[mask])
                retained_metrics = _binary_metrics(
                    retained_probability[mask], dataset.retained[mask]
                )
                retained_metrics.pop("average_precision")
                subset_results[name] = {
                    "samples": int(mask.sum()),
                    "goal_within_8s": {
                        **goal_metrics,
                        "calibration": _calibration(
                            goal_probability[mask],
                            dataset.goal[mask],
                            bins=int(diagnostic_cfg["calibration_bins"]),
                        ),
                    },
                    "retained_possession": retained_metrics,
                }
            results[condition][str(seed)] = {
                "checkpoint": str(descriptor["path"]),
                "checkpoint_sha256": descriptor["sha256"],
                "selected_by": "validation_factorized_next_touch_nll",
                "subsets": subset_results,
            }
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    differences: dict[str, dict[str, dict[str, float]]] = {}
    for seed in ("17", "23", "41"):
        differences[seed] = {}
        for condition in ("actor_only", "roster_only", "full"):
            differences[seed][condition] = {}
            for subset in subsets:
                anonymous = results["anonymous"][seed]["subsets"][subset]
                candidate = results[condition][seed]["subsets"][subset]
                differences[seed][condition][subset] = {
                    "goal_log_loss_anonymous_minus_condition": float(
                        anonymous["goal_within_8s"]["log_loss"]
                        - candidate["goal_within_8s"]["log_loss"]
                    ),
                    "goal_brier_anonymous_minus_condition": float(
                        anonymous["goal_within_8s"]["brier"]
                        - candidate["goal_within_8s"]["brier"]
                    ),
                    "retained_log_loss_anonymous_minus_condition": float(
                        anonymous["retained_possession"]["log_loss"]
                        - candidate["retained_possession"]["log_loss"]
                    ),
                    "retained_brier_anonymous_minus_condition": float(
                        anonymous["retained_possession"]["brier"]
                        - candidate["retained_possession"]["brier"]
                    ),
                }

    report = {
        "version": 1,
        "experiment": "rlcs_identity_matchup_v1_outcome_head_post_hoc_audit",
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "status": "diagnostic_only_v1_conclusion_unchanged",
        "data_boundary": "already_open_validation_only",
        "test_loaded": False,
        "test_unlock_created": False,
        "checkpoint_selection_unchanged": True,
        "v1_validation_summary": str(summary_path),
        "v1_validation_summary_sha256": file_sha256(summary_path),
        "v1_dataset_manifest": str(manifest_path),
        "v1_dataset_manifest_sha256": file_sha256(manifest_path),
        "counts": {name: int(mask.sum()) for name, mask in subsets.items()},
        "target_prevalence": {
            name: {
                "goal_within_8s": float(dataset.goal[mask].mean()),
                "retained_possession": float(dataset.retained[mask].mean()),
            }
            for name, mask in subsets.items()
        },
        "condition_seed_results": results,
        "anonymous_minus_condition_differences": differences,
        "interpretation_limit": (
            "Post hoc outcome-head behavior cannot rescue V1 because checkpoints were selected "
            "by next-touch NLL and the goal head was underweighted."
        ),
    }
    destination = args.output or Path(diagnostic_cfg["output"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    anonymous_goal = results["anonymous"]["17"]["subsets"]["all_validation"][
        "goal_within_8s"
    ]
    print(
        "V1 validation-only outcome audit complete: "
        f"prevalence={anonymous_goal['prevalence']:.4f}, "
        f"anonymous seed17 goal log loss={anonymous_goal['log_loss']:.4f}"
    )
    print(f"report: {destination}")


if __name__ == "__main__":
    main()
