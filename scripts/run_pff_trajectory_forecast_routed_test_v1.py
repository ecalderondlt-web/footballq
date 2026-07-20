"""Run the one-time confirmatory PFF test for the locked routed forecaster."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.constants import PITCH_CENTER_X_M, PITCH_CENTER_Y_M  # noqa: E402
from footballq.data.pff_forecasting import (  # noqa: E402
    PFFForecastDataset,
    prepare_pff_forecast_targets,
)
from footballq.data.sharded_td_dataset import ShardGroupedSampler  # noqa: E402
from footballq.data.td_jepa_projection import (  # noqa: E402
    project_td_jepa_feature_view,
)
from footballq.data.windows import ENTITY_BALL, ENTITY_PLAYER  # noqa: E402
from footballq.models.trajectory_forecaster import (  # noqa: E402
    MultiHorizonTrajectoryForecaster,
    predict_constant_velocity,
)
from footballq.repro.manifest import file_sha256  # noqa: E402
from footballq.repro.splits import load_split_manifest  # noqa: E402
from footballq.training.train import resolve_device  # noqa: E402
from footballq.training.train_td_jepa import create_td_jepa_model  # noqa: E402
from scripts.run_pff_trajectory_forecast_hybrid_context_v1 import (  # noqa: E402
    MEAN_KEYS,
    SEEDS,
    _hybrid_gate,
)

STUDY = "pff_trajectory_forecast_routed_test_v1"
LOCK_PATH = Path("configs/pff_trajectory_forecast_routed_test_v1.lock.json")
PROTOCOL_PATH = Path("docs/PFF_TRAJECTORY_FORECAST_ROUTED_TEST_PROTOCOL_V1.md")
SOURCE_MANIFEST = Path(
    "data/processed/pff_wc2022_td_jepa_v2/observed_only/dataset_manifest.json"
)
POSITION_ONLY_ROOT = Path(
    "data/processed/pff_wc2022_td_jepa_position_only_test_confirmatory_v1"
)
POSITION_ONLY_MANIFEST = POSITION_ONLY_ROOT / "observed_only/dataset_manifest.json"
FORECAST_ROOT = Path(
    "data/processed/pff_wc2022_trajectory_forecast_test_confirmatory_v1"
)
FORECAST_MANIFEST = FORECAST_ROOT / "dataset_manifest.json"
RUN_ROOT = Path("runs/pff_trajectory_forecast_routed_test_v1")
EXECUTION_PATH = RUN_ROOT / "execution_manifest.json"
METRICS_PATH = RUN_ROOT / "test_metrics.json"
SUMMARY_PATH = RUN_ROOT / "gate_summary.json"
AUDIT_PATH = Path(
    "runs/integrity/pff_trajectory_forecast_routed_test_v1_artifact_audit.json"
)
HORIZONS = (0.5, 1.0, 2.0, 4.0)
BOOTSTRAP_SEED = 20260717
BOOTSTRAP_REPLICATES = 10_000


def _path(value: str | Path) -> Path:
    value = Path(value)
    return value if value.is_absolute() else ROOT / value


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = _path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(_path(path).read_text(encoding="utf-8"))


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _load_lock() -> dict[str, Any]:
    lock = _read_json(LOCK_PATH)
    if lock.get("study") != STUDY or lock.get("status") != "frozen":
        raise ValueError("The routed-test lock is absent or not frozen.")
    if tuple(lock.get("seeds", [])) != SEEDS:
        raise ValueError("The routed-test seed set changed.")
    if tuple(lock.get("horizons_seconds", [])) != HORIZONS:
        raise ValueError("The routed-test horizons changed.")
    route = lock.get("route", {})
    if route != {
        "0.5": "constant_velocity",
        "1.0": "hybrid_context_raw",
        "2.0": "hybrid_context_raw",
        "4.0": "hybrid_context_raw",
    }:
        raise ValueError("The routed-test inference rule changed.")
    return lock


def _verify_frozen_inputs(lock: dict[str, Any]) -> dict[str, str]:
    actual: dict[str, str] = {}
    failures = []
    for name, record in lock["frozen_files"].items():
        path = _path(record["path"])
        if not path.exists():
            failures.append(f"{name}:missing")
            continue
        digest = file_sha256(path)
        actual[name] = digest
        if digest != record["sha256"]:
            failures.append(f"{name}:hash")
    if failures:
        raise ValueError("Frozen routed-test inputs changed: " + ", ".join(failures))

    split = load_split_manifest(_path(lock["split_manifest"]["path"]))
    if list(split.test_match_ids) != lock["test_match_ids"]:
        raise ValueError("Frozen routed-test match IDs changed.")
    if split.sha256 != lock["split_manifest"]["payload_sha256"]:
        raise ValueError("Frozen routed-test split payload changed.")
    return actual


def preflight() -> dict[str, Any]:
    lock = _load_lock()
    hashes = _verify_frozen_inputs(lock)
    existing = _path(SUMMARY_PATH).exists()
    state = {
        "version": 1,
        "study": STUDY,
        "status": "already_evaluated" if existing else "preflight_passed",
        "protocol_path": str(PROTOCOL_PATH),
        "lock_path": str(LOCK_PATH),
        "lock_sha256": file_sha256(_path(LOCK_PATH)),
        "runner_sha256": file_sha256(Path(__file__)),
        "frozen_file_hashes": hashes,
        "test_match_ids": list(lock["test_match_ids"]),
        "test_metric_previously_computed": existing,
        "test_targets_generated": _path(FORECAST_MANIFEST).exists(),
        "test_loaded_for_confirmation": existing,
        "created_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
    }
    if _path(EXECUTION_PATH).exists():
        previous = _read_json(EXECUTION_PATH)
        state["created_at_utc"] = previous.get("created_at_utc", state["created_at_utc"])
        if previous.get("lock_sha256") != state["lock_sha256"]:
            raise ValueError("The routed-test lock changed after preflight.")
    _write_json(EXECUTION_PATH, state)
    return state


def prepare_test_data(state: dict[str, Any]) -> dict[str, Any]:
    if _path(SUMMARY_PATH).exists():
        raise RuntimeError("The one-time routed confirmation has already been evaluated.")
    lock = _load_lock()
    _verify_frozen_inputs(lock)
    projected_path = project_td_jepa_feature_view(
        _path(SOURCE_MANIFEST),
        _path(POSITION_ONLY_ROOT),
        target_feature_view="position_only",
        included_splits={"test"},
    )
    if projected_path != _path(POSITION_ONLY_MANIFEST):
        raise ValueError(f"Unexpected test projection manifest: {projected_path}")
    projected = _read_json(POSITION_ONLY_MANIFEST)
    if (
        projected.get("included_splits") != ["test"]
        or projected.get("selected_match_ids") != lock["test_match_ids_sorted"]
        or projected.get("example_count") != lock["expected_test_examples"]
    ):
        raise ValueError("Projected PFF confirmation data does not match the frozen scope.")

    forecast = prepare_pff_forecast_targets(
        _path(POSITION_ONLY_MANIFEST),
        _path(FORECAST_ROOT),
        _path(lock["split_manifest"]["path"]),
        horizons_seconds=HORIZONS,
        included_splits=("test",),
        confirmatory_test=True,
    )
    if (
        forecast.get("included_splits") != ["test"]
        or forecast.get("test_included") is not True
        or forecast.get("included_match_ids") != lock["test_match_ids_sorted"]
        or forecast.get("example_count") != lock["expected_test_examples"]
    ):
        raise ValueError("Prepared PFF confirmation targets do not match the frozen scope.")

    state.update(
        {
            "status": "test_targets_prepared",
            "test_targets_generated": True,
            "test_metric_previously_computed": False,
            "test_loaded_for_confirmation": False,
            "position_only_manifest": str(POSITION_ONLY_MANIFEST),
            "position_only_manifest_sha256": file_sha256(projected_path),
            "position_only_manifest_payload_sha256": projected[
                "manifest_payload_sha256"
            ],
            "forecast_manifest": str(FORECAST_MANIFEST),
            "forecast_manifest_sha256": file_sha256(_path(FORECAST_MANIFEST)),
            "forecast_manifest_payload_sha256": forecast["manifest_payload_sha256"],
            "test_example_count": forecast["example_count"],
            "test_valid_endpoint_count": forecast["valid_endpoint_count"],
            "updated_at_utc": _utc_now(),
        }
    )
    _write_json(EXECUTION_PATH, state)
    return state


def _load_forecaster(
    checkpoint_path: str | Path,
    dataset: PFFForecastDataset,
    device: torch.device,
) -> MultiHorizonTrajectoryForecaster:
    checkpoint = torch.load(_path(checkpoint_path), map_location="cpu", weights_only=False)
    if checkpoint.get("family") != "raw" or int(checkpoint.get("step", -1)) != 2_000:
        raise ValueError(f"Unexpected forecast checkpoint scope: {checkpoint_path}")
    config = checkpoint["config"]
    source_path = _path(config["sources"]["tracking_checkpoint"])
    source_checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    source_model = create_td_jepa_model(source_checkpoint["config"], dataset.prototype)
    source_model_cfg = source_checkpoint["config"].get("model", {})
    model_cfg = config.get("model", {})
    model = MultiHorizonTrajectoryForecaster(
        source_model.online_encoder,
        family="raw",
        z_dim=int(source_model_cfg.get("z_dim", 128)),
        n_entities=int(dataset.prototype.state_t.shape[2]),
        horizons_seconds=HORIZONS,
        fps=float(dataset.prototype.fps),
        hidden_dim=int(model_cfg.get("hidden_dim", 512)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        representation_mode=str(model_cfg.get("representation_mode", "global")),
        token_dim=int(source_model_cfg.get("d_model", 128)),
        decoder_mode=str(model_cfg.get("decoder_mode", "shared")),
    )
    model.encoder.load_state_dict(checkpoint["encoder"], strict=True)
    model.decoder.load_state_dict(checkpoint["decoder"], strict=True)
    model.eval()
    return model.to(device)


def route_prediction(
    constant_velocity: torch.Tensor,
    hybrid: torch.Tensor,
) -> torch.Tensor:
    """Use constant velocity at 0.5 seconds and hybrid at later horizons."""

    if constant_velocity.shape != hybrid.shape or constant_velocity.shape[1] != 4:
        raise ValueError("Routed forecast inputs must share the frozen four-horizon shape.")
    routed = hybrid.clone()
    routed[:, 0] = constant_velocity[:, 0]
    return routed


def _new_accumulator() -> dict[str, Any]:
    return {
        "sums": {
            name: torch.zeros(len(HORIZONS), dtype=torch.float64)
            for name in ("all_entity", "player", "ball")
        },
        "counts": {
            name: torch.zeros(len(HORIZONS), dtype=torch.long)
            for name in ("all_entity", "player", "ball")
        },
        "by_match": defaultdict(_new_match_accumulator),
    }


def _new_match_accumulator() -> dict[str, Any]:
    return {
        "sums": {
            name: torch.zeros(len(HORIZONS), dtype=torch.float64)
            for name in ("all_entity", "player", "ball")
        },
        "counts": {
            name: torch.zeros(len(HORIZONS), dtype=torch.long)
            for name in ("all_entity", "player", "ball")
        },
        "num_examples": 0,
    }


def _accumulate(
    accumulator: dict[str, Any],
    prediction: torch.Tensor,
    batch: dict[str, Any],
    scale: torch.Tensor,
) -> None:
    distance = torch.linalg.vector_norm(
        (prediction - batch["future_xy"]) * scale.to(prediction.dtype), dim=-1
    )
    selectors = {
        "all_entity": torch.ones_like(batch["entity_type"], dtype=torch.bool),
        "player": batch["entity_type"] == ENTITY_PLAYER,
        "ball": batch["entity_type"] == ENTITY_BALL,
    }
    for name, selector in selectors.items():
        valid = batch["future_mask"] & selector.unsqueeze(1)
        accumulator["sums"][name] += (distance * valid).sum(dim=(0, 2)).double().cpu()
        accumulator["counts"][name] += valid.sum(dim=(0, 2)).long().cpu()

    match_ids = [str(value) for value in batch["match_id"]]
    for match_id in sorted(set(match_ids)):
        match_accumulator = accumulator["by_match"][match_id]
        example_selector = torch.tensor(
            [value == match_id for value in match_ids],
            dtype=torch.bool,
            device=prediction.device,
        )
        match_accumulator["num_examples"] += int(example_selector.sum().item())
        for name, selector in selectors.items():
            valid = (
                batch["future_mask"]
                & selector.unsqueeze(1)
                & example_selector.view(-1, 1, 1)
            )
            match_accumulator["sums"][name] += (
                (distance * valid).sum(dim=(0, 2)).double().cpu()
            )
            match_accumulator["counts"][name] += valid.sum(dim=(0, 2)).long().cpu()


def _metrics_from_totals(
    sums: dict[str, torch.Tensor],
    counts: dict[str, torch.Tensor],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {"horizons_seconds": list(HORIZONS)}
    for name in sums:
        per_horizon = sums[name] / counts[name].clamp_min(1)
        metrics[f"{name}_ADE_m"] = float(sums[name].sum() / counts[name].sum())
        metrics[f"{name}_FDE_m"] = float(per_horizon[-1])
        metrics[f"{name}_valid_endpoint_count"] = int(counts[name].sum())
        for index, horizon in enumerate(HORIZONS):
            label = str(horizon).replace(".", "p")
            metrics[f"{name}_error_h{label}s_m"] = float(per_horizon[index])
            metrics[f"{name}_count_h{label}s"] = int(counts[name][index])
    finite = [
        math.isfinite(value)
        for value in metrics.values()
        if isinstance(value, float)
    ]
    if not all(finite):
        raise ValueError("Confirmatory forecast evaluation produced non-finite metrics.")
    return metrics


def _finish_accumulator(
    accumulator: dict[str, Any],
    *,
    num_examples: int,
    sample_digest: str,
) -> dict[str, Any]:
    result = _metrics_from_totals(accumulator["sums"], accumulator["counts"])
    result.update({"num_examples": num_examples, "sample_id_sha256": sample_digest})
    result["by_match"] = {}
    for match_id, row in sorted(accumulator["by_match"].items()):
        metrics = _metrics_from_totals(row["sums"], row["counts"])
        metrics["num_examples"] = row["num_examples"]
        result["by_match"][match_id] = metrics
    return result


def _evaluate_seed(
    seed: int,
    checkpoints: dict[str, dict[str, str]],
    dataset: PFFForecastDataset,
    device: torch.device,
    *,
    include_constant_velocity: bool,
) -> dict[str, dict[str, Any]]:
    models = {
        name: _load_forecaster(checkpoints[name][str(seed)], dataset, device)
        for name in ("entity_raw", "global_raw", "hybrid_context_raw")
    }
    names = [*models, "routed_candidate"]
    if include_constant_velocity:
        names.append("constant_velocity")
    accumulators = {name: _new_accumulator() for name in names}
    loader = DataLoader(
        dataset,
        batch_size=128,
        sampler=ShardGroupedSampler(dataset, shuffle=False, seed=0),
        num_workers=0,
    )
    sample_digest = hashlib.sha256()
    num_examples = 0
    scale = torch.tensor([PITCH_CENTER_X_M, PITCH_CENTER_Y_M], device=device)
    with torch.inference_mode():
        for batch in loader:
            for sample_id in batch["sample_id"]:
                sample_digest.update(str(sample_id).encode("utf-8") + b"\n")
            num_examples += int(batch["state_t"].shape[0])
            device_batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            constant_velocity = predict_constant_velocity(
                device_batch["state_t"],
                device_batch["mask_t"],
                HORIZONS,
                fps=float(dataset.prototype.fps),
            )
            predictions = {
                name: model(device_batch["state_t"], device_batch["mask_t"])
                for name, model in models.items()
            }
            predictions["routed_candidate"] = route_prediction(
                constant_velocity, predictions["hybrid_context_raw"]
            )
            if include_constant_velocity:
                predictions["constant_velocity"] = constant_velocity
            for name, prediction in predictions.items():
                _accumulate(accumulators[name], prediction, device_batch, scale)
    digest = sample_digest.hexdigest()
    results = {
        name: _finish_accumulator(
            accumulator,
            num_examples=num_examples,
            sample_digest=digest,
        )
        for name, accumulator in accumulators.items()
    }
    del models
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def _means(rows: dict[int, dict[str, Any]]) -> dict[str, float]:
    return {
        key: sum(float(rows[seed][key]) for seed in SEEDS) / len(SEEDS)
        for key in MEAN_KEYS
    }


def _bootstrap_interval(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = rng.choice(array, size=(BOOTSTRAP_REPLICATES, len(array)), replace=True)
    means = sampled.mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {
        "unit": "held_out_match",
        "match_count": len(values),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "mean_improvement_m": float(array.mean()),
        "ci95_lower_m": float(lower),
        "ci95_upper_m": float(upper),
        "positive_means_candidate_is_better": True,
    }


def _uncertainty(
    routed_rows: dict[int, dict[str, Any]],
    entity_rows: dict[int, dict[str, Any]],
    global_rows: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    match_ids = sorted(routed_rows[SEEDS[0]]["by_match"])

    def paired_improvements(reference: dict[int, dict[str, Any]], key: str) -> list[float]:
        values = []
        for match_id in match_ids:
            candidate = np.mean(
                [routed_rows[seed]["by_match"][match_id][key] for seed in SEEDS]
            )
            baseline = np.mean(
                [reference[seed]["by_match"][match_id][key] for seed in SEEDS]
            )
            values.append(float(baseline - candidate))
        return values

    return {
        "player_ADE_vs_entity_raw": _bootstrap_interval(
            paired_improvements(entity_rows, "player_ADE_m")
        ),
        "ball_ADE_vs_global_raw": _bootstrap_interval(
            paired_improvements(global_rows, "ball_ADE_m")
        ),
        "all_entity_ADE_vs_global_raw": _bootstrap_interval(
            paired_improvements(global_rows, "all_entity_ADE_m")
        ),
    }


def evaluate_once(state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if _path(SUMMARY_PATH).exists() or _path(METRICS_PATH).exists():
        raise RuntimeError("The one-time routed confirmation result already exists.")
    lock = _load_lock()
    _verify_frozen_inputs(lock)
    if not _path(FORECAST_MANIFEST).exists():
        raise FileNotFoundError("Run the routed-test preparation stage first.")
    forecast = _read_json(FORECAST_MANIFEST)
    if (
        forecast.get("access_protocol") != "confirmatory_test_only_v1"
        or forecast.get("included_match_ids") != lock["test_match_ids_sorted"]
        or forecast.get("example_count") != lock["expected_test_examples"]
    ):
        raise ValueError("The routed-test forecast manifest does not match the frozen lock.")

    dataset = PFFForecastDataset(
        _path(FORECAST_MANIFEST),
        "test",
        allow_confirmatory_test=True,
    )
    device = resolve_device(lock["evaluation"]["device"])
    all_results: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in SEEDS:
        all_results[seed] = _evaluate_seed(
            seed,
            lock["checkpoints"],
            dataset,
            device,
            include_constant_velocity=seed == SEEDS[0],
        )

    digests = {
        result["sample_id_sha256"]
        for seed_results in all_results.values()
        for result in seed_results.values()
    }
    if len(digests) != 1:
        raise ValueError(f"Confirmatory evaluation sample digests differ: {digests}")
    routed_rows = {seed: all_results[seed]["routed_candidate"] for seed in SEEDS}
    entity_rows = {seed: all_results[seed]["entity_raw"] for seed in SEEDS}
    global_rows = {seed: all_results[seed]["global_raw"] for seed in SEEDS}
    hybrid_rows = {seed: all_results[seed]["hybrid_context_raw"] for seed in SEEDS}
    routed_mean = _means(routed_rows)
    entity_mean = _means(entity_rows)
    global_mean = _means(global_rows)
    hybrid_mean = _means(hybrid_rows)
    gate = _hybrid_gate(
        routed_rows,
        entity_rows,
        global_rows,
        routed_mean,
        entity_mean,
        global_mean,
    )
    status = "confirmed" if gate["passed"] else "not_confirmed"
    constant_velocity = all_results[SEEDS[0]]["constant_velocity"]
    metrics = {
        "version": 1,
        "study": STUDY,
        "split": "test",
        "test_match_ids": lock["test_match_ids"],
        "sample_id_sha256": next(iter(digests)),
        "example_count": len(dataset),
        "rows": {str(seed): all_results[seed] for seed in SEEDS},
        "constant_velocity": constant_velocity,
    }
    _write_json(METRICS_PATH, metrics)
    summary = {
        "version": 1,
        "study": STUDY,
        "status": status,
        "claim_scope": "held_out_trajectory_forecasting_only",
        "route": lock["route"],
        "seeds": list(SEEDS),
        "test_match_ids": lock["test_match_ids"],
        "test_example_count": len(dataset),
        "test_sample_id_sha256": next(iter(digests)),
        "routed_mean": routed_mean,
        "entity_raw_mean": entity_mean,
        "global_raw_mean": global_mean,
        "hybrid_context_raw_mean": hybrid_mean,
        "constant_velocity": {
            key: value
            for key, value in constant_velocity.items()
            if key != "by_match"
        },
        "gate": gate,
        "blocking_conditions": gate["blocking_conditions"],
        "paired_match_uncertainty": _uncertainty(
            routed_rows, entity_rows, global_rows
        ),
        "data_access": {
            "loaded_splits": ["test"],
            "test_targets_generated": True,
            "test_metric_computed_once": True,
            "training_performed": False,
            "checkpoint_selection_performed": False,
            "predictions_retained": False,
        },
        "integrity_note": (
            "The PFF test split was historically exposed to older representation code at "
            "the tensor level, but no prior trajectory outcome metric was computed."
        ),
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": file_sha256(_path(PROTOCOL_PATH)),
        "lock_path": str(LOCK_PATH),
        "lock_sha256": file_sha256(_path(LOCK_PATH)),
        "metrics_path": str(METRICS_PATH),
        "metrics_sha256": file_sha256(_path(METRICS_PATH)),
        "runner_sha256": file_sha256(Path(__file__)),
        "completed_at_utc": _utc_now(),
    }
    _write_json(SUMMARY_PATH, summary)
    state.update(
        {
            "status": status,
            "test_targets_generated": True,
            "test_loaded_for_confirmation": True,
            "test_metric_previously_computed": True,
            "test_metrics_path": str(METRICS_PATH),
            "test_metrics_sha256": file_sha256(_path(METRICS_PATH)),
            "gate_summary_path": str(SUMMARY_PATH),
            "gate_summary_sha256": file_sha256(_path(SUMMARY_PATH)),
            "updated_at_utc": _utc_now(),
        }
    )
    _write_json(EXECUTION_PATH, state)
    return metrics, summary


def verify_artifacts() -> dict[str, Any]:
    lock = _load_lock()
    frozen_hashes = _verify_frozen_inputs(lock)
    state = _read_json(EXECUTION_PATH)
    summary = _read_json(SUMMARY_PATH)
    checks = {
        "frozen_inputs_match": bool(frozen_hashes),
        "lock_hash_matches_preflight": state.get("lock_sha256")
        == file_sha256(_path(LOCK_PATH)),
        "runner_hash_matches_lock": lock["frozen_files"]["runner"]["sha256"]
        == file_sha256(Path(__file__)),
        "metrics_hash_matches": state.get("test_metrics_sha256")
        == file_sha256(_path(METRICS_PATH)),
        "summary_hash_matches": state.get("gate_summary_sha256")
        == file_sha256(_path(SUMMARY_PATH)),
        "test_only_access": summary.get("data_access", {}).get("loaded_splits")
        == ["test"],
        "no_training": summary.get("data_access", {}).get("training_performed") is False,
        "no_checkpoint_selection": summary.get("data_access", {}).get(
            "checkpoint_selection_performed"
        )
        is False,
        "complete_test_scope": summary.get("test_match_ids") == lock["test_match_ids"],
        "expected_example_count": summary.get("test_example_count")
        == lock["expected_test_examples"],
        "allowed_status": summary.get("status") in {"confirmed", "not_confirmed"},
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    report = {
        "version": 1,
        "study": STUDY,
        "status": "passed" if not failed else "blocked",
        "checks": checks,
        "failed_checks": failed,
        "gate_status": summary.get("status"),
        "lock_sha256": file_sha256(_path(LOCK_PATH)),
        "runner_sha256": file_sha256(Path(__file__)),
        "summary_sha256": file_sha256(_path(SUMMARY_PATH)),
        "metrics_sha256": file_sha256(_path(METRICS_PATH)),
    }
    _write_json(AUDIT_PATH, report)
    if failed:
        raise ValueError("Routed-test artifact audit failed: " + ", ".join(failed))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("preflight", "prepare", "evaluate", "verify", "all"),
        default="preflight",
    )
    args = parser.parse_args()
    state = preflight()
    if args.stage == "preflight":
        print(f"preflight_status: {state['status']}")
        print(f"lock_sha256: {state['lock_sha256']}")
        return
    if args.stage in {"prepare", "all"}:
        state = prepare_test_data(state)
        print(f"prepared_test_examples: {state['test_example_count']}")
    if args.stage in {"evaluate", "all"}:
        _metrics, summary = evaluate_once(state)
        print(f"confirmation_status: {summary['status']}")
        print(f"blocking_conditions: {summary['blocking_conditions']}")
    if args.stage in {"verify", "all"}:
        report = verify_artifacts()
        print(f"artifact_audit: {report['status']}")


if __name__ == "__main__":
    main()
