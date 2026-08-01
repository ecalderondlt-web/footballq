"""Run the sealed-development FOOTPASS player-history experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from footballq.analysis.footpass_player_history import (
    ExtractedFootpassData,
    build_footpass_feature_dataset,
    combine_extracted_footpass_data,
    evaluate_frozen_probes,
    extract_footpass_experiment_data,
    load_extracted_footpass_data,
    load_footpass_appearances,
    load_logistic_probes,
    run_development_probes,
    save_extracted_footpass_data,
    save_logistic_probes,
)
from footballq.repro.manifest import (
    build_run_manifest,
    file_sha256,
    write_run_manifest,
)
from footballq.repro.splits import SplitManifest, load_split_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PATH = REPO_ROOT / "src/footballq/analysis/footpass_player_history.py"
RUNNER_PATH = Path(__file__).resolve()


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"FOOTPASS config must be a mapping: {path}.")
    return payload


def _write_json(path: Path, payload: Any, *, exclusive: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _stable_payload_hash(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def _file_record(path: Path) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
    }


def _validate_protocol_inputs(
    config: dict[str, Any],
) -> tuple[Path, str, SplitManifest, list[Any], list[Path]]:
    data_config = config["data"]
    hdf5_path = _resolve_path(data_config["hdf5_path"])
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"FOOTPASS HDF5 is missing: {hdf5_path}.")
    expected_hdf5_sha = str(data_config["hdf5_sha256"])
    print("Hashing the 8.8 GB FOOTPASS source before reading labels.", flush=True)
    actual_hdf5_sha = file_sha256(hdf5_path)
    if actual_hdf5_sha != expected_hdf5_sha:
        raise ValueError(
            "FOOTPASS source hash mismatch: "
            f"expected {expected_hdf5_sha}, got {actual_hdf5_sha}."
        )

    split_path = _resolve_path(data_config["split_manifest"])
    split = load_split_manifest(split_path)
    if split.payload["dataset"] != "footpass":
        raise ValueError("The configured split is not a FOOTPASS split.")
    if str(split.payload["source_hdf5_sha256"]) != actual_hdf5_sha:
        raise ValueError("Split source hash does not match the FOOTPASS source.")

    identity_paths = [
        _resolve_path(value) for value in data_config["identity_manifests"]
    ]
    for identity_path in identity_paths:
        identity_payload = json.loads(identity_path.read_text(encoding="utf-8"))
        if str(identity_payload["source_hdf5_sha256"]) != actual_hdf5_sha:
            raise ValueError(
                f"Identity source hash mismatch in {identity_path}."
            )
    appearances = load_footpass_appearances(identity_paths)
    appearance_match_ids = {item.match_id for item in appearances}
    if appearance_match_ids != set(split.all_match_ids):
        raise ValueError(
            "Identity appearances and immutable split contain different matches."
        )
    confirmation_from_identity = {
        item.match_id
        for item in appearances
        if item.partition == "confirmatory_reserve_do_not_read_until_frozen"
    }
    if confirmation_from_identity != set(split.test_match_ids):
        raise ValueError(
            "Identity confirmation partition differs from the immutable split."
        )
    return hdf5_path, actual_hdf5_sha, split, appearances, identity_paths


def _verify_cache_manifest(
    *,
    cache_path: Path,
    cache_manifest_path: Path,
    source_sha256: str,
    split: SplitManifest,
    selected_match_ids: set[str],
    config: dict[str, Any],
) -> None:
    if not cache_manifest_path.is_file():
        raise ValueError(
            f"Existing cache lacks a provenance manifest: {cache_manifest_path}. "
            "Use --rebuild-cache."
        )
    manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    expected = {
        "source_hdf5_sha256": source_sha256,
        "split_manifest_sha256": split.sha256,
        "analysis_sha256": file_sha256(ANALYSIS_PATH),
        "selected_match_ids": sorted(selected_match_ids, key=int),
        "tracking_stride_frames": int(
            config["features"]["tracking_sample_stride_frames"]
        ),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"Existing cache provenance differs at {key!r}; use --rebuild-cache."
            )
    if file_sha256(cache_path) != str(manifest["cache_sha256"]):
        raise ValueError("Existing FOOTPASS extraction cache hash is invalid.")
    if manifest.get("confirmation_match_ids_included"):
        raise ValueError("Development cache provenance contains confirmation matches.")


def _extract_or_load_development_cache(
    *,
    config: dict[str, Any],
    hdf5_path: Path,
    source_sha256: str,
    split: SplitManifest,
    appearances: list[Any],
    rebuild_cache: bool,
) -> tuple[ExtractedFootpassData, Path, Path]:
    cache_path = _resolve_path(config["data"]["cache_path"])
    cache_manifest_path = cache_path.with_suffix(".manifest.json")
    selected_match_ids = set(split.train_match_ids) | set(split.val_match_ids)
    if selected_match_ids & set(split.test_match_ids):
        raise ValueError("Development extraction selection overlaps confirmation.")
    if cache_path.is_file() and not rebuild_cache:
        _verify_cache_manifest(
            cache_path=cache_path,
            cache_manifest_path=cache_manifest_path,
            source_sha256=source_sha256,
            split=split,
            selected_match_ids=selected_match_ids,
            config=config,
        )
        data = load_extracted_footpass_data(cache_path)
    else:
        data = extract_footpass_experiment_data(
            hdf5_path,
            appearances,
            selected_match_ids=selected_match_ids,
            tracking_stride_frames=int(
                config["features"]["tracking_sample_stride_frames"]
            ),
            x_edges=[
                float(value) for value in config["features"]["spatial_x_bins"]
            ],
            y_edges=[
                float(value) for value in config["features"]["spatial_y_bins"]
            ],
            progress=lambda message: print(message, flush=True),
        )
        save_extracted_footpass_data(cache_path, data)
        cache_manifest = {
            "name": "footpass_player_history_development_cache_v1",
            "version": 1,
            "source_hdf5_sha256": source_sha256,
            "split_manifest_sha256": split.sha256,
            "analysis_sha256": file_sha256(ANALYSIS_PATH),
            "selected_match_ids": sorted(selected_match_ids, key=int),
            "tracking_stride_frames": int(
                config["features"]["tracking_sample_stride_frames"]
            ),
            "confirmation_match_ids_included": data.metadata[
                "confirmation_match_ids_included"
            ],
            "cache_path": str(cache_path),
            "cache_sha256": file_sha256(cache_path),
        }
        cache_manifest["manifest_payload_sha256"] = _stable_payload_hash(
            cache_manifest
        )
        _write_json(cache_manifest_path, cache_manifest)

    if set(data.metadata["selected_match_ids"]) != selected_match_ids:
        raise ValueError("Development cache has the wrong match selection.")
    if data.metadata["confirmation_match_ids_included"]:
        raise ValueError("Development cache opened confirmation action labels.")
    return data, cache_path, cache_manifest_path


def _save_validation_predictions(
    path: Path,
    *,
    dataset: Any,
    validation_match_ids: set[str],
    probabilities: dict[str, np.ndarray],
) -> Path:
    selected = np.asarray(
        [
            index
            for index, match_id in enumerate(dataset.match_ids)
            if match_id in validation_match_ids
        ],
        dtype=np.int64,
    )
    metadata = {
        "prediction_keys": [],
        "row_count": len(selected),
    }
    arrays: dict[str, np.ndarray] = {
        "sample_id": np.asarray(
            [dataset.sample_ids[index] for index in selected.tolist()]
        ),
        "match_id": np.asarray(
            [dataset.match_ids[index] for index in selected.tolist()]
        ),
        "period": np.asarray(
            [dataset.periods[index] for index in selected.tolist()],
            dtype=np.int8,
        ),
        "frame": np.asarray(
            [dataset.frames[index] for index in selected.tolist()],
            dtype=np.int64,
        ),
    }
    for target, labels in sorted(dataset.labels.items()):
        arrays[f"label_{target}"] = labels[selected]
    for index, (key, values) in enumerate(sorted(probabilities.items())):
        array_key = f"prediction_{index}"
        metadata["prediction_keys"].append({"key": key, "array": array_key})
        arrays[array_key] = np.asarray(values, dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        **arrays,
    )
    return path


def _development_run(
    *,
    args: argparse.Namespace,
    config_path: Path,
    config: dict[str, Any],
    hdf5_path: Path,
    source_sha256: str,
    split: SplitManifest,
    appearances: list[Any],
    identity_paths: list[Path],
) -> None:
    data, cache_path, cache_manifest_path = _extract_or_load_development_cache(
        config=config,
        hdf5_path=hdf5_path,
        source_sha256=source_sha256,
        split=split,
        appearances=appearances,
        rebuild_cache=bool(args.rebuild_cache),
    )
    dataset = build_footpass_feature_dataset(
        data,
        appearances,
        config,
        query_partitions={"development_train", "development_validation"},
    )
    development_fit = {
        str(value) for value in split.payload["development_fit_match_ids"]
    }
    development_validation = {
        str(value)
        for value in split.payload["development_validation_match_ids"]
    }
    if development_fit != set(split.train_match_ids) - set(
        split.payload["profile_support_only_match_ids"]
    ):
        raise ValueError("Development fit IDs do not match train minus support.")
    if development_validation != set(split.val_match_ids):
        raise ValueError("Development validation IDs differ from val IDs.")

    result, probes, probabilities = run_development_probes(
        dataset,
        config,
        train_match_ids=development_fit,
        validation_match_ids=development_validation,
    )
    output_root = _resolve_path(config["data"]["output_root"])
    development_dir = output_root / "development"
    audit_path = development_dir / "feature_audit.json"
    model_path = development_dir / "models.npz"
    predictions_path = development_dir / "validation_predictions.npz"
    result_path = development_dir / "results.json"
    manifest_path = development_dir / "run_manifest.json"
    _write_json(audit_path, dataset.audit)
    save_logistic_probes(model_path, probes)
    _save_validation_predictions(
        predictions_path,
        dataset=dataset,
        validation_match_ids=development_validation,
        probabilities=probabilities,
    )
    result.update(
        {
            "experiment_protocol": str(config["experiment_protocol"]),
            "source_hdf5_sha256": source_sha256,
            "split_manifest_sha256": split.sha256,
            "config_sha256": file_sha256(config_path),
            "analysis_sha256": file_sha256(ANALYSIS_PATH),
            "runner_sha256": file_sha256(RUNNER_PATH),
            "cache_sha256": file_sha256(cache_path),
            "cache_manifest_sha256": file_sha256(cache_manifest_path),
            "model_sha256": file_sha256(model_path),
            "validation_predictions_sha256": file_sha256(predictions_path),
            "feature_audit_sha256": file_sha256(audit_path),
            "development_metrics_loaded": True,
            "confirmatory_metrics_loaded": False,
        }
    )
    result["result_payload_sha256"] = _stable_payload_hash(result)
    _write_json(result_path, result)

    dataset_paths = {
        "footpass_hdf5": hdf5_path,
        "development_cache": cache_path,
        **{
            f"identity_manifest_{index}": path
            for index, path in enumerate(identity_paths)
        },
    }
    manifest = build_run_manifest(
        command=sys.argv,
        config_path=config_path,
        split_manifest_path=split.path,
        evaluation_protocol=str(config["experiment_protocol"]),
        feature_view="geometry_to_strict_prior_player_history_ladder",
        objective_mode="unweighted_binary_logistic_nll",
        dataset_paths=dataset_paths,
        output_paths={
            "result": result_path,
            "models": model_path,
            "validation_predictions": predictions_path,
            "feature_audit": audit_path,
            "cache_manifest": cache_manifest_path,
        },
        warnings=[
            "FOOTPASS has no ball coordinates; the primary target is an action-location proxy.",
            "The cohort contains three repeated teams and is not a population estimate.",
            "Confirmation action labels remained unopened during this run.",
        ],
    )
    manifest.update(
        {
            "source_hdf5_sha256": source_sha256,
            "analysis_sha256": file_sha256(ANALYSIS_PATH),
            "runner_sha256": file_sha256(RUNNER_PATH),
            "result_sha256": file_sha256(result_path),
            "result_payload_sha256": result["result_payload_sha256"],
            "model_sha256": file_sha256(model_path),
            "cache_sha256": file_sha256(cache_path),
            "feature_audit_sha256": file_sha256(audit_path),
            "confirmation_match_ids_loaded": [],
            "confirmatory_metrics_loaded": False,
        }
    )
    write_run_manifest(manifest_path, manifest)
    print(
        json.dumps(
            {
                "results": str(result_path),
                "run_manifest": str(manifest_path),
                "gate": result["gate"],
                "primary_comparison": result["primary_comparison"],
                "secondary_comparison": result["secondary_comparison"],
            },
            indent=2,
        ),
        flush=True,
    )


def _verify_file_record(record: dict[str, Any], label: str) -> None:
    path = Path(str(record["path"]))
    if not path.is_file():
        raise ValueError(f"Frozen {label} is missing: {path}.")
    actual = file_sha256(path)
    if actual != str(record["sha256"]):
        raise ValueError(
            f"Frozen {label} changed: expected {record['sha256']}, got {actual}."
        )


def _verify_confirmatory_freeze(
    *,
    config: dict[str, Any],
    source_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    freeze_path = _resolve_path(config["confirmation"]["freeze_manifest"])
    if not freeze_path.is_file():
        raise ValueError("Confirmation requires a completed freeze manifest.")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "metric_sealed_before_first_confirmatory_read":
        raise ValueError("The confirmation freeze is not in a sealed state.")
    if freeze.get("source_hdf5_sha256") != source_sha256:
        raise ValueError("The frozen source hash differs from the current source.")
    for label, record in freeze["files"].items():
        _verify_file_record(record, label)
    development_result_path = Path(
        str(freeze["files"]["development_results"]["path"])
    )
    development_result = json.loads(
        development_result_path.read_text(encoding="utf-8")
    )
    if not bool(development_result["gate"]["passed"]):
        raise ValueError("A failed development gate cannot open confirmation.")
    if (
        development_result["result_payload_sha256"]
        != freeze["development_result_payload_sha256"]
    ):
        raise ValueError("Frozen development result payload has changed.")
    return freeze_path, freeze


def _confirmatory_run(
    *,
    config_path: Path,
    config: dict[str, Any],
    hdf5_path: Path,
    source_sha256: str,
    split: SplitManifest,
    appearances: list[Any],
    identity_paths: list[Path],
) -> None:
    freeze_path, freeze = _verify_confirmatory_freeze(
        config=config,
        source_sha256=source_sha256,
    )
    output_root = _resolve_path(config["data"]["output_root"])
    run_dir = output_root / "confirmatory"
    started_path = _resolve_path(config["confirmation"]["unseal_started"])
    completed_path = _resolve_path(config["confirmation"]["unseal_completed"])
    result_path = run_dir / "results.json"
    for path in (started_path, completed_path, result_path):
        if path.exists():
            raise ValueError(
                "Confirmation was already opened or attempted; reruns are refused."
            )
    started = {
        "name": "footpass_player_history_confirmation_unseal_v1",
        "version": 1,
        "status": "confirmation_action_labels_about_to_be_read",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "freeze_manifest_path": str(freeze_path),
        "freeze_manifest_sha256": file_sha256(freeze_path),
        "source_hdf5_sha256": source_sha256,
    }
    _write_json(started_path, started, exclusive=True)

    confirmation_match_ids = set(split.test_match_ids)
    confirmation_cache = extract_footpass_experiment_data(
        hdf5_path,
        appearances,
        selected_match_ids=confirmation_match_ids,
        tracking_stride_frames=int(
            config["features"]["tracking_sample_stride_frames"]
        ),
        x_edges=[
            float(value) for value in config["features"]["spatial_x_bins"]
        ],
        y_edges=[
            float(value) for value in config["features"]["spatial_y_bins"]
        ],
        progress=lambda message: print(message, flush=True),
    )
    confirmation_cache_path = run_dir / "confirmation_cache.npz"
    save_extracted_footpass_data(confirmation_cache_path, confirmation_cache)
    if set(
        confirmation_cache.metadata["confirmation_match_ids_included"]
    ) != confirmation_match_ids:
        raise ValueError("Confirmation cache does not contain the frozen test IDs.")

    development_cache_path = Path(
        str(freeze["files"]["development_cache"]["path"])
    )
    development_cache = load_extracted_footpass_data(development_cache_path)
    combined = combine_extracted_footpass_data(
        development_cache,
        confirmation_cache,
    )
    dataset = build_footpass_feature_dataset(
        combined,
        appearances,
        config,
        query_partitions={
            "confirmatory_reserve_do_not_read_until_frozen",
        },
    )
    model_path = Path(str(freeze["files"]["development_models"]["path"]))
    probes = load_logistic_probes(model_path)
    result = evaluate_frozen_probes(dataset, config, probes)
    audit_path = run_dir / "feature_audit.json"
    manifest_path = run_dir / "run_manifest.json"
    _write_json(audit_path, dataset.audit)
    result.update(
        {
            "experiment_protocol": str(config["experiment_protocol"]),
            "source_hdf5_sha256": source_sha256,
            "split_manifest_sha256": split.sha256,
            "freeze_manifest_sha256": file_sha256(freeze_path),
            "development_result_payload_sha256": freeze[
                "development_result_payload_sha256"
            ],
            "model_sha256": file_sha256(model_path),
            "confirmation_cache_sha256": file_sha256(confirmation_cache_path),
            "feature_audit_sha256": file_sha256(audit_path),
        }
    )
    result["result_payload_sha256"] = _stable_payload_hash(result)
    _write_json(result_path, result)

    manifest = build_run_manifest(
        command=sys.argv,
        config_path=config_path,
        split_manifest_path=split.path,
        evaluation_protocol=str(config["experiment_protocol"]),
        feature_view="frozen_strict_prior_player_history_confirmation",
        objective_mode="frozen_unweighted_binary_logistic_nll",
        dataset_paths={
            "footpass_hdf5": hdf5_path,
            "development_cache": development_cache_path,
            "confirmation_cache": confirmation_cache_path,
            **{
                f"identity_manifest_{index}": path
                for index, path in enumerate(identity_paths)
            },
        },
        output_paths={
            "result": result_path,
            "feature_audit": audit_path,
            "freeze_manifest": freeze_path,
        },
        warnings=[
            "Confirmation was opened once after a passing frozen development gate.",
            "FOOTPASS has no ball coordinates; the primary target is an action-location proxy.",
            "The cohort contains only one later match for each of three focal teams.",
        ],
    )
    manifest.update(
        {
            "source_hdf5_sha256": source_sha256,
            "analysis_sha256": file_sha256(ANALYSIS_PATH),
            "runner_sha256": file_sha256(RUNNER_PATH),
            "result_sha256": file_sha256(result_path),
            "result_payload_sha256": result["result_payload_sha256"],
            "model_sha256": file_sha256(model_path),
            "freeze_manifest_sha256": file_sha256(freeze_path),
            "confirmation_match_ids_loaded": sorted(
                confirmation_match_ids,
                key=int,
            ),
            "confirmatory_metrics_loaded": True,
        }
    )
    write_run_manifest(manifest_path, manifest)
    completion = {
        "name": "footpass_player_history_confirmation_completion_v1",
        "version": 1,
        "status": "confirmatory_run_completed_once",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "freeze_manifest_sha256": file_sha256(freeze_path),
        "unseal_started_sha256": file_sha256(started_path),
        "result_sha256": file_sha256(result_path),
        "result_payload_sha256": result["result_payload_sha256"],
        "run_manifest_sha256": file_sha256(manifest_path),
        "gate_passed": bool(result["gate"]["passed"]),
    }
    _write_json(completed_path, completion, exclusive=True)
    print(
        json.dumps(
            {
                "results": str(result_path),
                "run_manifest": str(manifest_path),
                "gate": result["gate"],
                "primary_comparison": result["primary_comparison"],
                "secondary_comparison": result["secondary_comparison"],
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/footpass_player_history_v1.yaml",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--unseal-confirmatory", action="store_true")
    args = parser.parse_args()
    config_path = _resolve_path(args.config)
    config = _load_config(config_path)
    (
        hdf5_path,
        source_sha256,
        split,
        appearances,
        identity_paths,
    ) = _validate_protocol_inputs(config)
    if args.unseal_confirmatory:
        if args.rebuild_cache:
            raise ValueError("--rebuild-cache cannot be used during confirmation.")
        _confirmatory_run(
            config_path=config_path,
            config=config,
            hdf5_path=hdf5_path,
            source_sha256=source_sha256,
            split=split,
            appearances=appearances,
            identity_paths=identity_paths,
        )
    else:
        _development_run(
            args=args,
            config_path=config_path,
            config=config,
            hdf5_path=hdf5_path,
            source_sha256=source_sha256,
            split=split,
            appearances=appearances,
            identity_paths=identity_paths,
        )


if __name__ == "__main__":
    main()
