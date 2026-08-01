"""Run the development-only FOOTPASS compact player-residual experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from footballq.analysis.footpass_compact_player_residual import (
    build_compact_player_residual_dataset,
    run_compact_player_residual_development,
)
from footballq.analysis.footpass_player_history import (
    load_extracted_footpass_data,
    load_footpass_appearances,
    save_logistic_probes,
)
from footballq.repro.manifest import (
    build_run_manifest,
    file_sha256,
    write_run_manifest,
)
from footballq.repro.splits import load_split_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PATH = (
    REPO_ROOT
    / "src/footballq/analysis/footpass_compact_player_residual.py"
)
BASE_ANALYSIS_PATH = (
    REPO_ROOT / "src/footballq/analysis/footpass_player_history.py"
)
RUNNER_PATH = Path(__file__).resolve()


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _stable_payload_hash(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


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
    metadata: dict[str, Any] = {
        "row_count": len(selected),
        "prediction_keys": [],
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
        metadata["prediction_keys"].append(
            {
                "key": key,
                "array": array_key,
            }
        )
        arrays[array_key] = np.asarray(values, dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        **arrays,
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/footpass_compact_player_residual_v2.yaml",
    )
    args = parser.parse_args()
    config_path = _resolve_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("V2 config must be a mapping.")

    split_path = _resolve_path(config["data"]["split_manifest"])
    split = load_split_manifest(split_path)
    expected_source_sha = str(config["data"]["source_hdf5_sha256"])
    if str(split.payload["source_hdf5_sha256"]) != expected_source_sha:
        raise ValueError("V2 config and split source hashes differ.")
    configured_confirmation = {
        str(value) for value in config["confirmation"]["match_ids"]
    }
    if configured_confirmation != set(split.test_match_ids):
        raise ValueError("V2 confirmation IDs differ from the immutable split.")
    if not bool(config["confirmation"]["development_runner_must_not_unseal"]):
        raise ValueError("V2 development runner requires an explicit no-unseal rule.")

    cache_path = _resolve_path(config["data"]["development_cache"])
    cache_manifest_path = _resolve_path(
        config["data"]["development_cache_manifest"]
    )
    cache_manifest = json.loads(
        cache_manifest_path.read_text(encoding="utf-8")
    )
    if str(cache_manifest["source_hdf5_sha256"]) != expected_source_sha:
        raise ValueError("Development cache source hash differs from V2.")
    if str(cache_manifest["split_manifest_sha256"]) != split.sha256:
        raise ValueError("Development cache split hash differs from V2.")
    if file_sha256(cache_path) != str(cache_manifest["cache_sha256"]):
        raise ValueError("Development cache file hash is invalid.")
    if cache_manifest["confirmation_match_ids_included"]:
        raise ValueError("Development cache manifest contains confirmation IDs.")
    expected_development_ids = set(split.train_match_ids) | set(
        split.val_match_ids
    )
    if set(cache_manifest["selected_match_ids"]) != expected_development_ids:
        raise ValueError("Development cache manifest has the wrong match IDs.")

    identity_paths = [
        _resolve_path(value) for value in config["data"]["identity_manifests"]
    ]
    appearances = load_footpass_appearances(identity_paths)
    if {item.match_id for item in appearances} != set(split.all_match_ids):
        raise ValueError("Identity manifests and immutable split differ.")
    data = load_extracted_footpass_data(cache_path)
    if set(data.metadata["selected_match_ids"]) != expected_development_ids:
        raise ValueError("Loaded cache has the wrong development match IDs.")
    if data.metadata["confirmation_match_ids_included"]:
        raise ValueError("Loaded cache contains confirmation matches.")

    protocol_path = _resolve_path(config["data"]["protocol_doc"])
    v1_result_path = _resolve_path(config["data"]["v1_development_results"])
    v1_result = json.loads(v1_result_path.read_text(encoding="utf-8"))
    if bool(v1_result["gate"]["passed"]):
        raise ValueError("V2 motivation expects the recorded V1 gate failure.")
    dataset = build_compact_player_residual_dataset(
        data,
        appearances,
        config,
        query_partitions={"development_train", "development_validation"},
    )
    development_fit_match_ids = {
        str(value) for value in split.payload["development_fit_match_ids"]
    }
    development_validation_match_ids = {
        str(value)
        for value in split.payload["development_validation_match_ids"]
    }
    result, probes, validation_probabilities = (
        run_compact_player_residual_development(
            dataset,
            config,
            development_fit_match_ids=development_fit_match_ids,
            development_validation_match_ids=(
                development_validation_match_ids
            ),
        )
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
        validation_match_ids=development_validation_match_ids,
        probabilities=validation_probabilities,
    )
    result.update(
        {
            "experiment_protocol": str(config["experiment_protocol"]),
            "source_hdf5_sha256": expected_source_sha,
            "split_manifest_sha256": split.sha256,
            "config_sha256": file_sha256(config_path),
            "protocol_sha256": file_sha256(protocol_path),
            "analysis_sha256": file_sha256(ANALYSIS_PATH),
            "base_analysis_sha256": file_sha256(BASE_ANALYSIS_PATH),
            "runner_sha256": file_sha256(RUNNER_PATH),
            "development_cache_sha256": file_sha256(cache_path),
            "development_cache_manifest_sha256": file_sha256(
                cache_manifest_path
            ),
            "v1_development_result_sha256": file_sha256(v1_result_path),
            "feature_audit_sha256": file_sha256(audit_path),
            "model_sha256": file_sha256(model_path),
            "validation_predictions_sha256": file_sha256(predictions_path),
            "confirmation_match_ids_loaded": [],
            "confirmatory_metrics_loaded": False,
        }
    )
    result["result_payload_sha256"] = _stable_payload_hash(result)
    _write_json(result_path, result)

    manifest = build_run_manifest(
        command=sys.argv,
        config_path=config_path,
        split_manifest_path=split_path,
        evaluation_protocol=str(config["experiment_protocol"]),
        feature_view="compact_role_prior_plus_shrunk_player_residual",
        objective_mode="unweighted_binary_logistic_nll",
        dataset_paths={
            "development_cache": cache_path,
            "development_cache_manifest": cache_manifest_path,
            "v1_development_results": v1_result_path,
            **{
                f"identity_manifest_{index}": path
                for index, path in enumerate(identity_paths)
            },
        },
        output_paths={
            "result": result_path,
            "models": model_path,
            "validation_predictions": predictions_path,
            "feature_audit": audit_path,
        },
        warnings=[
            "V2 design was informed by already-seen V1 development validation.",
            "Internal match-group CV is a development stability check, not an untouched test.",
            "FOOTPASS has no ball coordinates; the primary target is an action-location proxy.",
            "Confirmation matches 22, 40, and 43 were not loaded by this runner.",
        ],
    )
    manifest.update(
        {
            "source_hdf5_sha256": expected_source_sha,
            "protocol_sha256": file_sha256(protocol_path),
            "analysis_sha256": file_sha256(ANALYSIS_PATH),
            "base_analysis_sha256": file_sha256(BASE_ANALYSIS_PATH),
            "runner_sha256": file_sha256(RUNNER_PATH),
            "result_sha256": file_sha256(result_path),
            "result_payload_sha256": result["result_payload_sha256"],
            "model_sha256": file_sha256(model_path),
            "feature_audit_sha256": file_sha256(audit_path),
            "confirmation_match_ids_loaded": [],
            "confirmatory_metrics_loaded": False,
        }
    )
    write_run_manifest(manifest_path, manifest)

    if bool(result["gate"]["passed"]):
        eligibility_path = development_dir / "ELIGIBLE_FOR_FREEZE.json"
        eligibility = {
            "name": "footpass_compact_player_residual_v2_freeze_eligibility",
            "version": 1,
            "status": "development_gate_passed_confirmation_still_sealed",
            "result_path": str(result_path),
            "result_sha256": file_sha256(result_path),
            "result_payload_sha256": result["result_payload_sha256"],
            "run_manifest_path": str(manifest_path),
            "run_manifest_sha256": file_sha256(manifest_path),
            "confirmation_match_ids_loaded": [],
        }
        _write_json(eligibility_path, eligibility)

    print(
        json.dumps(
            {
                "results": str(result_path),
                "run_manifest": str(manifest_path),
                "gate": result["gate"],
                "primary_comparison": result["primary_comparison"],
                "secondary_comparison": result["secondary_comparison"],
                "internal_cv": {
                    "primary_relative_nll_improvement": result["internal_cv"][
                        "primary_relative_nll_improvement"
                    ],
                    "folds": {
                        name: {
                            "primary_nll_gain": values["primary_nll_gain"],
                            "primary_relative_nll_improvement": values[
                                "primary_relative_nll_improvement"
                            ],
                        }
                        for name, values in result["internal_cv"]["folds"].items()
                    },
                },
                "confirmation_match_ids_loaded": [],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
