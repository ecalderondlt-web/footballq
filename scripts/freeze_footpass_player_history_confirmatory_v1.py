"""Freeze a passing FOOTPASS development experiment before confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from footballq.repro.manifest import file_sha256
from footballq.repro.splits import load_split_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PATH = REPO_ROOT / "src/footballq/analysis/footpass_player_history.py"
RUNNER_PATH = REPO_ROOT / "scripts/run_footpass_player_history_v1.py"
FREEZER_PATH = Path(__file__).resolve()


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _stable_payload_hash(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def _file_record(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": file_sha256(path),
    }


def _git_metadata() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_line_count": len(status),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/footpass_player_history_v1.yaml",
    )
    args = parser.parse_args()
    config_path = _resolve_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split_path = _resolve_path(config["data"]["split_manifest"])
    split = load_split_manifest(split_path)
    output_root = _resolve_path(config["data"]["output_root"])
    development_dir = output_root / "development"
    development_result_path = development_dir / "results.json"
    development_result = json.loads(
        development_result_path.read_text(encoding="utf-8")
    )
    if not bool(development_result["gate"]["passed"]):
        raise ValueError(
            "Development gate failed; confirmation must remain sealed."
        )
    if bool(development_result.get("confirmatory_metrics_loaded")):
        raise ValueError("Development result says confirmation was already loaded.")
    if development_result["split_manifest_sha256"] != split.sha256:
        raise ValueError("Development result split hash differs from the split.")

    freeze_path = _resolve_path(config["confirmation"]["freeze_manifest"])
    started_path = _resolve_path(config["confirmation"]["unseal_started"])
    completed_path = _resolve_path(config["confirmation"]["unseal_completed"])
    confirmation_result_path = output_root / "confirmatory/results.json"
    for path in (
        freeze_path,
        started_path,
        completed_path,
        confirmation_result_path,
    ):
        if path.exists():
            raise ValueError(
                f"Refusing to overwrite an existing freeze/unseal artifact: {path}."
            )

    cache_path = _resolve_path(config["data"]["cache_path"])
    cache_manifest_path = cache_path.with_suffix(".manifest.json")
    protocol_path = _resolve_path(config["data"]["protocol_doc"])
    identity_paths = [
        _resolve_path(value) for value in config["data"]["identity_manifests"]
    ]
    files = {
        "config": _file_record(config_path),
        "split_manifest": _file_record(split_path),
        "protocol_doc": _file_record(protocol_path),
        "analysis": _file_record(ANALYSIS_PATH),
        "runner": _file_record(RUNNER_PATH),
        "freezer": _file_record(FREEZER_PATH),
        "development_cache": _file_record(cache_path),
        "development_cache_manifest": _file_record(cache_manifest_path),
        "development_results": _file_record(development_result_path),
        "development_run_manifest": _file_record(
            development_dir / "run_manifest.json"
        ),
        "development_feature_audit": _file_record(
            development_dir / "feature_audit.json"
        ),
        "development_models": _file_record(development_dir / "models.npz"),
        "development_predictions": _file_record(
            development_dir / "validation_predictions.npz"
        ),
    }
    for index, path in enumerate(identity_paths):
        files[f"identity_manifest_{index}"] = _file_record(path)

    payload = {
        "name": "footpass_player_history_confirmatory_freeze_v1",
        "version": 1,
        "status": "metric_sealed_before_first_confirmatory_read",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment_protocol": str(config["experiment_protocol"]),
        "source_hdf5_sha256": str(config["data"]["hdf5_sha256"]),
        "split_manifest_sha256": split.sha256,
        "development_result_payload_sha256": development_result[
            "result_payload_sha256"
        ],
        "development_gate": development_result["gate"],
        "confirmation_match_ids": split.test_match_ids,
        "confirmation_action_labels_read": False,
        "confirmatory_metrics_loaded": False,
        "files": files,
        "git": _git_metadata(),
    }
    payload["manifest_payload_sha256"] = _stable_payload_hash(payload)
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    with freeze_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "freeze_manifest": str(freeze_path),
                "manifest_payload_sha256": payload[
                    "manifest_payload_sha256"
                ],
                "confirmation_match_ids": split.test_match_ids,
                "status": payload["status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
