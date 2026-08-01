"""Unseal and run the frozen StatsBomb recipient-history replication once."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from footballq.analysis.statsbomb_recipient_history import (
    build_development_cache,
    evaluate_frozen_recipient_cache,
    load_config,
)
from footballq.analysis.wyscout_player_memory import (
    file_sha256,
    stable_payload_hash,
)

ANALYSIS_PATH = Path("src/footballq/analysis/statsbomb_recipient_history.py")
RUNNER_PATH = Path(
    "scripts/run_statsbomb_recipient_history_confirmatory_v1.py"
)


def _git_metadata() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_line_count": len(status),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _verify_file_record(record: dict[str, Any], label: str) -> None:
    path = Path(str(record["path"]))
    if not path.is_file():
        raise ValueError(f"Frozen {label} file is missing: {path}")
    actual = file_sha256(path)
    expected = str(record["sha256"])
    if actual != expected:
        raise ValueError(
            f"Frozen {label} hash changed: expected {expected}, got {actual}."
        )


def _verify_freeze(
    *,
    config: dict[str, Any],
    config_path: Path,
    output_root: Path,
    development_result: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    freeze_path = Path(config["data"]["confirmatory_freeze_manifest"])
    if not freeze_path.is_file():
        raise ValueError("Confirmatory run requires an existing freeze manifest.")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "metric_sealed_before_first_confirmatory_read":
        raise ValueError("Confirmatory freeze is not in the sealed state.")
    if str(freeze["output_root"]) != str(output_root):
        raise ValueError("Confirmatory output root differs from the frozen path.")
    for label in (
        "config",
        "analysis",
        "runner",
        "development_runner",
        "source_inventory_builder",
        "split_builder",
        "source_inventory",
        "split_manifest",
        "protocol_doc",
        "development_config",
        "development_results",
        "development_run_manifest",
        "development_source_audit",
    ):
        _verify_file_record(freeze["files"][label], label)
    if Path(str(freeze["files"]["config"]["path"])) != config_path:
        raise ValueError("Runtime config path differs from the frozen path.")
    if str(development_result["result_payload_sha256"]) != str(
        freeze["development_result_payload_sha256"]
    ):
        raise ValueError("Development result payload differs from the freeze.")
    run_dir = output_root / "confirmatory"
    if (run_dir / "UNSEAL_STARTED.json").exists() or (
        run_dir / "results.json"
    ).exists():
        raise ValueError(
            "Confirmatory tournament labels were already opened; reruns are refused."
        )
    return freeze_path, freeze


def _start_unseal(
    *,
    output_root: Path,
    freeze_path: Path,
    config_path: Path,
) -> Path:
    run_dir = output_root / "confirmatory"
    run_dir.mkdir(parents=True, exist_ok=True)
    sentinel_path = run_dir / "UNSEAL_STARTED.json"
    payload = {
        "name": "statsbomb_recipient_history_confirmatory_v1_unseal",
        "version": 1,
        "status": "confirmatory_recipient_metrics_about_to_be_loaded",
        "started_utc": datetime.now(UTC).isoformat(),
        "freeze_manifest_path": str(freeze_path),
        "freeze_manifest_sha256": file_sha256(freeze_path),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "git": _git_metadata(),
    }
    with sentinel_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
    return sentinel_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/statsbomb_recipient_history_confirmatory_v1.yaml",
    )
    parser.add_argument("--unseal-confirmatory", action="store_true")
    args = parser.parse_args()
    if not args.unseal_confirmatory:
        raise ValueError("Pass --unseal-confirmatory for the one permitted run.")

    config_path = Path(args.config)
    config = load_config(config_path)
    output_root = Path(config["output"]["run_dir"])
    development_result_path = Path(config["data"]["development_results"])
    development_result = json.loads(
        development_result_path.read_text(encoding="utf-8")
    )
    freeze_path, freeze = _verify_freeze(
        config=config,
        config_path=config_path,
        output_root=output_root,
        development_result=development_result,
    )
    sentinel_path = _start_unseal(
        output_root=output_root,
        freeze_path=freeze_path,
        config_path=config_path,
    )

    cache, audit = build_development_cache(config)
    audit = {
        **audit,
        "sealed_test_loaded": True,
        "confirmatory_metrics_loaded": True,
    }
    cache["audit"] = audit
    result = evaluate_frozen_recipient_cache(
        cache,
        config,
        development_result,
    )
    run_dir = output_root / "confirmatory"
    cache_path = run_dir / "confirmatory_cache.pt"
    audit_path = run_dir / "source_audit.json"
    result_path = run_dir / "results.json"
    manifest_path = run_dir / "run_manifest.json"
    torch.save(cache, cache_path)
    _write_json(audit_path, audit)
    _write_json(result_path, result)
    manifest = {
        "name": "statsbomb_recipient_history_confirmatory_v1_run_manifest",
        "version": 1,
        "status": "confirmatory_run_completed_once",
        "experiment_protocol": str(config["experiment_protocol"]),
        "result_path": str(result_path),
        "result_file_sha256": file_sha256(result_path),
        "result_payload_sha256": result["result_payload_sha256"],
        "cache_path": str(cache_path),
        "cache_file_sha256": file_sha256(cache_path),
        "source_audit_path": str(audit_path),
        "source_audit_sha256": file_sha256(audit_path),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "split_manifest_path": str(config["data"]["split_manifest"]),
        "split_manifest_sha256": file_sha256(
            config["data"]["split_manifest"]
        ),
        "source_inventory_path": str(config["data"]["source_inventory"]),
        "source_inventory_sha256": file_sha256(
            config["data"]["source_inventory"]
        ),
        "development_result_path": str(development_result_path),
        "development_result_sha256": file_sha256(development_result_path),
        "development_result_payload_sha256": development_result[
            "result_payload_sha256"
        ],
        "freeze_manifest_path": str(freeze_path),
        "freeze_manifest_sha256": file_sha256(freeze_path),
        "unseal_sentinel_path": str(sentinel_path),
        "unseal_sentinel_sha256": file_sha256(sentinel_path),
        "source_files": {
            "analysis": {
                "path": str(ANALYSIS_PATH),
                "sha256": file_sha256(ANALYSIS_PATH),
            },
            "runner": {
                "path": str(RUNNER_PATH),
                "sha256": file_sha256(RUNNER_PATH),
            },
        },
        "git": _git_metadata(),
        "confirmatory_metrics_loaded": True,
    }
    manifest["manifest_payload_sha256"] = stable_payload_hash(manifest)
    _write_json(manifest_path, manifest)
    completion = {
        "name": "statsbomb_recipient_history_confirmatory_v1_completion",
        "version": 1,
        "status": "confirmatory_run_completed_once",
        "completed_utc": datetime.now(UTC).isoformat(),
        "freeze_manifest_sha256": file_sha256(freeze_path),
        "unseal_sentinel_sha256": file_sha256(sentinel_path),
        "result_file_sha256": file_sha256(result_path),
        "run_manifest_sha256": file_sha256(manifest_path),
        "gate_passed": bool(result["gate"]["passed"]),
    }
    _write_json(run_dir / "UNSEAL_COMPLETED.json", completion)
    print(
        json.dumps(
            {
                "results": str(result_path),
                "run_manifest": str(manifest_path),
                "gate": result["gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
