"""Create the immutable record required before confirmatory metric access."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from footballq.analysis.wyscout_player_memory import (
    file_sha256,
    stable_payload_hash,
)

ANALYSIS_PATH = Path("src/footballq/analysis/wyscout_player_fingerprint.py")
RUNNER_PATH = Path("scripts/run_wyscout_player_fingerprint_v1.py")


def _file_record(path: str | Path) -> dict[str, str]:
    resolved = Path(path)
    if not resolved.is_file():
        raise ValueError(f"Cannot freeze missing file: {resolved}")
    return {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
    }


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/wyscout_player_fingerprint_v1.yaml",
    )
    parser.add_argument(
        "--output-root",
        default="runs/wyscout_player_fingerprint_v1",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = Path(args.output_root)
    development_result_path = output_root / "development" / "results.json"
    development_manifest_path = (
        output_root / "development" / "run_manifest.json"
    )
    development = json.loads(
        development_result_path.read_text(encoding="utf-8")
    )
    if not bool(development["gate"]["passed"]):
        raise ValueError("Cannot freeze a failed development experiment.")
    if bool(development["confirmatory_metrics_loaded"]):
        raise ValueError("Development artifact reports confirmatory access.")
    if development["config_sha256"] != file_sha256(config_path):
        raise ValueError("Development artifact uses a different config hash.")
    confirmatory_dir = output_root / "confirmatory"
    if (confirmatory_dir / "UNSEAL_STARTED.json").exists() or (
        confirmatory_dir / "results.json"
    ).exists():
        raise ValueError("Confirmatory metrics were already opened.")

    files = {
        "config": _file_record(config_path),
        "analysis": _file_record(ANALYSIS_PATH),
        "runner": _file_record(RUNNER_PATH),
        "dataset_manifest": _file_record(config["data"]["dataset_manifest"]),
        "development_split_manifest": _file_record(
            config["data"]["development_manifest"]
        ),
        "confirmatory_split_manifest": _file_record(
            config["data"]["confirmatory_manifest"]
        ),
        "protocol_doc": _file_record(
            config["data"]["confirmatory_protocol_doc"]
        ),
        "development_results": _file_record(development_result_path),
        "development_run_manifest": _file_record(development_manifest_path),
    }
    payload = {
        "name": "wyscout_player_fingerprint_confirmatory_freeze_v1",
        "version": 1,
        "status": "metric_sealed_before_first_confirmatory_read",
        "created_utc": datetime.now(UTC).isoformat(),
        "experiment_protocol": str(config["experiment_protocol"]),
        "output_root": str(output_root),
        "primary_claim": (
            "A player's outcome-free event-history behavior fingerprint "
            "persists across a club-to-national-team context change."
        ),
        "claim_exclusions": [
            "critical-event prediction improvement",
            "tracking-based tactical understanding",
            "opponent-specific matchup planning",
        ],
        "development_gate_passed": True,
        "development_result_payload_sha256": str(
            development["result_payload_sha256"]
        ),
        "confirmatory_metrics_read_before_freeze": False,
        "confirmatory_unseal_command": (
            "python scripts/run_wyscout_player_fingerprint_v1.py "
            "--unseal-confirmatory"
        ),
        "files": files,
        "git": _git_metadata(),
    }
    payload["freeze_payload_sha256"] = stable_payload_hash(payload)
    freeze_path = Path(config["data"]["confirmatory_freeze_manifest"])
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    with freeze_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "freeze_manifest": str(freeze_path),
                "freeze_payload_sha256": payload[
                    "freeze_payload_sha256"
                ],
                "status": payload["status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
