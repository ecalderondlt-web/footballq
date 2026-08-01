"""Run the sealed-development Wyscout player-memory experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from footballq.analysis.wyscout_player_memory import (
    file_sha256,
    load_config,
    run_development_experiment,
    stable_payload_hash,
)


def _git_metadata() -> dict[str, object]:
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
        default="configs/wyscout_player_memory_v1.yaml",
    )
    parser.add_argument(
        "--output-root",
        default="runs/wyscout_player_memory_v1",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    if bool(config["confirmatory"]["allow_metric_unseal"]):
        raise ValueError(
            "Development runner refuses a config with confirmatory metrics unsealed."
        )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results = run_development_experiment(config)
    results["config_path"] = str(config_path)
    results["config_sha256"] = file_sha256(config_path)
    results["result_sha256"] = stable_payload_hash(results)
    result_path = output_root / "development_results.json"
    result_path.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "name": "wyscout_player_memory_v1_run_manifest",
        "version": 1,
        "result_path": str(result_path),
        "result_file_sha256": file_sha256(result_path),
        "result_payload_sha256": results["result_sha256"],
        "config_path": str(config_path),
        "config_sha256": results["config_sha256"],
        "development_split_manifest": str(
            config["data"]["development_split_manifest"]
        ),
        "development_split_manifest_sha256": file_sha256(
            config["data"]["development_split_manifest"]
        ),
        "confirmatory_split_manifest": str(
            config["data"]["confirmatory_split_manifest"]
        ),
        "confirmatory_split_manifest_sha256": file_sha256(
            config["data"]["confirmatory_split_manifest"]
        ),
        "dataset_manifest": str(config["data"]["dataset_manifest"]),
        "dataset_manifest_sha256": file_sha256(
            config["data"]["dataset_manifest"]
        ),
        "source_files": {
            "analysis": {
                "path": "src/footballq/analysis/wyscout_player_memory.py",
                "sha256": file_sha256(
                    "src/footballq/analysis/wyscout_player_memory.py"
                ),
            },
            "runner": {
                "path": "scripts/run_wyscout_player_memory_v1.py",
                "sha256": file_sha256(
                    "scripts/run_wyscout_player_memory_v1.py"
                ),
            },
        },
        "git": _git_metadata(),
        "confirmatory_metrics_loaded": False,
    }
    manifest["manifest_payload_sha256"] = stable_payload_hash(manifest)
    manifest_path = output_root / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "result": str(result_path),
                "manifest": str(manifest_path),
                "development_gate": results["development_gate"],
                "development_effect": results["development_effect"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
