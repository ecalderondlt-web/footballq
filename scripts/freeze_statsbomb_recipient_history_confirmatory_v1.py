"""Freeze the recipient-history tournament protocol before metric access."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from footballq.analysis.statsbomb_recipient_history import load_config
from footballq.analysis.wyscout_player_memory import (
    file_sha256,
    stable_payload_hash,
)

ANALYSIS_PATH = Path("src/footballq/analysis/statsbomb_recipient_history.py")
DEVELOPMENT_RUNNER_PATH = Path(
    "scripts/run_statsbomb_recipient_history_v1.py"
)
CONFIRMATORY_RUNNER_PATH = Path(
    "scripts/run_statsbomb_recipient_history_confirmatory_v1.py"
)
INVENTORY_BUILDER_PATH = Path("scripts/build_statsbomb_source_inventory.py")
SPLIT_BUILDER_PATH = Path(
    "scripts/build_statsbomb_recipient_split_manifest.py"
)


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


def _development_eligibility(
    development: dict[str, Any],
) -> dict[str, Any]:
    baseline = development["conditions"]["E_rolling_involvement"][
        "development_test"
    ]
    profile = development["conditions"]["F_history_target_by_origin_zone"][
        "development_test"
    ]
    broad_shuffle = development["conditions"][
        "same_broad_role_shuffled_history"
    ]["development_test"]
    gain = float(baseline["nll"]) - float(profile["nll"])
    relative_gain = gain / float(baseline["nll"])
    bootstrap_lower = float(
        development["match_bootstrap_profile_vs_rolling"][
            "nll_improvement"
        ]["ci95"][0]
    )
    curve = development["support_size_curve"]
    checks = {
        "minimum_relative_nll_improvement_0_5_percent": (
            relative_gain >= 0.005
        ),
        "positive_match_bootstrap_lower_bound": bootstrap_lower > 0.0,
        "better_than_broad_role_shuffled_history": (
            float(profile["nll"]) < float(broad_shuffle["nll"])
        ),
        "ten_match_gain_exceeds_one_match_gain": (
            float(
                curve["10"]["development_test"][
                    "profile_minus_rolling_nll_improvement"
                ]
            )
            > float(
                curve["1"]["development_test"][
                    "profile_minus_rolling_nll_improvement"
                ]
            )
        ),
        "confirmatory_metrics_not_loaded": (
            not bool(development["sealed_test_loaded"])
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "observed": {
            "relative_nll_improvement": relative_gain,
            "absolute_nll_improvement": gain,
            "bootstrap_ci_lower": bootstrap_lower,
            "top3_gain": (
                float(profile["top3_accuracy"])
                - float(baseline["top3_accuracy"])
            ),
            "original_development_gate_status": development[
                "development_gate"
            ]["status"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/statsbomb_recipient_history_confirmatory_v1.yaml",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    output_root = Path(config["output"]["run_dir"])
    run_dir = output_root / "confirmatory"
    if (run_dir / "UNSEAL_STARTED.json").exists() or (
        run_dir / "results.json"
    ).exists():
        raise ValueError("Confirmatory recipient metrics were already opened.")

    development_result_path = Path(config["data"]["development_results"])
    development = json.loads(
        development_result_path.read_text(encoding="utf-8")
    )
    eligibility = _development_eligibility(development)
    if not bool(eligibility["passed"]):
        raise ValueError(
            "The narrower recipient-NLL development eligibility checks failed."
        )
    if int(development["selected_support_size"]) != int(
        config["confirmatory"]["frozen_support_size"]
    ):
        raise ValueError("Configured support size differs from development selection.")
    development_manifest_path = Path(
        config["data"]["development_run_manifest"]
    )
    development_manifest = json.loads(
        development_manifest_path.read_text(encoding="utf-8")
    )
    development_config_path = Path(config["data"]["development_config"])
    if development_manifest["config_sha256"] != file_sha256(
        development_config_path
    ):
        raise ValueError("Development config differs from its run manifest.")
    if development["run_manifest_split_sha256"] != str(
        development_manifest["split_manifest_sha256"]
    ):
        raise ValueError("Development split lineage differs from its run manifest.")

    source_inventory_path = Path(config["data"]["source_inventory"])
    source_inventory = json.loads(
        source_inventory_path.read_text(encoding="utf-8")
    )
    if str(source_inventory["source_commit"]) != str(
        config["data"]["source_commit"]
    ):
        raise ValueError("Source inventory commit differs from the config.")
    split_path = Path(config["data"]["split_manifest"])
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("status") != "metric_sealed":
        raise ValueError("Confirmatory split manifest is not metric-sealed.")

    files = {
        "config": _file_record(config_path),
        "analysis": _file_record(ANALYSIS_PATH),
        "runner": _file_record(CONFIRMATORY_RUNNER_PATH),
        "development_runner": _file_record(DEVELOPMENT_RUNNER_PATH),
        "source_inventory_builder": _file_record(INVENTORY_BUILDER_PATH),
        "split_builder": _file_record(SPLIT_BUILDER_PATH),
        "source_inventory": _file_record(source_inventory_path),
        "split_manifest": _file_record(split_path),
        "protocol_doc": _file_record(
            config["data"]["confirmatory_protocol_doc"]
        ),
        "development_config": _file_record(development_config_path),
        "development_results": _file_record(development_result_path),
        "development_run_manifest": _file_record(
            development_manifest_path
        ),
        "development_source_audit": _file_record(
            output_root.parent
            / "statsbomb_recipient_history_v1"
            / "source_audit.json"
        ),
    }
    payload = {
        "name": "statsbomb_recipient_history_confirmatory_freeze_v1",
        "version": 1,
        "status": "metric_sealed_before_first_confirmatory_read",
        "created_utc": datetime.now(UTC).isoformat(),
        "experiment_protocol": str(config["experiment_protocol"]),
        "output_root": str(output_root),
        "primary_claim": (
            "Strictly prior player receiving-location histories improve "
            "held-out pass-recipient probability ranking beyond role, current "
            "origin zone, fine position, static identity, and rolling frequency."
        ),
        "claim_exclusions": [
            "tracking-based critical-event prediction",
            "opponent-specific matchup understanding",
            "causal tactical intervention",
            "full-match planning",
            "top-k accuracy improvement as a primary endpoint",
        ],
        "development_eligibility": eligibility,
        "development_result_payload_sha256": str(
            development["result_payload_sha256"]
        ),
        "original_development_gate_note": (
            "The original exploratory gate remained blocked because its "
            "two-percentage-point top-3 threshold was not met. Before any "
            "tournament score was opened, the confirmatory claim was narrowed "
            "to NLL, a proper probability-ranking score."
        ),
        "confirmatory_cohorts": {
            "primary": str(config["confirmatory"]["primary_cohort"]),
            "external_replications": list(
                config["confirmatory"]["external_replication_cohorts"]
            ),
        },
        "confirmatory_gate": dict(config["confirmatory_gate"]),
        "confirmatory_metadata_accessed_before_freeze": [
            "match identifiers",
            "match dates",
            "lineups",
            "360 file availability",
            "raw file content hashes",
        ],
        "confirmatory_metrics_read_before_freeze": False,
        "confirmatory_unseal_command": (
            "python scripts/run_statsbomb_recipient_history_confirmatory_v1.py "
            "--unseal-confirmatory"
        ),
        "source_snapshot": {
            "source_commit": source_inventory["source_commit"],
            "file_count": source_inventory["file_count"],
            "total_size_bytes": source_inventory["total_size_bytes"],
            "inventory_payload_sha256": source_inventory[
                "inventory_payload_sha256"
            ],
        },
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
                "freeze_payload_sha256": payload["freeze_payload_sha256"],
                "development_eligibility": eligibility,
                "status": payload["status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
