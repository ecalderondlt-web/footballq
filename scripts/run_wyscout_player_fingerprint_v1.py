"""Run development or explicitly unsealed confirmatory fingerprint retrieval."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from footballq.analysis.wyscout_player_fingerprint import (
    build_player_vectors,
    eligible_support_and_query_player_ids,
    evaluate_cross_team_retrieval,
    evaluate_support_curve,
)
from footballq.analysis.wyscout_player_memory import (
    file_sha256,
    stable_payload_hash,
)
from footballq.repro.splits import load_split_manifest

ANALYSIS_PATH = Path("src/footballq/analysis/wyscout_player_fingerprint.py")
RUNNER_PATH = Path("scripts/run_wyscout_player_fingerprint_v1.py")


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


def _verify_confirmatory_freeze(
    *,
    config: dict[str, Any],
    config_path: Path,
    output_root: Path,
    development_result: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    freeze_path = Path(config["data"]["confirmatory_freeze_manifest"])
    if not freeze_path.is_file():
        raise ValueError(
            "Confirmatory unseal requires the checked-in freeze manifest."
        )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "metric_sealed_before_first_confirmatory_read":
        raise ValueError("Confirmatory freeze manifest is not metric-sealed.")
    if str(freeze["output_root"]) != str(output_root):
        raise ValueError("Confirmatory output root differs from the frozen path.")
    for label in (
        "config",
        "analysis",
        "runner",
        "dataset_manifest",
        "development_split_manifest",
        "confirmatory_split_manifest",
        "protocol_doc",
        "development_results",
        "development_run_manifest",
    ):
        _verify_file_record(freeze["files"][label], label)
    if Path(str(freeze["files"]["config"]["path"])) != config_path:
        raise ValueError("Runtime config path differs from the frozen config path.")
    if development_result["result_payload_sha256"] != str(
        freeze["development_result_payload_sha256"]
    ):
        raise ValueError(
            "Development result payload differs from the confirmatory freeze."
        )
    if not bool(development_result["gate"]["passed"]):
        raise ValueError("Frozen development result did not pass its gate.")
    if bool(development_result["confirmatory_metrics_loaded"]):
        raise ValueError("Development result reports confirmatory metric access.")
    run_dir = output_root / "confirmatory"
    if (run_dir / "UNSEAL_STARTED.json").exists() or (
        run_dir / "results.json"
    ).exists():
        raise ValueError(
            "Confirmatory cohort has already been opened; reruns are refused."
        )
    return freeze_path, freeze


def _start_confirmatory_unseal(
    *,
    output_root: Path,
    freeze_path: Path,
    config_path: Path,
) -> Path:
    run_dir = output_root / "confirmatory"
    run_dir.mkdir(parents=True, exist_ok=True)
    sentinel_path = run_dir / "UNSEAL_STARTED.json"
    payload = {
        "name": "wyscout_player_fingerprint_v1_confirmatory_unseal",
        "version": 1,
        "status": "confirmatory_metrics_about_to_be_loaded",
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


def _load_competitions(root: Path, competitions: list[str]) -> pd.DataFrame:
    return pd.concat(
        [
            pd.read_parquet(root / f"passes_{competition}.parquet")
            for competition in competitions
        ],
        ignore_index=True,
    )


def _validate_and_load(
    config: dict[str, Any],
    cohort_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cohort = config[cohort_name]
    manifest_key = (
        "development_manifest"
        if cohort_name == "development"
        else "confirmatory_manifest"
    )
    manifest = load_split_manifest(config["data"][manifest_key])
    dataset_manifest_path = Path(config["data"]["dataset_manifest"])
    dataset_hash = file_sha256(dataset_manifest_path)
    if dataset_hash != str(manifest.payload["dataset_manifest_sha256"]):
        raise ValueError(
            "Fingerprint split lineage does not match the current dataset manifest."
        )
    root = Path(config["data"]["dataset_root"])
    support = _load_competitions(
        root,
        [str(value) for value in cohort["support_competitions"]],
    )
    query = _load_competitions(
        root,
        [str(value) for value in cohort["query_competitions"]],
    )
    actual_support_ids = set(support["match_id"].astype(str).unique())
    expected_support_ids = {
        str(value) for value in manifest.payload["support_match_ids"]
    }
    actual_query_ids = set(query["match_id"].astype(str).unique())
    expected_query_ids = set(manifest.test_match_ids)
    if actual_support_ids != expected_support_ids:
        raise ValueError("Fingerprint support match IDs differ from frozen manifest.")
    if actual_query_ids != expected_query_ids:
        raise ValueError("Fingerprint query match IDs differ from frozen manifest.")
    latest_support = str(support["dateutc"].max())
    earliest_query = str(query["dateutc"].min())
    if latest_support >= earliest_query:
        raise ValueError("Fingerprint support is not strictly earlier than query.")
    return support, query, {
        **manifest.metadata(),
        "manifest_status": str(manifest.payload["status"]),
        "support_match_count": len(expected_support_ids),
        "query_match_count": len(expected_query_ids),
        "latest_support_dateutc": latest_support,
        "earliest_query_dateutc": earliest_query,
        "dataset_manifest_path": str(dataset_manifest_path),
        "dataset_manifest_sha256": dataset_hash,
    }


def _single_view(
    support: pd.DataFrame,
    query: pd.DataFrame,
    cohort: dict[str, Any],
    evaluation: dict[str, Any],
    *,
    match_cap: int,
    feature_view: str,
) -> dict[str, Any]:
    support_ids, query_ids = eligible_support_and_query_player_ids(
        support,
        query,
        support_minimum_matches=int(cohort["support_min_matches"]),
        support_minimum_passes=int(cohort["support_min_passes"]),
        query_minimum_matches=int(cohort["query_min_matches"]),
        query_minimum_passes=int(cohort["query_min_passes"]),
    )
    support_vectors = build_player_vectors(
        support,
        minimum_matches=1,
        minimum_passes=1,
        match_cap=match_cap,
        eligible_player_ids=support_ids,
    )
    query_vectors = build_player_vectors(
        query,
        minimum_matches=int(cohort["query_min_matches"]),
        minimum_passes=int(cohort["query_min_passes"]),
        eligible_player_ids=query_ids,
    )
    return evaluate_cross_team_retrieval(
        support_vectors,
        query_vectors,
        feature_view=feature_view,
        bootstrap_replicates=int(evaluation["bootstrap_replicates"]),
        bootstrap_seed=int(evaluation["bootstrap_seed"]) + match_cap,
        confidence_level=float(evaluation["confidence_level"]),
    )


def _chance_multiple(metrics: dict[str, Any]) -> float:
    return float(metrics["top1"]) / max(float(metrics["chance_top1"]), 1e-12)


def _gate(
    main: dict[str, Any],
    curve: dict[str, Any],
    cohort: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    support_caps = curve["support_match_caps"]
    ranking = main["ranking"]
    same_role = ranking["same_role"]
    support_team = ranking["same_support_team_and_role"]
    query_team = ranking["same_query_team_and_role"]
    bootstrap_positive = all(
        float(metrics["bootstrap"]["top1_minus_chance"]["ci_lower"]) > 0.0
        and float(metrics["bootstrap"]["mrr_minus_chance"]["ci_lower"]) > 0.0
        for metrics in (same_role, support_team, query_team)
    )
    main_cap = str(cohort["main_support_match_cap"])
    first_cap = str(cohort["support_match_caps"][0])
    main_mrr = float(support_caps[main_cap]["ranking"]["same_role"]["mrr"])
    first_mrr = float(support_caps[first_cap]["ranking"]["same_role"]["mrr"])
    mrr_gain = curve["main_vs_first_support"]["mrr_gain"]
    checks = {
        "same_role_pairwise_auc": float(
            main["same_role_pairwise_auc"]["point"]
        )
        >= float(gate["minimum_same_role_pairwise_auc"]),
        "same_role_top1_chance_multiple": _chance_multiple(same_role)
        >= float(gate["minimum_same_role_top1_chance_multiple"]),
        "support_team_top1_chance_multiple": _chance_multiple(support_team)
        >= float(gate["minimum_support_team_top1_chance_multiple"]),
        "query_team_top1_chance_multiple": _chance_multiple(query_team)
        >= float(gate["minimum_query_team_top1_chance_multiple"]),
        "positive_player_bootstrap_lower_bounds": (
            bootstrap_positive
            if bool(gate["require_positive_player_bootstrap_lower_bounds"])
            else True
        ),
    }
    if "require_positive_main_mrr_gain_lower_bound" in gate:
        checks["positive_main_mrr_gain_lower_bound"] = (
            float(mrr_gain["ci_lower"]) > 0.0
            if bool(gate["require_positive_main_mrr_gain_lower_bound"])
            else True
        )
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "observed": {
            "same_role_pairwise_auc": main["same_role_pairwise_auc"]["point"],
            "same_role_top1_chance_multiple": _chance_multiple(same_role),
            "support_team_top1_chance_multiple": _chance_multiple(support_team),
            "query_team_top1_chance_multiple": _chance_multiple(query_team),
            "main_same_role_mrr": main_mrr,
            "first_cap_same_role_mrr": first_mrr,
            "main_minus_first_mrr": mrr_gain["point"],
            "main_minus_first_mrr_ci_lower": mrr_gain["ci_lower"],
            "main_minus_first_mrr_ci_upper": mrr_gain["ci_upper"],
        },
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
    parser.add_argument("--unseal-confirmatory", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    cohort_name = "confirmatory" if args.unseal_confirmatory else "development"
    output_root = Path(args.output_root)
    development_path = output_root / "development" / "results.json"
    freeze_path: Path | None = None
    freeze: dict[str, Any] | None = None
    unseal_sentinel_path: Path | None = None
    if cohort_name == "confirmatory":
        if not development_path.is_file():
            raise ValueError(
                "Confirmatory unseal requires an existing development result."
            )
        development_result = json.loads(
            development_path.read_text(encoding="utf-8")
        )
        if not bool(development_result["gate"]["passed"]):
            raise ValueError(
                "Confirmatory unseal refused because the development gate failed."
            )
        if development_result["config_sha256"] != file_sha256(config_path):
            raise ValueError(
                "Confirmatory unseal refused because the frozen config changed."
            )
        freeze_path, freeze = _verify_confirmatory_freeze(
            config=config,
            config_path=config_path,
            output_root=output_root,
            development_result=development_result,
        )
        unseal_sentinel_path = _start_confirmatory_unseal(
            output_root=output_root,
            freeze_path=freeze_path,
            config_path=config_path,
        )

    support, query, lineage = _validate_and_load(config, cohort_name)
    cohort = config[cohort_name]
    evaluation = config["evaluation"]
    full_curve = evaluate_support_curve(
        support,
        query,
        cohort,
        evaluation,
        feature_view=str(config["profile"]["feature_view"]),
    )
    main_cap = int(cohort["main_support_match_cap"])
    main_result = full_curve["support_match_caps"][str(main_cap)]
    ablations = {
        feature_view: _single_view(
            support,
            query,
            cohort,
            evaluation,
            match_cap=main_cap,
            feature_view=feature_view,
        )
        for feature_view in config["profile"]["ablation_views"]
    }
    gate_config = config[
        "confirmatory_gate"
        if cohort_name == "confirmatory"
        else "development_gate"
    ]
    gate = _gate(
        main_result,
        full_curve,
        cohort,
        gate_config,
    )
    result = {
        "experiment_protocol": str(config["experiment_protocol"]),
        "cohort": cohort_name,
        "cohort_name": str(cohort["name"]),
        "status": (
            "confirmatory_unsealed_once"
            if cohort_name == "confirmatory"
            else "development_metric_opened"
        ),
        "claim_boundary": (
            "Event-history behavior fingerprints transfer across a team change. "
            "This does not by itself prove improved tracking-based critical-event "
            "prediction or matchup understanding."
        ),
        "lineage": lineage,
        "feature_contract": config["profile"],
        "eligible_player_universe": full_curve["eligible_player_universe"],
        "eligible_query_players": full_curve["eligible_query_players"],
        "eligible_support_candidates": full_curve[
            "eligible_support_candidates"
        ],
        "support_curve": full_curve["support_match_caps"],
        "main_vs_first_support": full_curve["main_vs_first_support"],
        "main_support_match_cap": main_cap,
        "main_result": main_result,
        "feature_ablations": ablations,
        "gate": gate,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "confirmatory_metrics_loaded": cohort_name == "confirmatory",
        "confirmatory_freeze_manifest": (
            {
                "path": str(freeze_path),
                "sha256": file_sha256(freeze_path),
            }
            if freeze_path is not None
            else None
        ),
    }
    result["result_payload_sha256"] = stable_payload_hash(result)
    run_dir = output_root / cohort_name
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "results.json"
    result_path.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_key = (
        "confirmatory_manifest"
        if cohort_name == "confirmatory"
        else "development_manifest"
    )
    manifest = {
        "name": f"wyscout_player_fingerprint_v1_{cohort_name}_run_manifest",
        "version": 1,
        "cohort": cohort_name,
        "result_path": str(result_path),
        "result_file_sha256": file_sha256(result_path),
        "result_payload_sha256": result["result_payload_sha256"],
        "config_path": str(config_path),
        "config_sha256": result["config_sha256"],
        "split_manifest_path": str(config["data"][manifest_key]),
        "split_manifest_sha256": file_sha256(config["data"][manifest_key]),
        "dataset_manifest_path": str(config["data"]["dataset_manifest"]),
        "dataset_manifest_sha256": file_sha256(
            config["data"]["dataset_manifest"]
        ),
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
        "confirmatory_metrics_loaded": cohort_name == "confirmatory",
        "confirmatory_freeze_manifest": (
            {
                "path": str(freeze_path),
                "sha256": file_sha256(freeze_path),
            }
            if freeze_path is not None
            else None
        ),
        "unseal_sentinel": (
            {
                "path": str(unseal_sentinel_path),
                "sha256": file_sha256(unseal_sentinel_path),
            }
            if unseal_sentinel_path is not None
            else None
        ),
    }
    manifest["manifest_payload_sha256"] = stable_payload_hash(manifest)
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    if cohort_name == "confirmatory":
        if freeze is None or unseal_sentinel_path is None:
            raise AssertionError("Confirmatory freeze metadata is unavailable.")
        completion = {
            "name": "wyscout_player_fingerprint_v1_confirmatory_completion",
            "version": 1,
            "status": "confirmatory_run_completed_once",
            "completed_utc": datetime.now(UTC).isoformat(),
            "freeze_manifest_sha256": file_sha256(freeze_path),
            "unseal_sentinel_sha256": file_sha256(unseal_sentinel_path),
            "result_file_sha256": file_sha256(result_path),
            "run_manifest_sha256": file_sha256(manifest_path),
            "gate_passed": bool(gate["passed"]),
        }
        (run_dir / "UNSEAL_COMPLETED.json").write_text(
            json.dumps(completion, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "cohort": cohort_name,
                "results": str(result_path),
                "run_manifest": str(manifest_path),
                "gate": gate,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
