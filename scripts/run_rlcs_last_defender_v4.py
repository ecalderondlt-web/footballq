from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import yaml

from footballq.analysis.rlcs_last_defender_v4 import (
    LastDefenderV4Error,
    build_overlap_support,
    evaluate_outcome_models,
    label_replay_success,
    outcome_volume_audit,
)
from footballq.data.rlcs_player_profiles import observations_and_roster
from footballq.repro.manifest import file_sha256

EVENT_OUTCOME_COLUMNS = (
    "event_number",
    "event_type",
    "observed_frame_number",
    "game_time_s_precise",
    "stint_number",
    "event_team",
    "event_player_1_id",
    "event_player_1_name",
    "event_player_1_team",
    "event_ball_pos_x",
    "event_ball_pos_y",
    "event_ball_pos_z",
    "ball_pos_x",
    "ball_pos_y",
    "ball_pos_z",
    "blue_score",
    "orange_score",
)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return path


def _source_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    data = config["data"]
    return {
        "v3_stop_ledger": file_sha256(data["v3_stop_ledger"]),
        "v3_opportunity_inventory": file_sha256(data["v3_opportunity_inventory"]),
        "replay_inventory": file_sha256(data["replay_inventory"]),
        "quality_report": file_sha256(data["quality_report"]),
    }


def _verify_preoutcome_freeze(
    config_path: Path, config: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    freeze_path = Path(config["preoutcome_freeze"])
    if not freeze_path.exists():
        raise PermissionError("V4 requires a committed pre-outcome freeze ledger.")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    expected = {
        "config_sha256": file_sha256(config_path),
        "protocol_sha256": file_sha256(config["protocol"]),
        "implementation_sha256": file_sha256(
            "src/footballq/analysis/rlcs_last_defender_v4.py"
        ),
        "runner_sha256": file_sha256("scripts/run_rlcs_last_defender_v4.py"),
        "test_sha256": file_sha256("tests/test_rlcs_last_defender_v4.py"),
    }
    if freeze.get("status") != "frozen_before_v4_support_and_outcomes":
        raise PermissionError("V4 pre-outcome ledger has the wrong status.")
    for key, value in expected.items():
        if freeze.get(key) != value:
            raise PermissionError(f"V4 pre-outcome freeze mismatch: {key}.")
    if freeze.get("source_sha256") != _source_hashes(config):
        raise PermissionError("V4 source hashes differ from the pre-outcome freeze.")
    outcome_state = freeze.get("outcome_artifact_state", {})
    if bool(outcome_state.get("success_labels_loaded")):
        raise PermissionError("V4 freeze ledger reports previously opened success labels.")
    return freeze_path, freeze


def _run_support(
    config_path: Path, config: Mapping[str, Any], freeze_path: Path
) -> None:
    data = config["data"]
    output_dir = Path(data["output_dir"])
    if output_dir.exists():
        raise FileExistsError("V4 output directory already exists; support will not overwrite it.")
    source_path = Path(data["v3_opportunity_inventory"])
    expected_hash = str(config["source_lock"]["opportunity_inventory_sha256"])
    if file_sha256(source_path) != expected_hash:
        raise LastDefenderV4Error("V4 opportunity inventory hash differs from the source lock.")
    frame = pd.read_parquet(source_path)
    weighted, support, pairs = build_overlap_support(
        frame,
        support_config=config["support"],
        source_lock=config["source_lock"],
    )
    status = "support_pass_outcomes_authorized" if support["all_gates_pass"] else "support_failed"
    with tempfile.TemporaryDirectory(prefix="rlcs_v4_support_", dir=output_dir.parent) as name:
        temporary = Path(name)
        weighted_path = temporary / "weighted_inventory.parquet"
        support_path = temporary / "support_audit.json"
        unlock_path = temporary / "outcome_unlock.json"
        weighted.sort_values(
            ["event_time_utc", "replay_id", "frame_idx"], kind="stable"
        ).to_parquet(weighted_path, index=False, compression="zstd")
        payload = {
            "version": 4,
            "experiment": config["experiment"],
            "created_at_utc": _utc_now(),
            "status": status,
            "preoutcome_freeze": str(freeze_path),
            "preoutcome_freeze_sha256": file_sha256(freeze_path),
            "config_path": str(config_path),
            "config_sha256": file_sha256(config_path),
            "protocol_path": str(config["protocol"]),
            "protocol_sha256": file_sha256(config["protocol"]),
            "source_sha256": _source_hashes(config),
            "weighted_inventory_sha256": file_sha256(weighted_path),
            "weighted_inventory_rows": int(len(weighted)),
            "exact_pairs": int(len(pairs)),
            "support": support,
            "opened_stages": ["train", "internal_development"],
            "success_labels_loaded": False,
            "split2_validation_loaded": False,
            "split2_test_loaded": False,
            "outcome_authorized": bool(support["all_gates_pass"]),
        }
        _atomic_json(support_path, payload)
        if support["all_gates_pass"]:
            _atomic_json(
                unlock_path,
                {
                    "version": 4,
                    "experiment": config["experiment"],
                    "status": "one_use_outcome_authorized",
                    "created_at_utc": payload["created_at_utc"],
                    "preoutcome_freeze_sha256": file_sha256(freeze_path),
                    "weighted_inventory_sha256": file_sha256(weighted_path),
                    "support_audit_sha256": file_sha256(support_path),
                    "success_labels_loaded": False,
                    "split2_validation_loaded": False,
                    "split2_test_loaded": False,
                },
            )
        output_dir.mkdir(parents=True, exist_ok=False)
        weighted_path.replace(Path(data["weighted_inventory"]))
        support_path.replace(Path(data["support_audit"]))
        if unlock_path.exists():
            unlock_path.replace(Path(data["outcome_unlock"]))
    print(f"V4 support status: {status}", flush=True)
    print(f"V4 support audit: {data['support_audit']}", flush=True)


def _read_outcome_events(path: Path) -> pd.DataFrame:
    available = set(pq.ParquetFile(path).schema_arrow.names)
    missing = set(EVENT_OUTCOME_COLUMNS).difference(available)
    if missing:
        raise LastDefenderV4Error(f"{path} is missing outcome columns {sorted(missing)}.")
    return pq.read_table(path, columns=list(EVENT_OUTCOME_COLUMNS)).to_pandas()


def _outcome_worker(job: Mapping[str, Any]) -> dict[str, Any]:
    events = _read_outcome_events(Path(job["events_path"]))
    observations, roster = observations_and_roster(job["quality_record"])
    labels = label_replay_success(
        events,
        pd.DataFrame(job["opportunities"]),
        observations=observations,
        roster_ids=roster,
        maximum_future_contacts=int(job["maximum_future_contacts"]),
        consecutive_opponent_contacts=int(job["consecutive_opponent_contacts"]),
        success_events=job["success_events"],
        boundary_events=job["boundary_events"],
    )
    return {"replay_id": str(job["replay_id"]), "labels": labels.to_dict(orient="records")}


def _parallel_outcomes(jobs: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_jobs = {
            executor.submit(_outcome_worker, job): (index, job)
            for index, job in enumerate(jobs)
        }
        completed = 0
        for future in as_completed(future_jobs):
            index, job = future_jobs[future]
            try:
                results[index] = future.result()
            except Exception as exc:
                failures.append(
                    {
                        "replay_id": str(job["replay_id"]),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            completed += 1
            if completed == len(jobs) or completed % 25 == 0:
                print(
                    f"V4 success labels: processed {completed}/{len(jobs)} replays; "
                    f"failures={len(failures)}",
                    flush=True,
                )
    if failures:
        raise RuntimeError(f"V4 success labeling failed closed: {failures[:5]}")
    return [results[index] for index in sorted(results)]


def _verify_unlock(config: Mapping[str, Any], freeze_path: Path) -> dict[str, Any]:
    data = config["data"]
    unlock_path = Path(data["outcome_unlock"])
    support_path = Path(data["support_audit"])
    weighted_path = Path(data["weighted_inventory"])
    if not unlock_path.exists() or not support_path.exists() or not weighted_path.exists():
        raise PermissionError("V4 outcome requires the complete passing support bundle.")
    unlock = json.loads(unlock_path.read_text(encoding="utf-8"))
    support = json.loads(support_path.read_text(encoding="utf-8"))
    expected = {
        "preoutcome_freeze_sha256": file_sha256(freeze_path),
        "weighted_inventory_sha256": file_sha256(weighted_path),
        "support_audit_sha256": file_sha256(support_path),
    }
    if unlock.get("status") != "one_use_outcome_authorized":
        raise PermissionError("V4 outcome unlock has the wrong status.")
    for key, value in expected.items():
        if unlock.get(key) != value:
            raise PermissionError(f"V4 outcome unlock mismatch: {key}.")
    if not bool(support.get("support", {}).get("all_gates_pass")):
        raise PermissionError("V4 support audit did not pass.")
    return unlock


def _consume_unlock(
    config: Mapping[str, Any], freeze_path: Path, unlock: Mapping[str, Any]
) -> Path:
    """Write an exclusive receipt immediately before any success event is read."""

    data = config["data"]
    receipt_path = Path(data["outcome_receipt"])
    payload = {
        "version": 4,
        "experiment": config["experiment"],
        "status": "outcome_read_started_unlock_consumed",
        "created_at_utc": _utc_now(),
        "preoutcome_freeze_sha256": file_sha256(freeze_path),
        "outcome_unlock_sha256": file_sha256(data["outcome_unlock"]),
        "weighted_inventory_sha256": str(unlock["weighted_inventory_sha256"]),
        "support_audit_sha256": str(unlock["support_audit_sha256"]),
        "success_labels_loaded_after_receipt_only": True,
        "split2_validation_loaded": False,
        "split2_test_loaded": False,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return receipt_path


def _run_outcome(
    config_path: Path,
    config: Mapping[str, Any],
    freeze_path: Path,
    *,
    workers: int,
) -> None:
    data = config["data"]
    outcome_path = Path(data["outcome_dataset"])
    result_path = Path(data["outcome_result"])
    receipt_path = Path(data["outcome_receipt"])
    if outcome_path.exists() or result_path.exists() or receipt_path.exists():
        raise FileExistsError("A V4 outcome artifact already exists; outcome will not rerun.")
    unlock = _verify_unlock(config, freeze_path)
    weighted = pd.read_parquet(data["weighted_inventory"])
    if len(weighted) != int(config["source_lock"]["opportunity_rows"]):
        raise LastDefenderV4Error("V4 weighted inventory row count changed after support.")
    quality_payload = json.loads(Path(data["quality_report"]).read_text(encoding="utf-8"))
    quality = {
        str(record["replay_id"]): record
        for record in quality_payload.get("replays", quality_payload.get("records", []))
        if bool(record.get("parse_success"))
        and bool(record.get("qc_accepted"))
        and bool(record.get("identity_accepted"))
    }
    jobs = []
    for replay_id, rows in weighted.groupby("replay_id", sort=True):
        replay_key = str(replay_id)
        if replay_key not in quality:
            raise LastDefenderV4Error(f"Missing accepted quality record for {replay_key}.")
        jobs.append(
            {
                "replay_id": replay_key,
                "events_path": str(Path(data["parser_cache"]) / replay_key / "events.parquet"),
                "quality_record": quality[replay_key],
                "opportunities": rows.to_dict(orient="records"),
                "maximum_future_contacts": config["outcome"][
                    "maximum_future_distinct_contacts"
                ],
                "consecutive_opponent_contacts": config["outcome"][
                    "stop_after_consecutive_opponent_contacts"
                ],
                "success_events": config["outcome"]["success_events"],
                "boundary_events": config["outcome"]["boundary_events"],
            }
        )
    receipt_path = _consume_unlock(config, freeze_path, unlock)
    label_results = _parallel_outcomes(jobs, workers)
    labels = pd.DataFrame(
        [row for replay in label_results for row in replay["labels"]]
    )
    if labels["sample_id"].duplicated().any() or len(labels) != len(weighted):
        raise LastDefenderV4Error("V4 success labels do not align one-to-one with support rows.")
    outcome_frame = weighted.merge(labels, on="sample_id", how="left", validate="one_to_one")
    volume = outcome_volume_audit(outcome_frame, config["outcome"])
    uncensored = outcome_frame.loc[outcome_frame["success_label"].notna()].copy()
    uncensored["success_label"] = uncensored["success_label"].astype("int8")
    if volume["all_gates_pass"]:
        evaluated, model_result = evaluate_outcome_models(
            uncensored,
            model_config=config["model"],
            uncertainty_config=config["uncertainty"],
            gate_config=config["gates"],
        )
        outcome_frame = outcome_frame.merge(
            evaluated[
                [
                    "sample_id",
                    "v4_outcome_fold",
                    "prediction_state",
                    "prediction_team_form",
                    "prediction_additive_profiles",
                    "prediction_full_matchup",
                ]
            ],
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
        status = "split1_pass_freeze_before_split2_validation" if model_result[
            "all_gates_pass"
        ] else "split1_outcome_gate_failed_close_v4"
    else:
        model_result = {
            "status": "not_run_outcome_volume_gate_failed",
            "all_gates_pass": False,
        }
        status = "outcome_volume_gate_failed_close_v4"

    with tempfile.TemporaryDirectory(
        prefix="rlcs_v4_outcome_", dir=outcome_path.parent
    ) as name:
        temporary = Path(name)
        dataset_tmp = temporary / "outcome_dataset.parquet"
        result_tmp = temporary / "outcome_result.json"
        outcome_frame.sort_values(
            ["event_time_utc", "replay_id", "frame_idx"], kind="stable"
        ).to_parquet(dataset_tmp, index=False, compression="zstd")
        result = {
            "version": 4,
            "experiment": config["experiment"],
            "created_at_utc": _utc_now(),
            "status": status,
            "preoutcome_freeze_sha256": file_sha256(freeze_path),
            "config_sha256": file_sha256(config_path),
            "protocol_sha256": file_sha256(config["protocol"]),
            "support_audit_sha256": file_sha256(data["support_audit"]),
            "weighted_inventory_sha256": file_sha256(data["weighted_inventory"]),
            "outcome_unlock_sha256": file_sha256(data["outcome_unlock"]),
            "outcome_receipt_sha256": file_sha256(receipt_path),
            "outcome_dataset_sha256": file_sha256(dataset_tmp),
            "outcome_volume": volume,
            "model_result": model_result,
            "opened_stages": ["train", "internal_development"],
            "success_labels_loaded": True,
            "action_labels_loaded": False,
            "split2_validation_loaded": False,
            "split2_test_loaded": False,
            "split2_validation_authorized": bool(model_result.get("all_gates_pass")),
            "unlock_consumed": unlock,
        }
        _atomic_json(result_tmp, result)
        outcome_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_tmp.replace(outcome_path)
        result_tmp.replace(result_path)
    print(f"V4 outcome status: {status}", flush=True)
    print(f"V4 outcome result: {result_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen RLCS last-defender overlap-weighted V4."
    )
    parser.add_argument(
        "--config", default="configs/rlcs_last_defender_overlap_value_v4.yaml"
    )
    parser.add_argument("--stage", choices=["support", "outcome"], required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Local event-labeling worker processes (default: up to four).",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least one")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("experiment") != "rlcs_last_defender_overlap_value_v4":
        raise LastDefenderV4Error("Wrong V4 experiment configuration.")
    freeze_path, _ = _verify_preoutcome_freeze(config_path, config)
    if args.stage == "support":
        _run_support(config_path, config, freeze_path)
    else:
        _run_outcome(config_path, config, freeze_path, workers=args.workers)


if __name__ == "__main__":
    main()
