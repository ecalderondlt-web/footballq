from __future__ import annotations

import argparse
import hashlib
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

from footballq.data.rlcs_last_defender_v3 import (
    LastDefenderV3Error,
    assign_favorable_matchup,
    build_replay_opportunities,
    calibrate_geometry_thresholds,
    common_support_audit,
    extract_replay_calibration,
    opportunity_volume_audit,
)
from footballq.data.rlcs_player_profiles import observations_and_roster
from footballq.repro.manifest import file_sha256

FRAME_BASE_COLUMNS = (
    "observed_frame_number",
    "game_time_s_precise",
    "stint_number",
    "ball_pos_x",
    "ball_pos_y",
    "ball_pos_z",
    "ball_vel_x",
    "ball_vel_y",
    "ball_vel_z",
    "ball_ang_vel_x",
    "ball_ang_vel_y",
    "ball_ang_vel_z",
)
EVENT_COLUMNS = (
    "event_number",
    "event_type",
    "observed_frame_number",
    "game_time_s_precise",
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
    "official_goal",
)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _player_frame_columns(quality_record: Mapping[str, Any]) -> list[str]:
    columns = list(FRAME_BASE_COLUMNS)
    for item in quality_record.get("observed_roster") or []:
        prefix = str(item["prefix"])
        for stem in ("pos", "vel", "ang_vel", "rot"):
            columns.extend(f"{prefix}_{stem}_{axis}" for axis in "xyz")
        columns.extend(
            [
                f"{prefix}_boost",
                f"{prefix}_jumped",
                f"{prefix}_flipped",
                f"{prefix}_double_jump_active",
            ]
        )
    return list(dict.fromkeys(columns))


def _read_columns(path: Path, requested: list[str] | tuple[str, ...]) -> pd.DataFrame:
    available = set(pq.ParquetFile(path).schema_arrow.names)
    required = {
        "observed_frame_number",
        "game_time_s_precise",
        "event_type" if path.name == "events.parquet" else "ball_pos_x",
    }
    missing = required.difference(available)
    if missing:
        raise LastDefenderV3Error(f"{path} is missing required columns {sorted(missing)}.")
    selected = [column for column in requested if column in available]
    return pq.read_table(path, columns=selected).to_pandas()


def _load_replay(job: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, Any, Any]:
    cache = Path(job["cache"])
    quality_record = job["quality_record"]
    frames = _read_columns(cache / "frames.parquet", _player_frame_columns(quality_record))
    events = _read_columns(cache / "events.parquet", EVENT_COLUMNS)
    observations, roster = observations_and_roster(quality_record)
    return frames, events, observations, roster


def _calibration_worker(job: Mapping[str, Any]) -> dict[str, Any]:
    frames, events, observations, roster = _load_replay(job)
    result = extract_replay_calibration(
        frames,
        events,
        observations=observations,
        roster_ids=roster,
        fps=float(job["fps"]),
        context_seconds=float(job["context_seconds"]),
        maximum_frame_lag_seconds=float(job["maximum_frame_lag_seconds"]),
    )
    return {"replay_id": job["replay_id"], **result}


def _opportunity_worker(job: Mapping[str, Any]) -> dict[str, Any]:
    frames, events, observations, roster = _load_replay(job)
    rows = build_replay_opportunities(
        frames,
        events,
        replay_id=str(job["replay_id"]),
        inventory=job["inventory"],
        stage=str(job["stage"]),
        observations=observations,
        roster_ids=roster,
        snapshots=job["snapshots"],
        priors=job["priors"],
        eligible_player_ids=set(job["eligible_player_ids"]),
        thresholds=job["thresholds"],
        minimum_prior_games_actor=int(job["minimum_prior_games_actor"]),
        minimum_prior_games_defender=int(job["minimum_prior_games_defender"]),
        standardized_profile_clip=job["standardized_profile_clip"],
        fps=float(job["fps"]),
        context_seconds=float(job["context_seconds"]),
        maximum_frame_lag_seconds=float(job["maximum_frame_lag_seconds"]),
    )
    return {"replay_id": job["replay_id"], "rows": rows}


def _parallel_replays(
    jobs: list[dict[str, Any]], worker: Any, *, workers: int, label: str
) -> list[dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_jobs = {
            executor.submit(worker, job): (index, job)
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
                    f"{label}: processed {completed}/{len(jobs)} replays; "
                    f"failures={len(failures)}",
                    flush=True,
                )
    if failures:
        preview = failures[:5]
        raise RuntimeError(
            f"{label} failed closed for {len(failures)} replays: {preview}"
        )
    return [results[index] for index in sorted(results)]


def _eligible_cohort(audit: Mapping[str, Any], required_players: int) -> list[str]:
    players = sorted(str(value) for value in audit.get("eligible_player_games", {}))
    if len(players) != int(required_players) or not bool(audit.get("all_gates_pass")):
        raise LastDefenderV3Error(
            "V3 requires the complete passing 48-player V2 stability cohort."
        )
    digest = hashlib.sha256()
    for player_id in players:
        digest.update(player_id.encode("utf-8") + b"\n")
    if digest.hexdigest() != str(audit.get("eligible_player_ids_sha256")):
        raise LastDefenderV3Error("Eligible-player cohort hash does not match the V2 audit.")
    return players


def _verify_preoutcome_freeze(
    config_path: Path, config: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    freeze_path = Path(config["preoutcome_freeze"])
    if not freeze_path.exists():
        raise PermissionError("V3 Stage 0 requires the committed pre-outcome freeze ledger.")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    expected = {
        "config_sha256": file_sha256(config_path),
        "protocol_sha256": file_sha256(config["protocol"]),
        "implementation_sha256": file_sha256("src/footballq/data/rlcs_last_defender_v3.py"),
        "runner_sha256": file_sha256("scripts/audit_rlcs_last_defender_v3.py"),
    }
    if freeze.get("status") != "frozen_before_stage0":
        raise PermissionError("V3 pre-outcome ledger is not frozen for Stage 0.")
    for key, value in expected.items():
        if freeze.get(key) != value:
            raise PermissionError(f"V3 pre-outcome freeze mismatch: {key}.")
    current_sources = _source_hashes(config)
    if freeze.get("source_sha256") != current_sources:
        raise PermissionError("V3 source data hashes differ from the pre-outcome freeze.")
    if any(bool(freeze.get(key)) for key in ("action_labels_loaded", "success_labels_loaded")):
        raise PermissionError("V3 pre-outcome ledger reports an opened downstream label.")
    return freeze_path, freeze


def _source_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    data = config["data"]
    paths = {
        "inventory": data["inventory"],
        "quality_report": data["quality_report"],
        "v2_split_manifest": data["v2_split_manifest"],
        "profile_snapshots": data["profile_snapshots"],
        "profile_priors": data["profile_priors"],
        "profile_audit": data["profile_audit"],
    }
    return {name: file_sha256(path) for name, path in paths.items()}


def _base_jobs(
    selected: pd.DataFrame,
    *,
    accepted: Mapping[str, Any],
    snapshots: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    data = config["data"]
    jobs = []
    for row in selected.to_dict(orient="records"):
        replay_id = str(row["replay_id"])
        if replay_id not in accepted:
            continue
        if replay_id not in snapshots:
            raise LastDefenderV3Error(f"Missing profile snapshots for {replay_id}.")
        jobs.append(
            {
                "replay_id": replay_id,
                "inventory": row,
                "stage": str(row["v3_stage"]),
                "quality_record": accepted[replay_id],
                "snapshots": snapshots[replay_id],
                "cache": str(Path(data["parser_cache"]) / replay_id),
                "fps": float(data["fps"]),
                "context_seconds": float(data["context_seconds"]),
                "maximum_frame_lag_seconds": float(data["maximum_frame_lag_seconds"]),
            }
        )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen outcome-blind RLCS last-defender V3 Stage 0 audit."
    )
    parser.add_argument("--config", default="configs/rlcs_last_defender_policy_value_v3.yaml")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Local replay worker processes (default: up to four).",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least one")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("experiment") != "rlcs_last_defender_policy_value_v3":
        raise LastDefenderV3Error("Wrong experiment configuration.")
    freeze_path, _ = _verify_preoutcome_freeze(config_path, config)
    data = config["data"]
    final_paths = [
        Path(data["threshold_output"]),
        Path(data["opportunity_inventory"]),
        Path(data["stage0_audit"]),
    ]
    if any(path.exists() for path in final_paths):
        raise FileExistsError(
            "A V3 Stage 0 artifact already exists; the frozen audit will not overwrite it."
        )

    source_hashes = _source_hashes(config)
    split_manifest = json.loads(Path(data["v2_split_manifest"]).read_text(encoding="utf-8"))
    inventory = pd.read_parquet(data["inventory"])
    stage_by_replay: dict[str, str] = {}
    for stage, payload in split_manifest["stages"].items():
        for replay_id in payload["replay_ids"]:
            stage_by_replay[str(replay_id)] = str(stage)
    inventory["v3_stage"] = inventory["replay_id"].astype(str).map(stage_by_replay)
    if inventory["v3_stage"].isna().any():
        raise LastDefenderV3Error("The frozen V2 split manifest does not cover inventory rows.")

    quality = json.loads(Path(data["quality_report"]).read_text(encoding="utf-8"))
    accepted = {
        str(record["replay_id"]): record
        for record in quality.get("replays", quality.get("records", []))
        if bool(record.get("parse_success"))
        and bool(record.get("qc_accepted"))
        and bool(record.get("identity_accepted"))
    }
    snapshot_frame = pd.read_parquet(data["profile_snapshots"])
    snapshot_groups = {
        str(replay_id): {
            str(row.player_id): {
                "profile": row.profile,
                "uncertainty": row.uncertainty,
                "n_prior_games": row.n_prior_games,
                "effective_sample_size": row.effective_sample_size,
                "prior_win_rate": row.prior_win_rate,
                "prior_goal_diff": row.prior_goal_diff,
                "latest_prior_time_utc": row.latest_prior_time_utc,
            }
            for row in rows.itertuples(index=False)
        }
        for replay_id, rows in snapshot_frame.groupby("replay_id", sort=False)
    }
    priors = json.loads(Path(data["profile_priors"]).read_text(encoding="utf-8"))
    profile_audit = json.loads(Path(data["profile_audit"]).read_text(encoding="utf-8"))
    eligible_players = _eligible_cohort(
        profile_audit, int(config["cohort"]["required_complete_players"])
    )

    calibration_rows = inventory.loc[inventory["v3_stage"] == "train"].copy()
    calibration_jobs = _base_jobs(
        calibration_rows,
        accepted=accepted,
        snapshots=snapshot_groups,
        config=config,
    )
    calibration_results = _parallel_replays(
        calibration_jobs,
        _calibration_worker,
        workers=args.workers,
        label="train geometry calibration",
    )
    combined_samples = {
        name: [
            value
            for result in calibration_results
            for value in result["samples"][name]
        ]
        for name in (
            "corridor_half_width",
            "last_defender_forward_distance",
            "immediate_intervention_range",
            "teammate_overload_range",
        )
    }
    calibration_base_contacts = sum(
        int(result["base_contacts"]) for result in calibration_results
    )
    thresholds, calibration_report = calibrate_geometry_thresholds(
        combined_samples, config["geometry_calibration"]
    )

    allowed_stages = set(config["split_locks"]["opportunity_stages"])
    if allowed_stages != {"train", "internal_development"}:
        raise LastDefenderV3Error("Stage 0 opportunity stages differ from the protocol freeze.")
    selected = inventory.loc[inventory["v3_stage"].isin(allowed_stages)].copy()
    opportunity_jobs = _base_jobs(
        selected,
        accepted=accepted,
        snapshots=snapshot_groups,
        config=config,
    )
    for job in opportunity_jobs:
        job.update(
            {
                "priors": priors,
                "eligible_player_ids": eligible_players,
                "thresholds": thresholds,
                "minimum_prior_games_actor": config["cohort"][
                    "minimum_prior_games_actor"
                ],
                "minimum_prior_games_defender": config["cohort"][
                    "minimum_prior_games_defender"
                ],
                "standardized_profile_clip": config["matchup_exposure"][
                    "standardized_clip"
                ],
            }
        )
    opportunity_results = _parallel_replays(
        opportunity_jobs,
        _opportunity_worker,
        workers=args.workers,
        label="last-defender inventory",
    )
    opportunity_rows = [row for result in opportunity_results for row in result["rows"]]
    opportunity_frame = pd.DataFrame(opportunity_rows)
    if opportunity_frame.empty:
        raise LastDefenderV3Error("The frozen detector produced no opportunity rows.")
    if opportunity_frame["sample_id"].duplicated().any():
        raise LastDefenderV3Error("Duplicate V3 sample IDs crossed replay shards.")
    opportunity_frame, exposure_threshold = assign_favorable_matchup(opportunity_frame)
    volume = opportunity_volume_audit(opportunity_frame, config["opportunity_gates"])

    common_support: dict[str, Any]
    pairs: list[dict[str, Any]] = []
    if volume["all_gates_pass"]:
        opportunity_frame, common_support, pairs = common_support_audit(
            opportunity_frame, config["common_support"]
        )
        status = (
            "stage0_pass_proceed_to_identity_blinded_label_audit"
            if common_support["all_gates_pass"]
            else "stopped_common_support_gate_failed"
        )
    else:
        common_support = {
            "status": "not_run_due_to_opportunity_volume_gate",
            "all_gates_pass": False,
        }
        status = "stopped_opportunity_volume_gate_failed"

    audit = {
        "version": 3,
        "experiment": config["experiment"],
        "created_at_utc": _utc_now(),
        "status": status,
        "preoutcome_freeze": str(freeze_path),
        "preoutcome_freeze_sha256": file_sha256(freeze_path),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "protocol_path": str(config["protocol"]),
        "protocol_sha256": file_sha256(config["protocol"]),
        "source_sha256": source_hashes,
        "opened_stages": ["train", "internal_development"],
        "action_labels_loaded": False,
        "success_labels_loaded": False,
        "split2_validation_loaded": False,
        "split2_test_loaded": False,
        "eligible_player_ids_sha256": profile_audit["eligible_player_ids_sha256"],
        "eligible_players": len(eligible_players),
        "calibration": {
            "fit_stage": "train",
            "base_contacts": calibration_base_contacts,
            **calibration_report,
        },
        "matchup_exposure": {
            "fit_stage": "train",
            "train_median_mismatch": exposure_threshold,
            "favorable_rows": int(opportunity_frame["favorable_matchup"].sum()),
            "unfavorable_rows": int(
                len(opportunity_frame) - opportunity_frame["favorable_matchup"].sum()
            ),
        },
        "opportunity_volume": volume,
        "common_support": common_support,
        "matched_sets": int(len(pairs)),
        "stage0_all_gates_pass": bool(
            volume["all_gates_pass"] and common_support.get("all_gates_pass", False)
        ),
        "downstream_training_authorized": False,
        "next_authorized_step": (
            "identity_blinded_100_example_label_audit"
            if status == "stage0_pass_proceed_to_identity_blinded_label_audit"
            else "close_rlcs_no_fourth_target"
        ),
    }

    output_dir = Path(data["output_dir"])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="rlcs_v3_stage0_", dir=output_dir.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        threshold_tmp = temporary / "geometry_thresholds.json"
        inventory_tmp = temporary / "opportunity_inventory.parquet"
        audit_tmp = temporary / "stage0_audit.json"
        _atomic_json(
            threshold_tmp,
            {
                "version": 3,
                "experiment": config["experiment"],
                "created_at_utc": audit["created_at_utc"],
                "fit_stage": "train",
                "base_contacts": calibration_base_contacts,
                **calibration_report,
                "action_labels_loaded": False,
                "success_labels_loaded": False,
            },
        )
        opportunity_frame.sort_values(
            ["event_time_utc", "replay_id", "frame_idx"], kind="stable"
        ).to_parquet(inventory_tmp, index=False, compression="zstd")
        audit["geometry_thresholds_sha256"] = file_sha256(threshold_tmp)
        audit["opportunity_inventory_sha256"] = file_sha256(inventory_tmp)
        _atomic_json(audit_tmp, audit)
        output_dir.mkdir(parents=True, exist_ok=False)
        threshold_tmp.replace(Path(data["threshold_output"]))
        inventory_tmp.replace(Path(data["opportunity_inventory"]))
        audit_tmp.replace(Path(data["stage0_audit"]))

    print(f"Stage 0 status: {status}", flush=True)
    print(f"Stage 0 audit: {data['stage0_audit']}", flush=True)


if __name__ == "__main__":
    main()
