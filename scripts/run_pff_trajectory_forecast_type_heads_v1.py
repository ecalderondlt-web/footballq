"""Run the frozen player/ball type-head PFF trajectory forecast study."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.training.train_trajectory_forecast import (  # noqa: E402
    train_trajectory_forecast_from_config,
)
from scripts.run_pff_trajectory_forecast_entity_v1 import (  # noqa: E402
    EXPECTED_VALIDATION_DIGEST,
    FAMILIES,
    HORIZON_KEYS,
    SEEDS,
    TRANSFERRED,
    _artifact_path,
    _comparison_gate,
    _means,
    _read_final,
    _record_valid,
    _relative_improvement,
    _sha256,
    _source_checkpoints,
    _write_json,
)

PROTOCOL = Path("docs/PFF_TRAJECTORY_FORECAST_TYPE_HEADS_PROTOCOL_V1.md")
CONFIG = Path("configs/pff_trajectory_forecast_type_heads_v1.yaml")
DATA_MANIFEST = Path("data/processed/pff_wc2022_trajectory_forecast_v1/dataset_manifest.json")
SPLIT_MANIFEST = Path("splits/pff_wc2022_64match_inductive_v1.json")
SOURCE_EXECUTION = Path("runs/pff_4x_tracking_complete_v1/execution_manifest.json")
PRIOR_ENTITY_SUMMARY = Path("runs/pff_trajectory_forecast_entity_v1/gate_summary.json")
PRIOR_ENTITY_AUDIT = Path(
    "runs/integrity/pff_trajectory_forecast_entity_v1_artifact_audit.json"
)
RUN_ROOT = Path("runs/pff_trajectory_forecast_type_heads_v1")
STATE_PATH = RUN_ROOT / "execution_manifest.json"
SUMMARY_PATH = RUN_ROOT / "gate_summary.json"
AUDIT_PATH = Path(
    "runs/integrity/pff_trajectory_forecast_type_heads_v1_artifact_audit.json"
)

FROZEN_FILES = {
    "protocol_sha256": (
        PROTOCOL,
        "beede94649a9119daca151ed271c60fe6c9271c5063f5080fe7b68d605f01792",
    ),
    "config_sha256": (
        CONFIG,
        "d9ac4ebf171da48da6d4187a17244a2e12b6d0aac87f802c39326fbdb3d6f775",
    ),
    "forecast_manifest_sha256": (
        DATA_MANIFEST,
        "688761b30c4fbe38d832d09d459e79153acc5851a397ceb600d5bc30c811b537",
    ),
    "split_manifest_file_sha256": (
        SPLIT_MANIFEST,
        "9f7d56184920e463f1aa5fdcee05dc9b59438184910afc93a7e0c12f4e322226",
    ),
    "source_execution_manifest_sha256": (
        SOURCE_EXECUTION,
        "463063455fa8850eb4b49ea5ba163db19a620dc1dda71f0c0ca308baf1ae9f00",
    ),
    "prior_entity_summary_sha256": (
        PRIOR_ENTITY_SUMMARY,
        "f4ed01a8aa1d470e66712e4b91c27fab1af02d692ac5c7477c2fd050843ece22",
    ),
    "prior_entity_audit_sha256": (
        PRIOR_ENTITY_AUDIT,
        "016a7324e9c3d9e87c5380a9d80109d7339588098feebc08cd01039d2a931011",
    ),
    "encoder_code_sha256": (
        Path("src/footballq/models/soccer_state_encoder.py"),
        "e21dda98c0605841be9df2fbd20c7ffa4ced5c7d9be910ae782d1ff3453c2eb0",
    ),
    "forecaster_code_sha256": (
        Path("src/footballq/models/trajectory_forecaster.py"),
        "2a98d7c74ed80caaa02aedbdc20746ae724a96390f9678c08640e7658f055226",
    ),
    "training_code_sha256": (
        Path("src/footballq/training/train_trajectory_forecast.py"),
        "ac530ea89dae69878cf4690a9a62a083fd16075c8fac20e32361b85cd1b9648d",
    ),
    "entity_runner_helper_sha256": (
        Path("scripts/run_pff_trajectory_forecast_entity_v1.py"),
        "1e2b0fc53f538cabc8d2b7b25c2699612575f68f4530a3d4f58236f5a6a1ec87",
    ),
}


def _current_hashes() -> dict[str, str]:
    return {name: _sha256(path) for name, (path, _expected) in FROZEN_FILES.items()}


def _expected_hashes() -> dict[str, str]:
    return {name: expected for name, (_path, expected) in FROZEN_FILES.items()}


def _new_state() -> dict[str, Any]:
    current = _current_hashes()
    expected = _expected_hashes()
    if current != expected:
        raise ValueError(f"Frozen type-head inputs changed: expected {expected}, got {current}")
    prior_audit = json.loads(_artifact_path(PRIOR_ENTITY_AUDIT).read_text(encoding="utf-8"))
    if prior_audit.get("status") != "passed":
        raise ValueError("Prior entity forecast artifact audit is not passed.")
    now = datetime.now(UTC).replace(microsecond=0).isoformat()
    return {
        "version": 1,
        "study": "pff_trajectory_forecast_type_heads_v1",
        **current,
        "tracking_checkpoints": {
            str(seed): row for seed, row in _source_checkpoints().items()
        },
        "seeds": list(SEEDS),
        "families": list(FAMILIES),
        "runs": {},
        "created_at_utc": now,
        "updated_at_utc": now,
    }


def _load_state() -> dict[str, Any]:
    path = _artifact_path(STATE_PATH)
    if not path.exists():
        return _new_state()
    state = json.loads(path.read_text(encoding="utf-8"))
    expected = _expected_hashes()
    current = _current_hashes()
    for key, value in expected.items():
        if state.get(key) != value or current.get(key) != value:
            raise ValueError(f"Frozen type-head state mismatch for {key}.")
    return state


def _save_state(state: dict[str, Any]) -> None:
    state["updated_at_utc"] = datetime.now(UTC).replace(microsecond=0).isoformat()
    _write_json(STATE_PATH, state)


def run_training(state: dict[str, Any]) -> None:
    for family in FAMILIES:
        for seed in SEEDS:
            key = f"type_heads:{family}:{seed}"
            existing = state["runs"].get(key)
            if existing and _record_valid(existing):
                print(f"[resume] {key}", flush=True)
                continue
            checkpoint = state["tracking_checkpoints"][str(seed)]["path"]
            print(f"[run] {key}", flush=True)
            result = train_trajectory_forecast_from_config(
                CONFIG,
                family=family,
                seed=seed,
                tracking_checkpoint=checkpoint,
            )
            run_dir = Path(result["run_dir"])
            metrics_path = run_dir / "metrics_val.jsonl"
            curve_path = run_dir / "metrics_val_curve.jsonl"
            state["runs"][key] = {
                "status": "complete",
                "family": family,
                "seed": seed,
                "tracking_checkpoint": checkpoint,
                "tracking_checkpoint_sha256": state["tracking_checkpoints"][str(seed)][
                    "sha256"
                ],
                "run_dir": str(run_dir),
                "latest_checkpoint": str(result["latest_checkpoint"]),
                "latest_checkpoint_sha256": _sha256(result["latest_checkpoint"]),
                "metrics_path": str(metrics_path),
                "metrics_sha256": _sha256(metrics_path),
                "curve_path": str(curve_path),
                "curve_sha256": _sha256(curve_path),
                "run_manifest_path": str(result["run_manifest"]),
                "run_manifest_sha256": _sha256(result["run_manifest"]),
                "validation_sample_id_sha256": result["metrics"]["sample_id_sha256"],
            }
            _save_state(state)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _redesign_gate(
    candidate_name: str,
    candidate_rows: dict[int, dict[str, Any]],
    entity_rows: dict[int, dict[str, Any]],
    candidate_mean: dict[str, float],
    entity_mean: dict[str, float],
    global_mean: dict[str, float],
) -> dict[str, Any]:
    player_non_degraded = sum(
        candidate_rows[seed]["player_ADE_m"]
        <= entity_rows[seed]["player_ADE_m"] * 1.01
        for seed in SEEDS
    )
    ball_wins = sum(
        candidate_rows[seed]["ball_ADE_m"] < entity_rows[seed]["ball_ADE_m"]
        for seed in SEEDS
    )
    player_improvement = _relative_improvement(
        entity_mean["player_ADE_m"], candidate_mean["player_ADE_m"]
    )
    ball_improvement = _relative_improvement(
        entity_mean["ball_ADE_m"], candidate_mean["ball_ADE_m"]
    )
    ball_fde_improvement = _relative_improvement(
        entity_mean["ball_FDE_m"], candidate_mean["ball_FDE_m"]
    )
    horizon_improvements = {
        key: _relative_improvement(entity_mean[key], candidate_mean[key])
        for key in HORIZON_KEYS
    }
    worst_horizon = min(horizon_improvements.values())
    criteria = {
        "player_non_degraded_seeds": {
            "value": player_non_degraded,
            "minimum": 2,
            "passed": player_non_degraded >= 2,
        },
        "mean_player_ADE_improvement": {
            "value": player_improvement,
            "minimum": -0.01,
            "passed": player_improvement >= -0.01,
        },
        "worst_player_horizon_improvement": {
            "value": worst_horizon,
            "minimum": -0.01,
            "passed": worst_horizon >= -0.01,
        },
        "ball_seed_wins": {
            "value": ball_wins,
            "minimum": 2,
            "passed": ball_wins >= 2,
        },
        "mean_ball_ADE_improvement": {
            "value": ball_improvement,
            "minimum": 0.05,
            "passed": ball_improvement >= 0.05,
        },
        "global_raw_ball_ADE_ceiling": {
            "value": candidate_mean["ball_ADE_m"],
            "maximum": global_mean["ball_ADE_m"],
            "passed": candidate_mean["ball_ADE_m"] <= global_mean["ball_ADE_m"],
        },
        "mean_ball_FDE_improvement": {
            "value": ball_fde_improvement,
            "minimum": 0.0,
            "passed": ball_fde_improvement >= 0.0,
        },
    }
    blockers = [name for name, criterion in criteria.items() if not criterion["passed"]]
    return {
        "candidate": candidate_name,
        "reference": "entity_raw",
        "passed": not blockers,
        "criteria": criteria,
        "player_horizon_improvements": horizon_improvements,
        "blocking_conditions": blockers,
    }


def summarize(state: dict[str, Any]) -> dict[str, Any]:
    if len(state.get("runs", {})) != 9:
        raise ValueError("All nine type-head forecast runs must complete before summarization.")
    prior = json.loads(_artifact_path(PRIOR_ENTITY_SUMMARY).read_text(encoding="utf-8"))
    entity_rows = {
        int(row["seed"]): row for row in prior["rows"] if row["family"] == "raw"
    }
    entity_mean = prior["entity_means"]["raw"]
    global_mean = prior["global_raw_mean"]
    rows_by_family: dict[str, dict[int, dict[str, Any]]] = {
        family: {} for family in FAMILIES
    }
    rows = []
    for family in FAMILIES:
        for seed in SEEDS:
            metrics = _read_final(
                state["runs"][f"type_heads:{family}:{seed}"]["metrics_path"]
            )
            rows_by_family[family][seed] = metrics
            rows.append({"family": family, "seed": seed, **metrics})
    digests = {row["sample_id_sha256"] for row in rows}
    if digests != {EXPECTED_VALIDATION_DIGEST}:
        raise ValueError(f"Type-head validation digest mismatch: {digests}")

    type_head_means = {family: _means(rows_by_family[family]) for family in FAMILIES}
    redesign_gates = {
        family: _redesign_gate(
            f"type_heads_{family}",
            rows_by_family[family],
            entity_rows,
            type_head_means[family],
            entity_mean,
            global_mean,
        )
        for family in FAMILIES
    }
    transfer_gates = {
        family: _comparison_gate(
            f"type_heads_{family}",
            "type_heads_raw",
            rows_by_family[family],
            rows_by_family["raw"],
            type_head_means[family],
            type_head_means["raw"],
            player_improvement_minimum=0.02,
        )
        for family in TRANSFERRED
    }
    passing = [family for family in FAMILIES if redesign_gates[family]["passed"]]
    if passing:
        family_order = {family: index for index, family in enumerate(FAMILIES)}
        selected = min(
            passing,
            key=lambda family: (
                type_head_means[family]["player_ADE_m"],
                type_head_means[family]["ball_ADE_m"],
                family_order[family],
            ),
        )
        operational = f"type_heads_{selected}"
        status = (
            "representation_passed"
            if selected in TRANSFERRED and transfer_gates[selected]["passed"]
            else "redesign_passed"
        )
        blockers = [
            f"transfer:{family}:{criterion}"
            for family in TRANSFERRED
            for criterion in transfer_gates[family]["blocking_conditions"]
        ]
    else:
        operational = "global_raw"
        status = "blocked"
        blockers = [
            f"redesign:{family}:{criterion}"
            for family in FAMILIES
            for criterion in redesign_gates[family]["blocking_conditions"]
        ]
        blockers.extend(
            f"transfer:{family}:{criterion}"
            for family in TRANSFERRED
            for criterion in transfer_gates[family]["blocking_conditions"]
        )
    summary = {
        "version": 1,
        "study": "pff_trajectory_forecast_type_heads_v1",
        "status": status,
        "operational_family": operational,
        "blocking_conditions": blockers,
        "seeds": list(SEEDS),
        "families": list(FAMILIES),
        "rows": rows,
        "type_head_means": type_head_means,
        "entity_raw_mean": entity_mean,
        "global_raw_mean": global_mean,
        "redesign_gates": redesign_gates,
        "transfer_gates": transfer_gates,
        "validation_sample_id_sha256": EXPECTED_VALIDATION_DIGEST,
        "protocol_path": str(PROTOCOL),
        "protocol_sha256": _sha256(PROTOCOL),
        "config_path": str(CONFIG),
        "config_sha256": _sha256(CONFIG),
        "prior_entity_summary_path": str(PRIOR_ENTITY_SUMMARY),
        "prior_entity_summary_sha256": _sha256(PRIOR_ENTITY_SUMMARY),
        "data_access": {
            "loaded_splits": ["train", "val"],
            "test_loaded": False,
            "test_targets_generated": False,
            "embedding_exported": False,
            "learned_run_count": 9,
        },
        "execution_manifest_path": str(STATE_PATH),
    }
    _write_json(SUMMARY_PATH, summary)
    state["gate_summary_path"] = str(SUMMARY_PATH)
    state["gate_summary_sha256"] = _sha256(SUMMARY_PATH)
    state["gate_status"] = status
    state["operational_family"] = operational
    _save_state(state)
    print(f"[done] gate status: {status}", flush=True)
    print(f"[done] operational family: {operational}", flush=True)
    return summary


def verify_artifacts(state: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "frozen_inputs_match": _current_hashes() == _expected_hashes(),
        "all_runs_complete": len(state.get("runs", {})) == 9
        and all(_record_valid(record) for record in state["runs"].values()),
        "summary_hash_matches": _sha256(SUMMARY_PATH) == state.get("gate_summary_sha256"),
    }
    run_reports = []
    for key, record in sorted(state["runs"].items()):
        manifest = json.loads(
            _artifact_path(record["run_manifest_path"]).read_text(encoding="utf-8")
        )
        boundary_ok = (
            manifest.get("loaded_splits") == ["train", "val"]
            and manifest.get("test_loaded") is False
            and manifest.get("embedding_exported") is False
            and manifest.get("source", {}).get("representation_mode") == "entity_tokens"
            and manifest.get("source", {}).get("decoder_mode") == "player_ball"
        )
        digest_ok = manifest.get("validation_sample_id_sha256") == EXPECTED_VALIDATION_DIGEST
        checks[f"{key}:access_boundary"] = boundary_ok
        checks[f"{key}:validation_digest"] = digest_ok
        run_reports.append(
            {"run": key, "access_boundary": boundary_ok, "validation_digest": digest_ok}
        )
    summary = json.loads(_artifact_path(SUMMARY_PATH).read_text(encoding="utf-8"))
    checks["summary_access_boundary"] = summary.get("data_access") == {
        "loaded_splits": ["train", "val"],
        "test_loaded": False,
        "test_targets_generated": False,
        "embedding_exported": False,
        "learned_run_count": 9,
    }
    failed = sorted(name for name, value in checks.items() if not value)
    report = {
        "version": 1,
        "study": "pff_trajectory_forecast_type_heads_v1",
        "status": "passed" if not failed else "blocked",
        "checks": checks,
        "failed_checks": failed,
        "runs": run_reports,
        "data_access": summary["data_access"],
    }
    _write_json(AUDIT_PATH, report)
    state["artifact_audit_path"] = str(AUDIT_PATH)
    state["artifact_audit_sha256"] = _sha256(AUDIT_PATH)
    state["artifact_audit_status"] = report["status"]
    _save_state(state)
    if failed:
        raise ValueError("Type-head artifact audit failed: " + ", ".join(failed))
    print("[done] artifact audit: passed", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("train", "summarize", "verify", "all"),
        default="all",
    )
    args = parser.parse_args()
    state = _load_state()
    _save_state(state)
    if args.stage in {"train", "all"}:
        run_training(state)
    if args.stage in {"summarize", "all"}:
        summarize(state)
    if args.stage in {"verify", "all"}:
        verify_artifacts(state)


if __name__ == "__main__":
    main()
