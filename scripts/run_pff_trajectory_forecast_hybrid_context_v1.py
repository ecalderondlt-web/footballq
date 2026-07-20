"""Run the frozen scratch-only hybrid-context PFF trajectory forecast study."""

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
    HORIZON_KEYS,
    SEEDS,
    _artifact_path,
    _read_final,
    _record_valid,
    _relative_improvement,
    _sha256,
    _source_checkpoints,
    _write_json,
)

BALL_HORIZON_KEYS = (
    "ball_error_h0p5s_m",
    "ball_error_h1p0s_m",
    "ball_error_h2p0s_m",
    "ball_error_h4p0s_m",
)
MEAN_KEYS = (
    "player_ADE_m",
    "player_FDE_m",
    "ball_ADE_m",
    "ball_FDE_m",
    "all_entity_ADE_m",
    "all_entity_FDE_m",
    *HORIZON_KEYS,
    *BALL_HORIZON_KEYS,
)
PROTOCOL = Path("docs/PFF_TRAJECTORY_FORECAST_HYBRID_CONTEXT_PROTOCOL_V1.md")
CONFIG = Path("configs/pff_trajectory_forecast_hybrid_context_v1.yaml")
DATA_MANIFEST = Path("data/processed/pff_wc2022_trajectory_forecast_v1/dataset_manifest.json")
SPLIT_MANIFEST = Path("splits/pff_wc2022_64match_inductive_v1.json")
SOURCE_EXECUTION = Path("runs/pff_4x_tracking_complete_v1/execution_manifest.json")
TYPE_HEAD_SUMMARY = Path("runs/pff_trajectory_forecast_type_heads_v1/gate_summary.json")
TYPE_HEAD_AUDIT = Path(
    "runs/integrity/pff_trajectory_forecast_type_heads_v1_artifact_audit.json"
)
ENTITY_SUMMARY = Path("runs/pff_trajectory_forecast_entity_v1/gate_summary.json")
ENTITY_AUDIT = Path("runs/integrity/pff_trajectory_forecast_entity_v1_artifact_audit.json")
GLOBAL_SUMMARY = Path("runs/pff_trajectory_forecast_v1/gate_summary.json")
GLOBAL_AUDIT = Path("runs/integrity/pff_trajectory_forecast_v1_artifact_audit.json")
RUN_ROOT = Path("runs/pff_trajectory_forecast_hybrid_context_v1")
STATE_PATH = RUN_ROOT / "execution_manifest.json"
SUMMARY_PATH = RUN_ROOT / "gate_summary.json"
AUDIT_PATH = Path(
    "runs/integrity/pff_trajectory_forecast_hybrid_context_v1_artifact_audit.json"
)

FROZEN_FILES = {
    "protocol_sha256": (
        PROTOCOL,
        "cb4dbf4b711dcd9cf28817d65d05fe7b7fe559950d34dadb865ca8541a695d2c",
    ),
    "config_sha256": (
        CONFIG,
        "bb5b55da1e80b22e828df421ca9782753349ad33b75eb1db15273c73b83a0d63",
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
    "type_head_summary_sha256": (
        TYPE_HEAD_SUMMARY,
        "bb145f9d195743f0d3b1d5d52b6f0d7fdf135a15deb08ea0eb198926e1b5573f",
    ),
    "type_head_audit_sha256": (
        TYPE_HEAD_AUDIT,
        "da12e4f4c57ab69ac5a623245eb2a303a63e16985b1a96872cd268ed461ba195",
    ),
    "entity_summary_sha256": (
        ENTITY_SUMMARY,
        "f4ed01a8aa1d470e66712e4b91c27fab1af02d692ac5c7477c2fd050843ece22",
    ),
    "entity_audit_sha256": (
        ENTITY_AUDIT,
        "016a7324e9c3d9e87c5380a9d80109d7339588098feebc08cd01039d2a931011",
    ),
    "global_summary_sha256": (
        GLOBAL_SUMMARY,
        "7905300e4d1784a1786c08920e528b54c93b3c82408c63b6f5defa7d233ec688",
    ),
    "global_audit_sha256": (
        GLOBAL_AUDIT,
        "8acd06a1d47d8f95c0d1461e0f3af9e13632eae323f68289fab2b109968b398d",
    ),
    "encoder_code_sha256": (
        Path("src/footballq/models/soccer_state_encoder.py"),
        "e21dda98c0605841be9df2fbd20c7ffa4ced5c7d9be910ae782d1ff3453c2eb0",
    ),
    "forecaster_code_sha256": (
        Path("src/footballq/models/trajectory_forecaster.py"),
        "d057fe6dacdc1f91762a7cdf74b2a6a107158faeb928c395d8d3aef59850a151",
    ),
    "training_code_sha256": (
        Path("src/footballq/training/train_trajectory_forecast.py"),
        "12440d2def20fb695baf26b0eb75a1844384854ede76aab989e8f91e02dfb4c7",
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


def _means(rows: dict[int, dict[str, Any]]) -> dict[str, float]:
    return {
        name: sum(float(rows[seed][name]) for seed in SEEDS) / len(SEEDS)
        for name in MEAN_KEYS
    }


def _new_state() -> dict[str, Any]:
    current = _current_hashes()
    expected = _expected_hashes()
    if current != expected:
        raise ValueError(f"Frozen hybrid inputs changed: expected {expected}, got {current}")
    for path in (TYPE_HEAD_AUDIT, ENTITY_AUDIT, GLOBAL_AUDIT):
        audit = json.loads(_artifact_path(path).read_text(encoding="utf-8"))
        if audit.get("status") != "passed":
            raise ValueError(f"Prior artifact audit is not passed: {path}")
    now = datetime.now(UTC).replace(microsecond=0).isoformat()
    return {
        "version": 1,
        "study": "pff_trajectory_forecast_hybrid_context_v1",
        **current,
        "tracking_checkpoints": {
            str(seed): row for seed, row in _source_checkpoints().items()
        },
        "seeds": list(SEEDS),
        "family": "raw",
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
            raise ValueError(f"Frozen hybrid state mismatch for {key}.")
    return state


def _save_state(state: dict[str, Any]) -> None:
    state["updated_at_utc"] = datetime.now(UTC).replace(microsecond=0).isoformat()
    _write_json(STATE_PATH, state)


def run_training(state: dict[str, Any]) -> None:
    for seed in SEEDS:
        key = f"hybrid_context:raw:{seed}"
        existing = state["runs"].get(key)
        if existing and _record_valid(existing):
            print(f"[resume] {key}", flush=True)
            continue
        checkpoint = state["tracking_checkpoints"][str(seed)]["path"]
        print(f"[run] {key}", flush=True)
        result = train_trajectory_forecast_from_config(
            CONFIG,
            family="raw",
            seed=seed,
            tracking_checkpoint=checkpoint,
        )
        run_dir = Path(result["run_dir"])
        metrics_path = run_dir / "metrics_val.jsonl"
        curve_path = run_dir / "metrics_val_curve.jsonl"
        state["runs"][key] = {
            "status": "complete",
            "family": "raw",
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


def _hybrid_gate(
    candidate_rows: dict[int, dict[str, Any]],
    entity_rows: dict[int, dict[str, Any]],
    global_rows: dict[int, dict[str, Any]],
    candidate_mean: dict[str, float],
    entity_mean: dict[str, float],
    global_mean: dict[str, float],
) -> dict[str, Any]:
    player_non_degraded = sum(
        candidate_rows[seed]["player_ADE_m"]
        <= entity_rows[seed]["player_ADE_m"] * 1.01
        for seed in SEEDS
    )
    ball_non_degraded = sum(
        candidate_rows[seed]["ball_ADE_m"] <= global_rows[seed]["ball_ADE_m"] * 1.01
        for seed in SEEDS
    )
    player_horizons = {
        key: _relative_improvement(entity_mean[key], candidate_mean[key])
        for key in HORIZON_KEYS
    }
    ball_horizons = {
        key: _relative_improvement(global_mean[key], candidate_mean[key])
        for key in BALL_HORIZON_KEYS
    }
    criteria = {
        "player_non_degraded_seeds": {
            "value": player_non_degraded,
            "minimum": 2,
            "passed": player_non_degraded >= 2,
        },
        "mean_player_ADE_improvement_vs_entity": {
            "value": _relative_improvement(
                entity_mean["player_ADE_m"], candidate_mean["player_ADE_m"]
            ),
            "minimum": -0.01,
        },
        "worst_player_horizon_improvement_vs_entity": {
            "value": min(player_horizons.values()),
            "minimum": -0.01,
        },
        "ball_non_degraded_seeds": {
            "value": ball_non_degraded,
            "minimum": 2,
            "passed": ball_non_degraded >= 2,
        },
        "mean_ball_ADE_improvement_vs_entity": {
            "value": _relative_improvement(
                entity_mean["ball_ADE_m"], candidate_mean["ball_ADE_m"]
            ),
            "minimum": 0.04,
        },
        "mean_ball_ADE_improvement_vs_global": {
            "value": _relative_improvement(
                global_mean["ball_ADE_m"], candidate_mean["ball_ADE_m"]
            ),
            "minimum": -0.01,
        },
        "mean_ball_FDE_improvement_vs_global": {
            "value": _relative_improvement(
                global_mean["ball_FDE_m"], candidate_mean["ball_FDE_m"]
            ),
            "minimum": -0.01,
        },
        "worst_ball_horizon_improvement_vs_global": {
            "value": min(ball_horizons.values()),
            "minimum": -0.02,
        },
        "mean_all_entity_ADE_improvement_vs_global": {
            "value": _relative_improvement(
                global_mean["all_entity_ADE_m"], candidate_mean["all_entity_ADE_m"]
            ),
            "minimum": 0.02,
        },
    }
    for criterion in criteria.values():
        criterion.setdefault("passed", criterion["value"] >= criterion["minimum"])
    blockers = [name for name, criterion in criteria.items() if not criterion["passed"]]
    return {
        "candidate": "hybrid_context_raw",
        "player_reference": "entity_raw",
        "ball_reference": "global_raw",
        "passed": not blockers,
        "criteria": criteria,
        "player_horizon_improvements_vs_entity": player_horizons,
        "ball_horizon_improvements_vs_global": ball_horizons,
        "blocking_conditions": blockers,
    }


def summarize(state: dict[str, Any]) -> dict[str, Any]:
    if len(state.get("runs", {})) != 3:
        raise ValueError("All three hybrid forecast runs must complete before summarization.")
    entity = json.loads(_artifact_path(ENTITY_SUMMARY).read_text(encoding="utf-8"))
    global_result = json.loads(_artifact_path(GLOBAL_SUMMARY).read_text(encoding="utf-8"))
    entity_rows = {
        int(row["seed"]): row for row in entity["rows"] if row["family"] == "raw"
    }
    global_rows = {
        int(row["seed"]): row
        for row in global_result["rows"]
        if row["family"] == "raw"
    }
    candidate_rows = {}
    rows = []
    for seed in SEEDS:
        metrics = _read_final(state["runs"][f"hybrid_context:raw:{seed}"]["metrics_path"])
        candidate_rows[seed] = metrics
        rows.append({"family": "raw", "seed": seed, **metrics})
    digests = {row["sample_id_sha256"] for row in rows}
    if digests != {EXPECTED_VALIDATION_DIGEST}:
        raise ValueError(f"Hybrid validation digest mismatch: {digests}")

    candidate_mean = _means(candidate_rows)
    entity_mean = _means(entity_rows)
    global_mean = _means(global_rows)
    gate = _hybrid_gate(
        candidate_rows,
        entity_rows,
        global_rows,
        candidate_mean,
        entity_mean,
        global_mean,
    )
    status = "hybrid_passed" if gate["passed"] else "blocked"
    operational = "hybrid_context_raw" if gate["passed"] else "global_raw"
    summary = {
        "version": 1,
        "study": "pff_trajectory_forecast_hybrid_context_v1",
        "status": status,
        "operational_family": operational,
        "blocking_conditions": gate["blocking_conditions"],
        "seeds": list(SEEDS),
        "family": "raw",
        "rows": rows,
        "hybrid_mean": candidate_mean,
        "entity_raw_mean": entity_mean,
        "global_raw_mean": global_mean,
        "gate": gate,
        "validation_sample_id_sha256": EXPECTED_VALIDATION_DIGEST,
        "protocol_path": str(PROTOCOL),
        "protocol_sha256": _sha256(PROTOCOL),
        "config_path": str(CONFIG),
        "config_sha256": _sha256(CONFIG),
        "data_access": {
            "loaded_splits": ["train", "val"],
            "test_loaded": False,
            "test_targets_generated": False,
            "embedding_exported": False,
            "learned_run_count": 3,
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
        "all_runs_complete": len(state.get("runs", {})) == 3
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
            and manifest.get("source", {}).get("online_encoder_weights_loaded") is False
            and manifest.get("source", {}).get("representation_mode") == "entity_tokens"
            and manifest.get("source", {}).get("decoder_mode") == "player_global_ball"
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
        "learned_run_count": 3,
    }
    failed = sorted(name for name, value in checks.items() if not value)
    report = {
        "version": 1,
        "study": "pff_trajectory_forecast_hybrid_context_v1",
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
        raise ValueError("Hybrid artifact audit failed: " + ", ".join(failed))
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
