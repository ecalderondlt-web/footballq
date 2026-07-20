"""Run the frozen entity-preserving PFF trajectory forecast study."""

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
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footballq.repro.manifest import file_sha256  # noqa: E402
from footballq.training.train_trajectory_forecast import (  # noqa: E402
    train_trajectory_forecast_from_config,
)

SEEDS = (7, 11, 23)
FAMILIES = ("raw", "frozen", "finetuned")
TRANSFERRED = ("frozen", "finetuned")
HORIZON_KEYS = (
    "player_error_h0p5s_m",
    "player_error_h1p0s_m",
    "player_error_h2p0s_m",
    "player_error_h4p0s_m",
)
PROTOCOL = Path("docs/PFF_TRAJECTORY_FORECAST_ENTITY_PROTOCOL_V1.md")
CONFIG = Path("configs/pff_trajectory_forecast_entity_v1.yaml")
DATA_MANIFEST = Path("data/processed/pff_wc2022_trajectory_forecast_v1/dataset_manifest.json")
SPLIT_MANIFEST = Path("splits/pff_wc2022_64match_inductive_v1.json")
SOURCE_EXECUTION = Path("runs/pff_4x_tracking_complete_v1/execution_manifest.json")
PRIOR_SUMMARY = Path("runs/pff_trajectory_forecast_v1/gate_summary.json")
PRIOR_AUDIT = Path("runs/integrity/pff_trajectory_forecast_v1_artifact_audit.json")
RUN_ROOT = Path("runs/pff_trajectory_forecast_entity_v1")
STATE_PATH = RUN_ROOT / "execution_manifest.json"
SUMMARY_PATH = RUN_ROOT / "gate_summary.json"
AUDIT_PATH = Path("runs/integrity/pff_trajectory_forecast_entity_v1_artifact_audit.json")
EXPECTED_VALIDATION_DIGEST = "5cf1ddab5ee33f318bd6c199674a00d58cdef4a4fd2732f3dcdd522dd0528d8d"

FROZEN_FILES = {
    "protocol_sha256": (
        PROTOCOL,
        "fedb63e62be0ac91d6a53e1e04eb6fddbc0a3396a62f1119acdc5645cdfba6ae",
    ),
    "config_sha256": (
        CONFIG,
        "8c0e56293b8130d916f85f15c6e0867abc23365dceb24ce38d25d55833b5b8a3",
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
    "prior_summary_sha256": (
        PRIOR_SUMMARY,
        "7905300e4d1784a1786c08920e528b54c93b3c82408c63b6f5defa7d233ec688",
    ),
    "prior_audit_sha256": (
        PRIOR_AUDIT,
        "8acd06a1d47d8f95c0d1461e0f3af9e13632eae323f68289fab2b109968b398d",
    ),
    "encoder_code_sha256": (
        Path("src/footballq/models/soccer_state_encoder.py"),
        "e21dda98c0605841be9df2fbd20c7ffa4ced5c7d9be910ae782d1ff3453c2eb0",
    ),
    "forecaster_code_sha256": (
        Path("src/footballq/models/trajectory_forecaster.py"),
        "1e48601f6cb86509735ea0907277ef2dc8d598316f425cac4fa7dd4c10796e59",
    ),
    "training_code_sha256": (
        Path("src/footballq/training/train_trajectory_forecast.py"),
        "68b9ca4d9073224d1ace0c991bb3ffdc6237c5bb878d87697d7c98679acbcb92",
    ),
}
CHECKPOINT_HASHES = {
    7: "267f907a9521fbec1ae31df11b36e931810d087d120b3d5822950c50f7aa7e9f",
    11: "1cc4bdc6fa6e11912baffa1bee8322b95ad2c4dff3f64898f433f0e95b8ae4ff",
    23: "ed6b3ad8b0de7b95ff693ccfe20e85012922ee86f128ed886469a02c1b37a366",
}


def _artifact_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _sha256(path: str | Path) -> str:
    return file_sha256(_artifact_path(path))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = _artifact_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


def _current_hashes() -> dict[str, str]:
    return {name: _sha256(path) for name, (path, _expected) in FROZEN_FILES.items()}


def _expected_hashes() -> dict[str, str]:
    return {name: expected for name, (_path, expected) in FROZEN_FILES.items()}


def _source_checkpoints() -> dict[int, dict[str, str]]:
    execution = json.loads(_artifact_path(SOURCE_EXECUTION).read_text(encoding="utf-8"))
    output = {}
    for seed in SEEDS:
        record = execution["runs"][f"pff:scratch:{seed}"]
        path = str(record["latest_checkpoint"])
        actual = _sha256(path)
        if actual != CHECKPOINT_HASHES[seed] or actual != record["latest_checkpoint_sha256"]:
            raise ValueError(f"Frozen tracking checkpoint hash mismatch for seed {seed}.")
        output[seed] = {"path": path, "sha256": actual}
    return output


def _new_state() -> dict[str, Any]:
    current = _current_hashes()
    expected = _expected_hashes()
    if current != expected:
        raise ValueError(
            f"Frozen entity forecast inputs changed: expected {expected}, got {current}"
        )
    prior_audit = json.loads(_artifact_path(PRIOR_AUDIT).read_text(encoding="utf-8"))
    if prior_audit.get("status") != "passed":
        raise ValueError("Prior global forecast artifact audit is not passed.")
    now = datetime.now(UTC).replace(microsecond=0).isoformat()
    return {
        "version": 1,
        "study": "pff_trajectory_forecast_entity_v1",
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
            raise ValueError(f"Frozen entity forecast state mismatch for {key}.")
    return state


def _save_state(state: dict[str, Any]) -> None:
    state["updated_at_utc"] = datetime.now(UTC).replace(microsecond=0).isoformat()
    _write_json(STATE_PATH, state)


def _record_valid(record: dict[str, Any]) -> bool:
    pairs = (
        ("latest_checkpoint", "latest_checkpoint_sha256"),
        ("metrics_path", "metrics_sha256"),
        ("curve_path", "curve_sha256"),
        ("run_manifest_path", "run_manifest_sha256"),
    )
    return bool(
        record.get("status") == "complete"
        and all(
            _artifact_path(record[path_key]).exists()
            and _sha256(record[path_key]) == record[hash_key]
            for path_key, hash_key in pairs
        )
    )


def run_training(state: dict[str, Any]) -> None:
    for family in FAMILIES:
        for seed in SEEDS:
            key = f"entity:{family}:{seed}"
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
                "tracking_checkpoint_sha256": state["tracking_checkpoints"][str(seed)]["sha256"],
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


def _read_final(path: str | Path) -> dict[str, Any]:
    lines = _artifact_path(path).read_text(encoding="utf-8").strip().splitlines()
    if len(lines) != 1:
        raise ValueError(f"Expected one final metric row in {path}.")
    return json.loads(lines[0])


def _relative_improvement(reference: float, candidate: float) -> float:
    return (float(reference) - float(candidate)) / float(reference)


def _comparison_gate(
    candidate_name: str,
    reference_name: str,
    candidate_rows: dict[int, dict[str, Any]],
    reference_rows: dict[int, dict[str, Any]],
    candidate_mean: dict[str, float],
    reference_mean: dict[str, float],
    *,
    player_improvement_minimum: float,
) -> dict[str, Any]:
    player_wins = sum(
        candidate_rows[seed]["player_ADE_m"] < reference_rows[seed]["player_ADE_m"]
        for seed in SEEDS
    )
    ball_wins = sum(
        candidate_rows[seed]["ball_ADE_m"] < reference_rows[seed]["ball_ADE_m"]
        for seed in SEEDS
    )
    player_improvement = _relative_improvement(
        reference_mean["player_ADE_m"], candidate_mean["player_ADE_m"]
    )
    ball_improvement = _relative_improvement(
        reference_mean["ball_ADE_m"], candidate_mean["ball_ADE_m"]
    )
    horizon_improvements = {
        key: _relative_improvement(reference_mean[key], candidate_mean[key])
        for key in HORIZON_KEYS
    }
    criteria = {
        "player_seed_wins": {"value": player_wins, "minimum": 2, "passed": player_wins >= 2},
        "mean_player_ADE_improvement": {
            "value": player_improvement,
            "minimum": player_improvement_minimum,
            "passed": player_improvement >= player_improvement_minimum,
        },
        "ball_seed_wins": {"value": ball_wins, "minimum": 2, "passed": ball_wins >= 2},
        "mean_ball_ADE_improvement": {
            "value": ball_improvement,
            "minimum": 0.05,
            "passed": ball_improvement >= 0.05,
        },
        "worst_player_horizon_improvement": {
            "value": min(horizon_improvements.values()),
            "minimum": -0.01,
            "passed": min(horizon_improvements.values()) >= -0.01,
        },
    }
    blockers = [name for name, criterion in criteria.items() if not criterion["passed"]]
    return {
        "candidate": candidate_name,
        "reference": reference_name,
        "passed": not blockers,
        "criteria": criteria,
        "player_horizon_improvements": horizon_improvements,
        "blocking_conditions": blockers,
    }


def _means(rows: dict[int, dict[str, Any]]) -> dict[str, float]:
    names = [
        "player_ADE_m",
        "player_FDE_m",
        "ball_ADE_m",
        "ball_FDE_m",
        "all_entity_ADE_m",
        "all_entity_FDE_m",
        *HORIZON_KEYS,
    ]
    return {
        name: sum(float(rows[seed][name]) for seed in SEEDS) / len(SEEDS)
        for name in names
    }


def summarize(state: dict[str, Any]) -> dict[str, Any]:
    if len(state.get("runs", {})) != 9:
        raise ValueError("All nine entity forecast runs must complete before summarization.")
    prior = json.loads(_artifact_path(PRIOR_SUMMARY).read_text(encoding="utf-8"))
    global_raw_rows = {
        int(row["seed"]): row for row in prior["rows"] if row["family"] == "raw"
    }
    rows_by_family: dict[str, dict[int, dict[str, Any]]] = {
        family: {} for family in FAMILIES
    }
    rows = []
    for family in FAMILIES:
        for seed in SEEDS:
            metrics = _read_final(state["runs"][f"entity:{family}:{seed}"]["metrics_path"])
            rows_by_family[family][seed] = metrics
            rows.append({"family": family, "seed": seed, **metrics})
    digests = {row["sample_id_sha256"] for row in rows}
    if digests != {EXPECTED_VALIDATION_DIGEST}:
        raise ValueError(f"Entity validation digest mismatch: {digests}")

    entity_means = {family: _means(rows_by_family[family]) for family in FAMILIES}
    global_raw_mean = _means(global_raw_rows)
    redesign_gate = _comparison_gate(
        "entity_raw",
        "global_raw",
        rows_by_family["raw"],
        global_raw_rows,
        entity_means["raw"],
        global_raw_mean,
        player_improvement_minimum=0.01,
    )
    transfer_gates = {
        family: _comparison_gate(
            f"entity_{family}",
            "entity_raw",
            rows_by_family[family],
            rows_by_family["raw"],
            entity_means[family],
            entity_means["raw"],
            player_improvement_minimum=0.02,
        )
        for family in TRANSFERRED
    }
    passing_transfer = [family for family in TRANSFERRED if transfer_gates[family]["passed"]]
    if passing_transfer:
        operational = "entity_" + min(
            passing_transfer, key=lambda family: entity_means[family]["player_ADE_m"]
        )
        status = "representation_passed"
        blockers = []
    elif redesign_gate["passed"]:
        operational = "entity_raw"
        status = "redesign_passed"
        blockers = [
            f"{family}:{criterion}"
            for family in TRANSFERRED
            for criterion in transfer_gates[family]["blocking_conditions"]
        ]
    else:
        operational = "global_raw"
        status = "blocked"
        blockers = [f"redesign:{name}" for name in redesign_gate["blocking_conditions"]]
        blockers.extend(
            f"{family}:{criterion}"
            for family in TRANSFERRED
            for criterion in transfer_gates[family]["blocking_conditions"]
        )
    summary = {
        "version": 1,
        "study": "pff_trajectory_forecast_entity_v1",
        "status": status,
        "operational_family": operational,
        "blocking_conditions": blockers,
        "seeds": list(SEEDS),
        "families": list(FAMILIES),
        "rows": rows,
        "entity_means": entity_means,
        "global_raw_mean": global_raw_mean,
        "redesign_gate": redesign_gate,
        "transfer_gates": transfer_gates,
        "validation_sample_id_sha256": EXPECTED_VALIDATION_DIGEST,
        "protocol_path": str(PROTOCOL),
        "protocol_sha256": _sha256(PROTOCOL),
        "config_path": str(CONFIG),
        "config_sha256": _sha256(CONFIG),
        "prior_summary_path": str(PRIOR_SUMMARY),
        "prior_summary_sha256": _sha256(PRIOR_SUMMARY),
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
        "study": "pff_trajectory_forecast_entity_v1",
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
        raise ValueError("Entity forecast artifact audit failed: " + ", ".join(failed))
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
