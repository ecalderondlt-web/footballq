"""Run and summarize the frozen StatsBomb semantic pretraining study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path("docs/STATSBOMB_SEMANTIC_PRETRAIN_PROTOCOL_V1.md")
CONFIGS = {
    "event_only": Path("configs/statsbomb_event_only_pretrain_v1.yaml"),
    "event_plus_360": Path("configs/statsbomb_event_plus_360_pretrain_v1.yaml"),
}
DATA_MANIFEST = Path("data/processed/statsbomb_event_sequence_v1/manifest.json")
BASELINES = Path("runs/integrity/statsbomb_event_baselines_v1.json")
TENSOR_AUDIT = Path("runs/integrity/statsbomb_event_sequence_v1_tensor_audit.json")
STATE_PATH = Path("runs/statsbomb_semantic_pretrain_v1/execution_manifest.json")
SUMMARY_PATH = Path("runs/statsbomb_semantic_pretrain_v1/gate_summary.json")
AUDIT_PATH = Path("runs/integrity/statsbomb_semantic_pretrain_v1_artifact_audit.json")
LOG_DIR = Path("runs/statsbomb_semantic_pretrain_v1/logs")
SEEDS = (7, 11, 23)
CURVE_STEPS = (100, 500, 1000, 2500, 5700)
FINAL_STEP = 5700
FINAL_WINDOWS = 93479
FINAL_EVENT_TARGETS = 2991328
FINAL_ANCHORED_EVENT_TARGETS = 278803


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with (ROOT / path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _save_state(state: dict[str, Any]) -> None:
    path = ROOT / STATE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _expected_hashes() -> dict[str, str]:
    return {
        "protocol_sha256": _sha256(PROTOCOL),
        "event_only_config_sha256": _sha256(CONFIGS["event_only"]),
        "event_plus_360_config_sha256": _sha256(CONFIGS["event_plus_360"]),
        "data_manifest_sha256": _sha256(DATA_MANIFEST),
        "baselines_sha256": _sha256(BASELINES),
        "tensor_audit_sha256": _sha256(TENSOR_AUDIT),
    }


def _load_state() -> dict[str, Any]:
    path = ROOT / STATE_PATH
    expected = _expected_hashes()
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        mismatches = [key for key, value in expected.items() if state.get(key) != value]
        if mismatches:
            raise ValueError(
                "Frozen StatsBomb study artifacts changed after execution began: "
                + ", ".join(mismatches)
            )
        return state
    return {
        "version": 1,
        "study": "statsbomb_semantic_pretrain_v1",
        "frozen_at_utc": _utc_now(),
        **expected,
        "seeds": list(SEEDS),
        "curve_steps": list(CURVE_STEPS),
        "final_step": FINAL_STEP,
        "runs": {},
        "current_run": None,
    }


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _run_one(state: dict[str, Any], family: str, seed: int) -> None:
    key = f"statsbomb:{family}:{seed}"
    if state["runs"].get(key, {}).get("status") == "complete":
        print(f"[skip] {key}", flush=True)
        return
    LOG_DIR_PATH = ROOT / LOG_DIR
    LOG_DIR_PATH.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR_PATH / f"{family}_seed{seed}.log"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_statsbomb_event.py"),
        "--config",
        str(ROOT / CONFIGS[family]),
        "--seed",
        str(seed),
    ]
    state["current_run"] = {
        "key": key,
        "family": family,
        "seed": seed,
        "started_at_utc": _utc_now(),
        "log_path": str(log_path.relative_to(ROOT)),
    }
    _save_state(state)
    print(f"[run] {key}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"StatsBomb training failed for {key}; see {log_path}")
    run_dir_line = next(
        (
            line
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("run_dir:")
        ),
        None,
    )
    if run_dir_line is None:
        raise ValueError(f"StatsBomb trainer did not report a run directory for {key}.")
    run_dir = _artifact_path(run_dir_line.split(":", 1)[1].strip())
    metrics_path = run_dir / "metrics_val.jsonl"
    curve_path = run_dir / "metrics_val_curve.jsonl"
    final = _read_rows(metrics_path)[-1]
    curve = _read_rows(curve_path)
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if int(final["step"]) != FINAL_STEP:
        raise ValueError(f"Unexpected final step for {key}: {final['step']}")
    if [int(row["step"]) for row in curve] != list(CURVE_STEPS):
        raise ValueError(f"Unexpected validation curve steps for {key}.")
    if run_manifest["data_access"] != {
        "loaded_splits": ["train", "val"],
        "test_loaded": False,
        "embedding_export_split": None,
    }:
        raise ValueError(f"Unexpected data access for {key}.")
    state["runs"][key] = {
        "status": "complete",
        "family": family,
        "seed": seed,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "latest_checkpoint": str((run_dir / "latest.pt").relative_to(ROOT)),
        "latest_checkpoint_sha256": _sha256((run_dir / "latest.pt").relative_to(ROOT)),
        "metrics_path": str(metrics_path.relative_to(ROOT)),
        "metrics_sha256": _sha256(metrics_path.relative_to(ROOT)),
        "curve_path": str(curve_path.relative_to(ROOT)),
        "curve_sha256": _sha256(curve_path.relative_to(ROOT)),
        "run_manifest_path": str(run_manifest_path.relative_to(ROOT)),
        "run_manifest_sha256": _sha256(run_manifest_path.relative_to(ROOT)),
        "started_at_utc": state["current_run"]["started_at_utc"],
        "completed_at_utc": _utc_now(),
    }
    state["current_run"] = None
    _save_state(state)


def run_training(state: dict[str, Any]) -> None:
    for family in ("event_only", "event_plus_360"):
        for seed in SEEDS:
            _run_one(state, family, seed)


def _relative_improvement(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference


def _relative_change(reference: float, candidate: float) -> float:
    return (candidate - reference) / reference


def summarize(state: dict[str, Any]) -> dict[str, Any]:
    if len(state["runs"]) != 6:
        raise ValueError("StatsBomb semantic summary requires all six runs.")
    rows = []
    metrics = (
        "event_type_loss",
        "location_mae",
        "anchored_event_type_loss",
        "anchored_location_mae",
    )
    for seed in SEEDS:
        row = {"seed": seed}
        for family in ("event_only", "event_plus_360"):
            record = state["runs"][f"statsbomb:{family}:{seed}"]
            final = _read_rows(_artifact_path(record["metrics_path"]))[-1]
            row[family] = {
                **{metric: float(final[metric]) for metric in metrics},
                "windows": int(final["windows"]),
                "event_targets": int(final["event_targets"]),
                "anchored_event_targets": int(final["anchored_event_targets"]),
                "metrics_path": record["metrics_path"],
                "metrics_sha256": record["metrics_sha256"],
            }
        rows.append(row)
    means = {
        family: {
            metric: sum(row[family][metric] for row in rows) / len(rows)
            for metric in metrics
        }
        for family in ("event_only", "event_plus_360")
    }
    baselines = json.loads((ROOT / BASELINES).read_text(encoding="utf-8"))
    markov = float(baselines["first_order_markov_event_type_nll"])
    anchored_event_wins = sum(
        row["event_plus_360"]["anchored_event_type_loss"]
        < row["event_only"]["anchored_event_type_loss"]
        for row in rows
    )
    anchored_location_wins = sum(
        row["event_plus_360"]["anchored_location_mae"]
        < row["event_only"]["anchored_location_mae"]
        for row in rows
    )
    finite = all(
        math.isfinite(row[family][metric])
        for row in rows
        for family in ("event_only", "event_plus_360")
        for metric in metrics
    )
    coverage = all(
        row[family]["windows"] == FINAL_WINDOWS
        and row[family]["event_targets"] == FINAL_EVENT_TARGETS
        and row[family]["anchored_event_targets"] > 0
        for row in rows
        for family in ("event_only", "event_plus_360")
    )
    criteria = {
        "finite_metrics": {"passed": finite},
        "event_only_beats_markov": {
            "value": _relative_improvement(markov, means["event_only"]["event_type_loss"]),
            "minimum": 0.01,
        },
        "anchored_event_type_wins": {
            "value": anchored_event_wins,
            "minimum": 2,
        },
        "anchored_event_type_improvement": {
            "value": _relative_improvement(
                means["event_only"]["anchored_event_type_loss"],
                means["event_plus_360"]["anchored_event_type_loss"],
            ),
            "minimum": 0.01,
        },
        "anchored_location_wins": {
            "value": anchored_location_wins,
            "minimum": 2,
        },
        "anchored_location_improvement": {
            "value": _relative_improvement(
                means["event_only"]["anchored_location_mae"],
                means["event_plus_360"]["anchored_location_mae"],
            ),
            "minimum": 0.01,
        },
        "overall_event_type_non_degradation": {
            "value": _relative_change(
                means["event_only"]["event_type_loss"],
                means["event_plus_360"]["event_type_loss"],
            ),
            "maximum": 0.01,
        },
        "overall_location_non_degradation": {
            "value": _relative_change(
                means["event_only"]["location_mae"],
                means["event_plus_360"]["location_mae"],
            ),
            "maximum": 0.01,
        },
        "final_coverage": {"passed": coverage},
    }
    for criterion in criteria.values():
        if "passed" in criterion:
            continue
        if "minimum" in criterion:
            criterion["passed"] = criterion["value"] >= criterion["minimum"]
        else:
            criterion["passed"] = criterion["value"] <= criterion["maximum"]
    blockers = [name for name, criterion in criteria.items() if not criterion["passed"]]
    baseline_passed = criteria["event_only_beats_markov"]["passed"]
    status = "controls_passed" if not blockers else "blocked"
    operational_family = (
        "event_plus_360"
        if status == "controls_passed"
        else ("event_only" if baseline_passed else "none")
    )
    mean_curves = {}
    for family in ("event_only", "event_plus_360"):
        by_step: dict[int, list[dict[str, Any]]] = {}
        for seed in SEEDS:
            record = state["runs"][f"statsbomb:{family}:{seed}"]
            for curve_row in _read_rows(_artifact_path(record["curve_path"])):
                by_step.setdefault(int(curve_row["step"]), []).append(curve_row)
        mean_curves[family] = {
            str(step): {
                metric: sum(float(row[metric]) for row in curve_rows) / len(curve_rows)
                for metric in metrics
            }
            for step, curve_rows in sorted(by_step.items())
        }
    summary = {
        "version": 1,
        "study": "statsbomb_semantic_pretrain_v1",
        "status": status,
        "operational_family": operational_family,
        "blocking_conditions": blockers,
        "seeds": list(SEEDS),
        "rows": rows,
        "means": means,
        "criteria": criteria,
        "raw_baselines": baselines,
        "mean_validation_curves": mean_curves,
        "protocol_path": str(PROTOCOL),
        "protocol_sha256": _sha256(PROTOCOL),
        "config_sha256": {family: _sha256(path) for family, path in CONFIGS.items()},
        "data_access": {
            "loaded_splits": ["train", "val"],
            "test_loaded": False,
            "run_count": 6,
        },
        "execution_manifest_path": str(STATE_PATH),
    }
    output = ROOT / SUMMARY_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    state["gate_summary_path"] = str(SUMMARY_PATH)
    state["gate_summary_sha256"] = _sha256(SUMMARY_PATH)
    state["gate_status"] = status
    state["operational_family"] = operational_family
    _save_state(state)
    print(f"[done] gate status: {status}", flush=True)
    print(f"[done] operational family: {operational_family}", flush=True)
    return summary


def verify_artifacts(state: dict[str, Any]) -> dict[str, Any]:
    expected_runs = {
        f"statsbomb:{family}:{seed}"
        for family in ("event_only", "event_plus_360")
        for seed in SEEDS
    }
    checks: dict[str, bool] = {
        "expected_runs_complete": (
            set(state.get("runs", {})) == expected_runs
            and all(record.get("status") == "complete" for record in state["runs"].values())
        ),
        "frozen_inputs_match_state": all(
            state.get(key) == value for key, value in _expected_hashes().items()
        ),
    }
    run_rows = []
    for key in sorted(expected_runs):
        record = state["runs"].get(key, {})
        family = record.get("family")
        seed = record.get("seed")
        artifact_fields = {
            "latest_checkpoint": "latest_checkpoint_sha256",
            "metrics_path": "metrics_sha256",
            "curve_path": "curve_sha256",
            "run_manifest_path": "run_manifest_sha256",
        }
        artifact_hashes = {}
        for path_field, hash_field in artifact_fields.items():
            path_value = record.get(path_field)
            expected_hash = record.get(hash_field)
            path = _artifact_path(path_value) if path_value else None
            actual_hash = _sha256(path.relative_to(ROOT)) if path and path.exists() else None
            artifact_hashes[path_field] = {
                "path": path_value,
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
            }
            checks[f"{key}:{path_field}_hash"] = actual_hash == expected_hash

        metrics_path = _artifact_path(record["metrics_path"])
        curve_path = _artifact_path(record["curve_path"])
        manifest_path = _artifact_path(record["run_manifest_path"])
        final = _read_rows(metrics_path)[-1]
        curve = _read_rows(curve_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_checks = {
            "identity_matches": (
                key == f"statsbomb:{family}:{seed}"
                and family in CONFIGS
                and seed in SEEDS
                and manifest["feature_view"] == family
                and int(manifest["training"]["seed"]) == seed
            ),
            "curve_steps_match": [int(row["step"]) for row in curve] == list(CURVE_STEPS),
            "final_step_matches": int(final["step"]) == FINAL_STEP,
            "final_windows_match": int(final["windows"]) == FINAL_WINDOWS,
            "final_event_targets_match": int(final["event_targets"])
            == FINAL_EVENT_TARGETS,
            "final_anchored_targets_match": int(final["anchored_event_targets"])
            == FINAL_ANCHORED_EVENT_TARGETS,
            "final_metrics_finite": all(
                math.isfinite(float(final[name]))
                for name in (
                    "total_loss",
                    "event_type_loss",
                    "location_mae",
                    "anchored_event_type_loss",
                    "anchored_location_mae",
                )
            ),
            "train_and_validation_only": manifest["data_access"]
            == {
                "loaded_splits": ["train", "val"],
                "test_loaded": False,
                "embedding_export_split": None,
            },
            "config_matches_family": manifest["config_sha256"]
            == state[f"{family}_config_sha256"],
            "data_manifest_matches": manifest["event_data"]["manifest_sha256"]
            == state["data_manifest_sha256"],
        }
        checks.update({f"{key}:{name}": passed for name, passed in run_checks.items()})
        run_rows.append(
            {
                "run": key,
                "family": family,
                "seed": seed,
                "artifacts": artifact_hashes,
                "checks": run_checks,
            }
        )

    summary_path = _artifact_path(state.get("gate_summary_path", SUMMARY_PATH))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    tensor_audit = json.loads((ROOT / TENSOR_AUDIT).read_text(encoding="utf-8"))
    baselines = json.loads((ROOT / BASELINES).read_text(encoding="utf-8"))
    summary_checks = {
        "gate_summary_hash_matches": _sha256(summary_path.relative_to(ROOT))
        == state.get("gate_summary_sha256"),
        "gate_status_blocked": summary.get("status") == "blocked",
        "operational_family_event_only": summary.get("operational_family") == "event_only",
        "blocking_conditions_match": summary.get("blocking_conditions")
        == [
            "anchored_event_type_improvement",
            "anchored_location_wins",
            "anchored_location_improvement",
            "overall_location_non_degradation",
        ],
        "summary_access_is_train_val_only": summary.get("data_access")
        == {"loaded_splits": ["train", "val"], "test_loaded": False, "run_count": 6},
        "tensor_audit_passed": (
            tensor_audit.get("status") == "passed"
            and tensor_audit.get("loaded_splits") == ["train", "val"]
            and tensor_audit.get("test_loaded") is False
        ),
        "baselines_access_is_train_val_only": (
            baselines.get("loaded_splits") == ["train", "val"]
            and baselines.get("test_loaded") is False
        ),
    }
    checks.update(summary_checks)
    failed = sorted(name for name, passed in checks.items() if not passed)
    report = {
        "version": 1,
        "study": "statsbomb_semantic_pretrain_v1",
        "scope": "frozen_inputs_and_six_result_bearing_runs",
        "status": "passed" if not failed else "blocked",
        "checks": checks,
        "failed_checks": failed,
        "runs": run_rows,
        "summary_checks": summary_checks,
        "data_access": {
            "loaded_splits": ["train", "val"],
            "test_loaded": False,
            "embedding_exported": False,
        },
    }
    output = ROOT / AUDIT_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    state["artifact_audit_path"] = str(AUDIT_PATH)
    state["artifact_audit_sha256"] = _sha256(AUDIT_PATH)
    state["artifact_audit_status"] = report["status"]
    _save_state(state)
    if failed:
        raise ValueError("StatsBomb artifact audit failed: " + ", ".join(failed))
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
