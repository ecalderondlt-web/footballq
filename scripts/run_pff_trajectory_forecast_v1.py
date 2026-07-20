"""Run and audit the frozen PFF multi-horizon trajectory forecast gate."""

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

from footballq.repro.manifest import (  # noqa: E402
    build_run_manifest,
    file_sha256,
    write_run_manifest,
)
from footballq.training.train_trajectory_forecast import (  # noqa: E402
    evaluate_forecast_baseline,
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
PROTOCOL = Path("docs/PFF_TRAJECTORY_FORECAST_PROTOCOL_V1.md")
CONFIG = Path("configs/pff_trajectory_forecast_v1.yaml")
DATA_MANIFEST = Path("data/processed/pff_wc2022_trajectory_forecast_v1/dataset_manifest.json")
SPLIT_MANIFEST = Path("splits/pff_wc2022_64match_inductive_v1.json")
SOURCE_EXECUTION = Path("runs/pff_4x_tracking_complete_v1/execution_manifest.json")
DATA_AUDIT = Path("runs/integrity/pff_trajectory_forecast_v1_data_audit.json")
RUN_ROOT = Path("runs/pff_trajectory_forecast_v1")
STATE_PATH = RUN_ROOT / "execution_manifest.json"
SUMMARY_PATH = RUN_ROOT / "gate_summary.json"
AUDIT_PATH = Path("runs/integrity/pff_trajectory_forecast_v1_artifact_audit.json")

FROZEN_HASHES = {
    "protocol_sha256": "2f3e03935a0ed6127403ab5e73e0f1bf0c9a6296c2a3faf768753a5fbc63bf97",
    "config_sha256": "b9e64d591c75269b5d1d4717b6a24b9ff51e662d5d55de4089544effad0359fc",
    "forecast_manifest_sha256": "688761b30c4fbe38d832d09d459e79153acc5851a397ceb600d5bc30c811b537",
    "split_manifest_sha256": "9f7d56184920e463f1aa5fdcee05dc9b59438184910afc93a7e0c12f4e322226",
    "source_execution_manifest_sha256": (
        "463063455fa8850eb4b49ea5ba163db19a620dc1dda71f0c0ca308baf1ae9f00"
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    output = _artifact_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


def _current_hashes() -> dict[str, str]:
    return {
        "protocol_sha256": _sha256(PROTOCOL),
        "config_sha256": _sha256(CONFIG),
        "forecast_manifest_sha256": _sha256(DATA_MANIFEST),
        "split_manifest_sha256": _sha256(SPLIT_MANIFEST),
        "source_execution_manifest_sha256": _sha256(SOURCE_EXECUTION),
    }


def _source_checkpoints() -> dict[int, dict[str, str]]:
    execution = json.loads(_artifact_path(SOURCE_EXECUTION).read_text(encoding="utf-8"))
    checkpoints = {}
    for seed in SEEDS:
        record = execution["runs"][f"pff:scratch:{seed}"]
        path = str(record["latest_checkpoint"])
        actual = _sha256(path)
        if actual != CHECKPOINT_HASHES[seed] or actual != record["latest_checkpoint_sha256"]:
            raise ValueError(f"Frozen scratch checkpoint hash mismatch for seed {seed}.")
        checkpoints[seed] = {"path": path, "sha256": actual}
    return checkpoints


def _new_state() -> dict[str, Any]:
    current = _current_hashes()
    if current != FROZEN_HASHES:
        raise ValueError(f"Frozen forecast inputs changed: expected {FROZEN_HASHES}, got {current}")
    data_audit = json.loads(_artifact_path(DATA_AUDIT).read_text(encoding="utf-8"))
    if data_audit.get("status") != "passed":
        raise ValueError("Forecast data audit must pass before result-bearing validation runs.")
    now = datetime.now(UTC).replace(microsecond=0).isoformat()
    return {
        "version": 1,
        "study": "pff_trajectory_forecast_v1",
        **current,
        "data_audit_path": str(DATA_AUDIT),
        "data_audit_sha256": _sha256(DATA_AUDIT),
        "tracking_checkpoints": {
            str(seed): value for seed, value in _source_checkpoints().items()
        },
        "seeds": list(SEEDS),
        "families": list(FAMILIES),
        "baselines": {},
        "runs": {},
        "created_at_utc": now,
        "updated_at_utc": now,
    }


def _load_state() -> dict[str, Any]:
    path = _artifact_path(STATE_PATH)
    if not path.exists():
        return _new_state()
    state = json.loads(path.read_text(encoding="utf-8"))
    for key, value in FROZEN_HASHES.items():
        if state.get(key) != value or _current_hashes()[key] != value:
            raise ValueError(f"Frozen forecast state mismatch for {key}.")
    return state


def _save_state(state: dict[str, Any]) -> None:
    state["updated_at_utc"] = datetime.now(UTC).replace(microsecond=0).isoformat()
    _write_json(STATE_PATH, state)


def _record_valid(record: dict[str, Any]) -> bool:
    required = {
        "latest_checkpoint": "latest_checkpoint_sha256",
        "metrics_path": "metrics_sha256",
        "run_manifest_path": "run_manifest_sha256",
    }
    return bool(
        record.get("status") == "complete"
        and all(
            _artifact_path(record[path_key]).exists()
            and _sha256(record[path_key]) == record[hash_key]
            for path_key, hash_key in required.items()
        )
    )


def run_baselines(state: dict[str, Any]) -> None:
    if state.get("baselines", {}).get("status") == "complete":
        metrics_path = state["baselines"]["metrics_path"]
        if _artifact_path(metrics_path).exists() and _sha256(metrics_path) == state["baselines"][
            "metrics_sha256"
        ]:
            print("[resume] forecast baselines", flush=True)
            return
    metrics = {
        baseline: evaluate_forecast_baseline(CONFIG, baseline=baseline, split="val")
        for baseline in ("last_position", "constant_velocity")
    }
    digests = {value["sample_id_sha256"] for value in metrics.values()}
    if len(digests) != 1:
        raise ValueError("Forecast baselines evaluated different validation samples.")
    metrics_path = RUN_ROOT / "baseline_metrics.json"
    _write_json(metrics_path, metrics)
    manifest = build_run_manifest(
        command=sys.argv,
        config_path=CONFIG,
        split_manifest_path=SPLIT_MANIFEST,
        evaluation_protocol="matched_multihorizon_forecast_validation_only_v1",
        feature_view="position_only_observed_tracking",
        objective_mode="deterministic_trajectory_baselines",
        dataset_paths={"forecast_manifest": DATA_MANIFEST},
        output_paths={"metrics": metrics_path},
        warnings=["PFF test tracking is sealed and was not loaded."],
    )
    manifest.update(
        {
            "loaded_splits": ["val"],
            "test_loaded": False,
            "validation_sample_id_sha256": next(iter(digests)),
        }
    )
    manifest_path = RUN_ROOT / "baseline_run_manifest.json"
    write_run_manifest(_artifact_path(manifest_path), manifest)
    state["baselines"] = {
        "status": "complete",
        "metrics_path": str(metrics_path),
        "metrics_sha256": _sha256(metrics_path),
        "run_manifest_path": str(manifest_path),
        "run_manifest_sha256": _sha256(manifest_path),
        "validation_sample_id_sha256": next(iter(digests)),
    }
    _save_state(state)


def run_training(state: dict[str, Any]) -> None:
    for family in FAMILIES:
        for seed in SEEDS:
            key = f"forecast:{family}:{seed}"
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
            record = {
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
            state["runs"][key] = record
            _save_state(state)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _read_final_metrics(path: str | Path) -> dict[str, Any]:
    lines = _artifact_path(path).read_text(encoding="utf-8").strip().splitlines()
    if len(lines) != 1:
        raise ValueError(f"Expected exactly one final metric row in {path}.")
    return json.loads(lines[0])


def _relative_improvement(reference: float, candidate: float) -> float:
    return (float(reference) - float(candidate)) / float(reference)


def _family_gate(
    family: str,
    rows_by_family: dict[str, dict[int, dict[str, Any]]],
    means: dict[str, dict[str, float]],
    constant_velocity: dict[str, Any],
) -> dict[str, Any]:
    wins = sum(
        rows_by_family[family][seed]["player_ADE_m"]
        < rows_by_family["raw"][seed]["player_ADE_m"]
        for seed in SEEDS
    )
    mean_improvement = _relative_improvement(
        means["raw"]["player_ADE_m"], means[family]["player_ADE_m"]
    )
    horizon_improvements = {
        key: _relative_improvement(means["raw"][key], means[family][key])
        for key in HORIZON_KEYS
    }
    cv_improvement = _relative_improvement(
        constant_velocity["player_ADE_m"], means[family]["player_ADE_m"]
    )
    criteria = {
        "seed_wins_vs_raw": {"value": wins, "minimum": 2, "passed": wins >= 2},
        "mean_player_ADE_improvement_vs_raw": {
            "value": mean_improvement,
            "minimum": 0.02,
            "passed": mean_improvement >= 0.02,
        },
        "worst_horizon_improvement_vs_raw": {
            "value": min(horizon_improvements.values()),
            "minimum": -0.01,
            "passed": min(horizon_improvements.values()) >= -0.01,
        },
        "mean_player_ADE_improvement_vs_constant_velocity": {
            "value": cv_improvement,
            "minimum": 0.01,
            "passed": cv_improvement >= 0.01,
        },
    }
    blockers = [name for name, criterion in criteria.items() if not criterion["passed"]]
    return {
        "family": family,
        "passed": not blockers,
        "criteria": criteria,
        "horizon_improvements_vs_raw": horizon_improvements,
        "blocking_conditions": blockers,
    }


def summarize(state: dict[str, Any]) -> dict[str, Any]:
    if len(state.get("runs", {})) != len(SEEDS) * len(FAMILIES):
        raise ValueError("All nine learned forecast runs must complete before summarization.")
    baselines = json.loads(
        _artifact_path(state["baselines"]["metrics_path"]).read_text(encoding="utf-8")
    )
    rows_by_family: dict[str, dict[int, dict[str, Any]]] = {
        family: {} for family in FAMILIES
    }
    rows = []
    for family in FAMILIES:
        for seed in SEEDS:
            record = state["runs"][f"forecast:{family}:{seed}"]
            metrics = _read_final_metrics(record["metrics_path"])
            rows_by_family[family][seed] = metrics
            rows.append({"family": family, "seed": seed, **metrics})
    digest_set = {
        state["baselines"]["validation_sample_id_sha256"],
        *(row["sample_id_sha256"] for row in rows),
    }
    if len(digest_set) != 1:
        raise ValueError("Forecast runs did not evaluate the same frozen validation samples.")
    metric_names = [
        "player_ADE_m",
        "player_FDE_m",
        "ball_ADE_m",
        "ball_FDE_m",
        "all_entity_ADE_m",
        "all_entity_FDE_m",
        *HORIZON_KEYS,
    ]
    means = {
        family: {
            name: sum(float(rows_by_family[family][seed][name]) for seed in SEEDS) / len(SEEDS)
            for name in metric_names
        }
        for family in FAMILIES
    }
    family_gates = {
        family: _family_gate(
            family, rows_by_family, means, baselines["constant_velocity"]
        )
        for family in TRANSFERRED
    }
    passing = [family for family in TRANSFERRED if family_gates[family]["passed"]]
    if passing:
        operational = min(passing, key=lambda family: means[family]["player_ADE_m"])
        status = "passed"
        blockers = []
    else:
        operational = (
            "raw"
            if means["raw"]["player_ADE_m"]
            < baselines["constant_velocity"]["player_ADE_m"]
            else "constant_velocity"
        )
        status = "blocked"
        blockers = [
            f"{family}:{criterion}"
            for family in TRANSFERRED
            for criterion in family_gates[family]["blocking_conditions"]
        ]
    summary = {
        "version": 1,
        "study": "pff_trajectory_forecast_v1",
        "status": status,
        "operational_family": operational,
        "blocking_conditions": blockers,
        "seeds": list(SEEDS),
        "families": list(FAMILIES),
        "baselines": baselines,
        "rows": rows,
        "means": means,
        "family_gates": family_gates,
        "validation_sample_id_sha256": next(iter(digest_set)),
        "protocol_path": str(PROTOCOL),
        "protocol_sha256": _sha256(PROTOCOL),
        "config_path": str(CONFIG),
        "config_sha256": _sha256(CONFIG),
        "forecast_manifest_path": str(DATA_MANIFEST),
        "forecast_manifest_sha256": _sha256(DATA_MANIFEST),
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
        "frozen_inputs_match": _current_hashes() == FROZEN_HASHES,
        "data_audit_hash_matches": _sha256(DATA_AUDIT) == state["data_audit_sha256"],
        "all_runs_complete": len(state.get("runs", {})) == 9
        and all(_record_valid(record) for record in state["runs"].values()),
        "baseline_metrics_hash_matches": _sha256(state["baselines"]["metrics_path"])
        == state["baselines"]["metrics_sha256"],
        "baseline_manifest_hash_matches": _sha256(state["baselines"]["run_manifest_path"])
        == state["baselines"]["run_manifest_sha256"],
        "summary_hash_matches": _sha256(SUMMARY_PATH) == state.get("gate_summary_sha256"),
    }
    run_reports = []
    expected_digest = state["baselines"]["validation_sample_id_sha256"]
    for key, record in sorted(state["runs"].items()):
        manifest = json.loads(
            _artifact_path(record["run_manifest_path"]).read_text(encoding="utf-8")
        )
        boundary_ok = (
            manifest.get("loaded_splits") == ["train", "val"]
            and manifest.get("test_loaded") is False
            and manifest.get("embedding_exported") is False
        )
        digest_ok = manifest.get("validation_sample_id_sha256") == expected_digest
        checks[f"{key}:access_boundary"] = boundary_ok
        checks[f"{key}:validation_digest"] = digest_ok
        checks[f"{key}:curve_hash"] = _sha256(record["curve_path"]) == record["curve_sha256"]
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
        "study": "pff_trajectory_forecast_v1",
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
        raise ValueError("Forecast artifact audit failed: " + ", ".join(failed))
    print("[done] artifact audit: passed", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("baseline", "train", "summarize", "verify", "all"),
        default="all",
    )
    args = parser.parse_args()
    state = _load_state()
    _save_state(state)
    if args.stage in {"baseline", "all"}:
        run_baselines(state)
    if args.stage in {"train", "all"}:
        run_training(state)
    if args.stage in {"summarize", "all"}:
        summarize(state)
    if args.stage in {"verify", "all"}:
        verify_artifacts(state)


if __name__ == "__main__":
    main()
