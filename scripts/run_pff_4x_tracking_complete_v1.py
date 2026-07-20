"""Run and resume the frozen scratch-vs-4x PFF tracking-backbone gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summarize_pff_transfer_gate import summarize_transfer_gate  # noqa: E402

SEEDS = (7, 11, 23)
CURVE_STEPS = (100, 250, 500, 1000, 2000, 5000, 10000)
CONFIG = Path("configs/td_jepa_pff_wc2022_4x_tracking_complete_v1.yaml")
PROTOCOL = Path("docs/PFF_4X_TRACKING_BACKBONE_COMPLETE_PROTOCOL_V1.md")
SOURCE_STATE = Path("runs/grf_position_scale_v1/execution_manifest.json")
RUN_ROOT = ROOT / "runs" / "pff_4x_tracking_complete_v1"
STATE_PATH = RUN_ROOT / "execution_manifest.json"
SUMMARY_PATH = RUN_ROOT / "gate_summary.json"
LOG_DIR = RUN_ROOT / "logs"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _artifact_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _source_checkpoints() -> dict[int, dict[str, str]]:
    state = json.loads((ROOT / SOURCE_STATE).read_text(encoding="utf-8"))
    checkpoints = {}
    for seed in SEEDS:
        record = state["runs"][f"synthetic:4x:{seed}"]
        path = _artifact_path(record["latest_checkpoint"])
        actual_hash = _sha256(path)
        if actual_hash != record["latest_checkpoint_sha256"]:
            raise ValueError(f"4x source checkpoint hash mismatch for seed {seed}.")
        checkpoints[seed] = {
            "path": record["latest_checkpoint"],
            "sha256": actual_hash,
        }
    return checkpoints


def _load_state() -> dict[str, Any]:
    checkpoints = _source_checkpoints()
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if state["protocol_sha256"] != _sha256(ROOT / PROTOCOL):
            raise ValueError("Frozen protocol changed after execution state was created.")
        if state["config_sha256"] != _sha256(ROOT / CONFIG):
            raise ValueError("Frozen training config changed after execution state was created.")
        return state
    return {
        "version": 1,
        "protocol_path": str(PROTOCOL),
        "protocol_sha256": _sha256(ROOT / PROTOCOL),
        "config_path": str(CONFIG),
        "config_sha256": _sha256(ROOT / CONFIG),
        "source_execution_manifest_path": str(SOURCE_STATE),
        "source_execution_manifest_sha256": _sha256(ROOT / SOURCE_STATE),
        "source_4x_checkpoints": {str(seed): value for seed, value in checkpoints.items()},
        "seeds": list(SEEDS),
        "runs": {},
        "created_at_utc": _utc_now(),
    }


def _save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at_utc"] = _utc_now()
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _completed(state: dict[str, Any], step: str) -> bool:
    record = state["runs"].get(step)
    return bool(
        record
        and record.get("status") == "complete"
        and _artifact_path(record["latest_checkpoint"]).exists()
        and _artifact_path(record["metrics_path"]).exists()
    )


def _parse_output_path(stdout: str, label: str) -> str:
    match = re.search(rf"^{re.escape(label)}:\s+(.+)$", stdout, flags=re.MULTILINE)
    if not match:
        raise RuntimeError(f"Training output did not contain {label!r}.")
    return match.group(1).strip()


def _run_one(
    state: dict[str, Any],
    *,
    family: str,
    seed: int,
    init_checkpoint: str | None,
) -> None:
    step = f"pff:{family}:{seed}"
    if _completed(state, step):
        print(f"[skip] {step}", flush=True)
        return
    command = [
        sys.executable,
        "scripts/train_td_jepa.py",
        "--config",
        str(CONFIG),
        "--seed",
        str(seed),
    ]
    if init_checkpoint is not None:
        command.extend(["--init-checkpoint", init_checkpoint])
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{family}_seed{seed}.log"
    print(f"[run ] {step}", flush=True)
    started = _utc_now()
    process = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_path.write_text(process.stdout, encoding="utf-8")
    if process.returncode != 0:
        print(process.stdout[-4000:], flush=True)
        raise RuntimeError(f"{step} failed with exit code {process.returncode}.")
    run_dir = _parse_output_path(process.stdout, "run_dir")
    latest = _parse_output_path(process.stdout, "latest")
    metrics = str(Path(run_dir) / "metrics_val.jsonl")
    state["runs"][step] = {
        "status": "complete",
        "family": family,
        "seed": seed,
        "run_dir": run_dir,
        "latest_checkpoint": latest,
        "latest_checkpoint_sha256": _sha256(_artifact_path(latest)),
        "metrics_path": metrics,
        "metrics_sha256": _sha256(_artifact_path(metrics)),
        "init_checkpoint": init_checkpoint,
        "init_checkpoint_sha256": (
            _sha256(_artifact_path(init_checkpoint)) if init_checkpoint else None
        ),
        "log_path": str(log_path.relative_to(ROOT)),
        "started_at_utc": started,
        "completed_at_utc": _utc_now(),
    }
    _save_state(state)


def run_training(state: dict[str, Any]) -> None:
    for seed in SEEDS:
        _run_one(state, family="scratch", seed=seed, init_checkpoint=None)
    for seed in SEEDS:
        checkpoint = state["source_4x_checkpoints"][str(seed)]["path"]
        _run_one(state, family="4x", seed=seed, init_checkpoint=checkpoint)


def _read_curve(metrics_path: Path) -> list[dict[str, Any]]:
    path = metrics_path.with_name("metrics_val_curve.jsonl")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _mean_curves(state: dict[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    output = {}
    for family in ("scratch", "4x"):
        by_step: dict[int, list[dict[str, Any]]] = {}
        for seed in SEEDS:
            metrics_path = _artifact_path(state["runs"][f"pff:{family}:{seed}"]["metrics_path"])
            for row in _read_curve(metrics_path):
                by_step.setdefault(int(row["step"]), []).append(row)
        output[family] = {
            str(step): {
                "total_loss": sum(float(row["total_loss"]) for row in rows) / len(rows),
                "td_loss": sum(float(row["td_loss"]) for row in rows) / len(rows),
            }
            for step, rows in sorted(by_step.items())
        }
    return output


def _access_audit(state: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {"expected_run_count": len(state["runs"]) == 6}
    rows = []
    for name, record in sorted(state["runs"].items()):
        run_dir = _artifact_path(record["run_dir"])
        manifest_path = run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        curve = _read_curve(_artifact_path(record["metrics_path"]))
        final = json.loads(
            _artifact_path(record["metrics_path"]).read_text(encoding="utf-8").splitlines()[-1]
        )
        row_checks = {
            "train_and_validation_only": (
                set(manifest["data_access"]["loaded_tensor_splits"]) == {"train", "val"}
            ),
            "test_not_loaded": "test" not in manifest["data_access"]["loaded_tensor_splits"],
            "embedding_export_disabled": (
                manifest["data_access"]["embedding_sample_split"] is None
            ),
            "curve_steps_match": [int(row["step"]) for row in curve] == list(CURVE_STEPS),
            "final_step_is_10000": int(final["step"]) == 10000,
            "final_validation_examples_are_64000": int(final["num_examples"]) == 64000,
        }
        checks.update({f"{name}:{key}": passed for key, passed in row_checks.items()})
        rows.append(
            {
                "run": name,
                "run_manifest_path": str(manifest_path.relative_to(ROOT)),
                "run_manifest_sha256": _sha256(manifest_path),
                "checks": row_checks,
            }
        )
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "status": "passed" if not failed else "blocked",
        "checks": checks,
        "failed_checks": failed,
        "runs": rows,
    }


def summarize(state: dict[str, Any]) -> dict[str, Any]:
    scratch = {
        seed: _artifact_path(state["runs"][f"pff:scratch:{seed}"]["metrics_path"])
        for seed in SEEDS
    }
    transfer = {
        seed: _artifact_path(state["runs"][f"pff:4x:{seed}"]["metrics_path"])
        for seed in SEEDS
    }
    summary = summarize_transfer_gate(
        scratch,
        transfer,
        min_total_wins=2,
        min_mean_total_improvement=0.01,
        max_mean_td_relative_change=0.0,
        min_z_online_std_mean=0.05,
    )
    access = _access_audit(state)
    if access["status"] != "passed":
        summary["status"] = "blocked"
        summary["blocking_conditions"] = [
            *summary["blocking_conditions"],
            "run_access_integrity",
        ]
    scratch_total = float(summary["means"]["scratch_total_loss"])
    transfer_total = float(summary["means"]["transfer_total_loss"])
    scratch_td = float(summary["means"]["scratch_td_loss"])
    transfer_td = float(summary["means"]["transfer_td_loss"])
    finite_means = all(
        math.isfinite(value)
        for value in (scratch_total, transfer_total, scratch_td, transfer_td)
    )
    operational_family = (
        "4x_grf"
        if finite_means and transfer_total < scratch_total and transfer_td <= scratch_td
        else "scratch"
    )
    summary.update(
        {
            "study": "pff_4x_tracking_backbone_complete_v1",
            "family_labels": {"scratch": "scratch", "transfer": "4x_grf"},
            "mean_validation_curves": _mean_curves(state),
            "run_access_audit": access,
            "operational_family": operational_family,
            "protocol_path": str(PROTOCOL),
            "protocol_sha256": _sha256(ROOT / PROTOCOL),
            "config_path": str(CONFIG),
            "config_sha256": _sha256(ROOT / CONFIG),
            "execution_manifest_path": str(STATE_PATH.relative_to(ROOT)),
        }
    )
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    state["gate_summary_path"] = str(SUMMARY_PATH.relative_to(ROOT))
    state["gate_summary_sha256"] = _sha256(SUMMARY_PATH)
    state["gate_status"] = summary["status"]
    state["operational_family"] = operational_family
    _save_state(state)
    print(f"[done] gate status: {summary['status']}", flush=True)
    print(f"[done] operational family: {operational_family}", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("train", "summarize", "all"), default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = _load_state()
    _save_state(state)
    if args.stage in {"train", "all"}:
        run_training(state)
    if args.stage in {"summarize", "all"}:
        summarize(state)


if __name__ == "__main__":
    main()
