"""Run and summarize the frozen PFF plus StatsBomb context residual study."""

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
PROTOCOL = Path("docs/PFF_STATSBOMB_CONTEXT_RESIDUAL_PROTOCOL_V1.md")
CONFIG = Path("configs/pff_statsbomb_context_residual_v1.yaml")
TRAIN_AUDIT = Path("runs/integrity/pff_statsbomb_event_context_v1_train_audit.json")
EVENT_MANIFEST = Path("data/processed/pff_statsbomb_event_context_v1/manifest.json")
EVENT_TENSOR_AUDIT = Path(
    "runs/integrity/pff_statsbomb_event_context_v1_train_val_audit.json"
)
TRACKING_MANIFEST = Path(
    "data/processed/pff_wc2022_td_jepa_position_only_train_val_v1/"
    "observed_only/dataset_manifest.json"
)
TRACKING_STATE = Path("runs/pff_4x_tracking_complete_v1/execution_manifest.json")
STATSBOMB_STATE = Path("runs/statsbomb_semantic_pretrain_v1/execution_manifest.json")
RUN_ROOT = Path("runs/pff_statsbomb_context_residual_v1")
STATE_PATH = RUN_ROOT / "execution_manifest.json"
SUMMARY_PATH = RUN_ROOT / "gate_summary.json"
AUDIT_PATH = Path(
    "runs/integrity/pff_statsbomb_context_residual_v1_artifact_audit.json"
)
LOG_DIR = RUN_ROOT / "logs"
FAMILIES = ("tracking", "raw", "random", "pretrained")
SEEDS = (7, 11, 23)
CURVE_STEPS = (100, 500, 1000, 2000)
FINAL_STEP = 2000
FINAL_EXAMPLES = 64000


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _source_checkpoints() -> tuple[dict[int, dict[str, str]], dict[int, dict[str, str]]]:
    tracking_state = json.loads((ROOT / TRACKING_STATE).read_text(encoding="utf-8"))
    event_state = json.loads((ROOT / STATSBOMB_STATE).read_text(encoding="utf-8"))
    tracking = {}
    event = {}
    for seed in SEEDS:
        tracking_record = tracking_state["runs"][f"pff:scratch:{seed}"]
        event_record = event_state["runs"][f"statsbomb:event_only:{seed}"]
        if _sha256(tracking_record["latest_checkpoint"]) != tracking_record[
            "latest_checkpoint_sha256"
        ]:
            raise ValueError(f"Tracking source checkpoint hash mismatch for seed {seed}.")
        if _sha256(event_record["latest_checkpoint"]) != event_record[
            "latest_checkpoint_sha256"
        ]:
            raise ValueError(f"Event source checkpoint hash mismatch for seed {seed}.")
        tracking[seed] = {
            "path": tracking_record["latest_checkpoint"],
            "sha256": tracking_record["latest_checkpoint_sha256"],
        }
        event[seed] = {
            "path": event_record["latest_checkpoint"],
            "sha256": event_record["latest_checkpoint_sha256"],
        }
    return tracking, event


def _expected_hashes() -> dict[str, str]:
    return {
        "protocol_sha256": _sha256(PROTOCOL),
        "config_sha256": _sha256(CONFIG),
        "train_audit_sha256": _sha256(TRAIN_AUDIT),
        "event_manifest_sha256": _sha256(EVENT_MANIFEST),
        "event_tensor_audit_sha256": _sha256(EVENT_TENSOR_AUDIT),
        "tracking_manifest_sha256": _sha256(TRACKING_MANIFEST),
        "tracking_state_sha256": _sha256(TRACKING_STATE),
        "statsbomb_state_sha256": _sha256(STATSBOMB_STATE),
    }


def _save_state(state: dict[str, Any]) -> None:
    path = ROOT / STATE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _load_state() -> dict[str, Any]:
    expected = _expected_hashes()
    tracking, event = _source_checkpoints()
    if (ROOT / STATE_PATH).is_file():
        state = json.loads((ROOT / STATE_PATH).read_text(encoding="utf-8"))
        changed = [name for name, value in expected.items() if state.get(name) != value]
        if changed:
            raise ValueError(
                "Frozen PFF StatsBomb study artifacts changed after execution began: "
                + ", ".join(changed)
            )
        return state
    return {
        "version": 1,
        "study": "pff_statsbomb_context_residual_v1",
        "frozen_at_utc": _utc_now(),
        **expected,
        "families": list(FAMILIES),
        "seeds": list(SEEDS),
        "curve_steps": list(CURVE_STEPS),
        "final_step": FINAL_STEP,
        "final_examples": FINAL_EXAMPLES,
        "tracking_checkpoints": {str(seed): row for seed, row in tracking.items()},
        "event_checkpoints": {str(seed): row for seed, row in event.items()},
        "runs": {},
        "current_run": None,
    }


def _parse_output_path(stdout: str, label: str) -> str:
    match = re.search(rf"^{re.escape(label)}:\s+(.+)$", stdout, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"Training output did not report {label!r}.")
    return match.group(1).strip()


def _run_one(state: dict[str, Any], family: str, seed: int) -> None:
    key = f"context:{family}:{seed}"
    if state["runs"].get(key, {}).get("status") == "complete":
        print(f"[skip] {key}", flush=True)
        return
    tracking = state["tracking_checkpoints"][str(seed)]
    event = state["event_checkpoints"][str(seed)]
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_event_context_residual.py"),
        "--config",
        str(ROOT / CONFIG),
        "--family",
        family,
        "--seed",
        str(seed),
        "--tracking-checkpoint",
        str(ROOT / tracking["path"]),
        "--event-checkpoint",
        str(ROOT / event["path"]),
    ]
    LOG_DIR_PATH = ROOT / LOG_DIR
    LOG_DIR_PATH.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR_PATH / f"{family}_seed{seed}.log"
    state["current_run"] = {
        "key": key,
        "family": family,
        "seed": seed,
        "started_at_utc": _utc_now(),
        "log_path": str(log_path.relative_to(ROOT)),
    }
    _save_state(state)
    print(f"[run] {key}", flush=True)
    process = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_path.write_text(process.stdout, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(f"{key} failed; see {log_path}.")
    run_dir = _artifact_path(_parse_output_path(process.stdout, "run_dir"))
    checkpoint = run_dir / "latest.pt"
    metrics_path = run_dir / "metrics_val.jsonl"
    curve_path = run_dir / "metrics_val_curve.jsonl"
    manifest_path = run_dir / "run_manifest.json"
    final = _read_rows(metrics_path)[-1]
    curve = _read_rows(curve_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(final["step"]) != FINAL_STEP or int(final["num_examples"]) != FINAL_EXAMPLES:
        raise ValueError(f"Unexpected final endpoint for {key}.")
    if [int(row["step"]) for row in curve] != list(CURVE_STEPS):
        raise ValueError(f"Unexpected validation curve for {key}.")
    if manifest["data_access"] != {
        "loaded_tensor_splits": ["train", "val"],
        "test_loaded": False,
        "embedding_export_split": None,
    }:
        raise ValueError(f"Unexpected data access for {key}.")
    if manifest["frozen_sources"]["tracking_checkpoint_sha256"] != tracking["sha256"]:
        raise ValueError(f"Tracking source mismatch for {key}.")
    if manifest["frozen_sources"]["event_checkpoint_sha256"] != event["sha256"]:
        raise ValueError(f"Event source mismatch for {key}.")
    state["runs"][key] = {
        "status": "complete",
        "family": family,
        "seed": seed,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "latest_checkpoint": str(checkpoint.relative_to(ROOT)),
        "latest_checkpoint_sha256": _sha256(checkpoint),
        "metrics_path": str(metrics_path.relative_to(ROOT)),
        "metrics_sha256": _sha256(metrics_path),
        "curve_path": str(curve_path.relative_to(ROOT)),
        "curve_sha256": _sha256(curve_path),
        "run_manifest_path": str(manifest_path.relative_to(ROOT)),
        "run_manifest_sha256": _sha256(manifest_path),
        "started_at_utc": state["current_run"]["started_at_utc"],
        "completed_at_utc": _utc_now(),
    }
    state["current_run"] = None
    _save_state(state)


def run_training(state: dict[str, Any]) -> None:
    for family in FAMILIES:
        for seed in SEEDS:
            _run_one(state, family, seed)


def _relative_improvement(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference


def summarize(state: dict[str, Any]) -> dict[str, Any]:
    expected = {f"context:{family}:{seed}" for family in FAMILIES for seed in SEEDS}
    if set(state["runs"]) != expected:
        raise ValueError("Context residual summary requires all 12 runs.")
    rows = []
    for seed in SEEDS:
        row: dict[str, Any] = {"seed": seed}
        for family in FAMILIES:
            record = state["runs"][f"context:{family}:{seed}"]
            final = _read_rows(_artifact_path(record["metrics_path"]))[-1]
            row[family] = {
                name: float(final[name])
                for name in (
                    "td_loss",
                    "base_td_loss",
                    "event_history_td_loss",
                    "ablated_event_td_loss",
                    "ablated_event_history_td_loss",
                    "num_examples",
                    "event_history_examples",
                    "no_event_history_examples",
                )
            }
            row[family]["metrics_path"] = record["metrics_path"]
            row[family]["metrics_sha256"] = record["metrics_sha256"]
        rows.append(row)
    means = {
        family: {
            name: sum(row[family][name] for row in rows) / len(rows)
            for name in (
                "td_loss",
                "base_td_loss",
                "event_history_td_loss",
                "ablated_event_td_loss",
                "ablated_event_history_td_loss",
            )
        }
        for family in FAMILIES
    }

    def wins(candidate: str, reference: str, metric: str = "td_loss") -> int:
        return sum(row[candidate][metric] < row[reference][metric] for row in rows)

    finite = all(
        math.isfinite(row[family][name])
        for row in rows
        for family in FAMILIES
        for name in (
            "td_loss",
            "base_td_loss",
            "event_history_td_loss",
            "ablated_event_td_loss",
        )
    )
    exact_coverage = all(
        int(row[family]["num_examples"]) == FINAL_EXAMPLES
        for row in rows
        for family in FAMILIES
    )
    matched_event_counts = all(
        len({int(row[family]["event_history_examples"]) for family in FAMILIES}) == 1
        for row in rows
    )
    matched_base = all(
        max(row[family]["base_td_loss"] for family in FAMILIES)
        - min(row[family]["base_td_loss"] for family in FAMILIES)
        <= 1e-10
        for row in rows
    )
    pretrained_ablation_wins = sum(
        row["pretrained"]["td_loss"] < row["pretrained"]["ablated_event_td_loss"]
        for row in rows
    )
    criteria = {
        "finite_metrics": {"passed": finite},
        "exact_final_coverage": {"passed": exact_coverage},
        "matched_event_history_counts": {"passed": matched_event_counts},
        "matched_base_tracking_loss": {"passed": matched_base},
        "pretrained_vs_tracking_wins": {
            "value": wins("pretrained", "tracking"),
            "minimum": 2,
        },
        "pretrained_vs_tracking_improvement": {
            "value": _relative_improvement(
                means["tracking"]["td_loss"], means["pretrained"]["td_loss"]
            ),
            "minimum": 0.01,
        },
        "pretrained_vs_raw_wins": {
            "value": wins("pretrained", "raw"),
            "minimum": 2,
        },
        "pretrained_vs_raw_improvement": {
            "value": _relative_improvement(
                means["raw"]["td_loss"], means["pretrained"]["td_loss"]
            ),
            "minimum": 0.005,
        },
        "pretrained_vs_random_wins": {
            "value": wins("pretrained", "random"),
            "minimum": 2,
        },
        "pretrained_vs_random_improvement": {
            "value": _relative_improvement(
                means["random"]["td_loss"], means["pretrained"]["td_loss"]
            ),
            "minimum": 0.01,
        },
        "pretrained_vs_ablation_wins": {
            "value": pretrained_ablation_wins,
            "minimum": 2,
        },
        "pretrained_vs_ablation_improvement": {
            "value": _relative_improvement(
                means["pretrained"]["ablated_event_td_loss"],
                means["pretrained"]["td_loss"],
            ),
            "minimum": 0.01,
        },
    }
    for criterion in criteria.values():
        if "passed" not in criterion:
            criterion["passed"] = criterion["value"] >= criterion["minimum"]
    blockers = [name for name, criterion in criteria.items() if not criterion["passed"]]
    status = "controls_passed" if not blockers else "blocked"
    raw_wins = wins("raw", "tracking")
    raw_improvement = _relative_improvement(
        means["tracking"]["td_loss"], means["raw"]["td_loss"]
    )
    if status == "controls_passed":
        operational = "pretrained"
    elif raw_wins >= 2 and raw_improvement >= 0.01:
        operational = "raw"
    else:
        operational = "tracking"

    mean_curves = {}
    for family in FAMILIES:
        by_step: dict[int, list[dict[str, Any]]] = {}
        for seed in SEEDS:
            record = state["runs"][f"context:{family}:{seed}"]
            for curve_row in _read_rows(_artifact_path(record["curve_path"])):
                by_step.setdefault(int(curve_row["step"]), []).append(curve_row)
        mean_curves[family] = {
            str(step): {
                name: sum(float(row[name]) for row in curve_rows) / len(curve_rows)
                for name in ("td_loss", "base_td_loss", "event_history_td_loss")
            }
            for step, curve_rows in sorted(by_step.items())
        }
    summary = {
        "version": 1,
        "study": "pff_statsbomb_context_residual_v1",
        "status": status,
        "operational_family": operational,
        "blocking_conditions": blockers,
        "seeds": list(SEEDS),
        "families": list(FAMILIES),
        "rows": rows,
        "means": means,
        "criteria": criteria,
        "raw_fallback": {
            "wins_vs_tracking": raw_wins,
            "mean_improvement_vs_tracking": raw_improvement,
        },
        "mean_validation_curves": mean_curves,
        "protocol_path": str(PROTOCOL),
        "protocol_sha256": _sha256(PROTOCOL),
        "config_path": str(CONFIG),
        "config_sha256": _sha256(CONFIG),
        "data_access": {
            "loaded_splits": ["train", "val"],
            "test_loaded": False,
            "embedding_exported": False,
            "run_count": 12,
        },
        "execution_manifest_path": str(STATE_PATH),
    }
    output = ROOT / SUMMARY_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    state["gate_summary_path"] = str(SUMMARY_PATH)
    state["gate_summary_sha256"] = _sha256(SUMMARY_PATH)
    state["gate_status"] = status
    state["operational_family"] = operational
    _save_state(state)
    print(f"[done] gate status: {status}", flush=True)
    print(f"[done] operational family: {operational}", flush=True)
    return summary


def verify_artifacts(state: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "frozen_inputs_match": all(
            state.get(name) == value for name, value in _expected_hashes().items()
        ),
        "all_runs_complete": len(state.get("runs", {})) == 12
        and all(row.get("status") == "complete" for row in state["runs"].values()),
    }
    rows = []
    for key, record in sorted(state["runs"].items()):
        artifacts = {}
        for path_key, hash_key in (
            ("latest_checkpoint", "latest_checkpoint_sha256"),
            ("metrics_path", "metrics_sha256"),
            ("curve_path", "curve_sha256"),
            ("run_manifest_path", "run_manifest_sha256"),
        ):
            actual = _sha256(record[path_key])
            expected = record[hash_key]
            checks[f"{key}:{path_key}_hash"] = actual == expected
            artifacts[path_key] = {"path": record[path_key], "sha256": actual}
        rows.append({"run": key, "artifacts": artifacts})
    summary = json.loads((ROOT / SUMMARY_PATH).read_text(encoding="utf-8"))
    checks["summary_hash_matches"] = _sha256(SUMMARY_PATH) == state.get(
        "gate_summary_sha256"
    )
    checks["summary_access_boundary"] = summary.get("data_access") == {
        "loaded_splits": ["train", "val"],
        "test_loaded": False,
        "embedding_exported": False,
        "run_count": 12,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    report = {
        "version": 1,
        "study": "pff_statsbomb_context_residual_v1",
        "status": "passed" if not failed else "blocked",
        "checks": checks,
        "failed_checks": failed,
        "runs": rows,
        "data_access": summary["data_access"],
    }
    output = ROOT / AUDIT_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    state["artifact_audit_path"] = str(AUDIT_PATH)
    state["artifact_audit_sha256"] = _sha256(AUDIT_PATH)
    state["artifact_audit_status"] = report["status"]
    _save_state(state)
    if failed:
        raise ValueError("Context residual artifact audit failed: " + ", ".join(failed))
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
