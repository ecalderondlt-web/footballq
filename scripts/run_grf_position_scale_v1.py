"""Run and resume the frozen GRF position-volume scaling experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summarize_grf_position_scale_gate import summarize_position_scale_gate  # noqa: E402

SEEDS = (7, 11, 23)
PROTOCOL = Path("docs/GRF_POSITION_SCALE_PROTOCOL_V1.md")
SYNTHETIC_CONFIGS = {
    "1x": Path("configs/td_jepa_gfootball_position_scale_v1_1x.yaml"),
    "1x_replay": Path("configs/td_jepa_gfootball_position_scale_v1_1x_replay.yaml"),
    "4x": Path("configs/td_jepa_gfootball_position_scale_v1_4x.yaml"),
    "8x": Path("configs/td_jepa_gfootball_position_scale_v1_8x.yaml"),
}
PFF_CONFIG = Path("configs/td_jepa_pff_wc2022_position_scale_v1.yaml")
STATE_PATH = ROOT / "runs" / "grf_position_scale_v1" / "execution_manifest.json"
LOG_DIR = ROOT / "runs" / "grf_position_scale_v1" / "logs"
SUMMARY_PATH = ROOT / "runs" / "grf_position_scale_v1" / "gate_summary.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _load_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {
        "version": 1,
        "protocol_path": str(PROTOCOL),
        "protocol_sha256": _sha256(ROOT / PROTOCOL),
        "seeds": list(SEEDS),
        "config_sha256": {
            str(path): _sha256(ROOT / path)
            for path in [*SYNTHETIC_CONFIGS.values(), PFF_CONFIG]
        },
        "runs": {},
        "created_at_utc": _utc_now(),
    }


def _save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at_utc"] = _utc_now()
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _artifact_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _completed_record(state: dict[str, Any], step: str, *, require_metrics: bool) -> bool:
    record = state["runs"].get(step)
    if not record or record.get("status") != "complete":
        return False
    if not _artifact_path(record["latest_checkpoint"]).exists():
        return False
    return not require_metrics or _artifact_path(record["metrics_path"]).exists()


def _parse_output_path(stdout: str, label: str) -> str:
    match = re.search(rf"^{re.escape(label)}:\s+(.+)$", stdout, flags=re.MULTILINE)
    if not match:
        raise RuntimeError(f"Training output did not contain {label!r}.")
    return match.group(1).strip()


def _run_training(
    state: dict[str, Any],
    *,
    step: str,
    family: str,
    phase: str,
    config: Path,
    seed: int,
    init_checkpoint: str | None = None,
) -> dict[str, Any]:
    require_metrics = phase == "pff"
    if _completed_record(state, step, require_metrics=require_metrics):
        print(f"[skip] {step}", flush=True)
        return state["runs"][step]

    command = [
        sys.executable,
        "scripts/train_td_jepa.py",
        "--config",
        str(config),
        "--seed",
        str(seed),
    ]
    if init_checkpoint is not None:
        command.extend(["--init-checkpoint", init_checkpoint])
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{step.replace(':', '_')}.log"
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
    metrics_path = str(Path(run_dir) / "metrics_val.jsonl") if require_metrics else None
    record = {
        "status": "complete",
        "phase": phase,
        "family": family,
        "seed": seed,
        "config_path": str(config),
        "config_sha256": _sha256(ROOT / config),
        "init_checkpoint": init_checkpoint,
        "init_checkpoint_sha256": (
            _sha256(_artifact_path(init_checkpoint)) if init_checkpoint else None
        ),
        "run_dir": run_dir,
        "latest_checkpoint": latest,
        "latest_checkpoint_sha256": _sha256(_artifact_path(latest)),
        "metrics_path": metrics_path,
        "metrics_sha256": (
            _sha256(_artifact_path(metrics_path)) if metrics_path is not None else None
        ),
        "log_path": str(log_path.relative_to(ROOT)),
        "started_at_utc": started,
        "completed_at_utc": _utc_now(),
    }
    state["runs"][step] = record
    _save_state(state)
    return record


def run_synthetic(state: dict[str, Any]) -> None:
    for family, config in SYNTHETIC_CONFIGS.items():
        for seed in SEEDS:
            _run_training(
                state,
                step=f"synthetic:{family}:{seed}",
                family=family,
                phase="synthetic",
                config=config,
                seed=seed,
            )


def run_pff(state: dict[str, Any]) -> None:
    for seed in SEEDS:
        _run_training(
            state,
            step=f"pff:scratch:{seed}",
            family="scratch",
            phase="pff",
            config=PFF_CONFIG,
            seed=seed,
        )
    for family in SYNTHETIC_CONFIGS:
        for seed in SEEDS:
            synthetic_step = f"synthetic:{family}:{seed}"
            if not _completed_record(state, synthetic_step, require_metrics=False):
                raise RuntimeError(f"Missing completed synthetic run: {synthetic_step}")
            init_checkpoint = state["runs"][synthetic_step]["latest_checkpoint"]
            _run_training(
                state,
                step=f"pff:{family}:{seed}",
                family=family,
                phase="pff",
                config=PFF_CONFIG,
                seed=seed,
                init_checkpoint=init_checkpoint,
            )


def summarize(state: dict[str, Any]) -> dict[str, Any]:
    scratch = {
        seed: _artifact_path(state["runs"][f"pff:scratch:{seed}"]["metrics_path"])
        for seed in SEEDS
    }
    families = {
        family: {
            seed: _artifact_path(state["runs"][f"pff:{family}:{seed}"]["metrics_path"])
            for seed in SEEDS
        }
        for family in SYNTHETIC_CONFIGS
    }
    summary = summarize_position_scale_gate(scratch, families)
    summary["protocol_path"] = str(PROTOCOL)
    summary["protocol_sha256"] = _sha256(ROOT / PROTOCOL)
    summary["execution_manifest_path"] = str(STATE_PATH.relative_to(ROOT))
    summary["execution_manifest_sha256_before_summary"] = _sha256(STATE_PATH)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    state["summary_path"] = str(SUMMARY_PATH.relative_to(ROOT))
    state["summary_sha256"] = _sha256(SUMMARY_PATH)
    state["gate_status"] = summary["status"]
    _save_state(state)
    print(f"[done] gate status: {summary['status']}", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("synthetic", "pff", "summarize", "all"),
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = _load_state()
    _save_state(state)
    if args.stage in {"synthetic", "all"}:
        run_synthetic(state)
    if args.stage in {"pff", "all"}:
        run_pff(state)
    if args.stage in {"summarize", "all"}:
        summarize(state)


if __name__ == "__main__":
    main()
