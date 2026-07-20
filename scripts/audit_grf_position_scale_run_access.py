"""Audit split access and fixed evaluation points for the GRF scale study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_CURVE_STEPS = [100, 250, 500, 1000, 2000]
EXPECTED_PFF_RUNS = 15
EXPECTED_SYNTHETIC_RUNS = 12


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_run_access(execution_manifest_path: str | Path) -> dict[str, Any]:
    path = Path(execution_manifest_path)
    state = json.loads(path.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    rows = []
    pff_records = {
        name: record for name, record in state["runs"].items() if name.startswith("pff:")
    }
    synthetic_records = {
        name: record
        for name, record in state["runs"].items()
        if name.startswith("synthetic:")
    }
    checks["expected_pff_run_count"] = len(pff_records) == EXPECTED_PFF_RUNS
    checks["expected_synthetic_run_count"] = (
        len(synthetic_records) == EXPECTED_SYNTHETIC_RUNS
    )

    for name, record in sorted(state["runs"].items()):
        run_dir = Path(record["run_dir"])
        run_manifest_path = run_dir / "run_manifest.json"
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        data_access = run_manifest["data_access"]
        is_pff = name.startswith("pff:")
        expected_splits = {"train", "val"} if is_pff else {"train"}
        row_checks = {
            "loaded_only_expected_splits": (
                set(data_access["loaded_tensor_splits"]) == expected_splits
            ),
            "test_not_loaded": "test" not in data_access["loaded_tensor_splits"],
            "embedding_export_disabled": data_access["embedding_sample_split"] is None,
            "checkpoint_exists": Path(record["latest_checkpoint"]).exists(),
        }
        if is_pff:
            curve_path = run_dir / "metrics_val_curve.jsonl"
            final_path = run_dir / "metrics_val.jsonl"
            curve = [
                json.loads(line)
                for line in curve_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            final = json.loads(final_path.read_text(encoding="utf-8").splitlines()[-1])
            row_checks.update(
                {
                    "validation_only_protocol": (
                        run_manifest["evaluation_protocol"]
                        == "inductive_match_holdout_validation_only"
                    ),
                    "curve_steps_match": [item["step"] for item in curve]
                    == EXPECTED_CURVE_STEPS,
                    "final_step_is_2000": int(final["step"]) == 2000,
                    "final_split_is_validation": final["split"] == "val",
                    "final_validation_examples_are_64000": (
                        int(final["num_examples"]) == 64000
                    ),
                }
            )
        checks.update({f"{name}:{key}": value for key, value in row_checks.items()})
        rows.append(
            {
                "run": name,
                "run_manifest_path": str(run_manifest_path),
                "run_manifest_sha256": _sha256(run_manifest_path),
                "loaded_tensor_splits": data_access["loaded_tensor_splits"],
                "checks": row_checks,
            }
        )

    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "status": "passed" if not failed else "blocked",
        "execution_manifest_path": str(path),
        "execution_manifest_sha256": _sha256(path),
        "checks": checks,
        "failed_checks": failed,
        "runs": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = audit_run_access(args.execution_manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"status: {audit['status']}")
    print(f"audit: {args.out}")
    for failure in audit["failed_checks"]:
        print(f"failed: {failure}")


if __name__ == "__main__":
    main()
