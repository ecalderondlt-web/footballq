"""Evaluate the frozen GRF motion feature-view selection rule."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_provider_neutral_preflight import (  # noqa: E402
    compare_train_tensor_subset_invariants,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_selection(
    lagged_invariants: dict[str, Any],
    position_invariants: dict[str, Any],
    lagged_motion_gate: dict[str, Any],
    *,
    position_projection_matches: bool,
    minimum_retention_fraction: float = 0.75,
) -> dict[str, Any]:
    """Apply the frozen lower-frequency-first, position-only-second rule."""

    lagged_retention = float(lagged_invariants.get("retention_fraction", 0.0))
    position_retention = float(position_invariants.get("retention_fraction", 0.0))
    criteria = {
        "lagged_integrity": bool(lagged_invariants.get("passed")),
        "lagged_retention": lagged_retention >= minimum_retention_fraction,
        "lagged_motion_gate": lagged_motion_gate.get("status") == "preflight_passed",
        "position_integrity": bool(position_invariants.get("passed")),
        "position_retention": position_retention >= minimum_retention_fraction,
        "position_projection_matches_lagged_source": bool(position_projection_matches),
    }
    if all(
        criteria[name]
        for name in ("lagged_integrity", "lagged_retention", "lagged_motion_gate")
    ):
        status = "lagged_motion_selected_for_future_model_protocol"
        selected = "jump_segmented_causal_position_difference_0p5s"
    elif all(
        criteria[name]
        for name in (
            "position_integrity",
            "position_retention",
            "position_projection_matches_lagged_source",
        )
    ):
        status = "position_only_selected_for_future_model_protocol"
        selected = "position_only"
    else:
        status = "blocked"
        selected = None
    return {
        "status": status,
        "selected_feature_view": selected,
        "model_training_authorized": False,
        "minimum_retention_fraction": minimum_retention_fraction,
        "lagged_retention_fraction": lagged_retention,
        "position_retention_fraction": position_retention,
        "criteria": criteria,
        "lagged_motion_blockers": lagged_motion_gate.get("blocking_conditions", []),
        "lagged_invariants": lagged_invariants,
        "position_invariants": position_invariants,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--lagged-manifest", type=Path, required=True)
    parser.add_argument("--position-manifest", type=Path, required=True)
    parser.add_argument("--lagged-motion-gate", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    lagged_invariants = compare_train_tensor_subset_invariants(
        args.baseline_manifest, args.lagged_manifest
    )
    position_invariants = compare_train_tensor_subset_invariants(
        args.baseline_manifest, args.position_manifest
    )
    lagged_manifest = json.loads(args.lagged_manifest.read_text(encoding="utf-8"))
    position_manifest = json.loads(args.position_manifest.read_text(encoding="utf-8"))
    lagged_motion_gate = json.loads(args.lagged_motion_gate.read_text(encoding="utf-8"))
    result = evaluate_selection(
        lagged_invariants,
        position_invariants,
        lagged_motion_gate,
        position_projection_matches=(
            position_manifest.get("source_feature_manifest_payload_sha256")
            == lagged_manifest.get("manifest_payload_sha256")
        ),
    )
    result.update(
        {
            "protocol_path": str(args.protocol),
            "protocol_sha256": _sha256(args.protocol),
            "baseline_manifest_path": str(args.baseline_manifest),
            "baseline_manifest_sha256": _sha256(args.baseline_manifest),
            "lagged_manifest_path": str(args.lagged_manifest),
            "lagged_manifest_sha256": _sha256(args.lagged_manifest),
            "position_manifest_path": str(args.position_manifest),
            "position_manifest_sha256": _sha256(args.position_manifest),
            "lagged_motion_gate_path": str(args.lagged_motion_gate),
            "lagged_motion_gate_sha256": _sha256(args.lagged_motion_gate),
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"status: {result['status']}")
    print(f"selected_feature_view: {result['selected_feature_view']}")
    print(f"model_training_authorized: {result['model_training_authorized']}")
    print(f"out: {args.out}")


if __name__ == "__main__":
    main()
