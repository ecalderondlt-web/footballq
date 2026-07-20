"""Evaluate the frozen provider-neutral GRF motion preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_root(path: Path) -> Path:
    return path.parent.parent


def compare_train_tensor_invariants(
    baseline_manifest_path: str | Path,
    candidate_manifest_path: str | Path,
) -> dict[str, Any]:
    """Verify that train tensors differ only in velocity feature channels."""

    baseline_path = Path(baseline_manifest_path)
    candidate_path = Path(candidate_manifest_path)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    baseline_shards = {
        str(shard["job_id"]): shard for shard in baseline["shards"] if shard["split"] == "train"
    }
    candidate_shards = {
        str(shard["job_id"]): shard for shard in candidate["shards"] if shard["split"] == "train"
    }
    checks = {
        "train_job_ids_match": set(baseline_shards) == set(candidate_shards),
        "collection_plan_sha256_match": (
            baseline["collection_plan_sha256"] == candidate["collection_plan_sha256"]
        ),
        "split_manifest_sha256_match": (
            baseline["split_manifest_sha256"] == candidate["split_manifest_sha256"]
        ),
        "visibility_profile_sha256_match": (
            baseline["visibility_profile_sha256"] == candidate["visibility_profile_sha256"]
        ),
        "train_example_count_match": (
            baseline["split_example_counts"]["train"]
            == candidate["split_example_counts"]["train"]
        ),
    }
    compared_examples = 0
    changed_velocity_values = 0
    for job_id in sorted(set(baseline_shards) & set(candidate_shards)):
        baseline_shard = baseline_shards[job_id]
        candidate_shard = candidate_shards[job_id]
        baseline_payload = torch.load(
            _manifest_root(baseline_path) / baseline_shard["path"],
            map_location="cpu",
            weights_only=False,
        )
        candidate_payload = torch.load(
            _manifest_root(candidate_path) / candidate_shard["path"],
            map_location="cpu",
            weights_only=False,
        )
        checks[f"{job_id}:sample_ids"] = (
            baseline_payload["sample_id"] == candidate_payload["sample_id"]
        )
        checks[f"{job_id}:match_ids"] = (
            baseline_payload["match_id"] == candidate_payload["match_id"]
        )
        checks[f"{job_id}:periods"] = baseline_payload["period"] == candidate_payload["period"]
        checks[f"{job_id}:frame_t"] = baseline_payload["frame_t"] == candidate_payload["frame_t"]
        for key in (
            "mask_t",
            "mask_t_plus_delta",
            "delta_mask",
            "context_frame_indices",
            "target_frame_indices",
            "delta_frame_indices",
        ):
            checks[f"{job_id}:{key}"] = torch.equal(
                baseline_payload[key], candidate_payload[key]
            )
        for key in ("state_t", "state_t_plus_delta"):
            checks[f"{job_id}:{key}:xy"] = torch.equal(
                baseline_payload[key][..., :2], candidate_payload[key][..., :2]
            )
            velocity_difference = (
                baseline_payload[key][..., 2:4] != candidate_payload[key][..., 2:4]
            )
            changed_velocity_values += int(velocity_difference.sum())
        checks[f"{job_id}:delta_state:xy"] = torch.equal(
            baseline_payload["delta_state"][..., :2], candidate_payload["delta_state"][..., :2]
        )
        compared_examples += len(baseline_payload["sample_id"])

    checks["velocity_channels_changed"] = changed_velocity_values > 0
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "passed": not failed,
        "compared_split": "train",
        "compared_examples": compared_examples,
        "changed_velocity_values": changed_velocity_values,
        "checks": checks,
        "failed_checks": failed,
    }


def compare_train_tensor_subset_invariants(
    baseline_manifest_path: str | Path,
    candidate_manifest_path: str | Path,
) -> dict[str, Any]:
    """Verify that event-segmented train tensors are a clean baseline subset."""

    baseline_path = Path(baseline_manifest_path)
    candidate_path = Path(candidate_manifest_path)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    baseline_shards = {
        str(shard["job_id"]): shard for shard in baseline["shards"] if shard["split"] == "train"
    }
    candidate_shards = {
        str(shard["job_id"]): shard for shard in candidate["shards"] if shard["split"] == "train"
    }
    velocity_mode = candidate.get("velocity_mode")
    feature_view = candidate.get("config", {}).get("feature_view")
    checks = {
        "train_job_ids_match": set(baseline_shards) == set(candidate_shards),
        "collection_plan_sha256_match": (
            baseline["collection_plan_sha256"] == candidate["collection_plan_sha256"]
        ),
        "split_manifest_sha256_match": (
            baseline["split_manifest_sha256"] == candidate["split_manifest_sha256"]
        ),
        "visibility_profile_sha256_match": (
            baseline["visibility_profile_sha256"] == candidate["visibility_profile_sha256"]
        ),
        "candidate_declares_train_only": candidate.get("included_splits") == ["train"],
        "candidate_has_no_held_out_shards": all(
            shard.get("split") == "train" for shard in candidate["shards"]
        ),
        "supported_subset_velocity_mode": velocity_mode
        in {
            "event_segmented_causal_position_difference",
            "jump_segmented_causal_position_difference_0p5s",
        },
        "supported_subset_feature_view": feature_view in {"geometry_only", "position_only"},
        "unsafe_tensor_references_are_zero": all(
            int(shard.get("unsafe_tensor_frame_reference_count", -1)) == 0
            for shard in candidate["shards"]
        ),
    }
    if velocity_mode == "event_segmented_causal_position_difference":
        checks.update(
            {
                "event_boundary_window_is_five": (
                    candidate.get("event_boundary_window_frames") == 5
                ),
                "event_boundary_totals_present": bool(
                    candidate.get("event_boundary_totals")
                ),
            }
        )
    if velocity_mode == "jump_segmented_causal_position_difference_0p5s":
        checks.update(
            {
                "jump_boundary_totals_present": bool(
                    candidate.get("jump_boundary_totals")
                ),
                "causal_velocity_lag_is_five": (
                    candidate.get("causal_velocity_lag_frames") == 5
                ),
                "jump_thresholds_match": (
                    candidate.get("player_jump_threshold_m") == 3.0
                    and candidate.get("ball_jump_threshold_m") == 10.0
                ),
                "boundary_crossings_are_zero": all(
                    int(shard.get("boundary_crossing_tensor_example_count", -1)) == 0
                    for shard in candidate["shards"]
                ),
            }
        )
    baseline_examples = 0
    candidate_examples = 0
    changed_velocity_values = 0
    for job_id in sorted(set(baseline_shards) & set(candidate_shards)):
        baseline_payload = torch.load(
            _manifest_root(baseline_path) / baseline_shards[job_id]["path"],
            map_location="cpu",
            weights_only=False,
        )
        candidate_payload = torch.load(
            _manifest_root(candidate_path) / candidate_shards[job_id]["path"],
            map_location="cpu",
            weights_only=False,
        )
        baseline_feature_names = list(map(str, baseline_payload["feature_names"]))
        candidate_feature_names = list(map(str, candidate_payload["feature_names"]))
        expected_candidate_names = (
            ["x_norm", "y_norm", "is_ball", "is_home", "is_away"]
            if feature_view == "position_only"
            else [
                "x_norm",
                "y_norm",
                "vx_norm",
                "vy_norm",
                "is_ball",
                "is_home",
                "is_away",
            ]
        )
        checks[f"{job_id}:feature_names"] = (
            candidate_feature_names == expected_candidate_names
        )
        baseline_ids = list(map(str, baseline_payload["sample_id"]))
        candidate_ids = list(map(str, candidate_payload["sample_id"]))
        baseline_examples += len(baseline_ids)
        candidate_examples += len(candidate_ids)
        baseline_index = {sample_id: index for index, sample_id in enumerate(baseline_ids)}
        checks[f"{job_id}:candidate_ids_unique"] = len(candidate_ids) == len(set(candidate_ids))
        checks[f"{job_id}:sample_id_subset"] = set(candidate_ids) <= set(baseline_ids)
        if not checks[f"{job_id}:sample_id_subset"]:
            continue
        selected = torch.tensor([baseline_index[sample_id] for sample_id in candidate_ids])
        checks[f"{job_id}:match_ids"] = [
            baseline_payload["match_id"][int(index)] for index in selected
        ] == candidate_payload["match_id"]
        checks[f"{job_id}:periods"] = [
            baseline_payload["period"][int(index)] for index in selected
        ] == candidate_payload["period"]
        checks[f"{job_id}:frame_t"] = [
            baseline_payload["frame_t"][int(index)] for index in selected
        ] == candidate_payload["frame_t"]
        for key in (
            "mask_t",
            "mask_t_plus_delta",
            "delta_mask",
            "context_frame_indices",
            "target_frame_indices",
            "delta_frame_indices",
        ):
            checks[f"{job_id}:{key}"] = torch.equal(
                baseline_payload[key][selected], candidate_payload[key]
            )
        for key in ("state_t", "state_t_plus_delta"):
            baseline_xy = [baseline_feature_names.index(name) for name in ("x_norm", "y_norm")]
            candidate_xy = [
                candidate_feature_names.index(name) for name in ("x_norm", "y_norm")
            ]
            checks[f"{job_id}:{key}:xy"] = torch.equal(
                baseline_payload[key][selected][..., baseline_xy],
                candidate_payload[key][..., candidate_xy],
            )
            if {"vx_norm", "vy_norm"} <= set(candidate_feature_names):
                baseline_velocity = [
                    baseline_feature_names.index(name) for name in ("vx_norm", "vy_norm")
                ]
                candidate_velocity = [
                    candidate_feature_names.index(name) for name in ("vx_norm", "vy_norm")
                ]
                changed_velocity_values += int(
                    (
                        baseline_payload[key][selected][..., baseline_velocity]
                        != candidate_payload[key][..., candidate_velocity]
                    ).sum()
                )
        baseline_xy = [baseline_feature_names.index(name) for name in ("x_norm", "y_norm")]
        candidate_xy = [
            candidate_feature_names.index(name) for name in ("x_norm", "y_norm")
        ]
        checks[f"{job_id}:delta_state:xy"] = torch.equal(
            baseline_payload["delta_state"][selected][..., baseline_xy],
            candidate_payload["delta_state"][..., candidate_xy],
        )

    retention_fraction = (
        candidate_examples / baseline_examples if baseline_examples else 0.0
    )
    checks["candidate_is_strict_subset"] = 0 < candidate_examples < baseline_examples
    checks["manifest_train_count_matches_loaded"] = (
        int(candidate["split_example_counts"]["train"]) == candidate_examples
    )
    checks["velocity_channels_changed_or_absent"] = (
        changed_velocity_values > 0 or feature_view == "position_only"
    )
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "passed": not failed,
        "comparison_mode": "candidate_train_subset",
        "compared_split": "train",
        "baseline_examples": baseline_examples,
        "candidate_examples": candidate_examples,
        "retention_fraction": retention_fraction,
        "changed_velocity_values": changed_velocity_values,
        "checks": checks,
        "failed_checks": failed,
    }


def _metric_row(report: dict[str, Any], metric: str) -> dict[str, Any]:
    for row in report["global_gap_ranking"]:
        if row["metric"] == metric:
            return row
    raise ValueError(f"Domain-gap report is missing metric {metric!r}.")


def _gap_score(report: dict[str, Any], metric: str) -> float:
    return float(_metric_row(report, metric)["gap_score"])


def _audit_is_train_only(report: dict[str, Any]) -> bool:
    if report.get("scope") != "train_only":
        return False
    paths = [
        row["path"]
        for family in ("real", "synthetic")
        for row in report["sampling"][family]["shards"]
    ]
    return all("\\train\\" in str(path) or "/train/" in str(path) for path in paths)


def evaluate_preflight_gate(
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
    invariants: dict[str, Any],
    *,
    minimum_retention_fraction: float | None = None,
) -> dict[str, Any]:
    baseline_acceleration = _gap_score(baseline_report, "player_acceleration_mps2")
    candidate_acceleration = _gap_score(candidate_report, "player_acceleration_mps2")
    baseline_turn = _gap_score(baseline_report, "player_turn_deg")
    candidate_turn = _gap_score(candidate_report, "player_turn_deg")
    baseline_speed = _gap_score(baseline_report, "player_speed_mps")
    candidate_speed = _gap_score(candidate_report, "player_speed_mps")
    baseline_acceleration_row = _metric_row(
        baseline_report, "player_acceleration_mps2"
    )
    candidate_acceleration_row = _metric_row(
        candidate_report, "player_acceleration_mps2"
    )
    baseline_acceleration_mean = float(baseline_acceleration_row["synthetic"]["mean"])
    candidate_acceleration_mean = float(
        candidate_acceleration_row["synthetic"]["mean"]
    )
    baseline_acceleration_p99 = float(baseline_acceleration_row["synthetic"]["p99"])
    candidate_acceleration_p99 = float(candidate_acceleration_row["synthetic"]["p99"])
    acceleration_reduction = (
        baseline_acceleration - candidate_acceleration
    ) / baseline_acceleration
    criteria = {
        "train_tensor_invariants": {"passed": bool(invariants["passed"])},
        "player_acceleration_gap_below_one": {
            "value": candidate_acceleration,
            "maximum_exclusive": 1.0,
            "passed": candidate_acceleration < 1.0,
        },
        "player_acceleration_gap_reduction": {
            "value": acceleration_reduction,
            "minimum": 0.25,
            "passed": acceleration_reduction >= 0.25,
        },
        "player_acceleration_mean_non_degradation": {
            "value": candidate_acceleration_mean,
            "maximum": baseline_acceleration_mean,
            "passed": candidate_acceleration_mean <= baseline_acceleration_mean,
        },
        "player_acceleration_p99_non_degradation": {
            "value": candidate_acceleration_p99,
            "maximum": baseline_acceleration_p99,
            "passed": candidate_acceleration_p99 <= baseline_acceleration_p99,
        },
        "player_turn_non_degradation": {
            "value": candidate_turn,
            "maximum": baseline_turn * 1.10,
            "passed": candidate_turn <= baseline_turn * 1.10,
        },
        "player_speed_non_degradation": {
            "value": candidate_speed,
            "maximum": baseline_speed * 1.10,
            "passed": candidate_speed <= baseline_speed * 1.10,
        },
        "train_only_audit_paths": {
            "passed": _audit_is_train_only(baseline_report)
            and _audit_is_train_only(candidate_report)
        },
    }
    if minimum_retention_fraction is not None:
        retention_fraction = float(invariants.get("retention_fraction", 0.0))
        criteria["train_example_retention"] = {
            "value": retention_fraction,
            "minimum": float(minimum_retention_fraction),
            "passed": retention_fraction >= minimum_retention_fraction,
        }
    blockers = [name for name, criterion in criteria.items() if not criterion["passed"]]
    return {
        "status": "preflight_passed" if not blockers else "blocked",
        "criteria": criteria,
        "blocking_conditions": blockers,
        "baseline_gap_scores": {
            "player_acceleration_mps2": baseline_acceleration,
            "player_turn_deg": baseline_turn,
            "player_speed_mps": baseline_speed,
        },
        "candidate_gap_scores": {
            "player_acceleration_mps2": candidate_acceleration,
            "player_turn_deg": candidate_turn,
            "player_speed_mps": candidate_speed,
        },
        "acceleration_physical_summary": {
            "baseline_synthetic_mean_mps2": baseline_acceleration_mean,
            "candidate_synthetic_mean_mps2": candidate_acceleration_mean,
            "baseline_synthetic_p99_mps2": baseline_acceleration_p99,
            "candidate_synthetic_p99_mps2": candidate_acceleration_p99,
        },
        "invariants": invariants,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-candidate-subset", action="store_true")
    parser.add_argument("--minimum-retention-fraction", type=float, default=None)
    args = parser.parse_args()

    comparison = (
        compare_train_tensor_subset_invariants
        if args.allow_candidate_subset
        else compare_train_tensor_invariants
    )
    invariants = comparison(args.baseline_manifest, args.candidate_manifest)
    baseline_report = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    candidate_report = json.loads(args.candidate_report.read_text(encoding="utf-8"))
    result = evaluate_preflight_gate(
        baseline_report,
        candidate_report,
        invariants,
        minimum_retention_fraction=args.minimum_retention_fraction,
    )
    result.update(
        {
            "protocol_path": str(args.protocol),
            "protocol_sha256": _sha256(args.protocol),
            "baseline_manifest_path": str(args.baseline_manifest),
            "baseline_manifest_sha256": _sha256(args.baseline_manifest),
            "candidate_manifest_path": str(args.candidate_manifest),
            "candidate_manifest_sha256": _sha256(args.candidate_manifest),
            "baseline_report_path": str(args.baseline_report),
            "baseline_report_sha256": _sha256(args.baseline_report),
            "candidate_report_path": str(args.candidate_report),
            "candidate_report_sha256": _sha256(args.candidate_report),
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"status: {result['status']}")
    print(f"out: {args.out}")
    for blocker in result["blocking_conditions"]:
        print(f"blocker: {blocker}")


if __name__ == "__main__":
    main()
