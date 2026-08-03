from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

from footballq.models.player_matchup_value import VALUE_CONDITIONS
from footballq.repro.manifest import file_sha256
from footballq.training.eval_rlcs_value import evaluate_value_bundle


def _atomic_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _latest_checkpoint(run_root: Path, condition: str, seed: int) -> dict[str, str]:
    manifests = sorted((run_root / condition / f"seed_{seed}").glob("*/run_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No completed V2 run for {condition}/seed {seed}.")
    manifest_path = manifests[-1]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = Path(manifest["checkpoint"])
    return {
        "path": str(checkpoint),
        "sha256": file_sha256(checkpoint),
        "run_manifest": str(manifest_path),
        "run_manifest_sha256": file_sha256(manifest_path),
    }


def _latest_completed_run(
    run_root: Path, condition: str, seed: int
) -> tuple[Path, dict] | None:
    manifests = sorted((run_root / condition / f"seed_{seed}").glob("*/run_manifest.json"))
    if not manifests:
        return None
    manifest_path = manifests[-1]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(manifest["checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Completed V2 run is missing {checkpoint_path}.")
    if file_sha256(checkpoint_path) != manifest["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint hash mismatch for {condition}/seed {seed}.")
    return manifest_path, manifest


def _write_irreversible_seed_gate_stop(
    *,
    config_path: Path,
    config: dict,
    profile_audit_path: Path,
    run_root: Path,
) -> Path | None:
    """Stop when the required passing-seed count is no longer attainable."""

    seeds = [int(value) for value in config["training"]["seeds"]]
    gate_cfg = config["evaluation"]["gates"]
    threshold = float(gate_cfg["full_vs_team_form_relative_log_loss_reduction"])
    required = int(gate_cfg["required_passing_seeds"])
    data_cfg = config["data"]
    dataset_manifest_path = Path(data_cfg["dataset_manifest"])
    expected_lineage = {
        "config_sha256": file_sha256(config_path),
        "dataset_manifest_sha256": file_sha256(dataset_manifest_path),
        "profile_audit_sha256": file_sha256(profile_audit_path),
    }
    completed: dict[str, dict] = {}
    condition_evidence: dict[str, dict[str, dict]] = {}
    for seed in seeds:
        per_condition: dict[str, dict] = {}
        for condition in VALUE_CONDITIONS:
            resolved = _latest_completed_run(run_root, condition, seed)
            if resolved is None:
                continue
            manifest_path, manifest = resolved
            for field, expected in expected_lineage.items():
                if manifest.get(field) != expected:
                    raise ValueError(
                        f"Run lineage mismatch for {condition}/seed {seed}: {field}."
                    )
            per_condition[condition] = {
                "run_manifest": str(manifest_path),
                "run_manifest_sha256": file_sha256(manifest_path),
                "checkpoint": manifest["checkpoint"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "best_step": int(manifest["best_step"]),
                "internal_development_log_loss": float(
                    manifest["best_internal_development_log_loss"]
                ),
            }
        condition_evidence[str(seed)] = per_condition
        if "team_form" not in per_condition or "full_matchup" not in per_condition:
            continue
        baseline = per_condition["team_form"]["internal_development_log_loss"]
        full = per_condition["full_matchup"]["internal_development_log_loss"]
        relative = (baseline - full) / baseline
        completed[str(seed)] = {
            "team_form_log_loss": baseline,
            "full_matchup_log_loss": full,
            "relative_log_loss_reduction": relative,
            "required_relative_log_loss_reduction": threshold,
            "passes": bool(relative >= threshold),
        }
    passing = sum(bool(item["passes"]) for item in completed.values())
    remaining = len(seeds) - len(completed)
    maximum_possible = passing + remaining
    if maximum_possible >= required:
        return None

    incomplete = []
    for checkpoint in sorted(run_root.glob("*/seed_*/*/best.pt")):
        if (checkpoint.parent / "run_manifest.json").exists():
            continue
        incomplete.append(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "status": "excluded_incomplete_run_without_manifest",
            }
        )
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    summary = {
        "version": 2,
        "experiment": "rlcs_player_matchup_value_v2",
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "status": "stopped_after_internal_development",
        "reason": "two_of_three_seed_gate_mathematically_impossible",
        "decision": "stop_before_remaining_training_and_controls",
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "profile_audit_path": str(profile_audit_path),
        "profile_audit_sha256": file_sha256(profile_audit_path),
        "dataset_manifest_path": str(dataset_manifest_path),
        "dataset_manifest_sha256": file_sha256(dataset_manifest_path),
        "completed_run_lineage_verified": True,
        "opened_stages": dataset_manifest.get("opened_stages", []),
        "seed_gate": {
            "required_passing_seeds": required,
            "configured_seeds": seeds,
            "completed_matched_seeds": sorted(int(value) for value in completed),
            "passing_completed_seeds": passing,
            "remaining_seeds": remaining,
            "maximum_possible_passing_seeds": maximum_possible,
            "results": completed,
        },
        "completed_condition_evidence": condition_evidence,
        "excluded_incomplete_artifacts": incomplete,
        "controls_evaluated": False,
        "controls_not_evaluated_reason": "primary seed gate already impossible",
        "architecture_frozen": False,
        "validation_loaded": False,
        "test_loaded": False,
    }
    return _atomic_json(run_root / "v2_stop_summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze matched V2 checkpoints and apply internal/validation gates."
    )
    parser.add_argument("--config", default="configs/rlcs_player_matchup_value_v2.yaml")
    parser.add_argument("--evaluate-validation", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    run_root = Path(config["training"]["run_root"])
    run_root.mkdir(parents=True, exist_ok=True)
    profile_audit_path = Path(data_cfg["profile_audit"])
    audit = json.loads(profile_audit_path.read_text(encoding="utf-8"))
    if not bool(audit.get("all_gates_pass")):
        stop = {
            "version": 2,
            "experiment": "rlcs_player_matchup_value_v2",
            "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "status": "stopped_before_outcome_training",
            "reason": "profile_stability_gate_failed",
            "profile_audit": str(profile_audit_path),
            "profile_audit_sha256": file_sha256(profile_audit_path),
            "failed_gates": [name for name, passed in audit["gates"].items() if not passed],
            "test_loaded": False,
            "validation_loaded": False,
        }
        path = _atomic_json(run_root / "v2_stop_summary.json", stop)
        print(f"V2 STOP: profile gate failed; summary: {path}")
        return

    irreversible_stop = _write_irreversible_seed_gate_stop(
        config_path=config_path,
        config=config,
        profile_audit_path=profile_audit_path,
        run_root=run_root,
    )
    if irreversible_stop is not None:
        print(f"V2 STOP: required passing-seed count is impossible; summary: {irreversible_stop}")
        return

    bundle_path = run_root / "checkpoint_bundle.json"
    if args.evaluate_validation:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        if not bool(bundle.get("architecture_frozen")):
            raise RuntimeError("Internal development did not freeze the V2 architecture.")
        paths = evaluate_value_bundle(
            config_path,
            bundle_path=bundle_path,
            stage="validation",
            output_dir=run_root / "validation",
        )
        print(f"validation results: {paths['results']}")
        return

    checkpoints = {
        condition: {
            str(seed): _latest_checkpoint(run_root, condition, int(seed))
            for seed in config["training"]["seeds"]
        }
        for condition in VALUE_CONDITIONS
    }
    bundle = {
        "version": 2,
        "experiment": "rlcs_player_matchup_value_v2",
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "architecture_frozen": False,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "dataset_manifest_path": str(data_cfg["dataset_manifest"]),
        "dataset_manifest_sha256": file_sha256(data_cfg["dataset_manifest"]),
        "profile_audit_sha256": file_sha256(profile_audit_path),
        "checkpoints": checkpoints,
        "test_loaded": False,
        "validation_loaded": False,
    }
    _atomic_json(bundle_path, bundle)
    paths = evaluate_value_bundle(
        config_path,
        bundle_path=bundle_path,
        stage="internal_development",
        output_dir=run_root / "internal_development",
    )
    results = json.loads(paths["results"].read_text(encoding="utf-8"))
    bundle["internal_development_results"] = str(paths["results"])
    bundle["internal_development_results_sha256"] = file_sha256(paths["results"])
    bundle["architecture_frozen"] = bool(results["all_gates_pass"])
    bundle["status"] = (
        "frozen_for_split2_validation"
        if bundle["architecture_frozen"]
        else "stopped_after_internal_development"
    )
    _atomic_json(bundle_path, bundle)
    print(f"internal results: {paths['results']}")
    print(f"bundle: {bundle_path}")
    print(f"status: {bundle['status']}")


if __name__ == "__main__":
    main()
