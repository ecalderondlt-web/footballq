from __future__ import annotations

import argparse
import json
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import yaml

from footballq.models.identity_matchup_transformer import IDENTITY_CONDITIONS
from footballq.repro.manifest import file_sha256

SEEDS = (17, 23, 41)


def _latest_complete_run(root: Path, condition: str, seed: int) -> tuple[Path, dict[str, Any]]:
    seed_root = root / condition / f"seed_{seed}"
    candidates = sorted(seed_root.glob("*/run_manifest.json"), reverse=True)
    for manifest_path in candidates:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint = manifest_path.parent / "best.pt"
        if checkpoint.exists() and not manifest.get("test_loaded", True):
            return checkpoint, manifest
    raise FileNotFoundError(f"No complete validation-only run for {condition}/seed_{seed}.")


def summarize_validation(config_path: Path, run_root: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    runs: dict[str, dict[str, Any]] = {}
    dataset_hashes: set[str] = set()
    for condition in IDENTITY_CONDITIONS:
        runs[condition] = {}
        for seed in SEEDS:
            checkpoint, manifest = _latest_complete_run(run_root, condition, seed)
            checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            dataset_hashes.add(str(checkpoint_payload["dataset_manifest_sha256"]))
            runs[condition][str(seed)] = {
                "path": str(checkpoint),
                "sha256": file_sha256(checkpoint),
                "validation_factorized_joint_nll": float(
                    checkpoint_payload["validation"]["factorized_joint_nll"]
                ),
                "best_step": int(checkpoint_payload["step"]),
                "run_manifest": str(checkpoint.parent / "run_manifest.json"),
                "run_manifest_sha256": file_sha256(checkpoint.parent / "run_manifest.json"),
            }
    if len(dataset_hashes) != 1:
        raise ValueError("The 12 runs do not share one frozen dataset manifest hash.")
    current_dataset_hash = file_sha256(cfg["data"]["manifest"])
    if dataset_hashes != {current_dataset_hash}:
        raise ValueError("The current dataset manifest differs from the 12 frozen runs.")
    validation_lifts: dict[str, float] = {}
    for seed in SEEDS:
        anonymous = runs["anonymous"][str(seed)]["validation_factorized_joint_nll"]
        full = runs["full"][str(seed)]["validation_factorized_joint_nll"]
        validation_lifts[str(seed)] = (anonymous - full) / anonymous
    passing_seeds = sum(value >= 0.02 for value in validation_lifts.values())
    return {
        "version": 1,
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "dataset_manifest_path": cfg["data"]["manifest"],
        "dataset_manifest_sha256": current_dataset_hash,
        "split_manifest_path": cfg["data"]["split_manifest"],
        "split_manifest_sha256": file_sha256(cfg["data"]["split_manifest"]),
        "validation_full_vs_anonymous_relative_lift_by_seed": validation_lifts,
        "validation_gate": {
            "threshold": 0.02,
            "required_passing_seeds": 2,
            "passing_seeds": passing_seeds,
            "passed": passing_seeds >= 2,
        },
        "runs": runs,
    }


def write_unlock(summary: dict[str, Any], path: Path) -> Path:
    if not summary["validation_gate"]["passed"]:
        raise ValueError("Validation gate failed; sealed test must remain locked.")
    checkpoints = {
        condition: {
            seed: {"path": row["path"], "sha256": row["sha256"]}
            for seed, row in seeds.items()
        }
        for condition, seeds in summary["runs"].items()
    }
    payload = {
        "version": 1,
        "protocol": "rlcs_identity_matchup_v1_sealed_test",
        "status": "unlocked_after_validation_gate",
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "nonce": secrets.token_hex(16),
        "dataset_manifest_sha256": summary["dataset_manifest_sha256"],
        "split_manifest_sha256": summary["split_manifest_sha256"],
        "validation_summary_sha256": None,
        "validation_gate": summary["validation_gate"],
        "checkpoints": checkpoints,
        "rule": "May be consumed once by eval_rlcs_matchup.py; no post-test model changes.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def render_test_figure(results_path: Path, output_path: Path) -> Path:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    comparisons = results["comparisons"]
    names = [
        "Anonymous",
        "Roster-only",
        "Actor-only",
        "Full actor + opponent",
        "Within-roster shuffle",
        "Matched opponent shuffle",
    ]
    values = [
        0.0,
        100
        * (
            results["mean_nll_primary"]["anonymous"]
            - results["mean_nll_primary"]["roster_only"]
        )
        / results["mean_nll_primary"]["anonymous"],
        100
        * (
            results["mean_nll_primary"]["anonymous"]
            - results["mean_nll_primary"]["actor_only"]
        )
        / results["mean_nll_primary"]["anonymous"],
        100 * comparisons["full_vs_anonymous"]["relative_nll_reduction"],
        100
        * (
            results["mean_nll_primary"]["anonymous"]
            - results["mean_nll_primary"]["within_roster_shuffle"]
        )
        / results["mean_nll_primary"]["anonymous"],
        100
        * (
            results["mean_nll_primary"]["anonymous"]
            - results["mean_nll_primary"]["matched_opponent_shuffle"]
        )
        / results["mean_nll_primary"]["anonymous"],
    ]
    figure, axis = plt.subplots(figsize=(9, 5.2))
    colors = ["#8A94A6", "#5B8FF9", "#5AD8A6", "#5D3FD3", "#E8684A", "#F6BD16"]
    axis.barh(names[::-1], values[::-1], color=colors[::-1])
    axis.axvline(5.0, color="black", linestyle="--", linewidth=1, label="5% gate")
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Critical-state joint-NLL reduction vs Anonymous (%)")
    counts = results["counts"]
    axis.set_title(
        f"RLCS identity-matchup test | {counts['primary_samples']:,} touches, "
        f"{counts['primary_series']} series"
    )
    axis.legend(frameon=False)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize validation or sealed RLCS results.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/rlcs_identity_matchup_v1.yaml")
    )
    parser.add_argument("--run-root", type=Path, default=Path("runs/rlcs_identity_matchup_v1"))
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=Path("runs/rlcs_identity_matchup_v1/validation_summary.json"),
    )
    parser.add_argument("--write-unlock", type=Path)
    parser.add_argument("--test-results", type=Path)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    if args.test_results:
        figure = args.figure or args.test_results.with_name("identity_matchup_test.png")
        print(f"figure: {render_test_figure(args.test_results, figure)}")
        return
    summary = summarize_validation(args.config, args.run_root)
    args.validation_output.parent.mkdir(parents=True, exist_ok=True)
    args.validation_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"validation_summary: {args.validation_output}")
    if args.write_unlock:
        print(f"test_unlock: {write_unlock(summary, args.write_unlock)}")


if __name__ == "__main__":
    main()
