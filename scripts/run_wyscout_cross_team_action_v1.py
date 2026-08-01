"""Run the development stage of cross-team pass-destination prediction."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from footballq.analysis.wyscout_cross_team_action import (
    build_destination_aggregates,
    build_penalty_aggregates,
    compare_destination_models,
    compare_penalty_models,
    penalty_metric_bundle,
    prepare_destination_cache,
    prepare_penalty_cache,
    selected_penalty_probabilities,
    selected_probabilities,
    shuffled_penalty_profile_results,
    tune_destination_models,
    tune_penalty_models,
)
from footballq.analysis.wyscout_player_memory import (
    file_sha256,
    load_development_frames,
    stable_payload_hash,
)

ANALYSIS_PATH = Path("src/footballq/analysis/wyscout_cross_team_action.py")
RUNNER_PATH = Path("scripts/run_wyscout_cross_team_action_v1.py")


def _git_metadata() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_line_count": len(status),
    }


def _development_gate(
    main: dict[str, Any],
    support_curve: dict[str, Any],
    shuffled: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    rolling = main["rolling_player"]
    conditional = main["conditional_player"]
    effect = main["effect"]
    main_cap = str(config["profiles"]["main_support_match_cap"])
    first_cap = str(config["profiles"]["support_match_caps"][0])
    gate_config = config["development_gate"]
    rolling_penalty = rolling["supported"]
    conditional_penalty = conditional["supported"]
    conditional_nll = float(conditional_penalty["nll"])
    relative_penalty_gain = float(effect["relative_nll_improvement"])
    checks = {
        "minimum_relative_penalty_nll_improvement": (
            relative_penalty_gain
            >= float(gate_config["minimum_relative_penalty_nll_improvement"])
        ),
        "positive_penalty_bootstrap_lower_bound": (
            float(effect["match_bootstrap"]["ci_lower"]) > 0.0
            if bool(gate_config["require_positive_penalty_bootstrap_lower_bound"])
            else True
        ),
        "penalty_brier_improvement": (
            float(conditional_penalty["brier"]) < float(rolling_penalty["brier"])
            if bool(gate_config["require_penalty_brier_improvement"])
            else True
        ),
        "penalty_average_precision_improvement": (
            float(conditional_penalty["average_precision"])
            > float(rolling_penalty["average_precision"])
            if bool(gate_config["require_penalty_average_precision_improvement"])
            else True
        ),
        "better_than_all_same_team_role_shuffles": (
            all(
                conditional_nll
                < float(row["metrics"]["supported"]["nll"])
                for row in shuffled
            )
            if bool(
                gate_config[
                    "require_better_than_all_same_team_role_shuffles"
                ]
            )
            else True
        ),
        "main_cap_better_than_first_cap": (
            float(
                support_curve[main_cap]["penalty_comparison"][
                    "conditional_player"
                ]["supported"]["nll"]
            )
            < float(
                support_curve[first_cap]["penalty_comparison"][
                    "conditional_player"
                ]["supported"]["nll"]
            )
            if bool(gate_config["require_main_cap_better_than_first_cap"])
            else True
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "observed": {
            "penalty_relative_nll_improvement": relative_penalty_gain,
            "penalty_entry_nll_gain": effect["rolling_minus_conditional_nll"],
            "penalty_entry_bootstrap_ci_lower": effect["match_bootstrap"][
                "ci_lower"
            ],
        },
    }


def run_development(config: dict[str, Any]) -> dict[str, Any]:
    frames, lineage = load_development_frames(config)
    profile_config = config["profiles"]
    uncertainty = config["uncertainty"]
    query_start_x_min = float(config["task"]["query_start_x_min"])
    minimum_matches = int(profile_config["minimum_prior_matches"])
    context_strength = float(profile_config["context_prior_strength"])
    residual_limit = float(profile_config["residual_ratio_limit"])
    prior_strengths = [
        float(value) for value in profile_config["prior_strengths"]
    ]
    support_curve: dict[str, Any] = {}
    penalty_aggregates_by_cap = {}
    penalty_selection_by_cap = {}
    for cap_raw in profile_config["support_match_caps"]:
        cap = int(cap_raw)
        destination_aggregates = build_destination_aggregates(
            frames["support"],
            match_cap=cap,
        )
        validation_cache = prepare_destination_cache(
            frames["validation"],
            destination_aggregates,
            minimum_prior_matches=minimum_matches,
        )
        destination_selection = tune_destination_models(
            validation_cache,
            minimum_prior_matches=minimum_matches,
            context_prior_strength=context_strength,
            prior_strengths=prior_strengths,
            residual_ratio_limit=residual_limit,
        )
        development_cache = prepare_destination_cache(
            frames["development"],
            destination_aggregates,
            minimum_prior_matches=minimum_matches,
        )
        destination_rolling, destination_conditional = selected_probabilities(
            development_cache,
            destination_selection,
            context_prior_strength=context_strength,
            residual_ratio_limit=residual_limit,
        )
        destination_comparison = compare_destination_models(
            development_cache,
            destination_rolling,
            destination_conditional,
            minimum_prior_matches=minimum_matches,
            bootstrap_replicates=int(uncertainty["bootstrap_replicates"]),
            bootstrap_seed=int(uncertainty["seed"]) + cap,
            confidence_level=float(uncertainty["confidence_level"]),
        )

        penalty_aggregates = build_penalty_aggregates(
            frames["support"],
            match_cap=cap,
            query_start_x_min=query_start_x_min,
        )
        penalty_aggregates_by_cap[cap] = penalty_aggregates
        validation_penalty_cache = prepare_penalty_cache(
            frames["validation"],
            penalty_aggregates,
            minimum_prior_matches=minimum_matches,
            query_start_x_min=query_start_x_min,
        )
        penalty_selection = tune_penalty_models(
            validation_penalty_cache,
            minimum_prior_matches=minimum_matches,
            context_prior_strength=context_strength,
            prior_strengths=prior_strengths,
            residual_ratio_limit=residual_limit,
        )
        penalty_selection_by_cap[cap] = penalty_selection
        development_penalty_cache = prepare_penalty_cache(
            frames["development"],
            penalty_aggregates,
            minimum_prior_matches=minimum_matches,
            query_start_x_min=query_start_x_min,
        )
        penalty_rolling, penalty_conditional = selected_penalty_probabilities(
            development_penalty_cache,
            penalty_selection,
            context_prior_strength=context_strength,
            residual_ratio_limit=residual_limit,
        )
        penalty_comparison = compare_penalty_models(
            development_penalty_cache,
            penalty_rolling,
            penalty_conditional,
            minimum_prior_matches=minimum_matches,
            bootstrap_replicates=int(uncertainty["bootstrap_replicates"]),
            bootstrap_seed=int(uncertainty["seed"]) + 10_000 + cap,
            confidence_level=float(uncertainty["confidence_level"]),
        )
        support_curve[str(cap)] = {
            "destination_selection": destination_selection,
            "penalty_selection": penalty_selection,
            "destination_comparison": destination_comparison,
            "penalty_comparison": penalty_comparison,
        }
    main_cap = int(profile_config["main_support_match_cap"])
    main_aggregates = penalty_aggregates_by_cap[main_cap]
    main_selection = penalty_selection_by_cap[main_cap]
    shuffled = shuffled_penalty_profile_results(
        frames["development"],
        main_aggregates,
        main_selection,
        minimum_prior_matches=minimum_matches,
        query_start_x_min=query_start_x_min,
        context_prior_strength=context_strength,
        residual_ratio_limit=residual_limit,
        seeds=[int(value) for value in config["falsification"]["shuffle_seeds"]],
    )
    main = support_curve[str(main_cap)]["penalty_comparison"]
    gate = _development_gate(main, support_curve, shuffled, config)
    validation_cache = prepare_penalty_cache(
        frames["validation"],
        main_aggregates,
        minimum_prior_matches=minimum_matches,
        query_start_x_min=query_start_x_min,
    )
    validation_rolling, validation_conditional = selected_penalty_probabilities(
        validation_cache,
        main_selection,
        context_prior_strength=context_strength,
        residual_ratio_limit=residual_limit,
    )
    return {
        "experiment_protocol": str(config["experiment_protocol"]),
        "status": "development_only",
        "claim_boundary": (
            "Support-only, context-specific player pass profiles improve exact "
            "held-out penalty-area-entry prediction beyond a player's rolling "
            "history after a complete team change. This is not tracking-based "
            "tactical understanding or multi-step critical-event prediction."
        ),
        "lineage": lineage,
        "feature_contract": {
            "causal_query_features": config["task"]["causal_context"],
            "primary_target": (
                "exact destination x>=83 and 21<=y<=79, conditional on a pass "
                "starting at x>=50 and outside that area"
            ),
            "secondary_target": "destination_zone_6_by_5",
            "rolling_baseline": (
                "support-only player penalty-entry rate adjusted for support "
                "team, broad role, and current start-zone context"
            ),
            "profile_model": (
                "support-only player-by-start-zone exact penalty-entry rate"
            ),
            "profile_history": "last K chronological support matches only",
        },
        "support_curve": support_curve,
        "main_support_match_cap": main_cap,
        "main_result": main,
        "development_destination_result": support_curve[str(main_cap)][
            "destination_comparison"
        ],
        "main_validation_metrics": {
            "rolling_player": penalty_metric_bundle(
                validation_cache,
                validation_rolling,
                minimum_prior_matches=minimum_matches,
            ),
            "conditional_player": penalty_metric_bundle(
                validation_cache,
                validation_conditional,
                minimum_prior_matches=minimum_matches,
            ),
        },
        "same_team_role_shuffles": shuffled,
        "gate": gate,
        "confirmatory_metrics_loaded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/wyscout_cross_team_action_v1.yaml",
    )
    parser.add_argument(
        "--output-root",
        default="runs/wyscout_cross_team_action_v1",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    result = run_development(config)
    result["config_path"] = str(config_path)
    result["config_sha256"] = file_sha256(config_path)
    result["result_payload_sha256"] = stable_payload_hash(result)
    output = Path(args.output_root) / "development"
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    result_path.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "name": "wyscout_cross_team_action_v1_development_run_manifest",
        "version": 1,
        "result_path": str(result_path),
        "result_file_sha256": file_sha256(result_path),
        "result_payload_sha256": result["result_payload_sha256"],
        "config_path": str(config_path),
        "config_sha256": result["config_sha256"],
        "dataset_manifest_path": str(config["data"]["dataset_manifest"]),
        "dataset_manifest_sha256": file_sha256(
            config["data"]["dataset_manifest"]
        ),
        "split_manifest_path": str(
            config["data"]["development_split_manifest"]
        ),
        "split_manifest_sha256": file_sha256(
            config["data"]["development_split_manifest"]
        ),
        "source_files": {
            "analysis": {
                "path": str(ANALYSIS_PATH),
                "sha256": file_sha256(ANALYSIS_PATH),
            },
            "runner": {
                "path": str(RUNNER_PATH),
                "sha256": file_sha256(RUNNER_PATH),
            },
        },
        "git": _git_metadata(),
        "confirmatory_metrics_loaded": False,
    }
    manifest["manifest_payload_sha256"] = stable_payload_hash(manifest)
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "results": str(result_path),
                "run_manifest": str(manifest_path),
                "gate": result["gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
