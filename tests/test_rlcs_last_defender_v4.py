from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from footballq.analysis.rlcs_last_defender_v4 import (
    ADDITIVE_PROFILE_FEATURE_NAMES,
    LastDefenderV4Error,
    build_overlap_support,
    evaluate_outcome_models,
    label_replay_success,
    outcome_volume_audit,
    weighted_absolute_smd,
)
from footballq.data.rlcs_last_defender_v3 import (
    MATCHING_FEATURE_NAMES,
    cross_fitted_propensity,
)
from tests.rlcs_test_utils import synthetic_replay


def _balanced_support_frame(pair_count: int = 120) -> pd.DataFrame:
    rows = []
    for pair in range(pair_count):
        region = "EU" if pair < pair_count // 2 else "NA"
        series = f"series-{pair % 12:02d}"
        base = {name: float(pair % 9) for name in MATCHING_FEATURE_NAMES}
        for favorable in (0, 1):
            rows.append(
                {
                    "sample_id": f"sample-{pair:03d}-{favorable}",
                    "v3_stage": "train" if pair % 2 == 0 else "internal_development",
                    "series_id": series,
                    "region": region,
                    "actor_team_roster_sha256": f"roster-{pair % 6}",
                    "favorable_matchup": favorable,
                    **deepcopy(base),
                }
            )
    frame = pd.DataFrame(rows)
    propensity, folds, _ = cross_fitted_propensity(
        frame, folds=5, seed=20260803, c_value=1.0
    )
    frame["propensity"] = propensity
    frame["propensity_fold"] = folds
    return frame


def _support_config() -> dict[str, object]:
    return {
        "propensity_folds": 5,
        "propensity_seed": 20260803,
        "propensity_c": 1.0,
        "propensity_interval": [0.10, 0.90],
        "minimum_propensity_interval_fraction": 0.70,
        "minimum_total_ess": 200,
        "minimum_exposure_group_ess": 100,
        "maximum_absolute_weighted_smd": 0.10,
        "exact_strata": ["series_id", "actor_team_roster_sha256"],
        "minimum_exact_pairs": 100,
        "maximum_absolute_exact_pair_smd": 0.10,
    }


def _source_lock(rows: int) -> dict[str, object]:
    return {
        "opportunity_rows": rows,
        "allowed_stages": ["train", "internal_development"],
        "forbidden_columns": ["action_label", "success_label", "outcome_label"],
    }


def test_overlap_support_reproduces_propensity_and_balances_weighted_and_exact_samples():
    frame = _balanced_support_frame()
    weighted, report, pairs = build_overlap_support(
        frame,
        support_config=_support_config(),
        source_lock=_source_lock(len(frame)),
    )
    assert report["all_gates_pass"] is True
    assert report["propensity_reproduction"]["maximum_absolute_error"] == 0.0
    assert report["propensity_reproduction"]["fold_mismatches"] == 0
    assert len(pairs) == 120
    assert weighted["v4_exact_pair_id"].ne("").all()
    assert report["overlap_weighting"]["maximum_absolute_smd"] == pytest.approx(0.0)


def test_support_rejects_an_outcome_column_before_balance():
    frame = _balanced_support_frame()
    frame["success_label"] = 0
    with pytest.raises(LastDefenderV4Error, match="forbidden"):
        build_overlap_support(
            frame,
            support_config=_support_config(),
            source_lock=_source_lock(len(frame)),
        )


def test_weighted_smd_balances_a_rawly_confounded_feature():
    frame = pd.DataFrame(
        {
            "favorable_matchup": [1, 1, 0, 0],
            "feature": [10.0, 0.0, 10.0, 0.0],
            "weight": [0.1, 0.9, 0.1, 0.9],
        }
    )
    result = weighted_absolute_smd(
        frame, feature_names=["feature"], weight_column="weight"
    )
    assert result["feature"] == pytest.approx(0.0)


def _event(
    *,
    number: int,
    frame: int,
    prefix: str,
    event_type: str,
    roster: dict[str, str],
) -> dict[str, object]:
    team = "blue" if prefix.startswith("blue") else "orange"
    platform_id = roster[prefix].split(":", 1)[1]
    return {
        "event_number": number,
        "event_type": event_type,
        "observed_frame_number": frame,
        "game_time_s_precise": frame / 10.0,
        "stint_number": 1,
        "event_team": team,
        "event_player_1_id": platform_id,
        "event_player_1_name": f"Player {platform_id}",
        "event_player_1_team": team,
        "event_ball_pos_x": float(frame),
        "event_ball_pos_y": 100.0,
        "event_ball_pos_z": 100.0,
        "ball_pos_x": float(frame),
        "ball_pos_y": 100.0,
        "ball_pos_z": 100.0,
        "blue_score": 0,
        "orange_score": 0,
    }


def test_success_label_counts_an_actor_team_shot_before_two_opponent_contacts():
    _, observations, roster, _ = synthetic_replay()
    events = pd.DataFrame(
        [
            _event(number=1, frame=20, prefix="blue_player_1", event_type="touch", roster=roster),
            _event(number=2, frame=24, prefix="blue_player_1", event_type="shot", roster=roster),
            _event(number=3, frame=30, prefix="orange_player_1", event_type="touch", roster=roster),
            _event(number=4, frame=40, prefix="orange_player_2", event_type="touch", roster=roster),
        ]
    )
    opportunities = pd.DataFrame(
        [
            {
                "sample_id": "opportunity-1",
                "stint_number": 1,
                "frame_idx": 20,
                "actor_player_id": roster["blue_player_1"],
            }
        ]
    )
    labeled = label_replay_success(
        events,
        opportunities,
        observations=observations,
        roster_ids=roster,
        maximum_future_contacts=5,
        consecutive_opponent_contacts=2,
        success_events=["shot", "goal"],
        boundary_events=["goal", "kickoff"],
    )
    assert labeled.loc[0, "success_label"] == 1
    assert labeled.loc[0, "success_horizon_termination"] == "consecutive_opponent_contacts"


def test_success_label_censors_an_incomplete_stint():
    _, observations, roster, _ = synthetic_replay()
    events = pd.DataFrame(
        [
            _event(number=1, frame=20, prefix="blue_player_1", event_type="touch", roster=roster),
            _event(number=2, frame=30, prefix="orange_player_1", event_type="touch", roster=roster),
        ]
    )
    opportunities = pd.DataFrame(
        [
            {
                "sample_id": "opportunity-1",
                "stint_number": 1,
                "frame_idx": 20,
                "actor_player_id": roster["blue_player_1"],
            }
        ]
    )
    labeled = label_replay_success(
        events,
        opportunities,
        observations=observations,
        roster_ids=roster,
        maximum_future_contacts=5,
        consecutive_opponent_contacts=2,
        success_events=["shot", "goal"],
        boundary_events=["goal", "kickoff"],
    )
    assert pd.isna(labeled.loc[0, "success_label"])
    assert labeled.loc[0, "success_horizon_termination"] == "censored_stint_end"


def test_outcome_volume_requires_complete_pairs_and_postlabel_balance():
    rows = []
    for region, pair_id in (("EU", "pair-eu"), ("NA", "pair-na")):
        for favorable in (0, 1):
            rows.append(
                {
                    "region": region,
                    "favorable_matchup": favorable,
                    "success_label": favorable,
                    "success_horizon_termination": "maximum_future_contacts",
                    "v4_overlap_weight": 1.0,
                    "v4_exact_pair_id": pair_id,
                    **{name: 0.0 for name in MATCHING_FEATURE_NAMES},
                }
            )
    report = outcome_volume_audit(
        pd.DataFrame(rows),
        {
            "minimum_uncensored_rows": 4,
            "minimum_successes": 2,
            "minimum_failures": 2,
            "minimum_region_successes": 1,
            "minimum_region_failures": 1,
            "minimum_complete_exact_pairs": 2,
            "maximum_postlabel_weighted_smd": 0.10,
            "maximum_complete_exact_pair_smd": 0.10,
        },
    )
    assert report["all_gates_pass"] is True
    assert report["complete_exact_pair_balance"]["pairs"] == 2


def _model_frame() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(17)
    for series in range(20):
        for row_index in range(8):
            outcome = row_index % 2
            row = {
                "sample_id": f"model-{series:02d}-{row_index:02d}",
                "series_id": f"series-{series:02d}",
                "region": "EU" if series < 10 else "NA",
                "success_label": outcome,
                "v4_overlap_weight": 1.0,
                "v4_exact_pair_id": f"pair-{series:02d}-{row_index // 2:02d}",
            }
            row.update(
                {name: float(rng.normal() + 0.1 * outcome) for name in MATCHING_FEATURE_NAMES}
            )
            row.update(
                {
                    name: float(rng.normal() + 0.1 * outcome)
                    for name in ADDITIVE_PROFILE_FEATURE_NAMES
                }
            )
            row["actor_attack_composite"] = float(rng.normal())
            row["defender_resistance_composite"] = float(rng.normal())
            row["matchup_mismatch"] = (
                row["actor_attack_composite"] - row["defender_resistance_composite"]
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_outcome_models_use_matched_series_folds_and_return_all_conditions():
    evaluated, result = evaluate_outcome_models(
        _model_frame(),
        model_config={
            "c_grid": [0.1],
            "outer_folds": 5,
            "inner_folds": 3,
            "seed": 20260808,
        },
        uncertainty_config={
            "bootstrap_resamples": 20,
            "bootstrap_seed": 20260809,
            "profile_permutations": 5,
            "permutation_seed": 20260810,
        },
        gate_config={
            "full_vs_team_form_relative_log_loss_reduction": 0.99,
            "full_vs_additive_relative_log_loss_reduction": 0.99,
            "official_series_bootstrap_lower_exclusive": 0.0,
            "profile_permutation_p_exclusive": 0.01,
        },
    )
    assert set(result["conditions"]) == {
        "state",
        "team_form",
        "additive_profiles",
        "full_matchup",
    }
    assert result["profile_permutation"]["status"] == (
        "not_run_point_estimate_gates_already_failed"
    )
    for fold in sorted(evaluated["v4_outcome_fold"].unique()):
        holdout = set(evaluated.loc[evaluated["v4_outcome_fold"].eq(fold), "series_id"])
        training = set(evaluated.loc[evaluated["v4_outcome_fold"].ne(fold), "series_id"])
        assert holdout.isdisjoint(training)
