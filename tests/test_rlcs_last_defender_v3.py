from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from rlcs_test_utils import synthetic_replay

from footballq.data.rlcs_last_defender_v3 import (
    MATCHING_FEATURE_NAMES,
    GeometryThresholds,
    assign_favorable_matchup,
    build_replay_opportunities,
    calibrate_geometry_thresholds,
    common_support_audit,
    extract_replay_calibration,
    opportunity_profile_traits,
    opportunity_volume_audit,
)
from footballq.data.rlcs_player_profiles import PROFILE_DIMENSION


def _last_defender_replay():
    parsed, observations, roster, inventory = synthetic_replay()
    frames = parsed.frames
    frames.loc[:, ["ball_pos_x", "ball_pos_y", "ball_pos_z"]] = [0.0, 1200.0, 100.0]
    frames.loc[:, ["ball_vel_x", "ball_vel_y", "ball_vel_z"]] = [0.0, 500.0, 0.0]
    positions = {
        "blue_player_1": (0.0, 1000.0, 17.0),
        "blue_player_2": (-3000.0, 0.0, 17.0),
        "blue_player_3": (3000.0, 0.0, 17.0),
        "orange_player_1": (0.0, 2500.0, 17.0),
        "orange_player_2": (-3000.0, 2000.0, 17.0),
        "orange_player_3": (3000.0, 2000.0, 17.0),
    }
    for prefix, position in positions.items():
        for axis, value in zip("xyz", position, strict=True):
            frames.loc[:, f"{prefix}_pos_{axis}"] = value
    snapshots = {
        player_id: {
            "profile": [0.0] * PROFILE_DIMENSION,
            "uncertainty": [1.0] * PROFILE_DIMENSION,
            "n_prior_games": 20,
            "effective_sample_size": 20.0,
            "prior_win_rate": 0.5,
            "prior_goal_diff": 0.0,
            "latest_prior_time_utc": "2024-12-31T00:00:00Z",
        }
        for player_id in roster.values()
    }
    priors = {
        "population_mean": [0.0] * PROFILE_DIMENSION,
        "uncertainty_scale": [1.0] * PROFILE_DIMENSION,
    }
    return parsed, observations, roster, inventory, snapshots, priors


def _build_rows(parsed, observations, roster, inventory, snapshots, priors):
    return build_replay_opportunities(
        parsed.frames,
        parsed.events,
        replay_id=parsed.replay_id,
        inventory=inventory,
        stage="train",
        observations=observations,
        roster_ids=roster,
        snapshots=snapshots,
        priors=priors,
        eligible_player_ids=set(roster.values()),
        thresholds=GeometryThresholds(
            corridor_half_width=1000.0,
            last_defender_forward_distance=3000.0,
            immediate_intervention_range=1000.0,
            teammate_overload_range=1000.0,
        ),
    )


def test_frozen_geometry_calibration_uses_declared_quantiles_and_clips():
    samples = {
        "corridor_half_width": np.linspace(0.0, 3000.0, 500),
        "last_defender_forward_distance": np.linspace(0.0, 6000.0, 500),
        "immediate_intervention_range": np.linspace(0.0, 4000.0, 500),
        "teammate_overload_range": np.linspace(0.0, 4000.0, 500),
    }
    config = {
        "quantile_method": "linear",
        "minimum_finite_observations": 500,
        "corridor_half_width": {"quantile": 0.60, "clip": [700.0, 1800.0]},
        "last_defender_forward_distance": {
            "quantile": 0.80,
            "clip": [1200.0, 4200.0],
        },
        "immediate_intervention_range": {
            "quantile": 0.25,
            "clip": [900.0, 2200.0],
        },
        "teammate_overload_range": {
            "quantile": 0.25,
            "clip": [900.0, 2200.0],
        },
    }
    thresholds, report = calibrate_geometry_thresholds(samples, config)
    assert thresholds.corridor_half_width == pytest.approx(1800.0)
    assert thresholds.last_defender_forward_distance == pytest.approx(4200.0)
    assert thresholds.immediate_intervention_range == pytest.approx(1000.0)
    assert thresholds.teammate_overload_range == pytest.approx(1000.0)
    assert set(report["finite_observation_counts"].values()) == {500}


def test_detector_builds_one_outcome_free_last_defender_opportunity():
    parsed, observations, roster, inventory, snapshots, priors = _last_defender_replay()
    rows = _build_rows(parsed, observations, roster, inventory, snapshots, priors)
    assert len(rows) == 1
    row = rows[0]
    assert row["actor_player_id"] == roster["blue_player_1"]
    assert row["defender_player_id"] == roster["orange_player_1"]
    assert row["actor_prior_games"] == 20
    assert row["defender_prior_games"] == 20
    assert set(MATCHING_FEATURE_NAMES).issubset(row)
    forbidden = {"action_label", "success_label", "outcome_label", "future_contacts"}
    assert forbidden.isdisjoint(row)


def test_future_telemetry_and_action_poison_cannot_change_current_opportunity():
    clean = _last_defender_replay()
    poisoned = _last_defender_replay()
    poisoned[0].frames.loc[
        poisoned[0].frames["observed_frame_number"] > 20, "ball_pos_x"
    ] = 999_999.0
    poisoned[0].events.loc[
        poisoned[0].events["observed_frame_number"] > 20, "event_type"
    ] = "shot"
    clean_row = _build_rows(*clean)[0]
    poisoned_row = _build_rows(*poisoned)[0]
    compared = [
        "sample_id",
        "actor_player_id",
        "defender_player_id",
        *MATCHING_FEATURE_NAMES,
        "matchup_mismatch",
    ]
    assert {key: clean_row[key] for key in compared} == {
        key: poisoned_row[key] for key in compared
    }


def test_stage0_rejects_validation_before_touch_extraction():
    parsed, observations, roster, inventory, snapshots, priors = _last_defender_replay()
    with pytest.raises(PermissionError, match="may not open"):
        build_replay_opportunities(
            parsed.frames.iloc[0:0],
            parsed.events.iloc[0:0],
            replay_id="does-not-matter",
            inventory=inventory,
            stage="validation",
            observations=observations,
            roster_ids=roster,
            snapshots=snapshots,
            priors=priors,
            eligible_player_ids=set(roster.values()),
            thresholds=GeometryThresholds(1000.0, 3000.0, 1000.0, 1000.0),
        )


def test_profile_mismatch_is_deterministic_and_train_median_is_inclusive():
    snapshot = {
        "profile": np.linspace(-1.0, 1.0, PROFILE_DIMENSION).tolist(),
    }
    priors = {
        "population_mean": [0.0] * PROFILE_DIMENSION,
        "uncertainty_scale": [1.0] * PROFILE_DIMENSION,
    }
    traits = opportunity_profile_traits(snapshot, snapshot, priors)
    assert np.isfinite(list(traits.values())).all()
    frame = pd.DataFrame(
        {
            "v3_stage": ["train", "train", "internal_development"],
            "matchup_mismatch": [-1.0, 1.0, 0.0],
        }
    )
    assigned, threshold = assign_favorable_matchup(frame)
    assert threshold == 0.0
    assert assigned["favorable_matchup"].tolist() == [0, 1, 1]


def test_opportunity_volume_gate_is_complete_and_concentration_aware():
    frame = pd.DataFrame(
        {
            "actor_player_id": [f"a{i % 10}" for i in range(100)],
            "defender_player_id": [f"d{i % 10}" for i in range(100)],
            "region": ["EU"] * 50 + ["NA"] * 50,
        }
    )
    report = opportunity_volume_audit(
        frame,
        {
            "minimum_opportunities": 100,
            "minimum_distinct_actors": 10,
            "minimum_distinct_defenders": 10,
            "minimum_eu_opportunities": 50,
            "minimum_na_opportunities": 50,
            "maximum_actor_share": 0.10,
            "maximum_defender_share": 0.10,
        },
    )
    assert report["all_gates_pass"]


def test_common_support_keeps_series_grouped_and_balances_exact_pairs():
    rows = []
    pair_count = 120
    for pair in range(pair_count):
        region = "EU" if pair < pair_count // 2 else "NA"
        series_id = f"series-{pair % 12:02d}"
        base = {name: float(pair % 7) for name in MATCHING_FEATURE_NAMES}
        for favorable in (0, 1):
            rows.append(
                {
                    "sample_id": f"sample-{pair:03d}-{favorable}",
                    "series_id": series_id,
                    "region": region,
                    "favorable_matchup": favorable,
                    **deepcopy(base),
                }
            )
    frame = pd.DataFrame(rows)
    audited, report, pairs = common_support_audit(
        frame,
        {
            "folds": 5,
            "seed": 20260803,
            "propensity_c": 1.0,
            "propensity_interval": [0.10, 0.90],
            "minimum_propensity_interval_fraction": 0.70,
            "overlap_weight_ess_minimum": 200,
            "logit_caliper_pooled_sd": 0.20,
            "minimum_matched_sets": 100,
            "maximum_absolute_smd": 0.10,
        },
    )
    assert report["all_gates_pass"]
    assert len(pairs) == pair_count
    assert audited["matched_set_id"].ne("").all()
    for fold in sorted(audited["propensity_fold"].unique()):
        held_out = set(audited.loc[audited["propensity_fold"] == fold, "series_id"])
        training = set(audited.loc[audited["propensity_fold"] != fold, "series_id"])
        assert held_out.isdisjoint(training)


def test_calibration_extracts_no_future_or_action_summary():
    parsed, observations, roster, *_ = _last_defender_replay()
    result = extract_replay_calibration(
        parsed.frames,
        parsed.events,
        observations=observations,
        roster_ids=roster,
    )
    assert result["base_contacts"] == 1
    assert set(result) == {"base_contacts", "samples"}
    assert set(result["samples"]) == {
        "corridor_half_width",
        "last_defender_forward_distance",
        "immediate_intervention_range",
        "teammate_overload_range",
    }
