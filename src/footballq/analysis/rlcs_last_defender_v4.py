"""Outcome-locked overlap-weighted RLCS last-defender V4 analysis."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from footballq.data.rlcs_last_defender_v3 import (
    MATCHING_FEATURE_NAMES,
    TEAM_FORM_FEATURE_NAMES,
    cross_fitted_propensity,
)
from footballq.data.rlcs_replay import IdentityObservation, normalize_handle
from footballq.data.rlcs_touch_windows import extract_touches

STATE_FEATURE_NAMES = tuple(
    name for name in MATCHING_FEATURE_NAMES if name not in TEAM_FORM_FEATURE_NAMES
)

ADDITIVE_PROFILE_FEATURE_NAMES = (
    "actor_carry_speed",
    "actor_boost_economy",
    "actor_take_on_control_frequency",
    "defender_goalside_recovery",
    "defender_challenge_win",
    "defender_boost_economy",
    "defender_turnover_pressure_proxy",
)

PROFILE_SOURCE_FEATURE_NAMES = (
    *ADDITIVE_PROFILE_FEATURE_NAMES,
    "actor_attack_composite",
    "defender_resistance_composite",
    "matchup_mismatch",
)

INTERACTION_SPECS = (
    ("actor_carry_speed", "defender_goalside_recovery"),
    ("actor_boost_economy", "defender_boost_economy"),
    ("actor_take_on_control_frequency", "defender_challenge_win"),
    ("actor_take_on_control_frequency", "defender_turnover_pressure_proxy"),
    ("actor_attack_composite", "defender_resistance_composite"),
)

INTERACTION_FEATURE_NAMES = tuple(
    f"interaction__{left}__x__{right}" for left, right in INTERACTION_SPECS
)

FULL_MATCHUP_EXTRA_FEATURE_NAMES = (
    "actor_attack_composite",
    "defender_resistance_composite",
    "matchup_mismatch",
    *INTERACTION_FEATURE_NAMES,
)

CONDITION_FEATURES: dict[str, tuple[str, ...]] = {
    "state": STATE_FEATURE_NAMES,
    "team_form": MATCHING_FEATURE_NAMES,
    "additive_profiles": (*MATCHING_FEATURE_NAMES, *ADDITIVE_PROFILE_FEATURE_NAMES),
    "full_matchup": (
        *MATCHING_FEATURE_NAMES,
        *ADDITIVE_PROFILE_FEATURE_NAMES,
        *FULL_MATCHUP_EXTRA_FEATURE_NAMES,
    ),
}


class LastDefenderV4Error(ValueError):
    """Raised when the frozen V4 design or outcome analysis must fail closed."""


@dataclass(frozen=True)
class SuccessLabel:
    """One mechanically labeled last-defender success horizon."""

    success: int | None
    actor_team: str
    contacts_observed: int
    termination: str
    horizon_end_frame: int | None


def effective_sample_size(weights: Sequence[float]) -> float:
    """Return Kish effective sample size for non-negative weights."""

    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        return 0.0
    if not np.isfinite(values).all() or bool((values < 0.0).any()):
        raise LastDefenderV4Error("Weights must be finite and non-negative.")
    denominator = float(np.square(values).sum())
    return 0.0 if denominator <= 0.0 else float(values.sum() ** 2 / denominator)


def _weighted_mean_variance(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    if len(values) == 0 or len(values) != len(weights):
        raise LastDefenderV4Error("Weighted moments require equally sized non-empty arrays.")
    total = float(weights.sum())
    if total <= 0.0:
        raise LastDefenderV4Error("Weighted moments require positive total weight.")
    mean = float(np.average(values, weights=weights))
    variance = float(np.average(np.square(values - mean), weights=weights))
    return mean, variance


def weighted_absolute_smd(
    frame: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    weight_column: str,
    exposure_column: str = "favorable_matchup",
) -> dict[str, float]:
    """Compute absolute SMDs with exposure-specific weighted moments."""

    from sklearn.impute import SimpleImputer

    missing = set((*feature_names, weight_column, exposure_column)).difference(frame.columns)
    if missing:
        raise LastDefenderV4Error(f"Missing weighted-balance columns: {sorted(missing)}")
    labels = frame[exposure_column].to_numpy(dtype=np.int8)
    if set(np.unique(labels)) != {0, 1}:
        raise LastDefenderV4Error("Weighted balance requires both exposure groups.")
    raw = frame.loc[:, list(feature_names)].to_numpy(dtype=np.float64)
    features = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(raw)
    weights = frame[weight_column].to_numpy(dtype=np.float64)
    output: dict[str, float] = {}
    for index, name in enumerate(feature_names):
        moments = []
        for label in (1, 0):
            selected = labels == label
            moments.append(_weighted_mean_variance(features[selected, index], weights[selected]))
        denominator = math.sqrt(max((moments[0][1] + moments[1][1]) / 2.0, 0.0))
        difference = abs(moments[0][0] - moments[1][0])
        if denominator <= 1e-12:
            output[str(name)] = 0.0 if difference <= 1e-12 else float("inf")
        else:
            output[str(name)] = float(difference / denominator)
    return output


def _pooled_standard_deviation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) + len(right) <= 2:
        return 0.0
    left_var = float(np.var(left, ddof=1)) if len(left) > 1 else 0.0
    right_var = float(np.var(right, ddof=1)) if len(right) > 1 else 0.0
    numerator = max(len(left) - 1, 0) * left_var + max(len(right) - 1, 0) * right_var
    return float(math.sqrt(max(numerator / max(len(left) + len(right) - 2, 1), 0.0)))


def exact_stratum_pairs(
    frame: pd.DataFrame,
    *,
    strata: Sequence[str],
    feature_names: Sequence[str] = MATCHING_FEATURE_NAMES,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, float]]:
    """Create deterministic optimal pairs within frozen exact strata."""

    from scipy.optimize import linear_sum_assignment
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    missing = set((*strata, *feature_names, "sample_id", "favorable_matchup")).difference(
        frame.columns
    )
    if missing:
        raise LastDefenderV4Error(f"Missing exact-pair columns: {sorted(missing)}")
    output = frame.copy().reset_index(drop=True)
    raw = output.loc[:, list(feature_names)].to_numpy(dtype=np.float64)
    imputed = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(raw)
    standardized = StandardScaler().fit_transform(imputed)
    pairs: list[dict[str, Any]] = []
    pair_number = 0
    grouped = output.groupby(list(strata), sort=True, dropna=False)
    for stratum_key, group in grouped:
        favorable = group.index[group["favorable_matchup"].eq(1)].to_numpy(dtype=np.int64)
        unfavorable = group.index[group["favorable_matchup"].eq(0)].to_numpy(dtype=np.int64)
        if len(favorable) == 0 or len(unfavorable) == 0:
            continue
        favorable = np.asarray(
            sorted(favorable, key=lambda index: str(output.at[index, "sample_id"])),
            dtype=np.int64,
        )
        unfavorable = np.asarray(
            sorted(unfavorable, key=lambda index: str(output.at[index, "sample_id"])),
            dtype=np.int64,
        )
        distances = np.linalg.norm(
            standardized[favorable][:, None, :] - standardized[unfavorable][None, :, :], axis=2
        )
        row_indices, column_indices = linear_sum_assignment(distances)
        key_values = stratum_key if isinstance(stratum_key, tuple) else (stratum_key,)
        for left_position, right_position in zip(row_indices, column_indices, strict=True):
            pair_number += 1
            left_index = int(favorable[int(left_position)])
            right_index = int(unfavorable[int(right_position)])
            pairs.append(
                {
                    "pair_id": f"v4_exact_{pair_number:06d}",
                    "favorable_index": left_index,
                    "unfavorable_index": right_index,
                    "distance": float(distances[int(left_position), int(right_position)]),
                    "stratum": {
                        str(name): str(value)
                        for name, value in zip(strata, key_values, strict=True)
                    },
                }
            )
    output["v4_exact_pair_id"] = ""
    output["v4_exact_pair_role"] = ""
    for pair in pairs:
        left = int(pair["favorable_index"])
        right = int(pair["unfavorable_index"])
        output.at[left, "v4_exact_pair_id"] = pair["pair_id"]
        output.at[left, "v4_exact_pair_role"] = "favorable"
        output.at[right, "v4_exact_pair_id"] = pair["pair_id"]
        output.at[right, "v4_exact_pair_role"] = "unfavorable"

    smd: dict[str, float] = {}
    if pairs:
        left_indices = np.asarray([pair["favorable_index"] for pair in pairs], dtype=np.int64)
        right_indices = np.asarray([pair["unfavorable_index"] for pair in pairs], dtype=np.int64)
        for feature_index, name in enumerate(feature_names):
            left = imputed[left_indices, feature_index]
            right = imputed[right_indices, feature_index]
            difference = abs(float(left.mean() - right.mean()))
            scale = _pooled_standard_deviation(left, right)
            smd[str(name)] = (
                0.0
                if scale <= 1e-12 and difference <= 1e-12
                else float("inf")
                if scale <= 1e-12
                else float(difference / scale)
            )
    else:
        smd = {str(name): float("inf") for name in feature_names}
    return output, pairs, smd


def build_overlap_support(
    frame: pd.DataFrame, *, support_config: Mapping[str, Any], source_lock: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    """Run the complete outcome-blind V4 support redesign."""

    expected_rows = int(source_lock["opportunity_rows"])
    if len(frame) != expected_rows:
        raise LastDefenderV4Error(f"V4 expected {expected_rows} source rows, found {len(frame)}.")
    allowed_stages = set(str(value) for value in source_lock["allowed_stages"])
    observed_stages = set(frame["v3_stage"].astype(str))
    if observed_stages != allowed_stages:
        raise LastDefenderV4Error(
            f"V4 source stages differ from the freeze: {sorted(observed_stages)}."
        )
    forbidden = set(str(value).casefold() for value in source_lock["forbidden_columns"])
    present_forbidden = sorted(column for column in frame.columns if column.casefold() in forbidden)
    if present_forbidden:
        raise LastDefenderV4Error(f"V4 source contains forbidden columns: {present_forbidden}")
    required = {
        "propensity",
        "propensity_fold",
        "favorable_matchup",
        "region",
        "series_id",
        "actor_team_roster_sha256",
        *MATCHING_FEATURE_NAMES,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise LastDefenderV4Error(f"V4 source is missing columns: {sorted(missing)}")

    predictions, assignments, fold_reports = cross_fitted_propensity(
        frame,
        folds=int(support_config["propensity_folds"]),
        seed=int(support_config["propensity_seed"]),
        c_value=float(support_config["propensity_c"]),
    )
    stored_predictions = frame["propensity"].to_numpy(dtype=np.float64)
    stored_assignments = frame["propensity_fold"].to_numpy(dtype=np.int16)
    maximum_reproduction_error = float(np.max(np.abs(predictions - stored_predictions)))
    fold_mismatches = int(np.sum(assignments != stored_assignments))
    if maximum_reproduction_error > 1e-12 or fold_mismatches:
        raise LastDefenderV4Error("V4 could not reproduce the frozen V3 propensity artifact.")

    output = frame.copy().reset_index(drop=True)
    labels = output["favorable_matchup"].to_numpy(dtype=np.int8)
    output["v4_overlap_weight"] = np.where(labels == 1, 1.0 - predictions, predictions)
    total_ess = effective_sample_size(output["v4_overlap_weight"].to_numpy(dtype=np.float64))
    exposure_ess = {
        str(label): effective_sample_size(
            output.loc[output["favorable_matchup"].eq(label), "v4_overlap_weight"].to_numpy(
                dtype=np.float64
            )
        )
        for label in (0, 1)
    }
    lower, upper = (float(value) for value in support_config["propensity_interval"])
    interval_fraction = float(((predictions >= lower) & (predictions <= upper)).mean())
    weighted_smd = weighted_absolute_smd(
        output, feature_names=MATCHING_FEATURE_NAMES, weight_column="v4_overlap_weight"
    )
    maximum_weighted_smd = max(weighted_smd.values(), default=float("inf"))
    regional_smd = {
        region: weighted_absolute_smd(
            output.loc[output["region"].eq(region)].copy(),
            feature_names=MATCHING_FEATURE_NAMES,
            weight_column="v4_overlap_weight",
        )
        for region in ("EU", "NA")
    }

    output, pairs, exact_smd = exact_stratum_pairs(
        output,
        strata=tuple(str(value) for value in support_config["exact_strata"]),
        feature_names=MATCHING_FEATURE_NAMES,
    )
    maximum_exact_smd = max(exact_smd.values(), default=float("inf"))
    decisions = {
        "minimum_propensity_interval_fraction": interval_fraction
        >= float(support_config["minimum_propensity_interval_fraction"]),
        "minimum_total_ess": total_ess >= float(support_config["minimum_total_ess"]),
        "minimum_exposure_group_ess": min(exposure_ess.values())
        >= float(support_config["minimum_exposure_group_ess"]),
        "maximum_absolute_weighted_smd": maximum_weighted_smd
        <= float(support_config["maximum_absolute_weighted_smd"]),
        "minimum_exact_pairs": len(pairs) >= int(support_config["minimum_exact_pairs"]),
        "maximum_absolute_exact_pair_smd": maximum_exact_smd
        <= float(support_config["maximum_absolute_exact_pair_smd"]),
    }
    report = {
        "rows": int(len(output)),
        "propensity_reproduction": {
            "maximum_absolute_error": maximum_reproduction_error,
            "fold_mismatches": fold_mismatches,
            "folds": fold_reports,
        },
        "propensity": {
            "interval": [lower, upper],
            "interval_fraction": interval_fraction,
            "minimum": float(predictions.min()),
            "median": float(np.median(predictions)),
            "maximum": float(predictions.max()),
        },
        "overlap_weighting": {
            "total_ess": total_ess,
            "exposure_group_ess": exposure_ess,
            "maximum_absolute_smd": float(maximum_weighted_smd),
            "absolute_smd_by_feature": {name: float(value) for name, value in weighted_smd.items()},
            "regional_descriptive_maximum_absolute_smd": {
                region: float(max(values.values())) for region, values in regional_smd.items()
            },
            "regional_descriptive_absolute_smd_by_feature": {
                region: {name: float(value) for name, value in values.items()}
                for region, values in regional_smd.items()
            },
        },
        "exact_stratum_sensitivity": {
            "strata": [str(value) for value in support_config["exact_strata"]],
            "pairs": int(len(pairs)),
            "rows": int(2 * len(pairs)),
            "maximum_absolute_smd": float(maximum_exact_smd),
            "absolute_smd_by_feature": {name: float(value) for name, value in exact_smd.items()},
        },
        "gates": decisions,
        "all_gates_pass": all(decisions.values()),
    }
    return output, report, pairs


def add_matchup_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Add only the five frozen V4 profile interactions."""

    missing = set(PROFILE_SOURCE_FEATURE_NAMES).difference(frame.columns)
    if missing:
        raise LastDefenderV4Error(f"Missing V4 profile features: {sorted(missing)}")
    output = frame.copy()
    for name, (left, right) in zip(INTERACTION_FEATURE_NAMES, INTERACTION_SPECS, strict=True):
        output[name] = output[left].to_numpy(dtype=np.float64) * output[right].to_numpy(
            dtype=np.float64
        )
    return output


def _event_team(row: Mapping[str, Any]) -> str:
    for key in ("event_team", "event_player_1_team"):
        value = str(row.get(key) or "").casefold()
        if value in {"blue", "orange"}:
            return value
    return ""


def label_replay_success(
    events: pd.DataFrame,
    opportunities: pd.DataFrame,
    *,
    observations: Sequence[IdentityObservation],
    roster_ids: Mapping[str, str],
    maximum_future_contacts: int,
    consecutive_opponent_contacts: int,
    success_events: Sequence[str],
    boundary_events: Sequence[str],
) -> pd.DataFrame:
    """Open and label only the frozen V4 success horizon for one replay."""

    if opportunities.empty:
        return pd.DataFrame()
    required = {
        "event_number",
        "event_type",
        "observed_frame_number",
        "game_time_s_precise",
        "stint_number",
        "event_team",
        "event_player_1_team",
        "event_player_1_id",
        "event_player_1_name",
        "event_ball_pos_x",
        "event_ball_pos_y",
        "event_ball_pos_z",
        "ball_pos_x",
        "ball_pos_y",
        "ball_pos_z",
        "blue_score",
        "orange_score",
    }
    missing = required.difference(events.columns)
    if missing:
        raise LastDefenderV4Error(f"Outcome event table is missing columns: {sorted(missing)}")
    if int(maximum_future_contacts) <= 0 or int(consecutive_opponent_contacts) <= 0:
        raise LastDefenderV4Error("V4 success horizons require positive contact thresholds.")
    success_kinds = {normalize_handle(value) for value in success_events}
    boundary_kinds = {normalize_handle(value) for value in boundary_events}
    rows: list[dict[str, Any]] = []
    for stint_value, opportunity_group in opportunities.groupby("stint_number", sort=True):
        stint_events = events.loc[
            pd.to_numeric(events["stint_number"], errors="coerce").eq(float(stint_value))
        ].copy()
        if stint_events.empty:
            raise LastDefenderV4Error(f"Replay has no events for opportunity stint {stint_value}.")
        stint_events = stint_events.sort_values(
            ["observed_frame_number", "event_number"], kind="stable"
        ).reset_index(drop=True)
        touches = extract_touches(
            stint_events, observations, roster_ids, scores_repaired=True
        )
        touch_lookup: dict[tuple[int, str], int] = {}
        for touch_index, touch in enumerate(touches):
            key = (int(touch.frame_idx), str(touch.player_id))
            if key in touch_lookup:
                raise LastDefenderV4Error(f"Duplicate de-duplicated touch key {key}.")
            touch_lookup[key] = touch_index
        event_records = stint_events.to_dict(orient="records")
        normalized_kinds = [normalize_handle(row.get("event_type")) for row in event_records]
        for opportunity in opportunity_group.to_dict(orient="records"):
            key = (int(opportunity["frame_idx"]), str(opportunity["actor_player_id"]))
            current_index = touch_lookup.get(key)
            if current_index is None:
                raise LastDefenderV4Error(f"Cannot recover frozen opportunity touch {key}.")
            current = touches[current_index]
            actor_team = str(current.team)
            current_frame = int(current.frame_idx)
            later_boundaries = [
                int(row["observed_frame_number"])
                for row, kind in zip(event_records, normalized_kinds, strict=True)
                if kind in boundary_kinds and int(row["observed_frame_number"]) > current_frame
            ]
            boundary_frame = min(later_boundaries) if later_boundaries else None
            contacts_observed = 0
            consecutive_opponents = 0
            horizon_end: int | None = None
            termination = "censored_stint_end"
            for touch in touches[current_index + 1 :]:
                touch_frame = int(touch.frame_idx)
                if boundary_frame is not None and boundary_frame < touch_frame:
                    horizon_end = boundary_frame
                    termination = "boundary"
                    break
                contacts_observed += 1
                if touch.team == actor_team:
                    consecutive_opponents = 0
                else:
                    consecutive_opponents += 1
                horizon_end = touch_frame
                if boundary_frame is not None and boundary_frame == touch_frame:
                    termination = "boundary"
                    break
                if consecutive_opponents >= int(consecutive_opponent_contacts):
                    termination = "consecutive_opponent_contacts"
                    break
                if contacts_observed >= int(maximum_future_contacts):
                    termination = "maximum_future_contacts"
                    break
            if boundary_frame is not None and (
                horizon_end is None or boundary_frame < horizon_end
            ):
                horizon_end = boundary_frame
                termination = "boundary"
            if horizon_end is None or termination == "censored_stint_end":
                label = SuccessLabel(
                    success=None,
                    actor_team=actor_team,
                    contacts_observed=contacts_observed,
                    termination="censored_stint_end",
                    horizon_end_frame=None,
                )
            else:
                success = 0
                for row, kind in zip(event_records, normalized_kinds, strict=True):
                    event_frame = int(row["observed_frame_number"])
                    if event_frame < current_frame or event_frame > horizon_end:
                        continue
                    if kind in success_kinds and _event_team(row) == actor_team:
                        success = 1
                        break
                label = SuccessLabel(
                    success=success,
                    actor_team=actor_team,
                    contacts_observed=contacts_observed,
                    termination=termination,
                    horizon_end_frame=horizon_end,
                )
            rows.append(
                {
                    "sample_id": str(opportunity["sample_id"]),
                    "success_label": label.success,
                    "actor_team": label.actor_team,
                    "future_contacts_observed": label.contacts_observed,
                    "success_horizon_termination": label.termination,
                    "success_horizon_end_frame": label.horizon_end_frame,
                }
            )
    output = pd.DataFrame(rows)
    if output["sample_id"].duplicated().any() or len(output) != len(opportunities):
        raise LastDefenderV4Error("V4 outcome labeling did not preserve unique source rows.")
    return output


def outcome_volume_audit(frame: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate frozen V4 uncensored class-volume gates."""

    labeled = frame.loc[frame["success_label"].notna()].copy()
    labels = labeled["success_label"].to_numpy(dtype=np.int8)
    successes = int(labels.sum())
    failures = int(len(labels) - successes)
    regions = {
        region: {
            "rows": int(len(group)),
            "successes": int(group["success_label"].sum()),
            "failures": int(len(group) - group["success_label"].sum()),
        }
        for region, group in labeled.groupby("region", sort=True)
    }
    postlabel_smd = weighted_absolute_smd(
        labeled,
        feature_names=MATCHING_FEATURE_NAMES,
        weight_column="v4_overlap_weight",
    )
    maximum_postlabel_smd = max(postlabel_smd.values(), default=float("inf"))
    candidate_pairs = labeled.loc[labeled["v4_exact_pair_id"].ne("")].copy()
    pair_counts = candidate_pairs["v4_exact_pair_id"].value_counts()
    complete_pair_ids = set(pair_counts.loc[pair_counts.eq(2)].index.astype(str))
    complete_pairs = candidate_pairs.loc[
        candidate_pairs["v4_exact_pair_id"].astype(str).isin(complete_pair_ids)
    ].copy()
    if complete_pairs.empty:
        complete_pair_smd = {name: float("inf") for name in MATCHING_FEATURE_NAMES}
    else:
        complete_pairs["_equal_pair_weight"] = 1.0
        complete_pair_smd = weighted_absolute_smd(
            complete_pairs,
            feature_names=MATCHING_FEATURE_NAMES,
            weight_column="_equal_pair_weight",
        )
    maximum_complete_pair_smd = max(complete_pair_smd.values(), default=float("inf"))
    decisions = {
        "minimum_uncensored_rows": len(labeled) >= int(config["minimum_uncensored_rows"]),
        "minimum_successes": successes >= int(config["minimum_successes"]),
        "minimum_failures": failures >= int(config["minimum_failures"]),
        "minimum_eu_successes": regions.get("EU", {}).get("successes", 0)
        >= int(config["minimum_region_successes"]),
        "minimum_eu_failures": regions.get("EU", {}).get("failures", 0)
        >= int(config["minimum_region_failures"]),
        "minimum_na_successes": regions.get("NA", {}).get("successes", 0)
        >= int(config["minimum_region_successes"]),
        "minimum_na_failures": regions.get("NA", {}).get("failures", 0)
        >= int(config["minimum_region_failures"]),
        "minimum_complete_exact_pairs": len(complete_pair_ids)
        >= int(config["minimum_complete_exact_pairs"]),
        "maximum_postlabel_weighted_smd": maximum_postlabel_smd
        <= float(config["maximum_postlabel_weighted_smd"]),
        "maximum_complete_exact_pair_smd": maximum_complete_pair_smd
        <= float(config["maximum_complete_exact_pair_smd"]),
    }
    return {
        "source_rows": int(len(frame)),
        "uncensored_rows": int(len(labeled)),
        "censored_rows": int(len(frame) - len(labeled)),
        "successes": successes,
        "failures": failures,
        "by_region": regions,
        "postlabel_overlap_balance": {
            "maximum_absolute_smd": float(maximum_postlabel_smd),
            "absolute_smd_by_feature": {
                name: float(value) for name, value in postlabel_smd.items()
            },
        },
        "complete_exact_pair_balance": {
            "pairs": int(len(complete_pair_ids)),
            "rows": int(len(complete_pairs)),
            "maximum_absolute_smd": float(maximum_complete_pair_smd),
            "absolute_smd_by_feature": {
                name: float(value) for name, value in complete_pair_smd.items()
            },
        },
        "termination_counts": {
            str(name): int(count)
            for name, count in frame["success_horizon_termination"].value_counts().items()
        },
        "gates": decisions,
        "all_gates_pass": all(decisions.values()),
    }


def weighted_binary_log_loss(
    labels: Sequence[int], probabilities: Sequence[float], weights: Sequence[float]
) -> float:
    """Compute clipped weighted binary log loss."""

    y = np.asarray(labels, dtype=np.float64)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    w = np.asarray(weights, dtype=np.float64)
    if len(y) == 0 or len(y) != len(p) or len(y) != len(w):
        raise LastDefenderV4Error("Weighted log loss requires aligned non-empty arrays.")
    if not np.isfinite(y).all() or not np.isfinite(p).all() or not np.isfinite(w).all():
        raise LastDefenderV4Error("Weighted log loss received non-finite values.")
    if float(w.sum()) <= 0.0:
        raise LastDefenderV4Error("Weighted log loss requires positive total weight.")
    losses = -(y * np.log(p) + (1.0 - y) * np.log1p(-p))
    return float(np.average(losses, weights=w))


def _fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    c_value: float,
    seed: int,
) -> Any:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=float(c_value),
                    solver="lbfgs",
                    max_iter=2_000,
                    random_state=int(seed),
                ),
            ),
        ]
    )
    model.fit(features, labels, model__sample_weight=weights)
    return model


def nested_group_predictions(
    frame: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    c_grid: Sequence[float],
    outer_folds: int,
    inner_folds: int,
    seed: int,
    label_column: str = "success_label",
    weight_column: str = "v4_overlap_weight",
    group_column: str = "series_id",
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Generate strictly series-out-of-fold weighted predictions with nested C selection."""

    from sklearn.model_selection import StratifiedGroupKFold

    missing = set((*feature_names, label_column, weight_column, group_column)).difference(
        frame.columns
    )
    if missing:
        raise LastDefenderV4Error(f"Missing model columns: {sorted(missing)}")
    x = frame.loc[:, list(feature_names)].to_numpy(dtype=np.float64)
    y = frame[label_column].to_numpy(dtype=np.int8)
    w = frame[weight_column].to_numpy(dtype=np.float64)
    groups = frame[group_column].astype(str).to_numpy()
    if set(np.unique(y)) != {0, 1}:
        raise LastDefenderV4Error("V4 outcome modeling requires both success classes.")
    if len(set(groups)) < int(outer_folds):
        raise LastDefenderV4Error("Too few official series for outer cross-fitting.")
    predictions = np.full(len(frame), np.nan, dtype=np.float64)
    assignments = np.full(len(frame), -1, dtype=np.int16)
    reports: list[dict[str, Any]] = []
    outer = StratifiedGroupKFold(
        n_splits=int(outer_folds), shuffle=True, random_state=int(seed)
    )
    for fold, (train_indices, holdout_indices) in enumerate(
        outer.split(x, y, groups), start=1
    ):
        train_groups = set(groups[train_indices])
        holdout_groups = set(groups[holdout_indices])
        if train_groups.intersection(holdout_groups):
            raise LastDefenderV4Error("An official series crossed outer folds.")
        inner_groups = groups[train_indices]
        if len(set(inner_groups)) < int(inner_folds):
            raise LastDefenderV4Error("Too few training series for inner cross-validation.")
        candidate_scores: list[tuple[float, float]] = []
        inner = StratifiedGroupKFold(
            n_splits=int(inner_folds),
            shuffle=True,
            random_state=int(seed) + fold,
        )
        inner_splits = list(
            inner.split(x[train_indices], y[train_indices], inner_groups)
        )
        for c_value in sorted(float(value) for value in c_grid):
            inner_predictions = np.full(len(train_indices), np.nan, dtype=np.float64)
            for inner_train, inner_holdout in inner_splits:
                model = _fit_logistic(
                    x[train_indices][inner_train],
                    y[train_indices][inner_train],
                    w[train_indices][inner_train],
                    c_value=c_value,
                    seed=int(seed) + fold,
                )
                inner_predictions[inner_holdout] = model.predict_proba(
                    x[train_indices][inner_holdout]
                )[:, 1]
            if not np.isfinite(inner_predictions).all():
                raise LastDefenderV4Error("Inner cross-validation left an unevaluated row.")
            score = weighted_binary_log_loss(
                y[train_indices], inner_predictions, w[train_indices]
            )
            candidate_scores.append((score, c_value))
        selected_score, selected_c = min(candidate_scores, key=lambda value: (value[0], value[1]))
        model = _fit_logistic(
            x[train_indices],
            y[train_indices],
            w[train_indices],
            c_value=selected_c,
            seed=int(seed) + fold,
        )
        predictions[holdout_indices] = model.predict_proba(x[holdout_indices])[:, 1]
        assignments[holdout_indices] = fold
        reports.append(
            {
                "fold": fold,
                "selected_c": selected_c,
                "selected_inner_log_loss": selected_score,
                "candidate_inner_log_loss": {
                    str(c_value): score for score, c_value in candidate_scores
                },
                "training_rows": int(len(train_indices)),
                "holdout_rows": int(len(holdout_indices)),
                "training_series": int(len(train_groups)),
                "holdout_series": int(len(holdout_groups)),
            }
        )
    if not np.isfinite(predictions).all() or bool((assignments < 0).any()):
        raise LastDefenderV4Error("Outer cross-fitting left an unevaluated row.")
    return predictions, assignments, reports


def relative_log_loss_reduction(baseline: float, challenger: float) -> float:
    """Return positive relative improvement for a lower-is-better log loss."""

    if baseline <= 0.0 or not math.isfinite(baseline) or not math.isfinite(challenger):
        raise LastDefenderV4Error("Relative log-loss reduction requires finite positive losses.")
    return float((baseline - challenger) / baseline)


def series_bootstrap_reduction(
    frame: pd.DataFrame,
    *,
    baseline_probabilities: Sequence[float],
    challenger_probabilities: Sequence[float],
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap a paired weighted log-loss reduction by official series."""

    y = frame["success_label"].to_numpy(dtype=np.float64)
    w = frame["v4_overlap_weight"].to_numpy(dtype=np.float64)
    baseline = np.clip(np.asarray(baseline_probabilities, dtype=np.float64), 1e-7, 1 - 1e-7)
    challenger = np.clip(
        np.asarray(challenger_probabilities, dtype=np.float64), 1e-7, 1 - 1e-7
    )
    baseline_row_loss = -(y * np.log(baseline) + (1.0 - y) * np.log1p(-baseline))
    challenger_row_loss = -(y * np.log(challenger) + (1.0 - y) * np.log1p(-challenger))
    series = frame["series_id"].astype(str).to_numpy()
    unique = np.asarray(sorted(set(series)), dtype=object)
    summaries = []
    for series_id in unique:
        selected = series == series_id
        summaries.append(
            (
                float(w[selected].sum()),
                float(np.sum(w[selected] * baseline_row_loss[selected])),
                float(np.sum(w[selected] * challenger_row_loss[selected])),
            )
        )
    values = np.asarray(summaries, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    reductions = np.empty(int(resamples), dtype=np.float64)
    for index in range(int(resamples)):
        selected = rng.integers(0, len(unique), size=len(unique))
        weight_sum, baseline_sum, challenger_sum = values[selected].sum(axis=0)
        baseline_loss = baseline_sum / weight_sum
        challenger_loss = challenger_sum / weight_sum
        reductions[index] = (baseline_loss - challenger_loss) / baseline_loss
    baseline_loss = weighted_binary_log_loss(y, baseline, w)
    challenger_loss = weighted_binary_log_loss(y, challenger, w)
    return {
        "resamples": int(resamples),
        "series": int(len(unique)),
        "point_estimate": relative_log_loss_reduction(baseline_loss, challenger_loss),
        "lower_95": float(np.quantile(reductions, 0.025, method="linear")),
        "upper_95": float(np.quantile(reductions, 0.975, method="linear")),
    }


def _fixed_fold_predictions(
    frame: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    assignments: np.ndarray,
    selected_c_by_fold: Mapping[int, float],
    seed: int,
) -> np.ndarray:
    x = frame.loc[:, list(feature_names)].to_numpy(dtype=np.float64)
    y = frame["success_label"].to_numpy(dtype=np.int8)
    w = frame["v4_overlap_weight"].to_numpy(dtype=np.float64)
    predictions = np.full(len(frame), np.nan, dtype=np.float64)
    for fold in sorted(int(value) for value in np.unique(assignments)):
        holdout = assignments == fold
        training = ~holdout
        model = _fit_logistic(
            x[training],
            y[training],
            w[training],
            c_value=float(selected_c_by_fold[fold]),
            seed=int(seed) + fold,
        )
        predictions[holdout] = model.predict_proba(x[holdout])[:, 1]
    if not np.isfinite(predictions).all():
        raise LastDefenderV4Error("Fixed-fold fitting left an unevaluated row.")
    return predictions


def profile_permutation_control(
    frame: pd.DataFrame,
    *,
    team_form_probabilities: Sequence[float],
    observed_full_probabilities: Sequence[float],
    assignments: np.ndarray,
    selected_c_by_fold: Mapping[int, float],
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    """Run the frozen within-series profile-row permutation control."""

    y = frame["success_label"].to_numpy(dtype=np.int8)
    w = frame["v4_overlap_weight"].to_numpy(dtype=np.float64)
    team_loss = weighted_binary_log_loss(y, team_form_probabilities, w)
    full_loss = weighted_binary_log_loss(y, observed_full_probabilities, w)
    observed = relative_log_loss_reduction(team_loss, full_loss)
    rng = np.random.default_rng(int(seed))
    profile_columns = list(PROFILE_SOURCE_FEATURE_NAMES)
    original_profiles = frame.loc[:, profile_columns].to_numpy(dtype=np.float64)
    series_groups = [
        group.index.to_numpy(dtype=np.int64)
        for _, group in frame.groupby("series_id", sort=True)
    ]
    null_reductions = np.empty(int(permutations), dtype=np.float64)
    for permutation in range(int(permutations)):
        shuffled = frame.copy()
        permuted_profiles = original_profiles.copy()
        for indices in series_groups:
            permuted_profiles[indices] = original_profiles[rng.permutation(indices)]
        shuffled.loc[:, profile_columns] = permuted_profiles
        shuffled = add_matchup_interactions(shuffled)
        probabilities = _fixed_fold_predictions(
            shuffled,
            feature_names=CONDITION_FEATURES["full_matchup"],
            assignments=assignments,
            selected_c_by_fold=selected_c_by_fold,
            seed=int(seed) + permutation + 1,
        )
        null_loss = weighted_binary_log_loss(y, probabilities, w)
        null_reductions[permutation] = relative_log_loss_reduction(team_loss, null_loss)
    exceedances = int(np.sum(null_reductions >= observed - 1e-15))
    return {
        "permutations": int(permutations),
        "observed_full_vs_team_form_reduction": observed,
        "null_mean": float(null_reductions.mean()),
        "null_95": [
            float(np.quantile(null_reductions, 0.025, method="linear")),
            float(np.quantile(null_reductions, 0.975, method="linear")),
        ],
        "exceedances": exceedances,
        "plus_one_one_sided_p": float((exceedances + 1) / (int(permutations) + 1)),
    }


def evaluate_outcome_models(
    frame: pd.DataFrame,
    *,
    model_config: Mapping[str, Any],
    uncertainty_config: Mapping[str, Any],
    gate_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the frozen Split 1 V4 model comparison and decision gates."""

    working = add_matchup_interactions(frame.reset_index(drop=True))
    condition_reports: dict[str, Any] = {}
    condition_probabilities: dict[str, np.ndarray] = {}
    common_assignments: np.ndarray | None = None
    for condition, features in CONDITION_FEATURES.items():
        probabilities, assignments, folds = nested_group_predictions(
            working,
            feature_names=features,
            c_grid=model_config["c_grid"],
            outer_folds=int(model_config["outer_folds"]),
            inner_folds=int(model_config["inner_folds"]),
            seed=int(model_config["seed"]),
        )
        if common_assignments is None:
            common_assignments = assignments
        elif not np.array_equal(common_assignments, assignments):
            raise LastDefenderV4Error("Matched model conditions received different outer folds.")
        loss = weighted_binary_log_loss(
            working["success_label"], probabilities, working["v4_overlap_weight"]
        )
        condition_probabilities[condition] = probabilities
        condition_reports[condition] = {
            "features": list(features),
            "feature_count": int(len(features)),
            "weighted_log_loss": loss,
            "folds": folds,
        }
        working[f"prediction_{condition}"] = probabilities
    assert common_assignments is not None
    working["v4_outcome_fold"] = common_assignments
    team_loss = float(condition_reports["team_form"]["weighted_log_loss"])
    additive_loss = float(condition_reports["additive_profiles"]["weighted_log_loss"])
    full_loss = float(condition_reports["full_matchup"]["weighted_log_loss"])
    full_vs_team = relative_log_loss_reduction(team_loss, full_loss)
    full_vs_additive = relative_log_loss_reduction(additive_loss, full_loss)
    point_gates = {
        "full_vs_team_form_minimum": full_vs_team
        >= float(gate_config["full_vs_team_form_relative_log_loss_reduction"]),
        "full_vs_additive_minimum": full_vs_additive
        >= float(gate_config["full_vs_additive_relative_log_loss_reduction"]),
    }

    bootstrap_team = series_bootstrap_reduction(
        working,
        baseline_probabilities=condition_probabilities["team_form"],
        challenger_probabilities=condition_probabilities["full_matchup"],
        resamples=int(uncertainty_config["bootstrap_resamples"]),
        seed=int(uncertainty_config["bootstrap_seed"]),
    )
    bootstrap_additive = series_bootstrap_reduction(
        working,
        baseline_probabilities=condition_probabilities["additive_profiles"],
        challenger_probabilities=condition_probabilities["full_matchup"],
        resamples=int(uncertainty_config["bootstrap_resamples"]),
        seed=int(uncertainty_config["bootstrap_seed"]) + 1,
    )
    regional: dict[str, Any] = {}
    for region in ("EU", "NA"):
        selected = working["region"].eq(region).to_numpy()
        region_y = working.loc[selected, "success_label"].to_numpy(dtype=np.int8)
        region_w = working.loc[selected, "v4_overlap_weight"].to_numpy(dtype=np.float64)
        region_team = weighted_binary_log_loss(
            region_y, condition_probabilities["team_form"][selected], region_w
        )
        region_full = weighted_binary_log_loss(
            region_y, condition_probabilities["full_matchup"][selected], region_w
        )
        regional[region] = {
            "rows": int(selected.sum()),
            "team_form_log_loss": region_team,
            "full_matchup_log_loss": region_full,
            "relative_reduction": relative_log_loss_reduction(region_team, region_full),
        }
    exact_candidates = working.loc[working["v4_exact_pair_id"].ne("")]
    exact_counts = exact_candidates["v4_exact_pair_id"].value_counts()
    complete_exact_ids = set(exact_counts.loc[exact_counts.eq(2)].index.astype(str))
    exact = working["v4_exact_pair_id"].astype(str).isin(complete_exact_ids).to_numpy()
    exact_y = working.loc[exact, "success_label"].to_numpy(dtype=np.int8)
    exact_weights = np.ones(int(exact.sum()), dtype=np.float64)
    exact_losses = {
        condition: weighted_binary_log_loss(
            exact_y, probabilities[exact], exact_weights
        )
        for condition, probabilities in condition_probabilities.items()
    }
    exact_sensitivity = {
        "rows": int(exact.sum()),
        "pairs": int(working.loc[exact, "v4_exact_pair_id"].nunique()),
        "log_loss": exact_losses,
        "full_vs_team_form_relative_reduction": relative_log_loss_reduction(
            exact_losses["team_form"], exact_losses["full_matchup"]
        ),
        "full_vs_additive_relative_reduction": relative_log_loss_reduction(
            exact_losses["additive_profiles"], exact_losses["full_matchup"]
        ),
    }

    permutation: dict[str, Any]
    if all(point_gates.values()):
        selected_c = {
            int(report["fold"]): float(report["selected_c"])
            for report in condition_reports["full_matchup"]["folds"]
        }
        permutation = profile_permutation_control(
            working,
            team_form_probabilities=condition_probabilities["team_form"],
            observed_full_probabilities=condition_probabilities["full_matchup"],
            assignments=common_assignments,
            selected_c_by_fold=selected_c,
            permutations=int(uncertainty_config["profile_permutations"]),
            seed=int(uncertainty_config["permutation_seed"]),
        )
    else:
        permutation = {
            "status": "not_run_point_estimate_gates_already_failed",
            "permutations": 0,
            "plus_one_one_sided_p": None,
        }

    lower_threshold = float(gate_config["official_series_bootstrap_lower_exclusive"])
    permutation_p = permutation.get("plus_one_one_sided_p")
    decisions = {
        **point_gates,
        "team_form_bootstrap_lower_positive": bootstrap_team["lower_95"] > lower_threshold,
        "additive_bootstrap_lower_positive": bootstrap_additive["lower_95"] > lower_threshold,
        "profile_permutation_p": permutation_p is not None
        and float(permutation_p) < float(gate_config["profile_permutation_p_exclusive"]),
        "positive_eu": regional["EU"]["relative_reduction"] > 0.0,
        "positive_na": regional["NA"]["relative_reduction"] > 0.0,
        "positive_exact_team_form": exact_sensitivity[
            "full_vs_team_form_relative_reduction"
        ]
        > 0.0,
        "positive_exact_additive": exact_sensitivity[
            "full_vs_additive_relative_reduction"
        ]
        > 0.0,
    }
    result = {
        "rows": int(len(working)),
        "series": int(working["series_id"].nunique()),
        "conditions": condition_reports,
        "primary_comparisons": {
            "full_vs_team_form_relative_reduction": full_vs_team,
            "full_vs_additive_relative_reduction": full_vs_additive,
        },
        "bootstrap": {
            "full_vs_team_form": bootstrap_team,
            "full_vs_additive": bootstrap_additive,
        },
        "regional": regional,
        "exact_stratum_sensitivity": exact_sensitivity,
        "profile_permutation": permutation,
        "gates": decisions,
        "all_gates_pass": all(decisions.values()),
    }
    return working, result
