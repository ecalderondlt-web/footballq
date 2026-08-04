"""Outcome-blind RLCS V3 last-defender opportunity and common-support audit."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from footballq.data.rlcs_player_profiles import PROFILE_FEATURES
from footballq.data.rlcs_replay import IdentityObservation
from footballq.data.rlcs_touch_windows import (
    ContextSelection,
    TouchWindowError,
    build_state_tensor,
    extract_touches,
    relative_player_order,
)

GOAL_CENTER = np.asarray([0.0, 5120.0], dtype=np.float64)
PROFILE_INDEX = {name: index for index, name in enumerate(PROFILE_FEATURES)}

TEAM_FORM_FEATURE_NAMES = (
    "team_form_actor_win_rate",
    "team_form_opponent_win_rate",
    "team_form_actor_goal_diff",
    "team_form_opponent_goal_diff",
    "team_form_actor_log_prior_games",
    "team_form_opponent_log_prior_games",
)

MATCHING_FEATURE_NAMES = (
    "ball_pos_x",
    "ball_pos_y",
    "ball_pos_z",
    "ball_vel_x",
    "ball_vel_y",
    "ball_vel_z",
    "actor_pos_x",
    "actor_pos_y",
    "actor_pos_z",
    "actor_vel_x",
    "actor_vel_y",
    "actor_vel_z",
    "actor_forward_x",
    "actor_forward_y",
    "actor_forward_z",
    "actor_boost",
    "defender_pos_x",
    "defender_pos_y",
    "defender_pos_z",
    "defender_vel_x",
    "defender_vel_y",
    "defender_vel_z",
    "defender_forward_x",
    "defender_forward_y",
    "defender_forward_z",
    "defender_boost",
    "actor_defender_distance",
    "actor_intercept_time",
    "defender_intercept_time",
    "ball_goal_distance",
    "teammate_support_distance",
    "second_defender_recovery_distance",
    "score_diff_actor",
    "seconds_remaining",
    "overtime",
    *TEAM_FORM_FEATURE_NAMES,
)

PROFILE_TRAIT_NAMES = (
    "actor_carry_speed",
    "actor_boost_economy",
    "actor_take_on_control_frequency",
    "defender_goalside_recovery",
    "defender_challenge_win",
    "defender_boost_economy",
    "defender_turnover_pressure_proxy",
    "actor_attack_composite",
    "defender_resistance_composite",
    "matchup_mismatch",
)


class LastDefenderV3Error(ValueError):
    """Raised when the frozen V3 audit cannot be evaluated safely."""


@dataclass(frozen=True)
class GeometryThresholds:
    """The four train-geometry thresholds frozen before inventory construction."""

    corridor_half_width: float
    last_defender_forward_distance: float
    immediate_intervention_range: float
    teammate_overload_range: float

    def to_dict(self) -> dict[str, float]:
        return {
            "corridor_half_width": float(self.corridor_half_width),
            "last_defender_forward_distance": float(
                self.last_defender_forward_distance
            ),
            "immediate_intervention_range": float(self.immediate_intervention_range),
            "teammate_overload_range": float(self.teammate_overload_range),
        }


def _finite_vector(row: Mapping[str, Any], prefix: str, stem: str) -> np.ndarray:
    values = np.asarray([row.get(f"{prefix}_{stem}_{axis}") for axis in "xyz"], dtype=float)
    if not np.isfinite(values).all():
        raise TouchWindowError(f"Missing {stem} vector for {prefix}.")
    return values


def _ball_vector(row: Mapping[str, Any], stem: str) -> np.ndarray:
    values = np.asarray([row.get(f"ball_{stem}_{axis}") for axis in "xyz"], dtype=float)
    if not np.isfinite(values).all():
        raise TouchWindowError(f"Missing ball {stem} vector.")
    return values


def _actor_orient(values: np.ndarray, actor_team: str) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    if str(actor_team).casefold() == "orange":
        output[1] *= -1.0
    return output


def _rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _forward_vector(row: Mapping[str, Any], prefix: str, actor_team: str) -> np.ndarray:
    angles = np.asarray([row.get(f"{prefix}_rot_{axis}") for axis in "xyz"], dtype=float)
    if not np.isfinite(angles).all():
        raise TouchWindowError(f"Missing orientation for {prefix}.")
    return _actor_orient(_rotation_matrix(*angles)[:, 0], actor_team)


def _planar_min_distance(
    position: np.ndarray, actor_position: np.ndarray, ball_position: np.ndarray
) -> float:
    return float(
        min(
            np.linalg.norm(position[:2] - actor_position[:2]),
            np.linalg.norm(position[:2] - ball_position[:2]),
        )
    )


def _goal_side_geometry(ball_xy: np.ndarray, player_xy: np.ndarray) -> tuple[float, float] | None:
    path = GOAL_CENTER - ball_xy
    squared = float(path @ path)
    if squared <= 1e-8:
        return None
    projection = float((player_xy - ball_xy) @ path / squared)
    if not 0.0 < projection <= 1.0:
        return None
    projected = ball_xy + projection * path
    lateral = float(np.linalg.norm(player_xy - projected))
    forward = float(projection * math.sqrt(squared))
    return lateral, forward


def _boundary_frames(events: pd.DataFrame) -> np.ndarray:
    event_type = events["event_type"].astype(str).str.casefold()
    boundary = event_type.isin({"goal", "kickoff"})
    if "official_goal" in events:
        boundary |= events["official_goal"].fillna(False).astype(bool)
    values = pd.to_numeric(
        events.loc[boundary, "observed_frame_number"], errors="coerce"
    ).dropna()
    return np.sort(values.to_numpy(dtype=np.int64))


def _select_context(
    frame_times: np.ndarray,
    frame_ids: np.ndarray,
    *,
    touch_frame_idx: int,
    touch_time_s: float,
    fps: float,
    context_seconds: float,
    maximum_frame_lag_seconds: float,
) -> ContextSelection:
    steps = int(round(float(fps) * float(context_seconds)))
    if steps != 20:
        raise TouchWindowError(f"Frozen V3 context requires 20 steps, got {steps}.")
    requested = np.round(
        float(touch_time_s) - np.arange(steps - 1, -1, -1, dtype=np.float64) / fps,
        6,
    )
    positions = np.searchsorted(frame_times, requested, side="right") - 1
    if np.any(positions < 0):
        raise TouchWindowError("Insufficient past context before current contact.")
    observed = frame_times[positions]
    selected_frames = frame_ids[positions]
    if np.any(observed > requested + 1e-7) or np.any(selected_frames > int(touch_frame_idx)):
        raise TouchWindowError("Future frame entered the V3 context.")
    if np.any(requested - observed > float(maximum_frame_lag_seconds)):
        raise TouchWindowError("V3 context crosses a parser gap.")
    if len(set(int(value) for value in positions)) != steps:
        raise TouchWindowError("V3 context grid reused a parser frame.")
    return ContextSelection(
        row_indices=tuple(int(value) for value in positions),
        requested_times=tuple(float(value) for value in requested),
        observed_times=tuple(float(value) for value in observed),
    )


def _base_geometry(
    frames: pd.DataFrame,
    events: pd.DataFrame,
    *,
    observations: Sequence[IdentityObservation],
    roster_ids: Mapping[str, str],
    fps: float,
    context_seconds: float,
    maximum_frame_lag_seconds: float,
) -> list[dict[str, Any]]:
    """Return only present/past geometry for contacts satisfying the frozen base."""

    if len(roster_ids) != 6:
        raise LastDefenderV3Error("V3 requires an exact six-player resolved roster.")
    frame_times = np.round(
        frames["game_time_s_precise"].to_numpy(dtype=np.float64), 6
    )
    frame_ids = frames["observed_frame_number"].to_numpy(dtype=np.int64)
    if np.any(np.diff(frame_times) < 0) or np.any(np.diff(frame_ids) <= 0):
        raise LastDefenderV3Error("V3 frames must be strictly ordered.")
    boundaries = _boundary_frames(events)
    touches = extract_touches(events, observations, roster_ids, scores_repaired=True)
    by_prefix = {item.prefix: item for item in observations}
    output: list[dict[str, Any]] = []

    for touch in touches:
        row_index = int(np.searchsorted(frame_ids, touch.frame_idx, side="right") - 1)
        if row_index < 0 or touch.player_prefix not in by_prefix:
            continue
        row = frames.iloc[row_index].to_dict()
        try:
            ball_position = _actor_orient(_ball_vector(row, "pos"), touch.team)
            actor_position = _actor_orient(
                _finite_vector(row, touch.player_prefix, "pos"), touch.team
            )
        except TouchWindowError:
            continue
        if not (ball_position[1] > 0.0 and actor_position[1] > 0.0):
            continue

        teammate_prefixes = sorted(
            item.prefix
            for item in observations
            if item.team == touch.team and item.prefix != touch.player_prefix
        )
        opponent_prefixes = sorted(
            item.prefix for item in observations if item.team != touch.team
        )
        if len(teammate_prefixes) != 2 or len(opponent_prefixes) != 3:
            continue
        try:
            team_distances = {
                prefix: float(
                    np.linalg.norm(
                        _actor_orient(_finite_vector(row, prefix, "pos"), touch.team)
                        - ball_position
                    )
                )
                for prefix in [touch.player_prefix, *teammate_prefixes]
            }
        except TouchWindowError:
            continue
        closest_team_prefix = min(team_distances, key=lambda key: (team_distances[key], key))
        if closest_team_prefix != touch.player_prefix:
            continue

        try:
            selection = _select_context(
                frame_times,
                frame_ids,
                touch_frame_idx=touch.frame_idx,
                touch_time_s=touch.game_time_s,
                fps=fps,
                context_seconds=context_seconds,
                maximum_frame_lag_seconds=maximum_frame_lag_seconds,
            )
            selected_indices = np.asarray(selection.row_indices, dtype=np.int64)
            if "stint_number" in frames:
                stints = pd.to_numeric(
                    frames.iloc[selected_indices]["stint_number"], errors="coerce"
                ).dropna()
                if stints.nunique() > 1:
                    continue
            context_start = int(frame_ids[selected_indices[0]])
            if bool(((boundaries >= context_start) & (boundaries <= touch.frame_idx)).any()):
                continue
            order = relative_player_order(
                row, actor_prefix=touch.player_prefix, observations=observations
            )
            # This call is deliberately retained even though its tensor is not persisted: it
            # enforces complete, finite ball and six-car state throughout the past-only window.
            state, state_mask = build_state_tensor(
                frames, selection, car_order=order, actor_team=touch.team
            )
            if not np.isfinite(state).all() or not bool(state_mask.all()):
                continue
            player_positions = {
                prefix: _actor_orient(_finite_vector(row, prefix, "pos"), touch.team)
                for prefix in order
            }
        except TouchWindowError:
            continue

        goal_side = []
        for prefix in opponent_prefixes:
            geometry = _goal_side_geometry(ball_position[:2], player_positions[prefix][:2])
            if geometry is not None:
                goal_side.append((prefix, geometry[0], geometry[1]))
        goal_side.sort(key=lambda value: (value[1], value[2], value[0]))
        teammate_ranges = {
            prefix: _planar_min_distance(
                player_positions[prefix], actor_position, ball_position
            )
            for prefix in teammate_prefixes
        }
        raw_stint = row.get("stint_number", 0)
        stint = int(raw_stint) if pd.notna(raw_stint) else 0
        output.append(
            {
                "touch": touch,
                "row": row,
                "order": order,
                "player_positions": player_positions,
                "ball_position": ball_position,
                "actor_position": actor_position,
                "teammate_prefixes": teammate_prefixes,
                "opponent_prefixes": opponent_prefixes,
                "goal_side": goal_side,
                "teammate_ranges": teammate_ranges,
                "stint": stint,
            }
        )
    return output


def extract_replay_calibration(
    frames: pd.DataFrame,
    events: pd.DataFrame,
    *,
    observations: Sequence[IdentityObservation],
    roster_ids: Mapping[str, str],
    fps: float = 10.0,
    context_seconds: float = 2.0,
    maximum_frame_lag_seconds: float = 0.15,
) -> dict[str, Any]:
    """Extract only the four permitted current-geometry calibration statistics."""

    base = _base_geometry(
        frames,
        events,
        observations=observations,
        roster_ids=roster_ids,
        fps=fps,
        context_seconds=context_seconds,
        maximum_frame_lag_seconds=maximum_frame_lag_seconds,
    )
    samples: dict[str, list[float]] = {
        "corridor_half_width": [],
        "last_defender_forward_distance": [],
        "immediate_intervention_range": [],
        "teammate_overload_range": [],
    }
    for candidate in base:
        samples["teammate_overload_range"].append(
            min(candidate["teammate_ranges"].values())
        )
        if not candidate["goal_side"]:
            continue
        selected_prefix, lateral, forward = candidate["goal_side"][0]
        samples["corridor_half_width"].append(float(lateral))
        samples["last_defender_forward_distance"].append(float(forward))
        remaining = [
            prefix
            for prefix in candidate["opponent_prefixes"]
            if prefix != selected_prefix
        ]
        ranges = [
            _planar_min_distance(
                candidate["player_positions"][prefix],
                candidate["actor_position"],
                candidate["ball_position"],
            )
            for prefix in remaining
        ]
        if ranges:
            samples["immediate_intervention_range"].append(min(ranges))
    return {"base_contacts": len(base), "samples": samples}


def calibrate_geometry_thresholds(
    samples: Mapping[str, Sequence[float]], config: Mapping[str, Any]
) -> tuple[GeometryThresholds, dict[str, Any]]:
    """Apply the predeclared quantiles and clips to train-only geometry."""

    minimum = int(config["minimum_finite_observations"])
    values: dict[str, float] = {}
    counts: dict[str, int] = {}
    raw_quantiles: dict[str, float] = {}
    for name in (
        "corridor_half_width",
        "last_defender_forward_distance",
        "immediate_intervention_range",
        "teammate_overload_range",
    ):
        array = np.asarray(samples.get(name, ()), dtype=np.float64)
        array = array[np.isfinite(array)]
        counts[name] = int(len(array))
        if len(array) < minimum:
            raise LastDefenderV3Error(
                f"{name} has {len(array)} finite calibration rows; {minimum} required."
            )
        specification = config[name]
        raw = float(
            np.quantile(
                array,
                float(specification["quantile"]),
                method=str(config.get("quantile_method", "linear")),
            )
        )
        lower, upper = (float(value) for value in specification["clip"])
        raw_quantiles[name] = raw
        values[name] = float(np.clip(raw, lower, upper))
    thresholds = GeometryThresholds(**values)
    return thresholds, {
        "finite_observation_counts": counts,
        "raw_quantiles": raw_quantiles,
        "clipped_thresholds": thresholds.to_dict(),
    }


def _standardized_profile(
    snapshot: Mapping[str, Any], priors: Mapping[str, Any], clip: Sequence[float]
) -> np.ndarray:
    profile = np.asarray(snapshot["profile"], dtype=np.float64)
    mean = np.asarray(priors["population_mean"], dtype=np.float64)
    scale = np.asarray(priors["uncertainty_scale"], dtype=np.float64)
    if profile.shape != mean.shape or profile.shape != scale.shape:
        raise LastDefenderV3Error("V3 profile/prior dimensions do not match.")
    finite = (
        np.isfinite(profile).all()
        and np.isfinite(mean).all()
        and np.isfinite(scale).all()
    )
    if not finite:
        raise LastDefenderV3Error("V3 profile/prior values must be finite.")
    scale = np.where(scale > 1e-8, scale, 1.0)
    lower, upper = (float(value) for value in clip)
    return np.clip((profile - mean) / scale, lower, upper)


def opportunity_profile_traits(
    actor_snapshot: Mapping[str, Any],
    defender_snapshot: Mapping[str, Any],
    priors: Mapping[str, Any],
    *,
    standardized_clip: Sequence[float] = (-5.0, 5.0),
) -> dict[str, float]:
    """Compute the frozen actor/defender composites without action or outcome data."""

    actor = _standardized_profile(actor_snapshot, priors, standardized_clip)
    defender = _standardized_profile(defender_snapshot, priors, standardized_clip)

    def value(vector: np.ndarray, name: str) -> float:
        return float(vector[PROFILE_INDEX[name]])

    actor_carry = value(actor, "mean_speed")
    actor_boost = float(
        np.mean([value(actor, "mean_boost"), -value(actor, "low_boost_fraction")])
    )
    actor_control = float(
        np.mean(
            [
                value(actor, "ground_dribble_per_touch"),
                value(actor, "aerial_control_per_touch"),
                value(actor, "flick_per_touch"),
            ]
        )
    )
    actor_attack = float(np.mean([actor_carry, actor_boost, actor_control]))

    defender_recovery = value(defender, "goalside_recovery_speed")
    defender_challenge = value(defender, "challenge_win_fraction")
    defender_boost = float(
        np.mean(
            [value(defender, "mean_boost"), -value(defender, "low_boost_fraction")]
        )
    )
    defender_pressure = float(
        np.mean(
            [
                value(defender, "retrieval_per_touch"),
                value(defender, "nearest_to_ball_fraction"),
                value(defender, "challenge_win_fraction"),
                -value(defender, "turnover_per_touch"),
            ]
        )
    )
    defender_resistance = float(
        np.mean(
            [
                defender_recovery,
                defender_challenge,
                defender_boost,
                defender_pressure,
            ]
        )
    )
    return {
        "actor_carry_speed": actor_carry,
        "actor_boost_economy": actor_boost,
        "actor_take_on_control_frequency": actor_control,
        "defender_goalside_recovery": defender_recovery,
        "defender_challenge_win": defender_challenge,
        "defender_boost_economy": defender_boost,
        "defender_turnover_pressure_proxy": defender_pressure,
        "actor_attack_composite": actor_attack,
        "defender_resistance_composite": defender_resistance,
        "matchup_mismatch": actor_attack - defender_resistance,
    }


def _team_form(
    order: Sequence[str],
    roster_ids: Mapping[str, str],
    snapshots: Mapping[str, Mapping[str, Any]],
    *,
    query_time: Any,
) -> dict[str, float]:
    if query_time is None or pd.isna(query_time):
        raise LastDefenderV3Error("V3 team form requires a chronology timestamp.")
    query_timestamp = pd.Timestamp(query_time)
    if query_timestamp.tzinfo is None:
        query_timestamp = query_timestamp.tz_localize("UTC")
    else:
        query_timestamp = query_timestamp.tz_convert("UTC")
    values = []
    for prefix in order:
        player_id = str(roster_ids[prefix])
        if player_id not in snapshots:
            raise LastDefenderV3Error(f"Missing chronology-safe snapshot for {player_id}.")
        snapshot = snapshots[player_id]
        prior_games = int(snapshot.get("n_prior_games", 0))
        raw_latest_prior = snapshot.get("latest_prior_time_utc")
        latest_prior = (
            pd.NaT
            if raw_latest_prior is None or pd.isna(raw_latest_prior)
            else pd.Timestamp(raw_latest_prior)
        )
        if not pd.isna(latest_prior):
            latest_prior = (
                latest_prior.tz_localize("UTC")
                if latest_prior.tzinfo is None
                else latest_prior.tz_convert("UTC")
            )
        if prior_games > 0 and (pd.isna(latest_prior) or latest_prior >= query_timestamp):
            raise LastDefenderV3Error(
                f"Profile snapshot is not strictly prior to the query for {player_id}."
            )
        values.append(
            [
                float(snapshot.get("prior_win_rate", 0.5)),
                float(snapshot.get("prior_goal_diff", 0.0)),
                math.log1p(float(prior_games)),
            ]
        )
    array = np.asarray(values, dtype=np.float64)
    output = [
        array[:3, 0].mean(),
        array[3:, 0].mean(),
        array[:3, 1].mean(),
        array[3:, 1].mean(),
        array[:3, 2].mean(),
        array[3:, 2].mean(),
    ]
    return dict(zip(TEAM_FORM_FEATURE_NAMES, (float(value) for value in output), strict=True))


def _roster_digest(player_ids: Sequence[str]) -> str:
    payload = "\n".join(sorted(str(value) for value in player_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _matching_state(
    candidate: Mapping[str, Any],
    *,
    defender_prefix: str,
    roster_ids: Mapping[str, str],
    snapshots: Mapping[str, Mapping[str, Any]],
    teammate_support_distance: float,
    second_defender_recovery_distance: float,
) -> dict[str, float]:
    row = candidate["row"]
    touch = candidate["touch"]
    actor_prefix = touch.player_prefix
    team = touch.team
    ball_position = candidate["ball_position"]
    ball_velocity = _actor_orient(_ball_vector(row, "vel"), team)
    actor_position = candidate["actor_position"]
    actor_velocity = _actor_orient(_finite_vector(row, actor_prefix, "vel"), team)
    defender_position = candidate["player_positions"][defender_prefix]
    defender_velocity = _actor_orient(_finite_vector(row, defender_prefix, "vel"), team)
    actor_forward = _forward_vector(row, actor_prefix, team)
    defender_forward = _forward_vector(row, defender_prefix, team)

    def intercept(position: np.ndarray, velocity: np.ndarray) -> float:
        distance = float(np.linalg.norm(position - ball_position))
        return float(min(distance / max(float(np.linalg.norm(velocity)), 500.0), 5.0))

    output: dict[str, float] = {}
    for stem, vector in (
        ("ball_pos", ball_position),
        ("ball_vel", ball_velocity),
        ("actor_pos", actor_position),
        ("actor_vel", actor_velocity),
        ("actor_forward", actor_forward),
        ("defender_pos", defender_position),
        ("defender_vel", defender_velocity),
        ("defender_forward", defender_forward),
    ):
        for axis, value in zip("xyz", vector, strict=True):
            output[f"{stem}_{axis}"] = float(value)
    output.update(
        {
            "actor_boost": float(row.get(f"{actor_prefix}_boost", 0.0) or 0.0) / 100.0,
            "defender_boost": float(row.get(f"{defender_prefix}_boost", 0.0) or 0.0)
            / 100.0,
            "actor_defender_distance": float(
                np.linalg.norm(actor_position[:2] - defender_position[:2])
            ),
            "actor_intercept_time": intercept(actor_position, actor_velocity),
            "defender_intercept_time": intercept(defender_position, defender_velocity),
            "ball_goal_distance": float(np.linalg.norm(ball_position[:2] - GOAL_CENTER)),
            "teammate_support_distance": float(teammate_support_distance),
            "second_defender_recovery_distance": float(
                second_defender_recovery_distance
            ),
            "score_diff_actor": float(
                touch.blue_score - touch.orange_score
                if team == "blue"
                else touch.orange_score - touch.blue_score
            ),
            "seconds_remaining": float(max(300.0 - touch.game_time_s, 0.0)),
            "overtime": float(touch.game_time_s > 300.0),
        }
    )
    output.update(
        _team_form(
            candidate["order"],
            roster_ids,
            snapshots,
            query_time=candidate["inventory_event_time_utc"],
        )
    )
    if set(output) != set(MATCHING_FEATURE_NAMES) or not np.isfinite(
        np.asarray(list(output.values()), dtype=np.float64)
    ).all():
        raise LastDefenderV3Error("V3 matching-state feature construction failed closed.")
    return output


def build_replay_opportunities(
    frames: pd.DataFrame,
    events: pd.DataFrame,
    *,
    replay_id: str,
    inventory: Mapping[str, Any],
    stage: str,
    observations: Sequence[IdentityObservation],
    roster_ids: Mapping[str, str],
    snapshots: Mapping[str, Mapping[str, Any]],
    priors: Mapping[str, Any],
    eligible_player_ids: set[str],
    thresholds: GeometryThresholds,
    minimum_prior_games_actor: int = 15,
    minimum_prior_games_defender: int = 15,
    standardized_profile_clip: Sequence[float] = (-5.0, 5.0),
    fps: float = 10.0,
    context_seconds: float = 2.0,
    maximum_frame_lag_seconds: float = 0.15,
) -> list[dict[str, Any]]:
    """Build outcome-free V3 opportunities for an allowed Split 1 stage."""

    if stage not in {"train", "internal_development"}:
        raise PermissionError(f"V3 Stage 0 may not open stage {stage!r}.")
    base = _base_geometry(
        frames,
        events,
        observations=observations,
        roster_ids=roster_ids,
        fps=fps,
        context_seconds=context_seconds,
        maximum_frame_lag_seconds=maximum_frame_lag_seconds,
    )
    rows: list[dict[str, Any]] = []
    for candidate in base:
        candidate["inventory_event_time_utc"] = inventory.get("event_time_utc")
        qualifying = [
            value
            for value in candidate["goal_side"]
            if value[1] <= thresholds.corridor_half_width
            and value[2] <= thresholds.last_defender_forward_distance
        ]
        if len(qualifying) != 1:
            continue
        defender_prefix = qualifying[0][0]
        remaining_opponents = [
            prefix
            for prefix in candidate["opponent_prefixes"]
            if prefix != defender_prefix
        ]
        opponent_ranges = {
            prefix: _planar_min_distance(
                candidate["player_positions"][prefix],
                candidate["actor_position"],
                candidate["ball_position"],
            )
            for prefix in remaining_opponents
        }
        if not opponent_ranges or any(
            value <= thresholds.immediate_intervention_range
            for value in opponent_ranges.values()
        ):
            continue
        if any(
            value <= thresholds.teammate_overload_range
            for value in candidate["teammate_ranges"].values()
        ):
            continue

        touch = candidate["touch"]
        actor_id = str(roster_ids[touch.player_prefix])
        defender_id = str(roster_ids[defender_prefix])
        if actor_id not in eligible_player_ids or defender_id not in eligible_player_ids:
            continue
        if actor_id not in snapshots or defender_id not in snapshots:
            continue
        actor_snapshot = snapshots[actor_id]
        defender_snapshot = snapshots[defender_id]
        if int(actor_snapshot.get("n_prior_games", 0)) < int(minimum_prior_games_actor):
            continue
        if int(defender_snapshot.get("n_prior_games", 0)) < int(
            minimum_prior_games_defender
        ):
            continue
        try:
            state = _matching_state(
                candidate,
                defender_prefix=defender_prefix,
                roster_ids=roster_ids,
                snapshots=snapshots,
                teammate_support_distance=min(candidate["teammate_ranges"].values()),
                second_defender_recovery_distance=min(opponent_ranges.values()),
            )
            traits = opportunity_profile_traits(
                actor_snapshot,
                defender_snapshot,
                priors,
                standardized_clip=standardized_profile_clip,
            )
        except (TouchWindowError, LastDefenderV3Error):
            continue

        actor_team_prefixes = [
            prefix for prefix in candidate["order"] if prefix.startswith(f"{touch.team}_")
        ]
        opponent_team = "orange" if touch.team == "blue" else "blue"
        opponent_team_prefixes = [
            prefix for prefix in candidate["order"] if prefix.startswith(f"{opponent_team}_")
        ]
        sample_id = (
            f"{replay_id}:stint_{candidate['stint']}:"
            f"touch_{touch.frame_idx}:last_defender_v3"
        )
        rows.append(
            {
                "sample_id": sample_id,
                "replay_id": str(replay_id),
                "series_id": str(inventory.get("series_id") or ""),
                "region": str(inventory.get("region") or "").upper(),
                "event_time_utc": inventory.get("event_time_utc"),
                "v3_stage": str(stage),
                "stint_number": int(candidate["stint"]),
                "frame_idx": int(touch.frame_idx),
                "game_time_s": float(touch.game_time_s),
                "actor_player_id": actor_id,
                "defender_player_id": defender_id,
                "actor_team_roster_sha256": _roster_digest(
                    [roster_ids[prefix] for prefix in actor_team_prefixes]
                ),
                "defender_team_roster_sha256": _roster_digest(
                    [roster_ids[prefix] for prefix in opponent_team_prefixes]
                ),
                "actor_prior_games": int(actor_snapshot["n_prior_games"]),
                "defender_prior_games": int(defender_snapshot["n_prior_games"]),
                **state,
                **traits,
            }
        )
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise LastDefenderV3Error(f"Duplicate V3 sample IDs in replay {replay_id}.")
    return rows


def assign_favorable_matchup(frame: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Freeze the exposure at the accepted-train mismatch median."""

    if frame.empty or "matchup_mismatch" not in frame:
        raise LastDefenderV3Error("Cannot assign matchup exposure to an empty inventory.")
    train = frame.loc[frame["v3_stage"] == "train", "matchup_mismatch"]
    if train.empty or not np.isfinite(train.to_numpy(dtype=np.float64)).all():
        raise LastDefenderV3Error("V3 train mismatch values are missing or non-finite.")
    threshold = float(np.median(train.to_numpy(dtype=np.float64)))
    output = frame.copy()
    output["favorable_matchup"] = (
        output["matchup_mismatch"].to_numpy(dtype=np.float64) >= threshold
    ).astype(np.int8)
    return output, threshold


def opportunity_volume_audit(
    frame: pd.DataFrame, gates: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate the frozen count, regional, and concentration gates."""

    rows = int(len(frame))
    actors = int(frame["actor_player_id"].nunique()) if rows else 0
    defenders = int(frame["defender_player_id"].nunique()) if rows else 0
    region_counts = {
        region: int((frame["region"] == region).sum()) if rows else 0
        for region in ("EU", "NA")
    }
    actor_share = (
        float(frame["actor_player_id"].value_counts(normalize=True).max()) if rows else 1.0
    )
    defender_share = (
        float(frame["defender_player_id"].value_counts(normalize=True).max())
        if rows
        else 1.0
    )
    decisions = {
        "minimum_opportunities": rows >= int(gates["minimum_opportunities"]),
        "minimum_distinct_actors": actors >= int(gates["minimum_distinct_actors"]),
        "minimum_distinct_defenders": defenders
        >= int(gates["minimum_distinct_defenders"]),
        "minimum_eu_opportunities": region_counts["EU"]
        >= int(gates["minimum_eu_opportunities"]),
        "minimum_na_opportunities": region_counts["NA"]
        >= int(gates["minimum_na_opportunities"]),
        "maximum_actor_share": actor_share <= float(gates["maximum_actor_share"]),
        "maximum_defender_share": defender_share
        <= float(gates["maximum_defender_share"]),
    }
    return {
        "counts": {
            "opportunities": rows,
            "distinct_actors": actors,
            "distinct_defenders": defenders,
            "by_region": region_counts,
        },
        "concentration": {
            "maximum_actor_share": actor_share,
            "maximum_defender_share": defender_share,
        },
        "gates": decisions,
        "all_gates_pass": all(decisions.values()),
    }


def cross_fitted_propensity(
    frame: pd.DataFrame,
    *,
    feature_names: Sequence[str] = MATCHING_FEATURE_NAMES,
    folds: int = 5,
    seed: int = 20_260_803,
    c_value: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Estimate strictly series-out-of-fold favorable-matchup propensities."""

    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    missing = set(feature_names).difference(frame.columns)
    if missing:
        raise LastDefenderV3Error(f"Missing common-support features: {sorted(missing)}")
    labels = frame["favorable_matchup"].to_numpy(dtype=np.int8)
    groups = frame["series_id"].astype(str).to_numpy()
    if set(np.unique(labels)) != {0, 1}:
        raise LastDefenderV3Error("Propensity fitting requires both matchup exposures.")
    if len(set(groups)) < int(folds):
        raise LastDefenderV3Error("Too few official series for frozen propensity folds.")
    features = frame.loc[:, list(feature_names)].to_numpy(dtype=np.float64)
    predictions = np.full(len(frame), np.nan, dtype=np.float64)
    assignments = np.full(len(frame), -1, dtype=np.int16)
    fold_reports: list[dict[str, Any]] = []
    splitter = StratifiedGroupKFold(
        n_splits=int(folds), shuffle=True, random_state=int(seed)
    )
    for fold, (train_indices, holdout_indices) in enumerate(
        splitter.split(features, labels, groups), start=1
    ):
        train_groups = set(groups[train_indices])
        holdout_groups = set(groups[holdout_indices])
        if train_groups.intersection(holdout_groups):
            raise LastDefenderV3Error("An official series crossed propensity folds.")
        if len(np.unique(labels[train_indices])) != 2:
            raise LastDefenderV3Error(f"Propensity fold {fold} has one training class.")
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
        model.fit(features[train_indices], labels[train_indices])
        predictions[holdout_indices] = model.predict_proba(features[holdout_indices])[:, 1]
        assignments[holdout_indices] = fold
        fold_reports.append(
            {
                "fold": fold,
                "training_rows": int(len(train_indices)),
                "holdout_rows": int(len(holdout_indices)),
                "training_series": int(len(train_groups)),
                "holdout_series": int(len(holdout_groups)),
                "training_favorable_fraction": float(labels[train_indices].mean()),
                "holdout_favorable_fraction": float(labels[holdout_indices].mean()),
            }
        )
    if not np.isfinite(predictions).all() or bool((assignments < 0).any()):
        raise LastDefenderV3Error("Propensity cross-fitting left an unevaluated row.")
    return np.clip(predictions, 1e-6, 1.0 - 1e-6), assignments, fold_reports


def _pooled_standard_deviation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) + len(right) <= 2:
        return 0.0
    numerator = max(len(left) - 1, 0) * float(np.var(left, ddof=1 if len(left) > 1 else 0))
    numerator += max(len(right) - 1, 0) * float(
        np.var(right, ddof=1 if len(right) > 1 else 0)
    )
    denominator = max(len(left) + len(right) - 2, 1)
    return float(math.sqrt(max(numerator / denominator, 0.0)))


def _deterministic_pairs(
    frame: pd.DataFrame,
    standardized_features: np.ndarray,
    *,
    caliper_scale: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    logits = np.log(frame["propensity"].to_numpy(dtype=np.float64)) - np.log1p(
        -frame["propensity"].to_numpy(dtype=np.float64)
    )
    labels = frame["favorable_matchup"].to_numpy(dtype=np.int8)
    samples = frame["sample_id"].astype(str).to_numpy()
    regions = frame["region"].astype(str).to_numpy()
    pairs: list[dict[str, Any]] = []
    region_reports: dict[str, Any] = {}
    set_number = 0

    for region in ("EU", "NA"):
        favorable = np.flatnonzero((regions == region) & (labels == 1))
        unfavorable = np.flatnonzero((regions == region) & (labels == 0))
        pooled_sd = _pooled_standard_deviation(logits[favorable], logits[unfavorable])
        caliper = float(caliper_scale) * pooled_sd
        initial_counts = {
            int(index): int(
                np.sum(np.abs(logits[unfavorable] - logits[index]) <= caliper + 1e-12)
            )
            for index in favorable
        }
        order = sorted(
            (int(index) for index in favorable),
            key=lambda index: (initial_counts[index], samples[index]),
        )
        available = set(int(index) for index in unfavorable)
        region_pairs = 0
        for favorable_index in order:
            candidates = [
                index
                for index in available
                if abs(logits[index] - logits[favorable_index]) <= caliper + 1e-12
            ]
            if not candidates:
                continue
            selected = min(
                candidates,
                key=lambda index: (
                    float(
                        np.linalg.norm(
                            standardized_features[favorable_index]
                            - standardized_features[index]
                        )
                    ),
                    samples[index],
                ),
            )
            set_number += 1
            region_pairs += 1
            available.remove(selected)
            pairs.append(
                {
                    "matched_set_id": f"v3_match_{set_number:06d}",
                    "region": region,
                    "favorable_index": favorable_index,
                    "unfavorable_index": selected,
                    "absolute_logit_difference": float(
                        abs(logits[favorable_index] - logits[selected])
                    ),
                    "standardized_state_distance": float(
                        np.linalg.norm(
                            standardized_features[favorable_index]
                            - standardized_features[selected]
                        )
                    ),
                }
            )
        region_reports[region] = {
            "favorable_rows": int(len(favorable)),
            "unfavorable_rows": int(len(unfavorable)),
            "pooled_logit_propensity_sd": pooled_sd,
            "caliper": caliper,
            "matched_sets": region_pairs,
        }
    return pairs, region_reports


def _matched_smd(
    imputed_features: np.ndarray,
    pairs: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
) -> dict[str, float]:
    if not pairs:
        return {name: float("inf") for name in feature_names}
    favorable_indices = np.asarray(
        [int(pair["favorable_index"]) for pair in pairs], dtype=np.int64
    )
    unfavorable_indices = np.asarray(
        [int(pair["unfavorable_index"]) for pair in pairs], dtype=np.int64
    )
    favorable = imputed_features[favorable_indices]
    unfavorable = imputed_features[unfavorable_indices]
    output: dict[str, float] = {}
    for index, name in enumerate(feature_names):
        left = favorable[:, index]
        right = unfavorable[:, index]
        difference = abs(float(left.mean() - right.mean()))
        scale = _pooled_standard_deviation(left, right)
        if scale <= 1e-12:
            output[str(name)] = 0.0 if difference <= 1e-12 else float("inf")
        else:
            output[str(name)] = difference / scale
    return output


def common_support_audit(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    feature_names: Sequence[str] = MATCHING_FEATURE_NAMES,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    """Run the complete frozen, outcome-blind propensity and matching audit."""

    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    predictions, assignments, fold_reports = cross_fitted_propensity(
        frame,
        feature_names=feature_names,
        folds=int(config["folds"]),
        seed=int(config["seed"]),
        c_value=float(config["propensity_c"]),
    )
    output = frame.copy().reset_index(drop=True)
    output["propensity"] = predictions
    output["propensity_fold"] = assignments

    raw_features = output.loc[:, list(feature_names)].to_numpy(dtype=np.float64)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputed = imputer.fit_transform(raw_features)
    standardized = StandardScaler().fit_transform(imputed)
    pairs, region_reports = _deterministic_pairs(
        output,
        standardized,
        caliper_scale=float(config["logit_caliper_pooled_sd"]),
    )
    output["matched_set_id"] = ""
    output["matched_role"] = ""
    for pair in pairs:
        favorable_index = int(pair["favorable_index"])
        unfavorable_index = int(pair["unfavorable_index"])
        output.at[favorable_index, "matched_set_id"] = pair["matched_set_id"]
        output.at[favorable_index, "matched_role"] = "favorable"
        output.at[unfavorable_index, "matched_set_id"] = pair["matched_set_id"]
        output.at[unfavorable_index, "matched_role"] = "unfavorable"

    lower, upper = (float(value) for value in config["propensity_interval"])
    interval_fraction = float(((predictions >= lower) & (predictions <= upper)).mean())
    labels = output["favorable_matchup"].to_numpy(dtype=np.int8)
    overlap_weights = np.where(labels == 1, 1.0 - predictions, predictions)
    ess = float(overlap_weights.sum() ** 2 / np.sum(overlap_weights**2))
    smd = _matched_smd(imputed, pairs, feature_names)
    maximum_smd = max(smd.values(), default=float("inf"))
    decisions = {
        "minimum_matched_sets": len(pairs) >= int(config["minimum_matched_sets"]),
        "minimum_overlap_weight_ess": ess
        >= float(config["overlap_weight_ess_minimum"]),
        "minimum_propensity_interval_fraction": interval_fraction
        >= float(config["minimum_propensity_interval_fraction"]),
        "maximum_absolute_smd_every_feature": maximum_smd
        <= float(config["maximum_absolute_smd"]),
    }
    serialized_smd = {
        name: (float(value) if math.isfinite(float(value)) else None)
        for name, value in smd.items()
    }
    report = {
        "propensity": {
            "method": "five_fold_stratified_group_l2_logistic",
            "rows": int(len(output)),
            "series": int(output["series_id"].nunique()),
            "favorable_fraction": float(labels.mean()),
            "interval": [lower, upper],
            "interval_fraction": interval_fraction,
            "minimum": float(predictions.min()),
            "median": float(np.median(predictions)),
            "maximum": float(predictions.max()),
            "folds": fold_reports,
        },
        "overlap_weight_effective_sample_size": ess,
        "matching": {
            "matched_sets": int(len(pairs)),
            "matched_rows": int(2 * len(pairs)),
            "regions": region_reports,
            "maximum_absolute_smd": (
                float(maximum_smd) if math.isfinite(float(maximum_smd)) else None
            ),
            "nonfinite_smd_features": [
                name for name, value in smd.items() if not math.isfinite(float(value))
            ],
            "absolute_smd_by_feature": serialized_smd,
        },
        "gates": decisions,
        "all_gates_pass": all(decisions.values()),
    }
    return output, report, pairs
