"""Chronology-safe player profiles for RLCS critical-value V2."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from footballq.data.rlcs_replay import IdentityObservation, normalize_handle, repair_score_columns
from footballq.data.rlcs_touch_windows import CONTACT_EVENT_TYPES, extract_touches

PROFILE_FEATURES = (
    "mean_speed",
    "supersonic_fraction",
    "mean_boost",
    "low_boost_fraction",
    "boost_active_fraction",
    "aerial_fraction",
    "goalside_fraction",
    "goalside_recovery_speed",
    "mean_distance_to_ball",
    "mean_distance_to_own_goal",
    "mean_teammate_spacing",
    "nearest_to_ball_fraction",
    "touches_per_minute",
    "shots_per_touch",
    "goals_per_shot",
    "saves_per_minute",
    "attacking_half_touch_fraction",
    "forward_ball_velocity_per_touch",
    "ground_dribble_per_touch",
    "aerial_control_per_touch",
    "flick_per_touch",
    "pass_per_touch",
    "rebound_per_touch",
    "retrieval_per_touch",
    "challenge_win_fraction",
    "turnover_per_touch",
    "double_commit_per_minute",
    "nearest_defender_conceded_shot_per_minute",
)
PROFILE_DIMENSION = len(PROFILE_FEATURES)

CORE_CONTINUOUS_TRAITS = (
    "mean_speed",
    "supersonic_fraction",
    "mean_boost",
    "low_boost_fraction",
    "aerial_fraction",
    "goalside_fraction",
    "goalside_recovery_speed",
    "mean_distance_to_ball",
    "mean_teammate_spacing",
    "nearest_to_ball_fraction",
    "touches_per_minute",
    "attacking_half_touch_fraction",
    "forward_ball_velocity_per_touch",
)

V2_STAGES = (
    "profile_support",
    "train",
    "internal_development",
    "validation",
    "test",
)

MAX_CAR_SPEED = 2300.0
MAX_ARENA_DISTANCE = 12000.0


class PlayerProfileError(ValueError):
    """Raised when a profile or chronological split violates the frozen protocol."""


def _largest_remainder_counts(total: int, fractions: Sequence[float]) -> list[int]:
    raw = np.asarray(fractions, dtype=np.float64) * int(total)
    counts = np.floor(raw).astype(int)
    remaining = int(total) - int(counts.sum())
    order = sorted(range(len(fractions)), key=lambda index: (-(raw[index] % 1), index))
    for index in order[:remaining]:
        counts[index] += 1
    return counts.tolist()


def build_v2_split_frame(
    inventory: pd.DataFrame,
    *,
    split1_fractions: Sequence[float] = (0.35, 0.45, 0.20),
) -> pd.DataFrame:
    """Assign complete official series to the five frozen V2 stages."""

    required = {
        "replay_id",
        "series_id",
        "region",
        "event_time_utc",
        "split_number",
        "regional_number",
    }
    missing = required.difference(inventory.columns)
    if missing:
        raise PlayerProfileError(f"Inventory is missing split columns: {sorted(missing)}")
    if len(split1_fractions) != 3 or not math.isclose(sum(split1_fractions), 1.0):
        raise PlayerProfileError("Split 1 fractions must contain three values summing to one.")

    frame = inventory.copy()
    frame["replay_id"] = frame["replay_id"].astype(str)
    frame["series_id"] = frame["series_id"].astype(str)
    frame["region"] = frame["region"].astype(str).str.upper()
    frame["event_time_utc"] = pd.to_datetime(frame["event_time_utc"], utc=True)
    if not set(frame["region"]).issubset({"EU", "NA"}):
        raise PlayerProfileError("V2 supports only EU and NA replay rows.")
    if frame["event_time_utc"].isna().any():
        raise PlayerProfileError("Every replay needs an event timestamp for chronological V2.")

    series_stage: dict[str, str] = {}
    for region in ("EU", "NA"):
        split1 = frame.loc[(frame["region"] == region) & (frame["split_number"] == 1)]
        series = (
            split1.groupby("series_id", as_index=False)["event_time_utc"]
            .min()
            .sort_values(["event_time_utc", "series_id"], kind="stable")
        )
        counts = _largest_remainder_counts(len(series), split1_fractions)
        boundaries = np.cumsum([0, *counts])
        names = ("profile_support", "train", "internal_development")
        for index, name in enumerate(names):
            for series_id in series.iloc[boundaries[index] : boundaries[index + 1]][
                "series_id"
            ]:
                series_stage[str(series_id)] = name

    split2 = frame.loc[frame["split_number"] == 2]
    for series_id, rows in split2.groupby("series_id", sort=False):
        regionals = set(pd.to_numeric(rows["regional_number"], errors="raise").astype(int))
        if len(regionals) != 1:
            raise PlayerProfileError(f"Series {series_id} spans multiple regional numbers.")
        regional = next(iter(regionals))
        if regional == 1:
            series_stage[str(series_id)] = "validation"
        elif regional in {2, 3}:
            series_stage[str(series_id)] = "test"
        else:
            raise PlayerProfileError(f"Series {series_id} is outside frozen Split 2 regionals.")

    frame["v2_stage"] = frame["series_id"].map(series_stage)
    if frame["v2_stage"].isna().any():
        unknown = sorted(frame.loc[frame["v2_stage"].isna(), "series_id"].unique())[:5]
        raise PlayerProfileError(f"Unassigned official series: {unknown}")
    if not set(frame["v2_stage"]).issubset(V2_STAGES):
        raise PlayerProfileError("Unknown V2 stage was assigned.")
    if frame.groupby("series_id")["v2_stage"].nunique().max() != 1:
        raise PlayerProfileError("An official series crosses V2 stages.")
    return frame.sort_values(["event_time_utc", "replay_id"], kind="stable").reset_index(
        drop=True
    )


def split_manifest_payload(split_frame: pd.DataFrame) -> dict[str, Any]:
    """Return the auditable JSON payload for a V2 replay/series split."""

    stages: dict[str, Any] = {}
    for stage in V2_STAGES:
        rows = split_frame.loc[split_frame["v2_stage"] == stage]
        stages[stage] = {
            "replay_ids": sorted(rows["replay_id"].astype(str).unique()),
            "series_ids": sorted(rows["series_id"].astype(str).unique()),
            "replays": int(rows["replay_id"].nunique()),
            "series": int(rows["series_id"].nunique()),
            "regions": {
                region: {
                    "replays": int((rows["region"] == region).sum()),
                    "series": int(rows.loc[rows["region"] == region, "series_id"].nunique()),
                }
                for region in ("EU", "NA")
            },
        }
    return {
        "version": 2,
        "experiment": "rlcs_player_matchup_value_v2",
        "status": "frozen",
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "split_unit": "official_series",
        "allocation": "largest_remainder_by_region_35_45_20",
        "test_stage": "sealed",
        "stages": stages,
    }


def observations_and_roster(
    quality_record: Mapping[str, Any],
) -> tuple[list[IdentityObservation], dict[str, str]]:
    """Reconstruct the audited prefix-to-player mapping from the V1 quality ledger."""

    observed = list(quality_record.get("observed_roster") or [])
    canonical = [str(value) for value in quality_record.get("canonical_roster") or []]
    if len(observed) != 6 or len(canonical) != 6:
        raise PlayerProfileError("Accepted replay must contain six audited roster identities.")
    observations: list[IdentityObservation] = []
    roster: dict[str, str] = {}
    for item, canonical_id in zip(observed, canonical, strict=True):
        prefix = str(item["prefix"])
        roster[prefix] = canonical_id
        handle = str(item.get("handle") or "")
        observations.append(
            IdentityObservation(
                replay_id=str(quality_record["replay_id"]),
                split=str(quality_record.get("split") or ""),
                event_time_utc=(
                    str(quality_record["event_time_utc"])
                    if quality_record.get("event_time_utc")
                    else None
                ),
                group_id=str(quality_record.get("series_id") or "") or None,
                prefix=prefix,
                team=str(item["team"]).casefold(),
                handle=handle,
                normalized_handle=normalize_handle(handle),
                platform=str(item.get("platform") or ""),
                platform_id=str(item.get("platform_id") or ""),
            )
        )
    return observations, roster


def profile_frame_columns(quality_record: Mapping[str, Any]) -> list[str]:
    """Columns needed to compute the frozen one-second telemetry profile."""

    observed = list(quality_record.get("observed_roster") or [])
    columns = ["observed_frame_number", "seconds_elapsed", "game_time_s_precise", "ball_pos_y"]
    for item in observed:
        prefix = str(item["prefix"])
        columns.extend(
            [
                f"{prefix}_pos_x",
                f"{prefix}_pos_y",
                f"{prefix}_pos_z",
                f"{prefix}_vel_x",
                f"{prefix}_vel_y",
                f"{prefix}_vel_z",
                f"{prefix}_boost",
                f"{prefix}_boost_active",
                f"{prefix}_supersonic",
                f"{prefix}_distance_to_ball",
                f"{prefix}_distance_to_own_net",
            ]
        )
    return list(dict.fromkeys(columns))


PROFILE_EVENT_COLUMNS = (
    "event_number",
    "event_type",
    "observed_frame_number",
    "game_time_s_precise",
    "event_team",
    "event_player_1_id",
    "event_player_1_name",
    "event_player_1_team",
    "event_ball_pos_x",
    "event_ball_pos_y",
    "event_ball_pos_z",
    "ball_pos_x",
    "ball_pos_y",
    "ball_pos_z",
    "ball_vel_y",
    "blue_score",
    "orange_score",
    "goal_number",
    "official_goal",
    "official_shot",
    "official_save",
    "off_challenge_win",
    "off_ground_dribble",
    "off_air_dribble",
    "off_flick",
    "off_pass",
    "off_retrieval",
)


def _safe_mean(values: Any, *, default: float = 0.0) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if len(finite) else float(default)


def _optional_score(value: Any) -> int | None:
    numeric = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(numeric) else int(numeric)


def _event_prefix(
    row: Mapping[str, Any], observations: Sequence[IdentityObservation]
) -> str | None:
    raw_id = str(row.get("event_player_1_id") or "").split(".0", 1)[0]
    handle = normalize_handle(row.get("event_player_1_name"))
    matches = {
        item.prefix
        for item in observations
        if (raw_id and raw_id == item.platform_id)
        or (handle and handle == item.normalized_handle)
    }
    return next(iter(matches)) if len(matches) == 1 else None


def compute_player_game_profiles(
    frames: pd.DataFrame,
    events: pd.DataFrame,
    *,
    quality_record: Mapping[str, Any],
    inventory_row: Mapping[str, Any],
    stage: str,
    sample_seconds: float = 1.0,
) -> list[dict[str, Any]]:
    """Compute one interpretable 28-feature row per player/game."""

    if stage not in V2_STAGES:
        raise PlayerProfileError(f"Unknown V2 stage {stage!r}.")
    observations, roster = observations_and_roster(quality_record)
    time_column = "game_time_s_precise" if "game_time_s_precise" in frames else "seconds_elapsed"
    times = pd.to_numeric(frames[time_column], errors="coerce")
    if times.notna().sum() < 2:
        raise PlayerProfileError("Replay has no usable telemetry clock.")
    ordered_frames = frames.assign(_time=times).dropna(subset=["_time"]).sort_values("_time")
    bucket = np.floor(ordered_frames["_time"].to_numpy() / float(sample_seconds)).astype(int)
    sampled = ordered_frames.loc[~pd.Series(bucket, index=ordered_frames.index).duplicated()].copy()
    duration_minutes = max(float(times.max() - times.min()) / 60.0, 1.0 / 60.0)

    repaired = repair_score_columns(
        events,
        expected_blue_score=_optional_score(inventory_row.get("blue_score")),
        expected_orange_score=_optional_score(inventory_row.get("orange_score")),
    )
    touches = extract_touches(repaired, observations, roster, scores_repaired=True)
    touches_by_prefix: dict[str, list[Any]] = {item.prefix: [] for item in observations}
    for touch in touches:
        touches_by_prefix[touch.player_prefix].append(touch)
    event_records = repaired.to_dict(orient="records")
    events_by_prefix: dict[str, list[dict[str, Any]]] = {
        item.prefix: [] for item in observations
    }
    for row in event_records:
        prefix = _event_prefix(row, observations)
        if prefix is not None:
            events_by_prefix[prefix].append(row)

    by_team = {
        team: [item.prefix for item in observations if item.team == team]
        for team in ("blue", "orange")
    }
    distance_to_ball = {
        prefix: pd.to_numeric(sampled[f"{prefix}_distance_to_ball"], errors="coerce").to_numpy()
        for prefix in roster
    }
    nearest_masks: dict[str, np.ndarray] = {}
    for team, prefixes in by_team.items():
        distances = np.column_stack([distance_to_ball[prefix] for prefix in prefixes])
        nearest = np.nanargmin(np.where(np.isfinite(distances), distances, np.inf), axis=1)
        for index, prefix in enumerate(prefixes):
            nearest_masks[prefix] = nearest == index

    reconstructed_blue = int(repaired["blue_score"].iloc[-1])
    reconstructed_orange = int(repaired["orange_score"].iloc[-1])
    blue_final = _optional_score(inventory_row.get("blue_score"))
    orange_final = _optional_score(inventory_row.get("orange_score"))
    blue_final = reconstructed_blue if blue_final is None else blue_final
    orange_final = reconstructed_orange if orange_final is None else orange_final
    output: list[dict[str, Any]] = []
    for item in observations:
        prefix = item.prefix
        team = item.team
        opponents = "orange" if team == "blue" else "blue"
        teammates = [value for value in by_team[team] if value != prefix]
        px = pd.to_numeric(sampled[f"{prefix}_pos_x"], errors="coerce").to_numpy()
        py = pd.to_numeric(sampled[f"{prefix}_pos_y"], errors="coerce").to_numpy()
        pz = pd.to_numeric(sampled[f"{prefix}_pos_z"], errors="coerce").to_numpy()
        vx = pd.to_numeric(sampled[f"{prefix}_vel_x"], errors="coerce").to_numpy()
        vy = pd.to_numeric(sampled[f"{prefix}_vel_y"], errors="coerce").to_numpy()
        vz = pd.to_numeric(sampled[f"{prefix}_vel_z"], errors="coerce").to_numpy()
        speed = np.sqrt(vx * vx + vy * vy + vz * vz) / MAX_CAR_SPEED
        boost = pd.to_numeric(sampled[f"{prefix}_boost"], errors="coerce").to_numpy() / 100.0
        ball_y = pd.to_numeric(sampled["ball_pos_y"], errors="coerce").to_numpy()
        goalside = py <= ball_y if team == "blue" else py >= ball_y
        not_goalside = ~goalside
        recovery_component = np.maximum(-vy if team == "blue" else vy, 0.0) / MAX_CAR_SPEED
        teammate_distances = []
        for teammate in teammates:
            tx = pd.to_numeric(sampled[f"{teammate}_pos_x"], errors="coerce").to_numpy()
            ty = pd.to_numeric(sampled[f"{teammate}_pos_y"], errors="coerce").to_numpy()
            tz = pd.to_numeric(sampled[f"{teammate}_pos_z"], errors="coerce").to_numpy()
            teammate_distances.append(np.sqrt((px - tx) ** 2 + (py - ty) ** 2 + (pz - tz) ** 2))

        player_events = events_by_prefix[prefix]
        event_types = [normalize_handle(row.get("event_type")) for row in player_events]
        player_touches = touches_by_prefix[prefix]
        touch_count = len(player_touches)
        denominator = max(touch_count, 1)
        shots = sum(kind == "shot" for kind in event_types)
        goals = sum(kind == "goal" for kind in event_types)
        saves = sum(kind == "save" for kind in event_types)
        challenge_rows = [
            row for row, kind in zip(player_events, event_types, strict=True) if kind == "challenge"
        ]
        attacking_touches = sum(
            (touch.ball_position[1] > 0 if team == "blue" else touch.ball_position[1] < 0)
            for touch in player_touches
        )
        touch_event_rows = [
            row
            for row, kind in zip(player_events, event_types, strict=True)
            if kind in CONTACT_EVENT_TYPES
        ]
        forward_velocity = []
        for row in touch_event_rows:
            value = pd.to_numeric(row.get("ball_vel_y"), errors="coerce")
            if not pd.isna(value):
                oriented_velocity = float(value) if team == "blue" else -float(value)
                forward_velocity.append(oriented_velocity / 6000.0)

        opponent_shots = [
            row
            for row in event_records
            if normalize_handle(row.get("event_type")) == "shot"
            and normalize_handle(row.get("event_team") or row.get("event_player_1_team"))
            == opponents
        ]
        nearest_defender_shots = 0
        for row in opponent_shots:
            frame_number = pd.to_numeric(row.get("observed_frame_number"), errors="coerce")
            if pd.isna(frame_number):
                continue
            frame_values = pd.to_numeric(
                sampled["observed_frame_number"], errors="coerce"
            ).to_numpy()
            index = int(np.argmin(np.abs(frame_values - float(frame_number))))
            if nearest_masks[prefix][index]:
                nearest_defender_shots += 1

        feature_values = (
            _safe_mean(speed),
            _safe_mean(sampled[f"{prefix}_supersonic"].astype(float)),
            _safe_mean(boost),
            _safe_mean(boost <= 0.20),
            _safe_mean(sampled[f"{prefix}_boost_active"].astype(float)),
            _safe_mean(pz > 300.0),
            _safe_mean(goalside),
            _safe_mean(recovery_component[not_goalside]),
            _safe_mean(distance_to_ball[prefix] / MAX_ARENA_DISTANCE),
            _safe_mean(
                pd.to_numeric(sampled[f"{prefix}_distance_to_own_net"], errors="coerce")
                / MAX_ARENA_DISTANCE
            ),
            _safe_mean(np.column_stack(teammate_distances) / MAX_ARENA_DISTANCE),
            _safe_mean(nearest_masks[prefix]),
            touch_count / duration_minutes,
            shots / denominator,
            goals / max(shots, 1),
            saves / duration_minutes,
            attacking_touches / denominator,
            _safe_mean(np.clip(forward_velocity, -1.0, 1.0)),
            sum(kind == "ground-dribble" for kind in event_types) / denominator,
            sum(kind in {"air-dribble", "double-tap", "flip-reset"} for kind in event_types)
            / denominator,
            sum(kind == "flick" for kind in event_types) / denominator,
            sum(kind == "pass" for kind in event_types) / denominator,
            sum(kind == "rebound" for kind in event_types) / denominator,
            sum(kind == "retrieval" for kind in event_types) / denominator,
            sum(bool(row.get("off_challenge_win")) for row in challenge_rows)
            / max(len(challenge_rows), 1),
            sum(kind == "turnover" for kind in event_types) / denominator,
            sum(kind == "double-commit" for kind in event_types) / duration_minutes,
            nearest_defender_shots / duration_minutes,
        )
        if len(feature_values) != PROFILE_DIMENSION or not np.isfinite(feature_values).all():
            replay_id = quality_record["replay_id"]
            raise PlayerProfileError(f"Non-finite profile for {replay_id}:{prefix}.")
        team_score = blue_final if team == "blue" else orange_final
        opponent_score = orange_final if team == "blue" else blue_final
        output.append(
            {
                "replay_id": str(quality_record["replay_id"]),
                "series_id": str(inventory_row.get("series_id") or ""),
                "region": str(inventory_row.get("region") or "").upper(),
                "event_time_utc": pd.Timestamp(inventory_row["event_time_utc"]),
                "v2_stage": stage,
                "player_id": roster[prefix],
                "player_prefix": prefix,
                "team": team,
                "profile": np.asarray(feature_values, dtype=np.float32).tolist(),
                "team_win": float(team_score > opponent_score),
                "team_goal_diff": int(team_score - opponent_score),
                "duration_minutes": np.float32(duration_minutes),
                "touch_count": int(touch_count),
            }
        )
    return output


def fit_profile_priors(
    game_profiles: pd.DataFrame,
    *,
    minimum_prior_games: float = 1.0,
    maximum_prior_games: float = 25.0,
) -> dict[str, Any]:
    """Fit empirical-Bayes hyperparameters using profile-support rows only."""

    support = game_profiles.loc[game_profiles["v2_stage"] == "profile_support"]
    if support.empty:
        raise PlayerProfileError("Profile priors require non-empty support rows.")
    values = np.stack(support["profile"].map(np.asarray))
    if values.shape[1] != PROFILE_DIMENSION:
        raise PlayerProfileError("Profile feature width differs from the frozen dimension.")
    population_mean = values.mean(axis=0)
    population_variance = values.var(axis=0, ddof=1 if len(values) > 1 else 0)
    player_means = []
    within_variances = []
    for _, rows in support.groupby("player_id", sort=False):
        matrix = np.stack(rows["profile"].map(np.asarray))
        player_means.append(matrix.mean(axis=0))
        if len(matrix) >= 2:
            within_variances.append(matrix.var(axis=0, ddof=1))
    between_variance = (
        np.var(np.stack(player_means), axis=0, ddof=1)
        if len(player_means) > 1
        else population_variance.copy()
    )
    within_variance = (
        np.nanmean(np.stack(within_variances), axis=0)
        if within_variances
        else population_variance
    )
    population_variance = np.nan_to_num(population_variance, nan=0.0)
    within_variance = np.where(np.isfinite(within_variance), within_variance, population_variance)
    between_variance = np.where(
        np.isfinite(between_variance), between_variance, population_variance
    )
    prior_games = np.clip(
        within_variance / np.maximum(between_variance, 1e-8),
        float(minimum_prior_games),
        float(maximum_prior_games),
    )
    uncertainty_scale = np.sqrt(np.maximum(population_variance, 1e-8))
    return {
        "version": 2,
        "fit_stage": "profile_support",
        "feature_names": list(PROFILE_FEATURES),
        "population_mean": population_mean.tolist(),
        "population_variance": population_variance.tolist(),
        "prior_games": prior_games.tolist(),
        "uncertainty_scale": uncertainty_scale.tolist(),
        "support_rows": int(len(support)),
        "support_players": int(support["player_id"].nunique()),
    }


def shrink_profile(history: pd.DataFrame, priors: Mapping[str, Any]) -> dict[str, Any]:
    """Create one shrunk profile from strictly prior game rows."""

    population = np.asarray(priors["population_mean"], dtype=np.float64)
    prior_games = np.asarray(priors["prior_games"], dtype=np.float64)
    scale = np.asarray(priors["uncertainty_scale"], dtype=np.float64)
    n_games = int(len(history))
    observed = (
        np.stack(history["profile"].map(np.asarray)).mean(axis=0)
        if n_games
        else population.copy()
    )
    effective = prior_games + n_games
    shrunk = (n_games * observed + prior_games * population) / effective
    uncertainty = scale / np.sqrt(effective)
    return {
        "profile": shrunk.astype(np.float32).tolist(),
        "uncertainty": uncertainty.astype(np.float32).tolist(),
        "n_prior_games": n_games,
        "effective_sample_size": float(np.mean(effective)),
        "prior_win_rate": float(history["team_win"].mean()) if n_games else 0.5,
        "prior_goal_diff": float(history["team_goal_diff"].mean()) if n_games else 0.0,
    }


def build_profile_snapshots(
    game_profiles: pd.DataFrame, priors: Mapping[str, Any]
) -> pd.DataFrame:
    """Build per-query snapshots with only games strictly earlier than the query."""

    frame = game_profiles.copy()
    frame["event_time_utc"] = pd.to_datetime(frame["event_time_utc"], utc=True)
    frame = frame.sort_values(["event_time_utc", "replay_id", "player_id"], kind="stable")
    player_rows = {
        str(player_id): rows.sort_values(["event_time_utc", "replay_id"], kind="stable")
        for player_id, rows in frame.groupby("player_id", sort=False)
    }
    snapshots: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        prior_rows = player_rows[str(row.player_id)]
        prior_rows = prior_rows.loc[prior_rows["event_time_utc"] < row.event_time_utc]
        snapshot = shrink_profile(prior_rows, priors)
        snapshots.append(
            {
                "replay_id": str(row.replay_id),
                "series_id": str(row.series_id),
                "region": str(row.region),
                "event_time_utc": row.event_time_utc,
                "v2_stage": str(row.v2_stage),
                "player_id": str(row.player_id),
                **snapshot,
                "latest_prior_time_utc": (
                    prior_rows["event_time_utc"].max() if len(prior_rows) else pd.NaT
                ),
            }
        )
    return pd.DataFrame(snapshots)


def _binary_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels_array = np.asarray(labels, dtype=np.int8)
    scores_array = np.asarray(scores, dtype=np.float64)
    positive = scores_array[labels_array == 1]
    negative = scores_array[labels_array == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    comparisons = positive[:, None] - negative[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def _median_spearman(early: np.ndarray, late: np.ndarray) -> float:
    from scipy.stats import rankdata

    early_rank = rankdata(early, axis=0, method="average")
    late_rank = rankdata(late, axis=0, method="average")
    early_centered = early_rank - early_rank.mean(axis=0, keepdims=True)
    late_centered = late_rank - late_rank.mean(axis=0, keepdims=True)
    numerator = np.sum(early_centered * late_centered, axis=0)
    denominator = np.sqrt(
        np.sum(early_centered**2, axis=0) * np.sum(late_centered**2, axis=0)
    )
    correlations = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator > 0,
    )
    finite = correlations[np.isfinite(correlations)]
    return float(np.median(finite)) if len(finite) else float("nan")


def _player_bootstrap(
    halves: Sequence[Mapping[str, Any]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    if int(resamples) <= 0:
        raise ValueError("Player bootstrap requires a positive resample count.")
    core_indices = np.asarray(
        [PROFILE_FEATURES.index(name) for name in CORE_CONTINUOUS_TRAITS], dtype=np.int64
    )
    early = np.stack([np.asarray(row["early"])[core_indices] for row in halves])
    late = np.stack([np.asarray(row["late"])[core_indices] for row in halves])
    positive = np.asarray([row["positive_similarity"] for row in halves], dtype=np.float64)
    negative = np.asarray([row["negative_similarity"] for row in halves], dtype=np.float64)
    regional_indices = {
        region: np.asarray(
            [index for index, row in enumerate(halves) if row["region"] == region],
            dtype=np.int64,
        )
        for region in ("EU", "NA")
    }
    if any(len(indices) == 0 for indices in regional_indices.values()):
        raise ValueError("Stratified player bootstrap requires eligible EU and NA players.")
    rng = np.random.default_rng(int(seed))
    auc_values = np.empty(int(resamples), dtype=np.float64)
    spearman_values = np.empty(int(resamples), dtype=np.float64)
    for replicate in range(int(resamples)):
        sampled = np.concatenate(
            [
                rng.choice(indices, size=len(indices), replace=True)
                for indices in regional_indices.values()
            ]
        )
        auc_values[replicate] = _binary_auc(
            [1] * len(sampled) + [0] * len(sampled),
            np.concatenate([positive[sampled], negative[sampled]]),
        )
        spearman_values[replicate] = _median_spearman(early[sampled], late[sampled])
    finite_auc = auc_values[np.isfinite(auc_values)]
    finite_spearman = spearman_values[np.isfinite(spearman_values)]
    if not len(finite_auc) or not len(finite_spearman):
        raise PlayerProfileError("Player bootstrap produced no finite stability replicates.")
    return {
        "method": "stratified_percentile_player_bootstrap_by_region",
        "seed": int(seed),
        "requested_resamples": int(resamples),
        "finite_auc_resamples": int(len(finite_auc)),
        "finite_spearman_resamples": int(len(finite_spearman)),
        "retrieval_auc_95pct": np.quantile(finite_auc, [0.025, 0.975]).tolist(),
        "median_spearman_95pct": np.quantile(finite_spearman, [0.025, 0.975]).tolist(),
    }


def audit_profile_stability(
    game_profiles: pd.DataFrame,
    priors: Mapping[str, Any],
    *,
    minimum_games: int = 15,
    required_complete_players: int = 48,
    minimum_players_per_region: int = 20,
    retrieval_auc_minimum: float = 0.75,
    retrieval_auc_bootstrap_lower_minimum: float = 0.65,
    regional_retrieval_auc_minimum_exclusive: float = 0.50,
    median_spearman_minimum: float = 0.35,
    median_spearman_bootstrap_lower_minimum_exclusive: float = 0.20,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 20_260_803,
) -> dict[str, Any]:
    """Evaluate the amended complete-cohort support-period persistence gate."""

    support = game_profiles.loc[game_profiles["v2_stage"] == "profile_support"].copy()
    support["event_time_utc"] = pd.to_datetime(support["event_time_utc"], utc=True)
    eligible = {
        str(player_id): rows.sort_values(["event_time_utc", "replay_id"], kind="stable")
        for player_id, rows in support.groupby("player_id", sort=True)
        if len(rows) >= int(minimum_games)
    }
    halves: list[dict[str, Any]] = []
    for player_id, rows in eligible.items():
        midpoint = len(rows) // 2
        early = rows.iloc[:midpoint]
        late = rows.iloc[midpoint:]
        halves.append(
            {
                "player_id": player_id,
                "region": str(rows["region"].mode().iloc[0]),
                "date": rows["event_time_utc"].median(),
                "team_strength": float(early["team_win"].mean()),
                "early": np.stack(early["profile"].map(np.asarray)).mean(axis=0),
                "late": np.stack(late["profile"].map(np.asarray)).mean(axis=0),
                "games": int(len(rows)),
            }
        )

    correlations: dict[str, float | None] = {}
    for name in CORE_CONTINUOUS_TRAITS:
        feature_index = PROFILE_FEATURES.index(name)
        early_values = [row["early"][feature_index] for row in halves]
        late_values = [row["late"][feature_index] for row in halves]
        correlation = pd.Series(early_values).corr(pd.Series(late_values), method="spearman")
        correlations[name] = None if pd.isna(correlation) else float(correlation)
    finite_correlations = [value for value in correlations.values() if value is not None]
    median_spearman = (
        float(np.median(finite_correlations)) if finite_correlations else float("nan")
    )

    scale = np.asarray(priors["uncertainty_scale"], dtype=np.float64)
    scale = np.where(scale > 1e-8, scale, 1.0)
    for row in halves:
        positive = -float(np.linalg.norm((row["early"] - row["late"]) / scale))
        candidates = [
            other
            for other in halves
            if other["player_id"] != row["player_id"] and other["region"] == row["region"]
        ]
        if not candidates:
            continue
        negative_player = min(
            candidates,
            key=lambda other: abs((other["date"] - row["date"]).total_seconds())
            / (14 * 86400)
            + 3.0 * abs(other["team_strength"] - row["team_strength"]),
        )
        negative = -float(
            np.linalg.norm((row["early"] - negative_player["late"]) / scale)
        )
        row["positive_similarity"] = positive
        row["negative_similarity"] = negative
        row["matched_negative_player_id"] = negative_player["player_id"]
    labels = [1] * len(halves) + [0] * len(halves)
    similarities = [row["positive_similarity"] for row in halves] + [
        row["negative_similarity"] for row in halves
    ]
    retrieval_auc = _binary_auc(labels, similarities)

    eligible_by_region = {
        region: sum(row["region"] == region for row in halves) for region in ("EU", "NA")
    }
    regional_retrieval_auc = {}
    for region in ("EU", "NA"):
        regional = [row for row in halves if row["region"] == region]
        regional_retrieval_auc[region] = _binary_auc(
            [1] * len(regional) + [0] * len(regional),
            [row["positive_similarity"] for row in regional]
            + [row["negative_similarity"] for row in regional],
        )
    bootstrap = _player_bootstrap(
        halves, resamples=int(bootstrap_resamples), seed=int(bootstrap_seed)
    )
    eligible_player_ids = sorted(row["player_id"] for row in halves)
    cohort_digest = hashlib.sha256()
    for player_id in eligible_player_ids:
        cohort_digest.update(player_id.encode("utf-8") + b"\n")
    minimum_observed_games = min((row["games"] for row in halves), default=0)
    gates = {
        "complete_available_cohort": len(halves) == int(required_complete_players),
        "minimum_prior_games": minimum_observed_games >= int(minimum_games),
        "minimum_eu_players": eligible_by_region["EU"] >= int(minimum_players_per_region),
        "minimum_na_players": eligible_by_region["NA"] >= int(minimum_players_per_region),
        "retrieval_auc_point": bool(np.isfinite(retrieval_auc))
        and retrieval_auc >= float(retrieval_auc_minimum),
        "retrieval_auc_bootstrap_lower": bootstrap["retrieval_auc_95pct"][0]
        >= float(retrieval_auc_bootstrap_lower_minimum),
        "median_core_spearman_point": bool(np.isfinite(median_spearman))
        and median_spearman >= float(median_spearman_minimum),
        "median_core_spearman_bootstrap_lower": bootstrap["median_spearman_95pct"][0]
        > float(median_spearman_bootstrap_lower_minimum_exclusive),
        "regional_retrieval_positive": all(
            value > float(regional_retrieval_auc_minimum_exclusive)
            for value in regional_retrieval_auc.values()
        ),
    }
    return {
        "version": 2,
        "experiment": "rlcs_player_matchup_value_v2",
        "gate_version": "pre_outcome_amendment_01_reduced_cohort",
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "fit_stage": "profile_support",
        "thresholds": {
            "minimum_games": int(minimum_games),
            "required_complete_players": int(required_complete_players),
            "minimum_players_per_region": int(minimum_players_per_region),
            "retrieval_auc_point": float(retrieval_auc_minimum),
            "retrieval_auc_bootstrap_lower": float(
                retrieval_auc_bootstrap_lower_minimum
            ),
            "regional_retrieval_auc_minimum_exclusive": float(
                regional_retrieval_auc_minimum_exclusive
            ),
            "median_core_spearman_point": float(median_spearman_minimum),
            "median_core_spearman_bootstrap_lower_exclusive": float(
                median_spearman_bootstrap_lower_minimum_exclusive
            ),
        },
        "counts": {
            "support_game_player_rows": int(len(support)),
            "support_replays": int(support["replay_id"].nunique()),
            "support_players": int(support["player_id"].nunique()),
            "eligible_players": int(len(halves)),
            "eligible_players_by_region": eligible_by_region,
            "minimum_prior_games_in_eligible_cohort": int(minimum_observed_games),
        },
        "cohort_selection": "all_players_with_at_least_15_support_games_no_manual_selection",
        "eligible_player_ids_sha256": cohort_digest.hexdigest(),
        "same_player_retrieval_auc": retrieval_auc,
        "regional_same_player_retrieval_auc": regional_retrieval_auc,
        "core_trait_spearman": correlations,
        "median_core_trait_spearman": median_spearman,
        "player_bootstrap": bootstrap,
        "eligible_player_games": {
            row["player_id"]: row["games"] for row in halves
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "stop_before_outcome_training": not all(gates.values()),
    }


def write_profile_parquet(frame: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False, compression="zstd")
    return destination
