"""Compact, shrunk FOOTPASS player-history development experiment."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from footballq.analysis.footpass_player_history import (
    ExtractedFootpassData,
    FootpassAppearance,
    LogisticProbe,
    PlayerMatchStats,
    _active_focal_players,
    _stable_choice_index,
    binary_metrics,
    blocked_bootstrap_nll_gain,
    broad_role,
    build_footpass_feature_dataset,
    build_player_match_stats,
    fit_logistic_probe,
)

COMPACT_PROFILE_FEATURE_NAMES = (
    "action_drive_probability",
    "action_pass_like_probability",
    "action_shot_probability",
    "action_defensive_probability",
    "event_mean_x_attack",
    "event_mean_y",
    "event_mean_vx_attack",
    "event_mean_vy",
    "event_mean_speed",
    "event_std_x_attack",
    "event_std_y",
    "event_std_speed",
    "prior_turnover_rate",
    "prior_penalty_area_action_rate",
    "tracking_mean_x_attack",
    "tracking_mean_y",
    "tracking_mean_vx_attack",
    "tracking_mean_vy",
    "tracking_mean_speed",
    "tracking_mean_team_relative_x",
    "tracking_mean_team_relative_y",
    "tracking_mean_nearest_teammate_distance",
    "tracking_mean_nearest_opponent_distance",
    "tracking_std_x_attack",
    "tracking_std_y",
    "tracking_std_speed",
    "tracking_std_nearest_teammate_distance",
    "tracking_std_nearest_opponent_distance",
)


def _neutral_compact_profile() -> np.ndarray:
    result = np.zeros(len(COMPACT_PROFILE_FEATURE_NAMES), dtype=np.float64)
    result[:4] = 0.25
    result[12:14] = 0.5
    return result


def compact_player_match_profile(stats: PlayerMatchStats) -> np.ndarray:
    """Summarize one player-match with the frozen 28-value V2 schema."""

    event = stats.event
    event_total = float(event.event_count)
    action_groups = np.asarray(
        [
            event.action_counts[0],
            event.action_counts[[1, 2, 3, 5]].sum(),
            event.action_counts[4],
            event.action_counts[[6, 7]].sum(),
        ],
        dtype=np.float64,
    )
    action_probability = (action_groups + 1.0) / (event_total + 4.0)
    event_denominator = max(event_total, 1.0)
    event_mean = event.continuous_sum / event_denominator
    event_variance = np.maximum(
        event.continuous_sumsq / event_denominator - event_mean * event_mean,
        0.0,
    )
    event_std = np.sqrt(event_variance)
    outcome_rates = np.asarray(
        [
            (event.turnover_positive + 1.0) / (event.outcome_count + 2.0),
            (event.entry_positive + 1.0) / (event.outcome_count + 2.0),
        ],
        dtype=np.float64,
    )

    tracking = stats.tracking
    tracking_total = float(tracking.sample_count)
    tracking_denominator = max(tracking_total, 1.0)
    tracking_mean = tracking.feature_sum / tracking_denominator
    tracking_variance = np.maximum(
        tracking.feature_sumsq / tracking_denominator
        - tracking_mean * tracking_mean,
        0.0,
    )
    tracking_std = np.sqrt(tracking_variance)
    result = np.concatenate(
        [
            action_probability,
            event_mean,
            event_std[[0, 1, 4]],
            outcome_rates,
            tracking_mean,
            tracking_std[[0, 1, 4, 7, 8]],
        ]
    ).astype(np.float64)
    if result.shape != (len(COMPACT_PROFILE_FEATURE_NAMES),):
        raise ValueError(f"Unexpected compact profile shape: {result.shape}.")
    if not np.isfinite(result).all():
        raise ValueError("Compact player-match profile contains non-finite values.")
    return result


@dataclass(frozen=True)
class HistoricalCompactProfile:
    values: np.ndarray
    support_matches: int
    available: bool
    support_appearance_ids: tuple[str, ...]


def _historical_profile(
    query: FootpassAppearance,
    player_id: str,
    appearances_by_team: dict[str, list[FootpassAppearance]],
    stats_by_appearance: dict[str, dict[str, PlayerMatchStats]],
    *,
    support_cap: int,
) -> HistoricalCompactProfile:
    candidates: list[tuple[Any, int, str, PlayerMatchStats]] = []
    for appearance in appearances_by_team[query.team_id]:
        if appearance.match_date >= query.match_date:
            continue
        stats = stats_by_appearance.get(appearance.appearance_id, {}).get(player_id)
        if stats is None:
            continue
        candidates.append(
            (
                appearance.match_date,
                int(appearance.match_id),
                appearance.appearance_id,
                stats,
            )
        )
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = candidates[: int(support_cap)]
    if not selected:
        return HistoricalCompactProfile(
            values=_neutral_compact_profile(),
            support_matches=0,
            available=False,
            support_appearance_ids=(),
        )
    vectors = [compact_player_match_profile(item[3]) for item in selected]
    return HistoricalCompactProfile(
        values=np.mean(np.stack(vectors), axis=0),
        support_matches=len(selected),
        available=True,
        support_appearance_ids=tuple(item[2] for item in selected),
    )


def shrunk_player_deviation(
    player_profile: np.ndarray,
    role_prior: np.ndarray,
    *,
    support_matches: int,
    shrinkage_match_equivalent: float,
) -> tuple[np.ndarray, float]:
    """Shrink a player-minus-role residual by match-level support."""

    support = float(support_matches)
    shrinkage = float(shrinkage_match_equivalent)
    if shrinkage < 0.0:
        raise ValueError("Shrinkage match-equivalent must be non-negative.")
    alpha = support / (support + shrinkage) if support + shrinkage > 0.0 else 0.0
    return alpha * (player_profile - role_prior), alpha


@dataclass
class CompactPlayerResidualDataset:
    sample_ids: list[str]
    team_ids: list[str]
    match_ids: list[str]
    periods: list[int]
    frames: list[int]
    actor_ids: list[str]
    labels: dict[str, np.ndarray]
    components: dict[str, np.ndarray]
    component_feature_names: dict[str, list[str]]
    audit: dict[str, Any]

    def feature_views(
        self,
        main_shrinkage: float,
    ) -> dict[str, tuple[np.ndarray, list[str]]]:
        def join(*names: str) -> tuple[np.ndarray, list[str]]:
            arrays = [self.components[name] for name in names]
            feature_names = [
                f"{component}:{feature}"
                for component in names
                for feature in self.component_feature_names[component]
            ]
            return np.concatenate(arrays, axis=1), feature_names

        prefix = ("geometry", "role")
        context = (*prefix, "role_context_profile")
        main_key = _lambda_component_name(main_shrinkage)
        views = {
            "geometry_role": join(*prefix),
            "geometry_role_identity": join(*prefix, "identity"),
            "role_context": join(*context),
            "player_residual": join(*context, main_key),
            "player_residual_unshrunk": join(
                *context,
                "player_residual_unshrunk",
            ),
        }
        for component_name in sorted(self.components):
            if component_name.startswith("player_residual_lambda_"):
                label = component_name.removeprefix("player_residual_")
                views[f"player_residual_{label}"] = join(
                    *context,
                    component_name,
                )
            if component_name.startswith("shuffled_player_residual_seed_"):
                views[component_name] = join(*context, component_name)
        return views


def _lambda_label(value: float) -> str:
    return f"lambda_{float(value):g}".replace(".", "p")


def _lambda_component_name(value: float) -> str:
    return f"player_residual_{_lambda_label(value)}"


def _event_index_by_sample_key(
    data: ExtractedFootpassData,
) -> dict[tuple[str, int, int, int], int]:
    result: dict[tuple[str, int, int, int], int] = {}
    for index in range(len(data.event_frame)):
        key = (
            str(data.event_match_id[index]),
            int(data.event_period[index]),
            int(data.event_frame[index]),
            int(data.event_player_id[index]),
        )
        if key in result:
            raise ValueError(f"Duplicate event lookup key: {key}.")
        result[key] = index
    return result


def _sample_raw_player_id(sample_id: str) -> int:
    marker = ":slot"
    if marker not in sample_id:
        raise ValueError(f"Unexpected FOOTPASS sample ID: {sample_id}.")
    return int(sample_id.rsplit(marker, 1)[1])


def _role_prior(
    active: list[tuple[str, int]],
    histories: dict[str, HistoricalCompactProfile],
    *,
    actor_id: str,
    actor_role: int,
) -> tuple[np.ndarray, str, int]:
    target_role = broad_role(actor_role)
    candidates = [
        histories[player_id].values
        for player_id, role_id in active
        if player_id != actor_id
        and broad_role(role_id) == target_role
        and histories[player_id].available
    ]
    mode = "same_broad_role"
    if not candidates:
        candidates = [
            histories[player_id].values
            for player_id, _role_id in active
            if player_id != actor_id and histories[player_id].available
        ]
        mode = "active_team_fallback"
    if not candidates:
        return _neutral_compact_profile(), "neutral_fallback", 0
    return np.mean(np.stack(candidates), axis=0), mode, len(candidates)


def _shuffled_donor(
    *,
    sample_id: str,
    seed: int,
    active: list[tuple[str, int]],
    histories: dict[str, HistoricalCompactProfile],
    actor_id: str,
    actor_role: int,
) -> str | None:
    candidates = [
        player_id
        for player_id, role_id in active
        if player_id != actor_id
        and broad_role(role_id) == broad_role(actor_role)
        and histories[player_id].available
    ]
    if not candidates:
        candidates = [
            player_id
            for player_id, _role_id in active
            if player_id != actor_id and histories[player_id].available
        ]
    if not candidates:
        return None
    candidates.sort()
    return candidates[
        _stable_choice_index(
            int(seed),
            f"compact-v2:{sample_id}:{actor_id}",
            len(candidates),
        )
    ]


def build_compact_player_residual_dataset(
    data: ExtractedFootpassData,
    appearances: list[FootpassAppearance],
    config: dict[str, Any],
    *,
    query_partitions: set[str],
) -> CompactPlayerResidualDataset:
    """Build compact, causal player residuals on top of current geometry."""

    if data.metadata["confirmation_match_ids_included"]:
        raise ValueError("V2 development refuses caches containing confirmation IDs.")
    base = build_footpass_feature_dataset(
        data,
        appearances,
        config,
        query_partitions=query_partitions,
    )
    available_match_ids = set(data.metadata["selected_match_ids"])
    selected_appearances = [
        item for item in appearances if item.match_id in available_match_ids
    ]
    stats_by_appearance, _outcomes = build_player_match_stats(
        data,
        selected_appearances,
        config,
    )
    appearances_by_team: dict[str, list[FootpassAppearance]] = defaultdict(list)
    for appearance in selected_appearances:
        appearances_by_team[appearance.team_id].append(appearance)
    for values in appearances_by_team.values():
        values.sort(key=lambda item: (item.match_date, int(item.match_id)))
    query_by_id = {
        item.appearance_id: item
        for item in selected_appearances
        if item.partition in query_partitions
    }
    all_player_ids_by_team = {
        team_id: sorted(
            {
                player_id
                for appearance in team_appearances
                for player_id in appearance.player_by_shirt.values()
            }
        )
        for team_id, team_appearances in appearances_by_team.items()
    }
    support_cap = int(config["features"]["main_history_support_cap"])
    history_cache: dict[
        tuple[str, str],
        HistoricalCompactProfile,
    ] = {}
    for query in query_by_id.values():
        for player_id in all_player_ids_by_team[query.team_id]:
            history_cache[(query.appearance_id, player_id)] = _historical_profile(
                query,
                player_id,
                appearances_by_team,
                stats_by_appearance,
                support_cap=support_cap,
            )

    event_lookup = _event_index_by_sample_key(data)
    shrinkages = [
        float(value)
        for value in config["features"]["shrinkage_match_equivalents"]
    ]
    main_shrinkage = float(
        config["features"]["main_shrinkage_match_equivalent"]
    )
    if main_shrinkage not in shrinkages:
        raise ValueError("Main shrinkage must be included in shrinkage sensitivities.")
    shuffle_seeds = [
        int(value) for value in config["features"]["shuffled_history_seeds"]
    ]

    role_context_rows: list[np.ndarray] = []
    residual_rows: dict[str, list[np.ndarray]] = defaultdict(list)
    support_values: list[int] = []
    alpha_values: list[float] = []
    role_prior_modes: dict[str, int] = defaultdict(int)
    role_prior_peer_counts: list[int] = []
    shuffle_missing = {seed: 0 for seed in shuffle_seeds}
    chronology_violations = 0

    for row_index, sample_id in enumerate(base.sample_ids):
        team_id = base.team_ids[row_index]
        match_id = base.match_ids[row_index]
        appearance_id = f"{team_id}:{match_id}"
        appearance = query_by_id[appearance_id]
        raw_player_id = _sample_raw_player_id(sample_id)
        event_key = (
            match_id,
            int(base.periods[row_index]),
            int(base.frames[row_index]),
            raw_player_id,
        )
        event_index = event_lookup[event_key]
        actor_id = base.actor_ids[row_index]
        actor_role = int(data.event_role_id[event_index])
        active = _active_focal_players(data, event_index, appearance)
        if actor_id not in {player_id for player_id, _role in active}:
            active.append((actor_id, actor_role))
        histories = {
            player_id: history_cache[(appearance_id, player_id)]
            for player_id, _role in active
        }
        actor_history = histories[actor_id]
        for support_appearance_id in actor_history.support_appearance_ids:
            support_match = next(
                item
                for item in appearances_by_team[team_id]
                if item.appearance_id == support_appearance_id
            )
            if support_match.match_date >= appearance.match_date:
                chronology_violations += 1

        role_prior, prior_mode, peer_count = _role_prior(
            active,
            histories,
            actor_id=actor_id,
            actor_role=actor_role,
        )
        role_context_rows.append(role_prior)
        role_prior_modes[prior_mode] += 1
        role_prior_peer_counts.append(peer_count)
        support = actor_history.support_matches
        support_values.append(support)
        actor_values = (
            actor_history.values if actor_history.available else role_prior
        )
        for shrinkage in shrinkages:
            residual, alpha = shrunk_player_deviation(
                actor_values,
                role_prior,
                support_matches=support,
                shrinkage_match_equivalent=shrinkage,
            )
            residual_rows[_lambda_component_name(shrinkage)].append(
                np.concatenate(
                    [
                        residual,
                        np.asarray(
                            [
                                math.log1p(support),
                                float(actor_history.available),
                                alpha,
                            ]
                        ),
                    ]
                )
            )
            if shrinkage == main_shrinkage:
                alpha_values.append(alpha)
        unshrunk = (
            actor_values - role_prior
            if actor_history.available
            else np.zeros_like(role_prior)
        )
        residual_rows["player_residual_unshrunk"].append(
            np.concatenate(
                [
                    unshrunk,
                    np.asarray(
                        [
                            math.log1p(support),
                            float(actor_history.available),
                            1.0 if actor_history.available else 0.0,
                        ]
                    ),
                ]
            )
        )
        for seed in shuffle_seeds:
            donor_id = _shuffled_donor(
                sample_id=sample_id,
                seed=seed,
                active=active,
                histories=histories,
                actor_id=actor_id,
                actor_role=actor_role,
            )
            if donor_id is None:
                donor_values = role_prior
                shuffle_missing[seed] += 1
            else:
                donor_values = histories[donor_id].values
            shuffled_residual, alpha = shrunk_player_deviation(
                donor_values,
                role_prior,
                support_matches=support,
                shrinkage_match_equivalent=main_shrinkage,
            )
            residual_rows[f"shuffled_player_residual_seed_{seed}"].append(
                np.concatenate(
                    [
                        shuffled_residual,
                        np.asarray(
                            [
                                math.log1p(support),
                                float(actor_history.available),
                                alpha,
                            ]
                        ),
                    ]
                )
            )

    components = {
        "geometry": np.asarray(base.components["geometry"], dtype=np.float64),
        "role": np.asarray(base.components["role"], dtype=np.float64),
        "identity": np.asarray(base.components["identity"], dtype=np.float64),
        "role_context_profile": np.stack(role_context_rows).astype(np.float64),
        **{
            name: np.stack(rows).astype(np.float64)
            for name, rows in residual_rows.items()
        },
    }
    residual_feature_names = [
        *[f"shrunk_{name}" for name in COMPACT_PROFILE_FEATURE_NAMES],
        "log_prior_match_support",
        "history_available",
        "shrinkage_alpha",
    ]
    component_names = {
        "geometry": list(base.component_feature_names["geometry"]),
        "role": list(base.component_feature_names["role"]),
        "identity": list(base.component_feature_names["identity"]),
        "role_context_profile": list(COMPACT_PROFILE_FEATURE_NAMES),
        **{
            name: list(residual_feature_names)
            for name in residual_rows
        },
    }
    finite = {
        name: bool(np.isfinite(values).all())
        for name, values in components.items()
    }
    audit_status = (
        "passed"
        if base.audit["status"] == "passed"
        and all(finite.values())
        and chronology_violations == 0
        and not data.metadata["confirmation_match_ids_included"]
        else "failed"
    )
    audit = {
        "status": audit_status,
        "base_feature_audit_status": base.audit["status"],
        "opportunity_count": len(base.sample_ids),
        "query_match_ids": sorted(set(base.match_ids), key=int),
        "confirmation_match_ids_loaded": data.metadata[
            "confirmation_match_ids_included"
        ],
        "chronology_violations": chronology_violations,
        "sample_id_duplicates": len(base.sample_ids) - len(set(base.sample_ids)),
        "compact_profile_features": list(COMPACT_PROFILE_FEATURE_NAMES),
        "compact_profile_dimension": len(COMPACT_PROFILE_FEATURE_NAMES),
        "use_all_strictly_prior_matches": support_cap == 99,
        "support_cap": support_cap,
        "mean_prior_match_support": float(np.mean(support_values)),
        "median_prior_match_support": float(np.median(support_values)),
        "maximum_prior_match_support": int(max(support_values)),
        "missing_actor_history_count": int(
            sum(value == 0 for value in support_values)
        ),
        "missing_actor_history_fraction": float(
            np.mean(np.asarray(support_values) == 0)
        ),
        "mean_main_shrinkage_alpha": float(np.mean(alpha_values)),
        "role_prior_leave_one_player_out": True,
        "role_prior_modes": dict(sorted(role_prior_modes.items())),
        "mean_role_prior_peer_count": float(np.mean(role_prior_peer_counts)),
        "shuffle_missing_donor_rows": {
            str(seed): count for seed, count in shuffle_missing.items()
        },
        "finite_components": finite,
        "v1_base_audit": base.audit,
    }
    return CompactPlayerResidualDataset(
        sample_ids=list(base.sample_ids),
        team_ids=list(base.team_ids),
        match_ids=list(base.match_ids),
        periods=list(base.periods),
        frames=list(base.frames),
        actor_ids=list(base.actor_ids),
        labels={key: np.asarray(value) for key, value in base.labels.items()},
        components=components,
        component_feature_names=component_names,
        audit=audit,
    )


def _subset_indices(match_ids: list[str], selected: set[str]) -> np.ndarray:
    return np.asarray(
        [
            index
            for index, match_id in enumerate(match_ids)
            if match_id in selected
        ],
        dtype=np.int64,
    )


def _fit_views(
    dataset: CompactPlayerResidualDataset,
    config: dict[str, Any],
    *,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    target: str,
    view_names: list[str],
) -> tuple[
    dict[str, LogisticProbe],
    dict[str, np.ndarray],
    dict[str, dict[str, Any]],
]:
    views = dataset.feature_views(
        float(config["features"]["main_shrinkage_match_equivalent"])
    )
    probes: dict[str, LogisticProbe] = {}
    probabilities: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, Any]] = {}
    labels = dataset.labels[target]
    for view_name in view_names:
        features, feature_names = views[view_name]
        probe = fit_logistic_probe(
            features[train_indices],
            labels[train_indices],
            feature_names,
            config["probe"],
        )
        predicted = probe.predict(features[validation_indices])
        probes[f"{target}::{view_name}"] = probe
        probabilities[view_name] = predicted
        metrics[view_name] = binary_metrics(
            labels[validation_indices],
            predicted,
            ece_bins=int(config["evaluation"]["ece_bins"]),
        )
    return probes, probabilities, metrics


def _nll_gain(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    ece_bins: int,
) -> float:
    return float(
        binary_metrics(labels, baseline, ece_bins=ece_bins)["nll"]
        - binary_metrics(labels, candidate, ece_bins=ece_bins)["nll"]
    )


def run_compact_player_residual_development(
    dataset: CompactPlayerResidualDataset,
    config: dict[str, Any],
    *,
    development_fit_match_ids: set[str],
    development_validation_match_ids: set[str],
) -> tuple[dict[str, Any], dict[str, LogisticProbe], dict[str, np.ndarray]]:
    """Fit the frozen V2 views and evaluate validation plus internal CV."""

    if dataset.audit["status"] != "passed":
        raise ValueError("Cannot run V2 from a failed feature audit.")
    train_indices = _subset_indices(
        dataset.match_ids,
        development_fit_match_ids,
    )
    validation_indices = _subset_indices(
        dataset.match_ids,
        development_validation_match_ids,
    )
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("V2 requires development fit and validation rows.")
    views = dataset.feature_views(
        float(config["features"]["main_shrinkage_match_equivalent"])
    )
    unique_view_names = sorted(views)
    primary_target = str(config["evaluation"]["primary_target"])
    secondary_target = str(config["evaluation"]["secondary_target"])
    probes: dict[str, LogisticProbe] = {}
    validation_probabilities: dict[str, np.ndarray] = {}
    all_metrics: dict[str, dict[str, Any]] = {}
    for target in (primary_target, secondary_target):
        target_probes, target_probabilities, target_metrics = _fit_views(
            dataset,
            config,
            train_indices=train_indices,
            validation_indices=validation_indices,
            target=target,
            view_names=unique_view_names,
        )
        probes.update(target_probes)
        validation_probabilities.update(
            {
                f"{target}::{view_name}": values
                for view_name, values in target_probabilities.items()
            }
        )
        all_metrics[target] = target_metrics

    primary_labels = dataset.labels[primary_target][validation_indices]
    primary_baseline = validation_probabilities[
        f"{primary_target}::geometry_role"
    ]
    primary_player = validation_probabilities[
        f"{primary_target}::player_residual"
    ]
    baseline_metrics = all_metrics[primary_target]["geometry_role"]
    player_metrics = all_metrics[primary_target]["player_residual"]
    primary_gain = float(baseline_metrics["nll"] - player_metrics["nll"])
    primary_relative_gain = primary_gain / float(baseline_metrics["nll"])
    block_ids = [
        f"{dataset.match_ids[index]}:{dataset.periods[index]}"
        for index in validation_indices.tolist()
    ]
    bootstrap = blocked_bootstrap_nll_gain(
        primary_labels,
        primary_baseline,
        primary_player,
        block_ids,
        replicates=int(config["evaluation"]["bootstrap_replicates"]),
        seed=int(config["evaluation"]["bootstrap_seed"]),
    )
    per_match: dict[str, dict[str, Any]] = {}
    for match_id in sorted(development_validation_match_ids, key=int):
        local = np.asarray(
            [
                offset
                for offset, index in enumerate(validation_indices.tolist())
                if dataset.match_ids[index] == match_id
            ],
            dtype=np.int64,
        )
        per_match[match_id] = {
            "examples": len(local),
            "nll_gain": _nll_gain(
                primary_labels[local],
                primary_baseline[local],
                primary_player[local],
                ece_bins=int(config["evaluation"]["ece_bins"]),
            ),
        }
    shuffle_names = sorted(
        name
        for name in views
        if name.startswith("shuffled_player_residual_seed_")
    )
    shuffle_nll = {
        name: float(all_metrics[primary_target][name]["nll"])
        for name in shuffle_names
    }

    secondary_baseline_metrics = all_metrics[secondary_target]["geometry_role"]
    secondary_player_metrics = all_metrics[secondary_target]["player_residual"]
    secondary_gain = float(
        secondary_baseline_metrics["nll"] - secondary_player_metrics["nll"]
    )
    secondary_relative_gain = secondary_gain / float(
        secondary_baseline_metrics["nll"]
    )

    cv_view_names = [
        "geometry_role",
        "role_context",
        "player_residual",
        *shuffle_names,
    ]
    cv_labels: list[np.ndarray] = []
    cv_probabilities: dict[str, list[np.ndarray]] = defaultdict(list)
    cv_block_ids: list[str] = []
    cv_fold_results: dict[str, dict[str, Any]] = {}
    covered_cv_matches: set[str] = set()
    for fold in config["evaluation"]["internal_cv_folds"]:
        fold_name = str(fold["name"])
        fold_validation_ids = {
            str(value) for value in fold["validation_match_ids"]
        }
        if fold_validation_ids & covered_cv_matches:
            raise ValueError("Internal CV validation folds overlap.")
        covered_cv_matches |= fold_validation_ids
        fold_train_ids = development_fit_match_ids - fold_validation_ids
        fold_train_indices = _subset_indices(dataset.match_ids, fold_train_ids)
        fold_validation_indices = _subset_indices(
            dataset.match_ids,
            fold_validation_ids,
        )
        _fold_probes, fold_probabilities, fold_metrics = _fit_views(
            dataset,
            config,
            train_indices=fold_train_indices,
            validation_indices=fold_validation_indices,
            target=primary_target,
            view_names=cv_view_names,
        )
        fold_labels = dataset.labels[primary_target][fold_validation_indices]
        fold_gain = float(
            fold_metrics["geometry_role"]["nll"]
            - fold_metrics["player_residual"]["nll"]
        )
        cv_fold_results[fold_name] = {
            "train_match_ids": sorted(fold_train_ids, key=int),
            "validation_match_ids": sorted(fold_validation_ids, key=int),
            "examples": len(fold_validation_indices),
            "metrics": fold_metrics,
            "primary_nll_gain": fold_gain,
            "primary_relative_nll_improvement": (
                fold_gain / float(fold_metrics["geometry_role"]["nll"])
            ),
        }
        cv_labels.append(fold_labels)
        for view_name in cv_view_names:
            cv_probabilities[view_name].append(fold_probabilities[view_name])
        cv_block_ids.extend(
            [
                f"{dataset.match_ids[index]}:{dataset.periods[index]}"
                for index in fold_validation_indices.tolist()
            ]
        )
    if covered_cv_matches != development_fit_match_ids:
        raise ValueError(
            "Internal CV folds must cover every development-fit match once."
        )
    pooled_labels = np.concatenate(cv_labels)
    pooled_probabilities = {
        name: np.concatenate(values)
        for name, values in cv_probabilities.items()
    }
    cv_pooled_metrics = {
        name: binary_metrics(
            pooled_labels,
            values,
            ece_bins=int(config["evaluation"]["ece_bins"]),
        )
        for name, values in pooled_probabilities.items()
    }
    cv_primary_gain = float(
        cv_pooled_metrics["geometry_role"]["nll"]
        - cv_pooled_metrics["player_residual"]["nll"]
    )
    cv_primary_relative_gain = cv_primary_gain / float(
        cv_pooled_metrics["geometry_role"]["nll"]
    )
    cv_bootstrap = blocked_bootstrap_nll_gain(
        pooled_labels,
        pooled_probabilities["geometry_role"],
        pooled_probabilities["player_residual"],
        cv_block_ids,
        replicates=int(config["evaluation"]["bootstrap_replicates"]),
        seed=int(config["evaluation"]["bootstrap_seed"]) + 50_000,
    )

    gate_config = config["development_gate"]
    checks = {
        "minimum_primary_relative_nll_improvement_vs_geometry_role": (
            primary_relative_gain
            >= float(
                gate_config[
                    "minimum_primary_relative_nll_improvement_vs_geometry_role"
                ]
            )
        ),
        "positive_primary_bootstrap_lower_bound": (
            float(bootstrap["ci95"][0]) > 0.0
            if bool(
                gate_config["require_positive_primary_bootstrap_lower_bound"]
            )
            else True
        ),
        "minimum_positive_validation_matches": (
            sum(float(value["nll_gain"]) > 0.0 for value in per_match.values())
            >= int(gate_config["minimum_positive_validation_matches"])
        ),
        "better_than_role_context": (
            float(player_metrics["nll"])
            < float(all_metrics[primary_target]["role_context"]["nll"])
            if bool(gate_config["require_better_than_role_context"])
            else True
        ),
        "better_than_every_player_shuffle": (
            all(float(player_metrics["nll"]) < value for value in shuffle_nll.values())
            if bool(gate_config["require_better_than_every_player_shuffle"])
            else True
        ),
        "primary_brier_noninferiority": (
            float(player_metrics["brier"]) <= float(baseline_metrics["brier"])
            if bool(gate_config["require_primary_brier_noninferiority"])
            else True
        ),
        "secondary_nll_noninferiority_vs_geometry_role": (
            secondary_relative_gain
            >= float(
                gate_config[
                    "minimum_secondary_relative_nll_change_vs_geometry_role"
                ]
            )
        ),
        "minimum_internal_cv_primary_relative_nll_improvement": (
            cv_primary_relative_gain
            >= float(
                gate_config[
                    "minimum_internal_cv_primary_relative_nll_improvement"
                ]
            )
        ),
        "minimum_positive_internal_cv_folds": (
            sum(
                float(value["primary_nll_gain"]) > 0.0
                for value in cv_fold_results.values()
            )
            >= int(gate_config["minimum_positive_internal_cv_folds"])
        ),
        "integrity_audits": (
            dataset.audit["status"] == "passed"
            if bool(gate_config["require_integrity_audits"])
            else True
        ),
    }
    result = {
        "status": "v2_development_metrics_opened",
        "claim_status": "development_only",
        "design_informed_by_v1_validation": True,
        "metrics": all_metrics,
        "primary_comparison": {
            "target": primary_target,
            "baseline": "geometry_role",
            "model": "player_residual",
            "nll_gain": primary_gain,
            "relative_nll_improvement": primary_relative_gain,
            "brier_gain": float(
                baseline_metrics["brier"] - player_metrics["brier"]
            ),
            "blocked_bootstrap": bootstrap,
            "per_match": per_match,
            "role_context_nll": float(
                all_metrics[primary_target]["role_context"]["nll"]
            ),
            "shuffle_nll": shuffle_nll,
        },
        "secondary_comparison": {
            "target": secondary_target,
            "baseline": "geometry_role",
            "model": "player_residual",
            "nll_gain": secondary_gain,
            "relative_nll_improvement": secondary_relative_gain,
        },
        "internal_cv": {
            "pooled_examples": len(pooled_labels),
            "pooled_metrics": cv_pooled_metrics,
            "primary_nll_gain": cv_primary_gain,
            "primary_relative_nll_improvement": cv_primary_relative_gain,
            "blocked_bootstrap": cv_bootstrap,
            "folds": cv_fold_results,
        },
        "gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "thresholds": dict(gate_config),
        },
        "train_examples": len(train_indices),
        "validation_examples": len(validation_indices),
        "feature_audit": dataset.audit,
        "confirmation_eligible": all(checks.values()),
        "confirmatory_metrics_loaded": False,
    }
    return result, probes, validation_probabilities
