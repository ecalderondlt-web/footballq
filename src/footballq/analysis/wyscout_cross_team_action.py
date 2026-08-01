"""Support-only player profiles for held-out pass-destination prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from footballq.analysis.wyscout_player_memory import (
    binary_metrics,
    binary_nll,
    match_bootstrap_nll_gain,
    same_team_role_shuffle,
    select_last_player_matches,
)

NUM_DESTINATION_ZONES = 30
COUNT_COLUMNS = tuple(f"zone_{index}" for index in range(NUM_DESTINATION_ZONES))
CONTEXT_KEYS = ["role", "start_zone"]
PENALTY_OUTCOME = "penalty_entry"


@dataclass(frozen=True)
class DestinationAggregates:
    global_counts: np.ndarray
    context: pd.DataFrame
    team_context: pd.DataFrame
    team_total: pd.DataFrame
    player_context: pd.DataFrame
    player_total: pd.DataFrame
    player_catalog: pd.DataFrame


@dataclass(frozen=True)
class DestinationCache:
    destination_zone: np.ndarray
    penalty_entry: np.ndarray
    penalty_entry_valid: np.ndarray
    match_ids: np.ndarray
    player_ids: np.ndarray
    support_match_count: np.ndarray
    global_counts: np.ndarray
    context_counts: np.ndarray
    team_context_counts: np.ndarray
    team_total_counts: np.ndarray
    player_context_counts: np.ndarray
    player_total_counts: np.ndarray


@dataclass(frozen=True)
class PenaltyAggregates:
    global_successes: float
    global_count: float
    context: pd.DataFrame
    team_context: pd.DataFrame
    team_total: pd.DataFrame
    player_context: pd.DataFrame
    player_total: pd.DataFrame
    player_catalog: pd.DataFrame


@dataclass(frozen=True)
class PenaltyCache:
    outcome: np.ndarray
    match_ids: np.ndarray
    player_ids: np.ndarray
    support_match_count: np.ndarray
    global_successes: float
    global_count: float
    context_successes: np.ndarray
    context_count: np.ndarray
    team_context_successes: np.ndarray
    team_context_count: np.ndarray
    team_total_successes: np.ndarray
    team_total_count: np.ndarray
    player_context_successes: np.ndarray
    player_context_count: np.ndarray
    player_total_successes: np.ndarray
    player_total_count: np.ndarray


def _count_table(
    frame: pd.DataFrame,
    keys: list[str],
    *,
    prefix: str,
) -> pd.DataFrame:
    table = (
        frame.groupby([*keys, "destination_zone"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(NUM_DESTINATION_ZONES), fill_value=0)
    )
    table.columns = [f"{prefix}_{column}" for column in COUNT_COLUMNS]
    return table.reset_index()


def _mode_catalog(support: pd.DataFrame) -> pd.DataFrame:
    team = (
        support.groupby(["player_id", "team_id"], observed=True)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(
            ["player_id", "count", "team_id"],
            ascending=[True, False, True],
        )
        .drop_duplicates("player_id")
    )
    role = (
        support.groupby(["player_id", "role"], observed=True)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(
            ["player_id", "count", "role"],
            ascending=[True, False, True],
        )
        .drop_duplicates("player_id")
    )
    matches = (
        support[["player_id", "match_id"]]
        .drop_duplicates()
        .groupby("player_id", observed=True)
        .size()
        .rename("support_match_count")
        .reset_index()
    )
    return (
        team[["player_id", "team_id"]]
        .rename(columns={"team_id": "support_team_id"})
        .merge(role[["player_id", "role"]], on="player_id", validate="one_to_one")
        .merge(matches, on="player_id", validate="one_to_one")
    )


def build_destination_aggregates(
    support: pd.DataFrame,
    *,
    match_cap: int,
) -> DestinationAggregates:
    """Build pooled and last-K player destination distributions."""

    profile_support = select_last_player_matches(support, match_cap)
    return DestinationAggregates(
        global_counts=np.bincount(
            support["destination_zone"].to_numpy(dtype=np.int64),
            minlength=NUM_DESTINATION_ZONES,
        ).astype(np.float64),
        context=_count_table(support, CONTEXT_KEYS, prefix="context"),
        team_context=_count_table(
            support,
            ["team_id", *CONTEXT_KEYS],
            prefix="team_context",
        ),
        team_total=_count_table(support, ["team_id"], prefix="team_total"),
        player_context=_count_table(
            profile_support,
            ["player_id", "start_zone"],
            prefix="player_context",
        ),
        player_total=_count_table(
            profile_support,
            ["player_id"],
            prefix="player_total",
        ),
        player_catalog=_mode_catalog(support),
    )


def _penalty_rows(
    frame: pd.DataFrame,
    *,
    query_start_x_min: float,
) -> pd.DataFrame:
    work = frame.loc[frame["start_x"] >= query_start_x_min].copy()
    starts_inside = (
        (work["start_x"] >= 83.0)
        & (work["start_y"] >= 21.0)
        & (work["start_y"] <= 79.0)
    )
    work = work.loc[~starts_inside].copy()
    work[PENALTY_OUTCOME] = (
        (work["destination_x"] >= 83.0)
        & (work["destination_y"] >= 21.0)
        & (work["destination_y"] <= 79.0)
    ).astype(np.float64)
    return work


def _outcome_table(
    frame: pd.DataFrame,
    keys: list[str],
    *,
    prefix: str,
) -> pd.DataFrame:
    return (
        frame.groupby(keys, observed=True)[PENALTY_OUTCOME]
        .agg(["sum", "count"])
        .reset_index()
        .rename(
            columns={
                "sum": f"{prefix}_successes",
                "count": f"{prefix}_count",
            }
        )
    )


def build_penalty_aggregates(
    support: pd.DataFrame,
    *,
    match_cap: int,
    query_start_x_min: float,
) -> PenaltyAggregates:
    """Build exact penalty-entry counts using strictly prior support rows."""

    eligible_support = _penalty_rows(
        support,
        query_start_x_min=query_start_x_min,
    )
    profile_support = _penalty_rows(
        select_last_player_matches(support, match_cap),
        query_start_x_min=query_start_x_min,
    )
    if eligible_support.empty:
        raise ValueError("No support passes are eligible for penalty-entry modeling.")
    return PenaltyAggregates(
        global_successes=float(eligible_support[PENALTY_OUTCOME].sum()),
        global_count=float(len(eligible_support)),
        context=_outcome_table(
            eligible_support,
            CONTEXT_KEYS,
            prefix="context",
        ),
        team_context=_outcome_table(
            eligible_support,
            ["team_id", *CONTEXT_KEYS],
            prefix="team_context",
        ),
        team_total=_outcome_table(
            eligible_support,
            ["team_id"],
            prefix="team_total",
        ),
        player_context=_outcome_table(
            profile_support,
            ["player_id", "start_zone"],
            prefix="player_context",
        ),
        player_total=_outcome_table(
            profile_support,
            ["player_id"],
            prefix="player_total",
        ),
        player_catalog=_mode_catalog(support),
    )


def _merge_outcome_counts(
    frame: pd.DataFrame,
    table: pd.DataFrame,
    *,
    keys: list[str],
    prefix: str,
) -> pd.DataFrame:
    merged = frame.merge(table, on=keys, how="left", validate="many_to_one")
    columns = [f"{prefix}_successes", f"{prefix}_count"]
    merged[columns] = merged[columns].fillna(0.0)
    return merged


def prepare_penalty_cache(
    query: pd.DataFrame,
    aggregates: PenaltyAggregates,
    *,
    minimum_prior_matches: int,
    query_start_x_min: float,
    profile_mapping: dict[int, int] | None = None,
) -> PenaltyCache:
    """Attach exact support-only penalty-entry counts to held-out query rows."""

    work = _penalty_rows(
        query,
        query_start_x_min=query_start_x_min,
    )[
        [
            "match_id",
            "player_id",
            "role",
            "start_zone",
            PENALTY_OUTCOME,
        ]
    ].copy()
    catalog = aggregates.player_catalog.rename(
        columns={"role": "support_role"}
    )
    work = work.merge(catalog, on="player_id", how="left", validate="many_to_one")
    work["support_team_id"] = work["support_team_id"].fillna(0).astype(int)
    work["support_match_count"] = (
        work["support_match_count"].fillna(0).astype(int)
    )
    if profile_mapping is None:
        work["profile_context_player_id"] = work["player_id"]
    else:
        work["profile_context_player_id"] = (
            work["player_id"]
            .map(profile_mapping)
            .fillna(work["player_id"])
            .astype(int)
        )
    work = _merge_outcome_counts(
        work,
        aggregates.context,
        keys=CONTEXT_KEYS,
        prefix="context",
    )
    team_context = aggregates.team_context.rename(
        columns={"team_id": "support_team_id"}
    )
    work = _merge_outcome_counts(
        work,
        team_context,
        keys=["support_team_id", *CONTEXT_KEYS],
        prefix="team_context",
    )
    team_total = aggregates.team_total.rename(
        columns={"team_id": "support_team_id"}
    )
    work = _merge_outcome_counts(
        work,
        team_total,
        keys=["support_team_id"],
        prefix="team_total",
    )
    player_context = aggregates.player_context.rename(
        columns={"player_id": "profile_context_player_id"}
    )
    work = _merge_outcome_counts(
        work,
        player_context,
        keys=["profile_context_player_id", "start_zone"],
        prefix="player_context",
    )
    work = _merge_outcome_counts(
        work,
        aggregates.player_total,
        keys=["player_id"],
        prefix="player_total",
    )
    unsupported = work["support_match_count"] < minimum_prior_matches
    for prefix in ("player_context", "player_total"):
        columns = [f"{prefix}_successes", f"{prefix}_count"]
        work.loc[unsupported, columns] = 0.0
    return PenaltyCache(
        outcome=work[PENALTY_OUTCOME].to_numpy(dtype=np.float64),
        match_ids=work["match_id"].to_numpy(),
        player_ids=work["player_id"].to_numpy(dtype=np.int64),
        support_match_count=work["support_match_count"].to_numpy(dtype=np.int64),
        global_successes=aggregates.global_successes,
        global_count=aggregates.global_count,
        context_successes=work["context_successes"].to_numpy(dtype=np.float64),
        context_count=work["context_count"].to_numpy(dtype=np.float64),
        team_context_successes=work["team_context_successes"].to_numpy(
            dtype=np.float64
        ),
        team_context_count=work["team_context_count"].to_numpy(dtype=np.float64),
        team_total_successes=work["team_total_successes"].to_numpy(
            dtype=np.float64
        ),
        team_total_count=work["team_total_count"].to_numpy(dtype=np.float64),
        player_context_successes=work["player_context_successes"].to_numpy(
            dtype=np.float64
        ),
        player_context_count=work["player_context_count"].to_numpy(
            dtype=np.float64
        ),
        player_total_successes=work["player_total_successes"].to_numpy(
            dtype=np.float64
        ),
        player_total_count=work["player_total_count"].to_numpy(dtype=np.float64),
    )


def _merge_counts(
    frame: pd.DataFrame,
    table: pd.DataFrame,
    *,
    keys: list[str],
    prefix: str,
) -> pd.DataFrame:
    columns = [f"{prefix}_{column}" for column in COUNT_COLUMNS]
    merged = frame.merge(table, on=keys, how="left", validate="many_to_one")
    merged[columns] = merged[columns].fillna(0.0)
    return merged


def _count_array(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    return frame[
        [f"{prefix}_{column}" for column in COUNT_COLUMNS]
    ].to_numpy(dtype=np.float64)


def prepare_destination_cache(
    query: pd.DataFrame,
    aggregates: DestinationAggregates,
    *,
    minimum_prior_matches: int,
    profile_mapping: dict[int, int] | None = None,
) -> DestinationCache:
    """Attach support-only destination counts to held-out query passes."""

    columns = [
        "match_id",
        "player_id",
        "role",
        "start_zone",
        "start_x",
        "start_y",
        "destination_x",
        "destination_y",
        "destination_zone",
    ]
    work = query[columns].copy()
    catalog = aggregates.player_catalog.rename(
        columns={"role": "support_role"}
    )
    work = work.merge(catalog, on="player_id", how="left", validate="many_to_one")
    work["support_team_id"] = work["support_team_id"].fillna(0).astype(int)
    work["support_match_count"] = (
        work["support_match_count"].fillna(0).astype(int)
    )
    if profile_mapping is None:
        work["profile_context_player_id"] = work["player_id"]
    else:
        work["profile_context_player_id"] = (
            work["player_id"]
            .map(profile_mapping)
            .fillna(work["player_id"])
            .astype(int)
        )
    work = _merge_counts(
        work,
        aggregates.context,
        keys=CONTEXT_KEYS,
        prefix="context",
    )
    team_context = aggregates.team_context.rename(
        columns={"team_id": "support_team_id"}
    )
    work = _merge_counts(
        work,
        team_context,
        keys=["support_team_id", *CONTEXT_KEYS],
        prefix="team_context",
    )
    team_total = aggregates.team_total.rename(
        columns={"team_id": "support_team_id"}
    )
    work = _merge_counts(
        work,
        team_total,
        keys=["support_team_id"],
        prefix="team_total",
    )
    player_context = aggregates.player_context.rename(
        columns={"player_id": "profile_context_player_id"}
    )
    work = _merge_counts(
        work,
        player_context,
        keys=["profile_context_player_id", "start_zone"],
        prefix="player_context",
    )
    work = _merge_counts(
        work,
        aggregates.player_total,
        keys=["player_id"],
        prefix="player_total",
    )
    unsupported = work["support_match_count"] < minimum_prior_matches
    for prefix in ("player_context", "player_total"):
        columns = [f"{prefix}_{column}" for column in COUNT_COLUMNS]
        work.loc[unsupported, columns] = 0.0
    starts_inside_penalty_area = (
        (work["start_x"] >= 83.0)
        & (work["start_y"] >= 21.0)
        & (work["start_y"] <= 79.0)
    )
    penalty_entry = (
        (work["destination_x"] >= 83.0)
        & (work["destination_y"] >= 21.0)
        & (work["destination_y"] <= 79.0)
    )
    return DestinationCache(
        destination_zone=work["destination_zone"].to_numpy(dtype=np.int64),
        penalty_entry=penalty_entry.to_numpy(dtype=np.float64),
        penalty_entry_valid=(~starts_inside_penalty_area).to_numpy(dtype=bool),
        match_ids=work["match_id"].to_numpy(),
        player_ids=work["player_id"].to_numpy(dtype=np.int64),
        support_match_count=work["support_match_count"].to_numpy(dtype=np.int64),
        global_counts=aggregates.global_counts,
        context_counts=_count_array(work, "context"),
        team_context_counts=_count_array(work, "team_context"),
        team_total_counts=_count_array(work, "team_total"),
        player_context_counts=_count_array(work, "player_context"),
        player_total_counts=_count_array(work, "player_total"),
    )


def _smooth(
    counts: np.ndarray,
    prior: np.ndarray,
    strength: float,
) -> np.ndarray:
    denominator = counts.sum(axis=1, keepdims=True) + strength
    return (counts + strength * prior) / np.maximum(denominator, 1e-12)


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def predict_penalty_probabilities(
    cache: PenaltyCache,
    *,
    context_prior_strength: float,
    team_prior_strength: float,
    player_prior_strength: float,
    residual_ratio_limit: float,
) -> dict[str, np.ndarray]:
    """Predict exact penalty entry from context and prior player history."""

    global_rate = cache.global_successes / max(cache.global_count, 1.0)
    context = (
        cache.context_successes + context_prior_strength * global_rate
    ) / (cache.context_count + context_prior_strength)
    team_total = (
        cache.team_total_successes + team_prior_strength * global_rate
    ) / (cache.team_total_count + team_prior_strength)
    team_context = (
        cache.team_context_successes + team_prior_strength * context
    ) / (cache.team_context_count + team_prior_strength)
    player_total = (
        cache.player_total_successes + player_prior_strength * team_total
    ) / (cache.player_total_count + player_prior_strength)
    residual_limit = np.log(max(residual_ratio_limit, 1.0))
    player_residual = np.clip(
        _logit(player_total) - _logit(team_total),
        -residual_limit,
        residual_limit,
    )
    rolling_player = _sigmoid(_logit(team_context) + player_residual)
    conditional_player = (
        cache.player_context_successes
        + player_prior_strength * rolling_player
    ) / (cache.player_context_count + player_prior_strength)
    return {
        "context": np.clip(context, 1e-6, 1.0 - 1e-6),
        "team": np.clip(team_context, 1e-6, 1.0 - 1e-6),
        "rolling_player": np.clip(
            rolling_player,
            1e-6,
            1.0 - 1e-6,
        ),
        "conditional_player": np.clip(
            conditional_player,
            1e-6,
            1.0 - 1e-6,
        ),
    }


def penalty_metric_bundle(
    cache: PenaltyCache,
    probability: np.ndarray,
    *,
    minimum_prior_matches: int,
) -> dict[str, Any]:
    supported = cache.support_match_count >= minimum_prior_matches
    if not np.any(supported):
        raise ValueError("No query rows meet the minimum prior-match threshold.")
    return {
        "all_rows": binary_metrics(cache.outcome, probability),
        "supported": binary_metrics(
            cache.outcome[supported],
            probability[supported],
        ),
        "support_coverage": float(np.mean(supported)),
    }


def compare_penalty_models(
    cache: PenaltyCache,
    rolling: np.ndarray,
    conditional: np.ndarray,
    *,
    minimum_prior_matches: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    supported = cache.support_match_count >= minimum_prior_matches
    rolling_metrics = penalty_metric_bundle(
        cache,
        rolling,
        minimum_prior_matches=minimum_prior_matches,
    )
    conditional_metrics = penalty_metric_bundle(
        cache,
        conditional,
        minimum_prior_matches=minimum_prior_matches,
    )
    bootstrap = match_bootstrap_nll_gain(
        cache.outcome[supported],
        rolling[supported],
        conditional[supported],
        cache.match_ids[supported],
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    rolling_nll = float(rolling_metrics["supported"]["nll"])
    conditional_nll = float(conditional_metrics["supported"]["nll"])
    return {
        "rolling_player": rolling_metrics,
        "conditional_player": conditional_metrics,
        "effect": {
            "rolling_minus_conditional_nll": rolling_nll - conditional_nll,
            "relative_nll_improvement": (
                rolling_nll - conditional_nll
            )
            / rolling_nll,
            "match_bootstrap": bootstrap,
        },
    }


def tune_penalty_models(
    cache: PenaltyCache,
    *,
    minimum_prior_matches: int,
    context_prior_strength: float,
    prior_strengths: list[float],
    residual_ratio_limit: float,
) -> dict[str, Any]:
    """Select exact binary smoothing on the validation cohort only."""

    supported = cache.support_match_count >= minimum_prior_matches
    if not np.any(supported):
        raise ValueError("Penalty model tuning has no supported validation rows.")
    candidates = []
    for team_strength in prior_strengths:
        for player_strength in prior_strengths:
            probabilities = predict_penalty_probabilities(
                cache,
                context_prior_strength=context_prior_strength,
                team_prior_strength=team_strength,
                player_prior_strength=player_strength,
                residual_ratio_limit=residual_ratio_limit,
            )
            candidates.append(
                {
                    "team_prior_strength": team_strength,
                    "player_prior_strength": player_strength,
                    "rolling_nll": binary_nll(
                        cache.outcome[supported],
                        probabilities["rolling_player"][supported],
                    ),
                    "conditional_nll": binary_nll(
                        cache.outcome[supported],
                        probabilities["conditional_player"][supported],
                    ),
                }
            )
    best_rolling = min(
        candidates,
        key=lambda row: (
            float(row["rolling_nll"]),
            float(row["team_prior_strength"]),
            float(row["player_prior_strength"]),
        ),
    )
    best_conditional = min(
        candidates,
        key=lambda row: (
            float(row["conditional_nll"]),
            float(row["team_prior_strength"]),
            float(row["player_prior_strength"]),
        ),
    )
    return {
        "best_rolling": best_rolling,
        "best_conditional": best_conditional,
        "candidate_count": len(candidates),
        "selection_metric": "supported_exact_penalty_entry_nll",
    }


def selected_penalty_probabilities(
    cache: PenaltyCache,
    selection: dict[str, Any],
    *,
    context_prior_strength: float,
    residual_ratio_limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    rolling_config = selection["best_rolling"]
    conditional_config = selection["best_conditional"]
    rolling = predict_penalty_probabilities(
        cache,
        context_prior_strength=context_prior_strength,
        team_prior_strength=float(rolling_config["team_prior_strength"]),
        player_prior_strength=float(rolling_config["player_prior_strength"]),
        residual_ratio_limit=residual_ratio_limit,
    )["rolling_player"]
    conditional = predict_penalty_probabilities(
        cache,
        context_prior_strength=context_prior_strength,
        team_prior_strength=float(conditional_config["team_prior_strength"]),
        player_prior_strength=float(conditional_config["player_prior_strength"]),
        residual_ratio_limit=residual_ratio_limit,
    )["conditional_player"]
    return rolling, conditional


def shuffled_penalty_profile_results(
    query: pd.DataFrame,
    aggregates: PenaltyAggregates,
    selection: dict[str, Any],
    *,
    minimum_prior_matches: int,
    query_start_x_min: float,
    context_prior_strength: float,
    residual_ratio_limit: float,
    seeds: list[int],
) -> list[dict[str, Any]]:
    """Evaluate wrong context profiles while preserving rolling player history."""

    results = []
    for seed in seeds:
        mapping = same_team_role_shuffle(aggregates.player_catalog, seed)
        cache = prepare_penalty_cache(
            query,
            aggregates,
            minimum_prior_matches=minimum_prior_matches,
            query_start_x_min=query_start_x_min,
            profile_mapping=mapping,
        )
        _, conditional = selected_penalty_probabilities(
            cache,
            selection,
            context_prior_strength=context_prior_strength,
            residual_ratio_limit=residual_ratio_limit,
        )
        supported = cache.support_match_count >= minimum_prior_matches
        changed = np.asarray(
            [
                mapping.get(int(player_id), int(player_id)) != int(player_id)
                for player_id in cache.player_ids
            ],
            dtype=bool,
        )
        results.append(
            {
                "seed": seed,
                "changed_example_fraction": float(np.mean(changed[supported])),
                "metrics": penalty_metric_bundle(
                    cache,
                    conditional,
                    minimum_prior_matches=minimum_prior_matches,
                ),
            }
        )
    return results


def predict_destination_probabilities(
    cache: DestinationCache,
    *,
    context_prior_strength: float,
    team_prior_strength: float,
    player_prior_strength: float,
    residual_ratio_limit: float,
) -> dict[str, np.ndarray]:
    """Return context, team, rolling-player, and conditional-player models."""

    global_probability = cache.global_counts / cache.global_counts.sum()
    global_rows = np.broadcast_to(
        global_probability,
        cache.context_counts.shape,
    )
    context = _smooth(
        cache.context_counts,
        global_rows,
        context_prior_strength,
    )
    team_total = _smooth(
        cache.team_total_counts,
        global_rows,
        team_prior_strength,
    )
    team_context = _smooth(
        cache.team_context_counts,
        context,
        team_prior_strength,
    )
    player_total = _smooth(
        cache.player_total_counts,
        team_total,
        player_prior_strength,
    )
    ratio = np.clip(
        player_total / np.maximum(team_total, 1e-9),
        1.0 / residual_ratio_limit,
        residual_ratio_limit,
    )
    rolling_player = team_context * ratio
    rolling_player /= np.maximum(
        rolling_player.sum(axis=1, keepdims=True),
        1e-12,
    )
    conditional_player = _smooth(
        cache.player_context_counts,
        rolling_player,
        player_prior_strength,
    )
    return {
        "context": context,
        "team": team_context,
        "rolling_player": rolling_player,
        "conditional_player": conditional_player,
    }


def destination_metrics(
    destination_zone: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float | int]:
    row = np.arange(len(destination_zone))
    true_probability = np.clip(
        probability[row, destination_zone],
        1e-9,
        1.0,
    )
    order = np.argsort(-probability, axis=1, kind="stable")
    target = destination_zone[:, None]
    one_hot = np.zeros_like(probability)
    one_hot[row, destination_zone] = 1.0
    return {
        "examples": int(len(destination_zone)),
        "nll": float(-np.log(true_probability).mean()),
        "brier": float(np.square(probability - one_hot).sum(axis=1).mean()),
        "top1": float(np.mean(order[:, :1] == target)),
        "top3": float(np.mean(np.any(order[:, :3] == target, axis=1))),
        "top5": float(np.mean(np.any(order[:, :5] == target, axis=1))),
    }


def _penalty_probability(probability: np.ndarray) -> np.ndarray:
    return probability[:, [26, 27, 28]].sum(axis=1)


def metric_bundle(
    cache: DestinationCache,
    probability: np.ndarray,
    *,
    minimum_prior_matches: int,
) -> dict[str, Any]:
    supported = cache.support_match_count >= minimum_prior_matches
    penalty = supported & cache.penalty_entry_valid
    return {
        "all_rows": destination_metrics(cache.destination_zone, probability),
        "supported": destination_metrics(
            cache.destination_zone[supported],
            probability[supported],
        ),
        "penalty_entry_supported": binary_metrics(
            cache.penalty_entry[penalty],
            _penalty_probability(probability)[penalty],
        ),
        "support_coverage": float(np.mean(supported)),
    }


def match_bootstrap_destination_gain(
    cache: DestinationCache,
    baseline: np.ndarray,
    challenger: np.ndarray,
    *,
    minimum_prior_matches: int,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> dict[str, float | int]:
    supported = cache.support_match_count >= minimum_prior_matches
    labels = cache.destination_zone[supported]
    matches = cache.match_ids[supported]
    row = np.arange(len(labels))
    baseline_loss = -np.log(np.clip(baseline[supported][row, labels], 1e-9, 1.0))
    challenger_loss = -np.log(
        np.clip(challenger[supported][row, labels], 1e-9, 1.0)
    )
    unique_matches, inverse = np.unique(matches, return_inverse=True)
    gains = np.bincount(
        inverse,
        weights=baseline_loss - challenger_loss,
        minlength=len(unique_matches),
    )
    counts = np.bincount(inverse, minlength=len(unique_matches))
    rng = np.random.default_rng(seed)
    draws = rng.integers(
        0,
        len(unique_matches),
        size=(replicates, len(unique_matches)),
    )
    sampled = gains[draws].sum(axis=1) / counts[draws].sum(axis=1)
    tail = (1.0 - confidence_level) / 2.0
    return {
        "bootstrap_unit": "match_id",
        "bootstrap_unit_count": int(len(unique_matches)),
        "replicates": replicates,
        "point_gain": float(np.mean(baseline_loss - challenger_loss)),
        "ci_lower": float(np.quantile(sampled, tail)),
        "ci_upper": float(np.quantile(sampled, 1.0 - tail)),
    }


def compare_destination_models(
    cache: DestinationCache,
    rolling: np.ndarray,
    conditional: np.ndarray,
    *,
    minimum_prior_matches: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    supported = cache.support_match_count >= minimum_prior_matches
    penalty = supported & cache.penalty_entry_valid
    rolling_metrics = metric_bundle(
        cache,
        rolling,
        minimum_prior_matches=minimum_prior_matches,
    )
    conditional_metrics = metric_bundle(
        cache,
        conditional,
        minimum_prior_matches=minimum_prior_matches,
    )
    destination_bootstrap = match_bootstrap_destination_gain(
        cache,
        rolling,
        conditional,
        minimum_prior_matches=minimum_prior_matches,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    penalty_bootstrap = match_bootstrap_nll_gain(
        cache.penalty_entry[penalty],
        _penalty_probability(rolling)[penalty],
        _penalty_probability(conditional)[penalty],
        cache.match_ids[penalty],
        replicates=bootstrap_replicates,
        seed=bootstrap_seed + 1,
        confidence_level=confidence_level,
    )
    rolling_nll = float(rolling_metrics["supported"]["nll"])
    conditional_nll = float(conditional_metrics["supported"]["nll"])
    return {
        "rolling_player": rolling_metrics,
        "conditional_player": conditional_metrics,
        "destination_effect": {
            "rolling_minus_conditional_nll": rolling_nll - conditional_nll,
            "relative_nll_improvement": (
                rolling_nll - conditional_nll
            )
            / rolling_nll,
            "match_bootstrap": destination_bootstrap,
        },
        "penalty_entry_effect": {
            "rolling_minus_conditional_nll": float(
                rolling_metrics["penalty_entry_supported"]["nll"]
            )
            - float(conditional_metrics["penalty_entry_supported"]["nll"]),
            "match_bootstrap": penalty_bootstrap,
        },
    }


def tune_destination_models(
    cache: DestinationCache,
    *,
    minimum_prior_matches: int,
    context_prior_strength: float,
    prior_strengths: list[float],
    residual_ratio_limit: float,
) -> dict[str, Any]:
    """Select rolling and conditional smoothing on development validation only."""

    supported = cache.support_match_count >= minimum_prior_matches
    candidates = []
    for team_strength in prior_strengths:
        for player_strength in prior_strengths:
            probabilities = predict_destination_probabilities(
                cache,
                context_prior_strength=context_prior_strength,
                team_prior_strength=team_strength,
                player_prior_strength=player_strength,
                residual_ratio_limit=residual_ratio_limit,
            )
            candidates.append(
                {
                    "team_prior_strength": team_strength,
                    "player_prior_strength": player_strength,
                    "rolling_nll": destination_metrics(
                        cache.destination_zone[supported],
                        probabilities["rolling_player"][supported],
                    )["nll"],
                    "conditional_nll": destination_metrics(
                        cache.destination_zone[supported],
                        probabilities["conditional_player"][supported],
                    )["nll"],
                }
            )
    best_rolling = min(
        candidates,
        key=lambda row: (
            float(row["rolling_nll"]),
            float(row["team_prior_strength"]),
            float(row["player_prior_strength"]),
        ),
    )
    best_conditional = min(
        candidates,
        key=lambda row: (
            float(row["conditional_nll"]),
            float(row["team_prior_strength"]),
            float(row["player_prior_strength"]),
        ),
    )
    return {
        "best_rolling": best_rolling,
        "best_conditional": best_conditional,
        "candidate_count": len(candidates),
        "selection_metric": "supported_destination_nll",
    }


def tune_penalty_entry_models(
    cache: DestinationCache,
    *,
    minimum_prior_matches: int,
    context_prior_strength: float,
    prior_strengths: list[float],
    residual_ratio_limit: float,
) -> dict[str, Any]:
    """Select smoothing using validation penalty-entry NLL only."""

    eligible = (
        (cache.support_match_count >= minimum_prior_matches)
        & cache.penalty_entry_valid
    )
    candidates = []
    for team_strength in prior_strengths:
        for player_strength in prior_strengths:
            probabilities = predict_destination_probabilities(
                cache,
                context_prior_strength=context_prior_strength,
                team_prior_strength=team_strength,
                player_prior_strength=player_strength,
                residual_ratio_limit=residual_ratio_limit,
            )
            rolling_probability = _penalty_probability(
                probabilities["rolling_player"]
            )[eligible]
            conditional_probability = _penalty_probability(
                probabilities["conditional_player"]
            )[eligible]
            candidates.append(
                {
                    "team_prior_strength": team_strength,
                    "player_prior_strength": player_strength,
                    "rolling_nll": binary_metrics(
                        cache.penalty_entry[eligible],
                        rolling_probability,
                    )["nll"],
                    "conditional_nll": binary_metrics(
                        cache.penalty_entry[eligible],
                        conditional_probability,
                    )["nll"],
                }
            )
    best_rolling = min(
        candidates,
        key=lambda row: (
            float(row["rolling_nll"]),
            float(row["team_prior_strength"]),
            float(row["player_prior_strength"]),
        ),
    )
    best_conditional = min(
        candidates,
        key=lambda row: (
            float(row["conditional_nll"]),
            float(row["team_prior_strength"]),
            float(row["player_prior_strength"]),
        ),
    )
    return {
        "best_rolling": best_rolling,
        "best_conditional": best_conditional,
        "candidate_count": len(candidates),
        "selection_metric": "supported_penalty_entry_nll",
    }


def selected_probabilities(
    cache: DestinationCache,
    selection: dict[str, Any],
    *,
    context_prior_strength: float,
    residual_ratio_limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    rolling_config = selection["best_rolling"]
    conditional_config = selection["best_conditional"]
    rolling = predict_destination_probabilities(
        cache,
        context_prior_strength=context_prior_strength,
        team_prior_strength=float(rolling_config["team_prior_strength"]),
        player_prior_strength=float(rolling_config["player_prior_strength"]),
        residual_ratio_limit=residual_ratio_limit,
    )["rolling_player"]
    conditional = predict_destination_probabilities(
        cache,
        context_prior_strength=context_prior_strength,
        team_prior_strength=float(conditional_config["team_prior_strength"]),
        player_prior_strength=float(conditional_config["player_prior_strength"]),
        residual_ratio_limit=residual_ratio_limit,
    )["conditional_player"]
    return rolling, conditional


def shuffled_profile_results(
    query: pd.DataFrame,
    aggregates: DestinationAggregates,
    selection: dict[str, Any],
    *,
    minimum_prior_matches: int,
    context_prior_strength: float,
    residual_ratio_limit: float,
    seeds: list[int],
) -> list[dict[str, Any]]:
    results = []
    for seed in seeds:
        mapping = same_team_role_shuffle(aggregates.player_catalog, seed)
        cache = prepare_destination_cache(
            query,
            aggregates,
            minimum_prior_matches=minimum_prior_matches,
            profile_mapping=mapping,
        )
        _, conditional = selected_probabilities(
            cache,
            selection,
            context_prior_strength=context_prior_strength,
            residual_ratio_limit=residual_ratio_limit,
        )
        supported = cache.support_match_count >= minimum_prior_matches
        changed = np.asarray(
            [
                mapping.get(int(player_id), int(player_id)) != int(player_id)
                for player_id in cache.player_ids
            ],
            dtype=bool,
        )
        results.append(
            {
                "seed": seed,
                "changed_example_fraction": float(np.mean(changed[supported])),
                "metrics": metric_bundle(
                    cache,
                    conditional,
                    minimum_prior_matches=minimum_prior_matches,
                ),
            }
        )
    return results
