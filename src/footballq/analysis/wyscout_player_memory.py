"""Leakage-controlled player-memory tests on public Wyscout pass events."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from footballq.repro.splits import load_split_manifest

CONTEXT_KEYS = ["role", "start_zone", "subevent_id"]
PLAYER_CONTEXT_KEYS = ["player_id", "start_zone", "subevent_id"]


@dataclass(frozen=True)
class AggregateTables:
    global_successes: float
    global_count: float
    context: pd.DataFrame
    team_context: pd.DataFrame
    team_total: pd.DataFrame
    player_context: pd.DataFrame
    player_total: pd.DataFrame
    player_catalog: pd.DataFrame


@dataclass(frozen=True)
class PredictionCache:
    outcome: np.ndarray
    match_ids: np.ndarray
    query_player_ids: np.ndarray
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


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def stable_payload_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_competitions(
    dataset_root: Path,
    competitions: list[str],
) -> pd.DataFrame:
    frames = [
        pd.read_parquet(dataset_root / f"passes_{competition}.parquet")
        for competition in competitions
    ]
    return pd.concat(frames, ignore_index=True)


def load_development_frames(
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load only development data and validate manifest lineage and chronology."""

    dataset_root = Path(config["data"]["dataset_root"])
    dataset_manifest_path = Path(config["data"]["dataset_manifest"])
    split = load_split_manifest(config["data"]["development_split_manifest"])
    expected_dataset_hash = str(split.payload["dataset_manifest_sha256"])
    actual_dataset_hash = file_sha256(dataset_manifest_path)
    if actual_dataset_hash != expected_dataset_hash:
        raise ValueError(
            "Wyscout dataset manifest hash differs from the frozen split lineage."
        )
    competitions = [
        str(value) for value in config["development"]["support_competitions"]
    ]
    all_domestic = _load_competitions(dataset_root, competitions)
    all_domestic["match_id_text"] = all_domestic["match_id"].astype(str)
    present_ids = set(all_domestic["match_id_text"].unique())
    support_ids = {str(value) for value in split.payload["support_match_ids"]}
    query_ids = set(split.all_match_ids)
    absent = sorted((support_ids | query_ids) - present_ids)
    if absent:
        raise ValueError(
            "Frozen Wyscout manifest references missing match IDs: "
            + ", ".join(absent[:10])
        )
    if not support_ids.isdisjoint(query_ids):
        raise ValueError("Support and query match IDs overlap.")

    support = all_domestic[all_domestic["match_id_text"].isin(support_ids)].copy()
    train = all_domestic[
        all_domestic["match_id_text"].isin(split.train_match_ids)
    ].copy()
    validation = all_domestic[
        all_domestic["match_id_text"].isin(split.val_match_ids)
    ].copy()
    development = all_domestic[
        all_domestic["match_id_text"].isin(split.test_match_ids)
    ].copy()
    latest_support = str(support["dateutc"].max())
    earliest_query = min(
        str(frame["dateutc"].min())
        for frame in (train, validation, development)
    )
    if latest_support >= earliest_query:
        raise ValueError("Wyscout support rows are not strictly earlier than queries.")
    start_x_min = float(config["task"]["query_start_x_min"])
    queries = {
        name: frame.loc[frame["start_x"] >= start_x_min].copy()
        for name, frame in {
            "train": train,
            "validation": validation,
            "development": development,
        }.items()
    }
    expected_counts = {
        "train": len(split.train_match_ids),
        "validation": len(split.val_match_ids),
        "development": len(split.test_match_ids),
    }
    for name, frame in queries.items():
        actual = int(frame["match_id_text"].nunique())
        if actual != expected_counts[name]:
            raise ValueError(
                f"Wyscout {name} query match count mismatch: "
                f"expected {expected_counts[name]}, got {actual}."
            )
    metadata = {
        "dataset_manifest_path": str(dataset_manifest_path),
        "dataset_manifest_sha256": actual_dataset_hash,
        **split.metadata(),
        "support_match_count": len(support_ids),
        "support_row_count": int(len(support)),
        "latest_support_dateutc": latest_support,
        "earliest_query_dateutc": earliest_query,
        "query_row_counts": {
            name: int(len(frame)) for name, frame in queries.items()
        },
        "query_match_counts": expected_counts,
        "confirmatory_metrics_loaded": False,
    }
    return {"support": support, **queries}, metadata


def select_last_player_matches(
    support: pd.DataFrame,
    match_cap: int,
) -> pd.DataFrame:
    """Keep rows from each player's last K distinct support matches."""

    appearances = (
        support[["player_id", "match_id", "dateutc"]]
        .drop_duplicates(["player_id", "match_id"])
        .sort_values(["player_id", "dateutc", "match_id"])
    )
    selected = appearances.groupby("player_id", sort=False).tail(match_cap)
    return support.merge(
        selected[["player_id", "match_id"]],
        on=["player_id", "match_id"],
        how="inner",
        validate="many_to_one",
    )


def _outcome_counts(
    frame: pd.DataFrame,
    keys: list[str],
    outcome: str,
    *,
    prefix: str,
) -> pd.DataFrame:
    return (
        frame.groupby(keys, observed=True)[outcome]
        .agg(["sum", "count"])
        .reset_index()
        .rename(
            columns={
                "sum": f"{prefix}_successes",
                "count": f"{prefix}_count",
            }
        )
    )


def build_aggregate_tables(
    support: pd.DataFrame,
    *,
    outcome: str,
    match_cap: int,
) -> AggregateTables:
    """Build pooled, club, and last-K player outcome summaries."""

    profile_support = select_last_player_matches(support, match_cap)
    context = _outcome_counts(
        support,
        CONTEXT_KEYS,
        outcome,
        prefix="context",
    )
    team_context = _outcome_counts(
        support,
        ["team_id", *CONTEXT_KEYS],
        outcome,
        prefix="team_context",
    )
    team_total = _outcome_counts(
        support,
        ["team_id"],
        outcome,
        prefix="team_total",
    )
    player_context = _outcome_counts(
        profile_support,
        PLAYER_CONTEXT_KEYS,
        outcome,
        prefix="player_context",
    )
    player_total = _outcome_counts(
        profile_support,
        ["player_id"],
        outcome,
        prefix="player_total",
    )
    player_matches = (
        support[["player_id", "match_id"]]
        .drop_duplicates()
        .groupby("player_id", observed=True)
        .size()
        .rename("support_match_count")
        .reset_index()
    )
    team_counts = (
        support.groupby(["player_id", "team_id"], observed=True)
        .size()
        .rename("support_pass_count")
        .reset_index()
        .sort_values(
            ["player_id", "support_pass_count", "team_id"],
            ascending=[True, False, True],
        )
        .drop_duplicates("player_id")
    )
    player_roles = (
        support.groupby(["player_id", "role"], observed=True)
        .size()
        .rename("role_count")
        .reset_index()
        .sort_values(
            ["player_id", "role_count", "role"],
            ascending=[True, False, True],
        )
        .drop_duplicates("player_id")
    )
    player_catalog = (
        team_counts[["player_id", "team_id"]]
        .merge(
            player_roles[["player_id", "role"]],
            on="player_id",
            validate="one_to_one",
        )
        .merge(player_matches, on="player_id", validate="one_to_one")
        .rename(columns={"team_id": "support_team_id"})
    )
    return AggregateTables(
        global_successes=float(support[outcome].sum()),
        global_count=float(len(support)),
        context=context,
        team_context=team_context,
        team_total=team_total,
        player_context=player_context,
        player_total=player_total,
        player_catalog=player_catalog,
    )


def same_team_role_shuffle(
    player_catalog: pd.DataFrame,
    seed: int,
) -> dict[int, int]:
    """Derange profile IDs within support club and role when possible."""

    rng = random.Random(seed)
    mapping: dict[int, int] = {}
    groups = player_catalog.groupby(["support_team_id", "role"], observed=True)
    for _key, group in groups:
        player_ids = sorted(int(value) for value in group["player_id"])
        if len(player_ids) < 2:
            mapping[player_ids[0]] = player_ids[0]
            continue
        rng.shuffle(player_ids)
        shift = rng.randrange(1, len(player_ids))
        for index, player_id in enumerate(player_ids):
            mapping[player_id] = player_ids[(index + shift) % len(player_ids)]
    return mapping


def prepare_prediction_cache(
    query: pd.DataFrame,
    aggregates: AggregateTables,
    *,
    outcome: str,
    minimum_prior_matches: int,
    profile_mapping: dict[int, int] | None = None,
) -> PredictionCache:
    """Attach aggregate counts to query rows without using query outcomes."""

    columns = [
        "match_id",
        "player_id",
        *CONTEXT_KEYS,
        outcome,
    ]
    work = query[columns].copy()
    catalog = aggregates.player_catalog.rename(
        columns={
            "player_id": "query_player_id",
            "role": "support_role",
        }
    )
    work = work.merge(
        catalog,
        left_on="player_id",
        right_on="query_player_id",
        how="left",
        validate="many_to_one",
    )
    work["support_team_id"] = work["support_team_id"].fillna(0).astype(int)
    work["support_match_count"] = (
        work["support_match_count"].fillna(0).astype(int)
    )
    if profile_mapping is None:
        work["profile_player_id"] = work["player_id"]
    else:
        work["profile_player_id"] = (
            work["player_id"].map(profile_mapping).fillna(work["player_id"]).astype(int)
        )
    work = work.merge(
        aggregates.context,
        on=CONTEXT_KEYS,
        how="left",
        validate="many_to_one",
    )
    team_context = aggregates.team_context.rename(
        columns={"team_id": "support_team_id"}
    )
    work = work.merge(
        team_context,
        on=["support_team_id", *CONTEXT_KEYS],
        how="left",
        validate="many_to_one",
    )
    team_total = aggregates.team_total.rename(
        columns={"team_id": "support_team_id"}
    )
    work = work.merge(
        team_total,
        on="support_team_id",
        how="left",
        validate="many_to_one",
    )
    player_context = aggregates.player_context.rename(
        columns={"player_id": "profile_player_id"}
    )
    work = work.merge(
        player_context,
        left_on=["profile_player_id", "start_zone", "subevent_id"],
        right_on=["profile_player_id", "start_zone", "subevent_id"],
        how="left",
        validate="many_to_one",
    )
    player_total = aggregates.player_total.rename(
        columns={"player_id": "profile_player_id"}
    )
    work = work.merge(
        player_total,
        on="profile_player_id",
        how="left",
        validate="many_to_one",
    )
    count_columns = [
        "context_successes",
        "context_count",
        "team_context_successes",
        "team_context_count",
        "team_total_successes",
        "team_total_count",
        "player_context_successes",
        "player_context_count",
        "player_total_successes",
        "player_total_count",
    ]
    work[count_columns] = work[count_columns].fillna(0.0)
    unsupported = work["support_match_count"] < minimum_prior_matches
    work.loc[
        unsupported,
        [
            "player_context_successes",
            "player_context_count",
            "player_total_successes",
            "player_total_count",
        ],
    ] = 0.0
    return PredictionCache(
        outcome=work[outcome].to_numpy(dtype=np.float64),
        match_ids=work["match_id"].to_numpy(),
        query_player_ids=work["player_id"].to_numpy(),
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


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def predict_probabilities(
    cache: PredictionCache,
    *,
    context_prior_strength: float,
    team_prior_strength: float,
    player_prior_strength: float,
) -> dict[str, np.ndarray]:
    """Return nested context, club-history, and player-history probabilities."""

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
    residual = np.clip(_logit(player_total) - _logit(team_total), -3.0, 3.0)
    player_context_prior = _sigmoid(_logit(team_context) + residual)
    player_context = (
        cache.player_context_successes
        + player_prior_strength * player_context_prior
    ) / (cache.player_context_count + player_prior_strength)
    return {
        "context": np.clip(context, 1e-6, 1.0 - 1e-6),
        "team": np.clip(team_context, 1e-6, 1.0 - 1e-6),
        "player": np.clip(player_context, 1e-6, 1.0 - 1e-6),
    }


def binary_nll(outcome: np.ndarray, probability: np.ndarray) -> float:
    probability = np.clip(probability, 1e-9, 1.0 - 1e-9)
    return float(
        -np.mean(
            outcome * np.log(probability)
            + (1.0 - outcome) * np.log(1.0 - probability)
        )
    )


def average_precision(outcome: np.ndarray, probability: np.ndarray) -> float:
    positives = float(outcome.sum())
    if positives <= 0:
        return 0.0
    order = np.argsort(-probability, kind="stable")
    ranked = outcome[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision * ranked) / positives)


def binary_metrics(
    outcome: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float | int]:
    return {
        "examples": int(len(outcome)),
        "positive_rate": float(np.mean(outcome)),
        "nll": binary_nll(outcome, probability),
        "brier": float(np.mean((probability - outcome) ** 2)),
        "average_precision": average_precision(outcome, probability),
    }


def match_bootstrap_nll_gain(
    outcome: np.ndarray,
    baseline: np.ndarray,
    challenger: np.ndarray,
    match_ids: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> dict[str, float | int]:
    """Bootstrap the paired baseline-minus-challenger NLL gain by match."""

    baseline_loss = -(
        outcome * np.log(np.clip(baseline, 1e-9, 1.0))
        + (1.0 - outcome) * np.log(np.clip(1.0 - baseline, 1e-9, 1.0))
    )
    challenger_loss = -(
        outcome * np.log(np.clip(challenger, 1e-9, 1.0))
        + (1.0 - outcome) * np.log(np.clip(1.0 - challenger, 1e-9, 1.0))
    )
    unique_matches, inverse = np.unique(match_ids, return_inverse=True)
    gains = np.bincount(
        inverse,
        weights=baseline_loss - challenger_loss,
        minlength=len(unique_matches),
    )
    counts = np.bincount(inverse, minlength=len(unique_matches))
    rng = np.random.default_rng(seed)
    draw = rng.integers(
        0,
        len(unique_matches),
        size=(replicates, len(unique_matches)),
    )
    sampled = gains[draw].sum(axis=1) / counts[draw].sum(axis=1)
    tail = (1.0 - confidence_level) / 2.0
    return {
        "bootstrap_unit_count": int(len(unique_matches)),
        "replicates": int(replicates),
        "point_gain": float(np.mean(baseline_loss - challenger_loss)),
        "bootstrap_mean_gain": float(np.mean(sampled)),
        "ci_lower": float(np.quantile(sampled, tail)),
        "ci_upper": float(np.quantile(sampled, 1.0 - tail)),
    }


def _coverage(cache: PredictionCache, thresholds: list[int]) -> dict[str, float]:
    return {
        str(threshold): float(np.mean(cache.support_match_count >= threshold))
        for threshold in thresholds
    }


def _metric_bundle(
    cache: PredictionCache,
    probabilities: dict[str, np.ndarray],
    minimum_prior_matches: int,
) -> dict[str, Any]:
    supported = cache.support_match_count >= minimum_prior_matches
    return {
        name: binary_metrics(cache.outcome, prediction)
        for name, prediction in probabilities.items()
    } | {
        "supported_subset": {
            "minimum_prior_matches": minimum_prior_matches,
            "examples": int(supported.sum()),
            "team": binary_metrics(
                cache.outcome[supported],
                probabilities["team"][supported],
            ),
            "player": binary_metrics(
                cache.outcome[supported],
                probabilities["player"][supported],
            ),
        }
    }


def run_development_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Tune on Spain and evaluate once on the held-out France development cohort."""

    frames, metadata = load_development_frames(config)
    outcome = str(config["task"]["primary_outcome"])
    profile_config = config["profiles"]
    caps = [int(value) for value in profile_config["support_match_caps"]]
    strengths = [
        float(value) for value in profile_config["beta_prior_strengths"]
    ]
    context_strength = float(profile_config["context_prior_strength"])
    minimum_matches = int(profile_config["minimum_prior_matches"])

    validation_candidates: list[dict[str, float | int]] = []
    aggregate_by_cap: dict[int, AggregateTables] = {}
    validation_cache_by_cap: dict[int, PredictionCache] = {}
    for cap in caps:
        aggregates = build_aggregate_tables(
            frames["support"],
            outcome=outcome,
            match_cap=cap,
        )
        aggregate_by_cap[cap] = aggregates
        cache = prepare_prediction_cache(
            frames["validation"],
            aggregates,
            outcome=outcome,
            minimum_prior_matches=minimum_matches,
        )
        validation_cache_by_cap[cap] = cache
        for team_strength in strengths:
            for player_strength in strengths:
                predictions = predict_probabilities(
                    cache,
                    context_prior_strength=context_strength,
                    team_prior_strength=team_strength,
                    player_prior_strength=player_strength,
                )
                validation_candidates.append(
                    {
                        "match_cap": cap,
                        "team_prior_strength": team_strength,
                        "player_prior_strength": player_strength,
                        "context_nll": binary_nll(
                            cache.outcome,
                            predictions["context"],
                        ),
                        "team_nll": binary_nll(
                            cache.outcome,
                            predictions["team"],
                        ),
                        "player_nll": binary_nll(
                            cache.outcome,
                            predictions["player"],
                        ),
                    }
                )
    best_team = min(
        validation_candidates,
        key=lambda row: (
            float(row["team_nll"]),
            int(row["match_cap"]),
            float(row["team_prior_strength"]),
        ),
    )
    best_player = min(
        validation_candidates,
        key=lambda row: (
            float(row["player_nll"]),
            int(row["match_cap"]),
            float(row["team_prior_strength"]),
            float(row["player_prior_strength"]),
        ),
    )
    selected_cap = int(best_player["match_cap"])
    selected_aggregates = aggregate_by_cap[selected_cap]

    def selected_predictions(
        cache: PredictionCache,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        player_bundle = predict_probabilities(
            cache,
            context_prior_strength=context_strength,
            team_prior_strength=float(best_player["team_prior_strength"]),
            player_prior_strength=float(best_player["player_prior_strength"]),
        )
        team_bundle = predict_probabilities(
            cache,
            context_prior_strength=context_strength,
            team_prior_strength=float(best_team["team_prior_strength"]),
            player_prior_strength=float(best_player["player_prior_strength"]),
        )
        return (
            player_bundle["context"],
            team_bundle["team"],
            player_bundle["player"],
        )

    split_results: dict[str, Any] = {}
    selected_caches: dict[str, PredictionCache] = {}
    selected_probabilities: dict[str, dict[str, np.ndarray]] = {}
    for name in ("train", "validation", "development"):
        cache = prepare_prediction_cache(
            frames[name],
            selected_aggregates,
            outcome=outcome,
            minimum_prior_matches=minimum_matches,
        )
        context_probability, team_probability, player_probability = (
            selected_predictions(cache)
        )
        bundle = {
            "context": context_probability,
            "team": team_probability,
            "player": player_probability,
        }
        selected_caches[name] = cache
        selected_probabilities[name] = bundle
        split_results[name] = {
            **_metric_bundle(cache, bundle, minimum_matches),
            "support_match_coverage": _coverage(cache, [1, 3, 5, 10, 15, 20]),
        }

    development_cache = selected_caches["development"]
    development_probability = selected_probabilities["development"]
    bootstrap_config = config["uncertainty"]
    bootstrap = match_bootstrap_nll_gain(
        development_cache.outcome,
        development_probability["team"],
        development_probability["player"],
        development_cache.match_ids,
        replicates=int(bootstrap_config["bootstrap_replicates"]),
        seed=20260729,
        confidence_level=float(bootstrap_config["confidence_level"]),
    )
    shuffle_results: list[dict[str, Any]] = []
    for seed in config["falsification"]["shuffle_seeds"]:
        mapping = same_team_role_shuffle(
            selected_aggregates.player_catalog,
            int(seed),
        )
        shuffle_cache = prepare_prediction_cache(
            frames["development"],
            selected_aggregates,
            outcome=outcome,
            minimum_prior_matches=minimum_matches,
            profile_mapping=mapping,
        )
        shuffled = predict_probabilities(
            shuffle_cache,
            context_prior_strength=context_strength,
            team_prior_strength=float(best_player["team_prior_strength"]),
            player_prior_strength=float(best_player["player_prior_strength"]),
        )["player"]
        changed = np.array(
            [
                mapping.get(int(player_id), int(player_id)) != int(player_id)
                for player_id in shuffle_cache.query_player_ids
            ],
            dtype=bool,
        )
        shuffle_results.append(
            {
                "seed": int(seed),
                "changed_example_fraction": float(np.mean(changed)),
                "metrics": binary_metrics(shuffle_cache.outcome, shuffled),
            }
        )

    team_nll = float(split_results["development"]["team"]["nll"])
    player_nll = float(split_results["development"]["player"]["nll"])
    supported_team_nll = float(
        split_results["development"]["supported_subset"]["team"]["nll"]
    )
    supported_player_nll = float(
        split_results["development"]["supported_subset"]["player"]["nll"]
    )
    relative_gain = (team_nll - player_nll) / team_nll
    gate_config = config["development_gate"]
    gate_checks = {
        "minimum_relative_nll_improvement": relative_gain
        >= float(gate_config["minimum_relative_nll_improvement"]),
        "bootstrap_lower_bound_above_zero": float(bootstrap["ci_lower"]) > 0.0,
        "better_than_all_shuffled_controls": all(
            player_nll < float(row["metrics"]["nll"]) for row in shuffle_results
        ),
        "supported_subset_consistency": supported_player_nll
        < supported_team_nll,
    }
    return {
        "experiment_protocol": str(config["experiment_protocol"]),
        "status": "development_only",
        "primary_claim_test": (
            "Does a player's strictly earlier club history improve prediction "
            "beyond pooled context and the same club's history?"
        ),
        "metadata": metadata,
        "selection": {
            "context_prior_strength": context_strength,
            "best_team": best_team,
            "best_player": best_player,
            "candidate_count": len(validation_candidates),
            "top_player_candidates": sorted(
                validation_candidates,
                key=lambda row: float(row["player_nll"]),
            )[:10],
        },
        "splits": split_results,
        "development_effect": {
            "team_minus_player_nll": team_nll - player_nll,
            "relative_nll_improvement": relative_gain,
            "match_bootstrap": bootstrap,
        },
        "same_team_role_shuffles": shuffle_results,
        "development_gate": {
            "checks": gate_checks,
            "passed": all(gate_checks.values()),
        },
        "confirmatory": {
            "competition": str(config["confirmatory"]["competition"]),
            "metrics_loaded": False,
            "manifest": str(config["data"]["confirmatory_split_manifest"]),
        },
    }
