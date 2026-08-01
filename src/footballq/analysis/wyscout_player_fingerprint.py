"""Cross-team player fingerprint retrieval from public event histories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from footballq.analysis.wyscout_player_memory import select_last_player_matches

SUBEVENT_IDS = (80, 81, 82, 83, 84, 85, 86)
SUMMARY_FEATURE_NAMES = (
    "pass_accuracy",
    "mean_start_x",
    "mean_start_y",
    "mean_destination_x",
    "mean_destination_y",
    "mean_progression",
    "mean_absolute_lateral_displacement",
    "mean_pass_length",
    "progression_std",
    "lateral_displacement_std",
    "forward_pass_rate",
    "backward_pass_rate",
    "long_pass_rate",
    "cross_rate",
    "hand_pass_rate",
    "head_pass_rate",
    "high_pass_rate",
    "launch_rate",
    "simple_pass_rate",
    "smart_pass_rate",
    "key_pass_rate",
    "shot_chain_rate",
    "attacking_half_origin_rate",
    "final_third_destination_rate",
    "penalty_area_destination_rate",
)
FEATURE_NAMES = (
    *SUMMARY_FEATURE_NAMES,
    *(f"destination_zone_{index}" for index in range(30)),
    *(f"start_zone_{index}" for index in range(20)),
)
OUTCOME_FEATURE_NAMES = (
    "pass_accuracy",
    "key_pass_rate",
    "shot_chain_rate",
)
ROLE_NAMES = {
    0: "goalkeeper",
    1: "defender",
    2: "midfielder",
    3: "forward",
    4: "unknown",
}


@dataclass(frozen=True)
class PlayerVectorSet:
    player_ids: np.ndarray
    roles: np.ndarray
    team_ids: np.ndarray
    pass_counts: np.ndarray
    match_counts: np.ndarray
    vectors: np.ndarray

    def subset(self, mask: np.ndarray) -> PlayerVectorSet:
        return PlayerVectorSet(
            player_ids=self.player_ids[mask],
            roles=self.roles[mask],
            team_ids=self.team_ids[mask],
            pass_counts=self.pass_counts[mask],
            match_counts=self.match_counts[mask],
            vectors=self.vectors[mask],
        )


def _mode_int(values: pd.Series) -> int:
    counts = values.value_counts()
    top_count = int(counts.iloc[0])
    return int(min(int(value) for value in counts[counts == top_count].index))


def _player_vector(frame: pd.DataFrame) -> np.ndarray:
    delta_x = (frame["destination_x"] - frame["start_x"]).to_numpy(
        dtype=np.float64
    )
    delta_y = (frame["destination_y"] - frame["start_y"]).to_numpy(
        dtype=np.float64
    )
    length = np.hypot(delta_x, delta_y)
    destination_histogram = np.bincount(
        frame["destination_zone"].to_numpy(dtype=np.int64),
        minlength=30,
    ).astype(np.float64)
    destination_histogram /= len(frame)
    start_histogram = np.bincount(
        frame["start_zone"].to_numpy(dtype=np.int64),
        minlength=20,
    ).astype(np.float64)
    start_histogram /= len(frame)
    summary = [
        float(frame["accurate"].mean()),
        float(frame["start_x"].mean()) / 100.0,
        float(frame["start_y"].mean()) / 100.0,
        float(frame["destination_x"].mean()) / 100.0,
        float(frame["destination_y"].mean()) / 100.0,
        float(delta_x.mean()) / 100.0,
        float(np.abs(delta_y).mean()) / 100.0,
        float(length.mean()) / 100.0,
        float(delta_x.std()) / 100.0,
        float(delta_y.std()) / 100.0,
        float(np.mean(delta_x > 10.0)),
        float(np.mean(delta_x < -5.0)),
        float(np.mean(length > 30.0)),
        *[
            float(np.mean(frame["subevent_id"].to_numpy() == subevent_id))
            for subevent_id in SUBEVENT_IDS
        ],
        float(frame["key_pass"].mean()),
        float(frame["shot_within_horizon"].mean()),
        float(np.mean(frame["start_x"].to_numpy() >= 50.0)),
        float(np.mean(frame["destination_x"].to_numpy() >= 67.0)),
        float(
            np.mean(
                (frame["destination_x"].to_numpy() >= 83.0)
                & (frame["destination_y"].to_numpy() >= 21.0)
                & (frame["destination_y"].to_numpy() <= 79.0)
            )
        ),
    ]
    vector = np.concatenate(
        [
            np.asarray(summary, dtype=np.float64),
            destination_histogram,
            start_histogram,
        ]
    )
    if len(vector) != len(FEATURE_NAMES):
        raise AssertionError(
            f"Expected {len(FEATURE_NAMES)} fingerprint features, got {len(vector)}."
        )
    return vector


def build_player_vectors(
    frame: pd.DataFrame,
    *,
    minimum_matches: int,
    minimum_passes: int,
    match_cap: int | None = None,
    eligible_player_ids: set[int] | None = None,
) -> PlayerVectorSet:
    """Aggregate one deterministic behavior vector per eligible player."""

    selected = (
        select_last_player_matches(frame, match_cap)
        if match_cap is not None
        else frame
    )
    records: list[tuple[int, int, int, int, int, np.ndarray]] = []
    for player_id, player_frame in selected.groupby("player_id", observed=True):
        player_id_int = int(player_id)
        if (
            eligible_player_ids is not None
            and player_id_int not in eligible_player_ids
        ):
            continue
        match_count = int(player_frame["match_id"].nunique())
        pass_count = int(len(player_frame))
        if match_count < minimum_matches or pass_count < minimum_passes:
            continue
        records.append(
            (
                player_id_int,
                _mode_int(player_frame["role"]),
                _mode_int(player_frame["team_id"]),
                pass_count,
                match_count,
                _player_vector(player_frame),
            )
        )
    if not records:
        raise ValueError("No players satisfy the fingerprint vector thresholds.")
    records.sort(key=lambda row: row[0])
    return PlayerVectorSet(
        player_ids=np.asarray([row[0] for row in records], dtype=np.int64),
        roles=np.asarray([row[1] for row in records], dtype=np.int64),
        team_ids=np.asarray([row[2] for row in records], dtype=np.int64),
        pass_counts=np.asarray([row[3] for row in records], dtype=np.int64),
        match_counts=np.asarray([row[4] for row in records], dtype=np.int64),
        vectors=np.stack([row[5] for row in records]),
    )


def eligible_player_universe(
    support: pd.DataFrame,
    query: pd.DataFrame,
    *,
    support_minimum_matches: int,
    support_minimum_passes: int,
    query_minimum_matches: int,
    query_minimum_passes: int,
) -> set[int]:
    """Return query identities with enough evidence in both time periods."""

    _, query_ids = eligible_support_and_query_player_ids(
        support,
        query,
        support_minimum_matches=support_minimum_matches,
        support_minimum_passes=support_minimum_passes,
        query_minimum_matches=query_minimum_matches,
        query_minimum_passes=query_minimum_passes,
    )
    return query_ids


def eligible_support_and_query_player_ids(
    support: pd.DataFrame,
    query: pd.DataFrame,
    *,
    support_minimum_matches: int,
    support_minimum_passes: int,
    query_minimum_matches: int,
    query_minimum_passes: int,
) -> tuple[set[int], set[int]]:
    """Return all support candidates and the supported query identities.

    Players absent from the query cohort remain in the support candidate pool.
    This prevents tournament-roster knowledge from making retrieval artificially
    easy.
    """

    support_counts = support.groupby("player_id", observed=True).agg(
        passes=("player_id", "size"),
        matches=("match_id", "nunique"),
    )
    query_counts = query.groupby("player_id", observed=True).agg(
        passes=("player_id", "size"),
        matches=("match_id", "nunique"),
    )
    support_ids = set(
        int(value)
        for value in support_counts.loc[
            (support_counts["passes"] >= support_minimum_passes)
            & (support_counts["matches"] >= support_minimum_matches)
        ].index
    )
    query_ids = set(
        int(value)
        for value in query_counts.loc[
            (query_counts["passes"] >= query_minimum_passes)
            & (query_counts["matches"] >= query_minimum_matches)
        ].index
    )
    return support_ids, support_ids & query_ids


def normalized_support_and_query(
    support: PlayerVectorSet,
    query: PlayerVectorSet,
    *,
    feature_view: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit z-scoring and role residuals on support only, then L2 normalize."""

    outcome_indices = np.asarray(
        [FEATURE_NAMES.index(name) for name in OUTCOME_FEATURE_NAMES],
        dtype=np.int64,
    )
    if feature_view == "summary_behavior_22":
        feature_indices = np.asarray(
            [
                index
                for index in range(len(SUMMARY_FEATURE_NAMES))
                if index not in set(outcome_indices)
            ],
            dtype=np.int64,
        )
    elif feature_view == "summary_25":
        feature_indices = np.arange(len(SUMMARY_FEATURE_NAMES))
    elif feature_view == "spatial_50":
        feature_indices = np.arange(len(SUMMARY_FEATURE_NAMES), len(FEATURE_NAMES))
    elif feature_view == "outcome_3":
        feature_indices = outcome_indices
    elif feature_view == "behavior_72":
        feature_indices = np.asarray(
            [
                index
                for index in range(len(FEATURE_NAMES))
                if index not in set(outcome_indices)
            ],
            dtype=np.int64,
        )
    elif feature_view == "full_75":
        feature_indices = np.arange(len(FEATURE_NAMES))
    else:
        raise ValueError(f"Unknown fingerprint feature view: {feature_view}")
    support_values = support.vectors[:, feature_indices].astype(np.float64)
    query_values = query.vectors[:, feature_indices].astype(np.float64)
    mean = support_values.mean(axis=0)
    scale = support_values.std(axis=0)
    scale[scale < 1e-6] = 1.0
    support_values = (support_values - mean) / scale
    query_values = (query_values - mean) / scale
    role_means = {
        int(role): support_values[support.roles == role].mean(axis=0)
        for role in np.unique(support.roles)
    }
    support_values = np.stack(
        [
            value - role_means[int(role)]
            for value, role in zip(
                support_values,
                support.roles,
                strict=True,
            )
        ]
    )
    query_values = np.stack(
        [
            value - role_means.get(int(role), np.zeros_like(value))
            for value, role in zip(
                query_values,
                query.roles,
                strict=True,
            )
        ]
    )
    support_norm = np.linalg.norm(support_values, axis=1, keepdims=True)
    query_norm = np.linalg.norm(query_values, axis=1, keepdims=True)
    return (
        support_values / np.maximum(support_norm, 1e-9),
        query_values / np.maximum(query_norm, 1e-9),
    )


def _ranking_summary(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    rank = np.asarray([int(row["rank"]) for row in rows])
    candidates = np.asarray([int(row["candidates"]) for row in rows])
    chance_mrr = np.asarray(
        [
            sum(1.0 / index for index in range(1, count + 1)) / count
            for count in candidates
        ]
    )
    return {
        "queries": int(len(rows)),
        "mean_candidates": float(candidates.mean()),
        "top1": float(np.mean(rank <= 1)),
        "chance_top1": float(np.mean(1.0 / candidates)),
        "top3": float(np.mean(rank <= 3)),
        "chance_top3": float(np.mean(np.minimum(3.0 / candidates, 1.0))),
        "top5": float(np.mean(rank <= 5)),
        "chance_top5": float(np.mean(np.minimum(5.0 / candidates, 1.0))),
        "top10": float(np.mean(rank <= 10)),
        "chance_top10": float(np.mean(np.minimum(10.0 / candidates, 1.0))),
        "mrr": float(np.mean(1.0 / rank)),
        "chance_mrr": float(chance_mrr.mean()),
        "median_rank": float(np.median(rank)),
    }


def _bootstrap_ranking_gains(
    rows: list[dict[str, float | int]],
    *,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    rank = np.asarray([int(row["rank"]) for row in rows])
    candidates = np.asarray([int(row["candidates"]) for row in rows])
    top1_gain = (rank <= 1).astype(np.float64) - 1.0 / candidates
    chance_mrr = np.asarray(
        [
            sum(1.0 / index for index in range(1, count + 1)) / count
            for count in candidates
        ]
    )
    mrr_gain = 1.0 / rank - chance_mrr
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(rows), size=(replicates, len(rows)))
    tail = (1.0 - confidence_level) / 2.0

    def interval(values: np.ndarray) -> dict[str, float]:
        sampled = values[draws].mean(axis=1)
        return {
            "point": float(values.mean()),
            "ci_lower": float(np.quantile(sampled, tail)),
            "ci_upper": float(np.quantile(sampled, 1.0 - tail)),
        }

    return {
        "bootstrap_unit": "player_id",
        "replicates": replicates,
        "top1_minus_chance": interval(top1_gain),
        "mrr_minus_chance": interval(mrr_gain),
    }


def _bootstrap_mean(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(replicates, len(values)))
    sampled = values[draws].mean(axis=1)
    tail = (1.0 - confidence_level) / 2.0
    return {
        "point": float(values.mean()),
        "ci_lower": float(np.quantile(sampled, tail)),
        "ci_upper": float(np.quantile(sampled, 1.0 - tail)),
    }


def _paired_support_gain(
    first_rows: list[dict[str, float | int]],
    main_rows: list[dict[str, float | int]],
    *,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    first_by_player = {int(row["player_id"]): row for row in first_rows}
    main_by_player = {int(row["player_id"]): row for row in main_rows}
    common_ids = sorted(set(first_by_player) & set(main_by_player))
    if not common_ids:
        raise ValueError("No common players exist for the support-size comparison.")
    first_rank = np.asarray(
        [int(first_by_player[player_id]["rank"]) for player_id in common_ids]
    )
    main_rank = np.asarray(
        [int(main_by_player[player_id]["rank"]) for player_id in common_ids]
    )
    return {
        "bootstrap_unit": "player_id",
        "replicates": replicates,
        "common_players": len(common_ids),
        "mrr_gain": _bootstrap_mean(
            (1.0 / main_rank) - (1.0 / first_rank),
            replicates=replicates,
            seed=seed,
            confidence_level=confidence_level,
        ),
        "top1_gain": _bootstrap_mean(
            (main_rank <= 1).astype(np.float64)
            - (first_rank <= 1).astype(np.float64),
            replicates=replicates,
            seed=seed + 1,
            confidence_level=confidence_level,
        ),
    }


def evaluate_cross_team_retrieval(
    support: PlayerVectorSet,
    query: PlayerVectorSet,
    *,
    feature_view: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    """Retrieve each query identity from support under role and team controls."""

    support_by_id = {
        int(player_id): index
        for index, player_id in enumerate(support.player_ids)
    }
    query_mask = np.asarray(
        [int(player_id) in support_by_id for player_id in query.player_ids]
    )
    query = query.subset(query_mask)
    support_values, query_values = normalized_support_and_query(
        support,
        query,
        feature_view=feature_view,
    )
    query_rosters: dict[tuple[int, int], set[int]] = {}
    for player_id, team_id, role in zip(
        query.player_ids,
        query.team_ids,
        query.roles,
        strict=True,
    ):
        query_rosters.setdefault((int(team_id), int(role)), set()).add(
            int(player_id)
        )
    rows_by_set: dict[str, list[dict[str, float | int]]] = {
        "same_role": [],
        "same_support_team_and_role": [],
        "same_query_team_and_role": [],
    }
    pairwise_auc_by_player: list[float] = []
    changed_team: list[bool] = []
    for query_index, player_id_raw in enumerate(query.player_ids):
        player_id = int(player_id_raw)
        support_index = support_by_id[player_id]
        role = int(query.roles[query_index])
        support_team = int(support.team_ids[support_index])
        query_team = int(query.team_ids[query_index])
        changed_team.append(support_team != query_team)
        similarities = support_values @ query_values[query_index]
        candidate_masks = {
            "same_role": support.roles == role,
            "same_support_team_and_role": (
                (support.roles == role) & (support.team_ids == support_team)
            ),
            "same_query_team_and_role": np.asarray(
                [
                    int(candidate_id)
                    in query_rosters.get((query_team, role), set())
                    for candidate_id in support.player_ids
                ]
            ),
        }
        for name, mask in candidate_masks.items():
            candidate_ids = support.player_ids[mask]
            if len(candidate_ids) < 2 or player_id not in set(candidate_ids):
                continue
            candidate_similarity = similarities[mask]
            order = np.argsort(-candidate_similarity, kind="stable")
            rank = int(
                np.flatnonzero(candidate_ids[order] == player_id)[0]
            ) + 1
            rows_by_set[name].append(
                {
                    "player_id": player_id,
                    "role": role,
                    "rank": rank,
                    "candidates": int(len(candidate_ids)),
                }
            )
        role_mask = (support.roles == role) & (support.player_ids != player_id)
        true_similarity = float(similarities[support_index])
        negative_similarity = similarities[role_mask]
        pairwise_auc_by_player.append(
            {
                "player_id": player_id,
                "role": role,
                "auc": float(
                    np.mean(true_similarity > negative_similarity)
                    + 0.5 * np.mean(true_similarity == negative_similarity)
                ),
            }
        )
    ranking = {
        name: {
            **_ranking_summary(rows),
            "bootstrap": _bootstrap_ranking_gains(
                rows,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + index,
                confidence_level=confidence_level,
            ),
        }
        for index, (name, rows) in enumerate(rows_by_set.items())
    }
    pairwise = np.asarray(
        [float(row["auc"]) for row in pairwise_auc_by_player]
    )
    by_role = {}
    for role in sorted(set(int(row["role"]) for row in pairwise_auc_by_player)):
        role_rows = [
            row for row in rows_by_set["same_role"] if int(row["role"]) == role
        ]
        role_auc = np.asarray(
            [
                float(row["auc"])
                for row in pairwise_auc_by_player
                if int(row["role"]) == role
            ]
        )
        by_role[ROLE_NAMES.get(role, str(role))] = {
            **_ranking_summary(role_rows),
            "bootstrap": _bootstrap_ranking_gains(
                role_rows,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + 200 + role,
                confidence_level=confidence_level,
            ),
            "pairwise_auc": _bootstrap_mean(
                role_auc,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + 300 + role,
                confidence_level=confidence_level,
            ),
        }
    return {
        "feature_view": feature_view,
        "query_players": int(len(query.player_ids)),
        "support_candidates": int(len(support.player_ids)),
        "changed_team_fraction": float(np.mean(changed_team)),
        "ranking": ranking,
        "same_role_pairwise_auc": _bootstrap_mean(
            pairwise,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + 100,
            confidence_level=confidence_level,
        ),
        "same_role_by_role": by_role,
        "per_player_ranks": rows_by_set,
        "per_player_pairwise_auc": pairwise_auc_by_player,
    }


def evaluate_support_curve(
    support_frame: pd.DataFrame,
    query_frame: pd.DataFrame,
    cohort: dict[str, Any],
    evaluation: dict[str, Any],
    *,
    feature_view: str,
) -> dict[str, Any]:
    support_ids, query_ids = eligible_support_and_query_player_ids(
        support_frame,
        query_frame,
        support_minimum_matches=int(cohort["support_min_matches"]),
        support_minimum_passes=int(cohort["support_min_passes"]),
        query_minimum_matches=int(cohort["query_min_matches"]),
        query_minimum_passes=int(cohort["query_min_passes"]),
    )
    query_vectors = build_player_vectors(
        query_frame,
        minimum_matches=int(cohort["query_min_matches"]),
        minimum_passes=int(cohort["query_min_passes"]),
        eligible_player_ids=query_ids,
    )
    curve: dict[str, Any] = {}
    for match_cap_raw in cohort["support_match_caps"]:
        match_cap = int(match_cap_raw)
        support_vectors = build_player_vectors(
            support_frame,
            minimum_matches=1,
            minimum_passes=1,
            match_cap=match_cap,
            eligible_player_ids=support_ids,
        )
        curve[str(match_cap)] = evaluate_cross_team_retrieval(
            support_vectors,
            query_vectors,
            feature_view=feature_view,
            bootstrap_replicates=int(evaluation["bootstrap_replicates"]),
            bootstrap_seed=int(evaluation["bootstrap_seed"]) + match_cap * 1000,
            confidence_level=float(evaluation["confidence_level"]),
        )
    first_cap = str(cohort["support_match_caps"][0])
    main_cap = str(cohort["main_support_match_cap"])
    support_gain = _paired_support_gain(
        curve[first_cap]["per_player_ranks"]["same_role"],
        curve[main_cap]["per_player_ranks"]["same_role"],
        replicates=int(evaluation["bootstrap_replicates"]),
        seed=int(evaluation["bootstrap_seed"]) + 900_000,
        confidence_level=float(evaluation["confidence_level"]),
    )
    return {
        "eligible_player_universe": len(query_ids),
        "eligible_query_players": len(query_ids),
        "eligible_support_candidates": len(support_ids),
        "support_match_caps": curve,
        "main_vs_first_support": {
            "first_match_cap": int(first_cap),
            "main_match_cap": int(main_cap),
            **support_gain,
        },
    }
