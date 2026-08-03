from __future__ import annotations

import numpy as np
import pandas as pd

from footballq.data.rlcs_player_profiles import (
    PROFILE_DIMENSION,
    audit_profile_stability,
    fit_profile_priors,
)


def _stable_profiles(players: int = 48, games: int = 16) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for player in range(players):
        base = rng.normal(size=PROFILE_DIMENSION) + player * 0.15
        for game in range(games):
            rows.append(
                {
                    "replay_id": f"r-{player}-{game}",
                    "series_id": f"s-{player}-{game // 3}",
                    "region": "EU" if player < players // 2 else "NA",
                    "event_time_utc": pd.Timestamp("2025-01-01", tz="UTC")
                    + pd.Timedelta(days=game),
                    "v2_stage": "profile_support",
                    "player_id": f"p-{player}",
                    "profile": (base + rng.normal(scale=0.01, size=PROFILE_DIMENSION)).tolist(),
                    "team_win": float((player + game) % 2),
                    "team_goal_diff": (player + game) % 5 - 2,
                }
            )
    return pd.DataFrame(rows)


def test_stable_profiles_pass_retrieval_and_correlation_gates():
    games = _stable_profiles()
    priors = fit_profile_priors(games)
    report = audit_profile_stability(games, priors, bootstrap_resamples=200)
    assert report["counts"]["eligible_players"] == 48
    assert report["counts"]["eligible_players_by_region"] == {"EU": 24, "NA": 24}
    assert report["same_player_retrieval_auc"] >= 0.75
    assert report["player_bootstrap"]["retrieval_auc_95pct"][0] >= 0.65
    assert report["median_core_trait_spearman"] >= 0.35
    assert report["player_bootstrap"]["median_spearman_95pct"][0] > 0.20
    assert all(value > 0.50 for value in report["regional_same_player_retrieval_auc"].values())
    assert report["all_gates_pass"] is True


def test_player_count_gate_fails_closed():
    games = _stable_profiles(players=12)
    priors = fit_profile_priors(games)
    report = audit_profile_stability(games, priors, bootstrap_resamples=100)
    assert report["gates"]["complete_available_cohort"] is False
    assert report["stop_before_outcome_training"] is True
