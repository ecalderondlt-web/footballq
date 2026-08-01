from __future__ import annotations

from datetime import date

import numpy as np

from footballq.analysis.footpass_compact_player_residual import (
    COMPACT_PROFILE_FEATURE_NAMES,
    HistoricalCompactProfile,
    _historical_profile,
    _role_prior,
    compact_player_match_profile,
    shrunk_player_deviation,
)
from footballq.analysis.footpass_player_history import (
    FootpassAppearance,
    PlayerMatchStats,
    _role_mean_profile,
)


def _appearance(match_id: str, day: int) -> FootpassAppearance:
    return FootpassAppearance(
        team_id="team",
        team_name="Team",
        match_id=match_id,
        match_date=date(2024, 1, day),
        focal_team_index=0,
        partition="development_train",
        player_by_shirt={10: "team:actor", 11: "team:peer"},
        player_name_by_id={
            "team:actor": "Actor",
            "team:peer": "Peer",
        },
    )


def _stats(action_class: int, event_repeats: int) -> PlayerMatchStats:
    result = PlayerMatchStats()
    for _ in range(event_repeats):
        result.event.update_event(
            action_class=action_class,
            x_attack=0.6,
            y=0.4,
            vx_attack=0.02,
            vy=-0.01,
            x_edges=[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0001],
            y_edges=[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0001],
        )
    result.event.update_outcome(
        action_class=action_class,
        x_attack=0.6,
        turnover=1,
        penalty_entry=0,
        x_edges=[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0001],
    )
    tracking_values = np.asarray(
        [0.6, 0.4, 0.02, -0.01, 0.03, 0.1, -0.1, 0.2, 0.15]
    )
    result.tracking.update(
        tracking_values,
        x_bin=1,
        y_bin=1,
        role_id=8,
    )
    result.role_counts[8] += 1
    return result


def test_compact_profile_has_frozen_shape_and_action_simplex() -> None:
    profile = compact_player_match_profile(_stats(2, 3))
    assert profile.shape == (len(COMPACT_PROFILE_FEATURE_NAMES),)
    assert np.isfinite(profile).all()
    assert np.isclose(profile[:4].sum(), 1.0)


def test_historical_profile_weights_prior_matches_equally() -> None:
    first = _appearance("1", 1)
    second = _appearance("2", 2)
    query = _appearance("3", 3)
    first_stats = _stats(1, 100)
    second_stats = _stats(8, 1)
    by_appearance = {
        first.appearance_id: {"team:actor": first_stats},
        second.appearance_id: {"team:actor": second_stats},
    }
    profile = _historical_profile(
        query,
        "team:actor",
        {"team": [first, second, query]},
        by_appearance,
        support_cap=99,
    )
    expected = np.mean(
        np.stack(
            [
                compact_player_match_profile(first_stats),
                compact_player_match_profile(second_stats),
            ]
        ),
        axis=0,
    )
    assert profile.support_matches == 2
    assert profile.available
    assert np.allclose(profile.values, expected)


def test_shrinkage_uses_match_equivalent_strength() -> None:
    player = np.ones(len(COMPACT_PROFILE_FEATURE_NAMES))
    role = np.zeros(len(COMPACT_PROFILE_FEATURE_NAMES))
    residual, alpha = shrunk_player_deviation(
        player,
        role,
        support_matches=3,
        shrinkage_match_equivalent=3.0,
    )
    assert alpha == 0.5
    assert np.allclose(residual, 0.5)


def test_compact_role_prior_excludes_focal_player() -> None:
    actor = HistoricalCompactProfile(
        values=np.full(len(COMPACT_PROFILE_FEATURE_NAMES), 9.0),
        support_matches=2,
        available=True,
        support_appearance_ids=("team:1", "team:2"),
    )
    peer = HistoricalCompactProfile(
        values=np.full(len(COMPACT_PROFILE_FEATURE_NAMES), 2.0),
        support_matches=2,
        available=True,
        support_appearance_ids=("team:1", "team:2"),
    )
    prior, mode, peer_count = _role_prior(
        [("team:actor", 8), ("team:peer", 9)],
        {"team:actor": actor, "team:peer": peer},
        actor_id="team:actor",
        actor_role=8,
    )
    assert mode == "same_broad_role"
    assert peer_count == 1
    assert np.allclose(prior, peer.values)


def test_v1_role_mean_control_excludes_requested_player() -> None:
    actor = np.asarray([9.0, 1.0])
    peer = np.asarray([2.0, 1.0])
    result = _role_mean_profile(
        [("actor", 8), ("peer", 9)],
        {"actor": actor, "peer": peer},
        8,
        2,
        exclude_player_id="actor",
    )
    assert np.array_equal(result, peer)


def test_missing_history_shrinks_to_zero_deviation() -> None:
    player = np.full(len(COMPACT_PROFILE_FEATURE_NAMES), 3.0)
    role = np.full(len(COMPACT_PROFILE_FEATURE_NAMES), 1.0)
    residual, alpha = shrunk_player_deviation(
        player,
        role,
        support_matches=0,
        shrinkage_match_equivalent=3.0,
    )
    assert alpha == 0.0
    assert np.array_equal(residual, np.zeros_like(residual))
