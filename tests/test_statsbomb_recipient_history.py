from __future__ import annotations

from collections import defaultdict

import numpy as np

from footballq.analysis.statsbomb_recipient_history import (
    START_X_BINS,
    START_Y_BINS,
    _appeared_player_ids_by_team,
    _event_sample_id,
    _history_components,
    _match_bootstrap_gain,
    _ranking_metrics,
    _row_probabilities,
    evaluate_development_cache,
    evaluate_frozen_recipient_cache,
    frozen_recipient_conditions,
)


def _row(
    *,
    split: str = "validation",
    match_id: str = "match-1",
    target_index: int = 0,
    cohort: str = "development",
) -> dict:
    return {
        "match_id": match_id,
        "split": split,
        "cohort": cohort,
        "recipient_id": f"p{target_index}",
        "start_zone": 0,
        "candidates": ["p0", "p1", "p2", "p3"],
        "candidate_roles": ["midfielder"] * 4,
        "candidate_positions": ["Center Midfield"] * 4,
        "target_index": target_index,
        "support": {
            1: {
                "global": [10.0, 10.0, 10.0, 10.0],
                "zone": [9.0, 1.0, 1.0, 1.0],
                "pair": [1.0, 1.0, 1.0, 1.0],
                "shuffled_zone": [1.0, 9.0, 1.0, 1.0],
                "shuffled_global": [10.0, 10.0, 10.0, 10.0],
                "shuffled_position_zone": [1.0, 9.0, 1.0, 1.0],
                "shuffled_position_global": [10.0, 10.0, 10.0, 10.0],
                "appearance_count": [1.0, 1.0, 1.0, 1.0],
                "pair_coappearance_count": [1.0, 1.0, 1.0, 1.0],
            }
        },
    }


def _priors() -> dict:
    return {
        "role_zone_selected": {(0, "midfielder"): 4},
        "position_zone_selected": {(0, "Center Midfield"): 4},
        "position_selected": {"Center Midfield": 4},
        "static_selected": {},
        "static_available": {},
    }


def test_event_sample_identity_includes_period() -> None:
    assert _event_sample_id("123", 2, "event-7") == "123:2:event-7"


def test_support_appearance_includes_players_with_positions_and_excludes_bench() -> None:
    lineups = [
        {
            "team_name": "A",
            "lineup": [
                {"player_id": 1, "positions": [{"position": "Center Back"}]},
                {"player_id": 2, "positions": []},
                {"player_id": 3, "positions": [{"position": "Forward"}]},
            ],
        }
    ]
    assert _appeared_player_ids_by_team(lineups, {"1", "2"}) == {"A": ["1"]}


def test_history_components_use_only_last_k_matches() -> None:
    old_zone = np.zeros(START_X_BINS * START_Y_BINS)
    middle_zone = np.zeros(START_X_BINS * START_Y_BINS)
    recent_zone = np.zeros(START_X_BINS * START_Y_BINS)
    old_zone[0] = 100.0
    middle_zone[0] = 2.0
    recent_zone[0] = 3.0
    receiver_history = defaultdict(
        list,
        {
            "receiver": [
                ("2024-01-01", "m1", 100),
                ("2024-01-02", "m2", 2),
                ("2024-01-03", "m3", 3),
            ]
        },
    )
    zone_history = defaultdict(
        list,
        {
            "receiver": [
                ("2024-01-01", "m1", old_zone),
                ("2024-01-02", "m2", middle_zone),
                ("2024-01-03", "m3", recent_zone),
            ]
        },
    )
    pair_history = defaultdict(
        list,
        {
            ("actor", "receiver"): [
                ("2024-01-01", "m1", 100),
                ("2024-01-02", "m2", 2),
                ("2024-01-03", "m3", 3),
            ]
        },
    )
    result = _history_components(
        candidates=["receiver"],
        actor_id="actor",
        start_zone=0,
        support_size=2,
        receiver_history=receiver_history,
        zone_history=zone_history,
        pair_history=pair_history,
    )
    assert result == {
        "global": [5.0],
        "zone": [5.0],
        "pair": [5.0],
        "appearance_count": [2.0],
        "pair_coappearance_count": [2.0],
    }

    frozen = _history_components(
        candidates=["receiver"],
        actor_id="actor",
        start_zone=0,
        support_size=2,
        receiver_history=receiver_history,
        zone_history=zone_history,
        pair_history=pair_history,
        support_before="2024-01-03",
    )
    assert frozen["global"] == [102.0]
    assert frozen["appearance_count"] == [2.0]


def test_history_changes_ranking_and_same_role_shuffle_breaks_identity() -> None:
    row = _row()
    baseline = _row_probabilities(
        row,
        _priors(),
        support_size=1,
        global_weight=1.0,
    )
    history = _row_probabilities(
        row,
        _priors(),
        support_size=1,
        global_weight=1.0,
        zone_weight=1.0,
    )
    shuffled = _row_probabilities(
        row,
        _priors(),
        support_size=1,
        global_weight=1.0,
        zone_weight=1.0,
        shuffled_zone=True,
    )
    assert np.isclose(baseline.sum(), 1.0)
    assert np.isclose(history.sum(), 1.0)
    assert int(np.argmax(history)) == 0
    assert int(np.argmax(shuffled)) == 1
    assert history[0] > baseline[0]


def test_cold_start_profile_shrinks_exactly_to_the_position_prior() -> None:
    row = _row()
    row["support"][1]["global"] = [0.0] * 4
    row["support"][1]["zone"] = [0.0] * 4
    baseline = _row_probabilities(
        row,
        _priors(),
        support_size=1,
        position_weight=1.0,
    )
    with_profile = _row_probabilities(
        row,
        _priors(),
        support_size=1,
        position_weight=1.0,
        zone_weight=1.0,
        profile_prior_strength=3.0,
    )
    assert np.allclose(with_profile, baseline)


def test_ranking_metrics_and_match_bootstrap_report_positive_nll_gain() -> None:
    rows = [
        _row(match_id="m1"),
        _row(match_id="m1"),
        _row(match_id="m2"),
        _row(match_id="m2"),
    ]
    baseline = [np.full(4, 0.25)] * len(rows)
    profile = [np.array([0.7, 0.1, 0.1, 0.1])] * len(rows)
    metrics = _ranking_metrics(rows, profile)
    bootstrap = _match_bootstrap_gain(
        rows,
        baseline,
        profile,
        samples=50,
        seed=7,
    )
    assert metrics["top1_accuracy"] == 1.0
    assert bootstrap["nll_improvement"]["ci95"][0] > 0.0
    assert bootstrap["nll_improvement"]["positive_fraction"] == 1.0


def test_development_gate_is_explicit_and_provenance_is_preserved() -> None:
    rows = [
        _row(split="train", match_id="train"),
        _row(split="validation", match_id="val"),
        _row(split="development_test", match_id="dev"),
    ]
    cache = {
        "rows": rows,
        "audit": {
            "config_sha256": "config-hash",
            "split_manifest_path": "split.json",
            "split_manifest_sha256": "split-hash",
            "source_commit": "source-commit",
            "sample_identity": "match_id:period:event_uuid",
            "feature_view": "event-only",
            "objective_mode": "recipient-ranking",
            "support_policy": "online_strictly_prior_appearances",
            "chronology_rule": "strictly-before-query",
        },
    }
    config = {
        "profiles": {"support_sizes": [1]},
        "model_selection": {
            "weights": [0.0, 1.0],
            "profile_prior_strengths": [1.0],
        },
        "evaluation": {
            "match_bootstrap_samples": 20,
            "bootstrap_seed": 11,
        },
        "gates": {
            "minimum_nll_improvement_fraction": 0.0,
            "minimum_top3_gain": 0.0,
            "require_positive_match_bootstrap_mean": True,
            "require_positive_match_bootstrap_ci": True,
            "require_profile_better_than_same_role_shuffle": True,
        },
    }
    result = evaluate_development_cache(cache, config)
    assert result["development_gate"]["status"] == "controls_passed"
    assert result["provenance"]["split_manifest_sha256"] == "split-hash"
    assert result["sealed_test_loaded"] is False


def test_frozen_confirmatory_evaluation_reproduces_validation_and_gates() -> None:
    development_rows = [
        _row(split="train", match_id="train"),
        _row(split="validation", match_id="val"),
        _row(split="development_test", match_id="dev"),
    ]
    audit = {
        "config_sha256": "config-hash",
        "split_manifest_path": "split.json",
        "split_manifest_sha256": "split-hash",
        "source_commit": "source-commit",
        "sample_identity": "match_id:period:event_uuid",
        "feature_view": "event-only",
        "objective_mode": "recipient-ranking",
        "support_policy": "online_strictly_prior_appearances",
        "chronology_rule": "strictly-before-query",
    }
    development_config = {
        "profiles": {"support_sizes": [1]},
        "model_selection": {
            "weights": [0.0, 1.0],
            "profile_prior_strengths": [1.0],
        },
        "evaluation": {
            "match_bootstrap_samples": 20,
            "bootstrap_seed": 11,
        },
        "gates": {
            "minimum_nll_improvement_fraction": 0.0,
            "minimum_top3_gain": 0.0,
            "require_positive_match_bootstrap_mean": True,
            "require_positive_match_bootstrap_ci": True,
            "require_profile_better_than_same_role_shuffle": True,
        },
    }
    development = evaluate_development_cache(
        {"rows": development_rows, "audit": audit},
        development_config,
    )
    assert frozen_recipient_conditions(development)["profile"][
        "support_size"
    ] == 1

    confirmatory_rows = [
        *development_rows[:2],
        _row(
            split="development_test",
            match_id="primary",
            cohort="primary",
        ),
        _row(
            split="development_test",
            match_id="external",
            cohort="external",
        ),
    ]
    confirmatory_config = {
        "experiment_protocol": "frozen-recipient-test",
        "profiles": {"support_sizes": [1]},
        "evaluation": {
            "match_bootstrap_samples": 20,
            "bootstrap_seed": 17,
        },
        "confirmatory": {
            "cohort_order": ["primary", "external"],
            "primary_cohort": "primary",
            "external_replication_cohorts": ["external"],
        },
        "confirmatory_gate": {
            "primary_minimum_relative_nll_improvement": 0.0,
        },
    }
    result = evaluate_frozen_recipient_cache(
        {"rows": confirmatory_rows, "audit": audit},
        confirmatory_config,
        development,
    )

    assert result["validation_reproduced_exactly"] is True
    assert result["gate"]["passed"] is True
    assert result["cohorts"]["primary"]["effects"][
        "profile_minus_rolling_nll_improvement"
    ] > 0.0
