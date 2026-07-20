from scripts.run_pff_trajectory_forecast_entity_v1 import (
    FAMILIES,
    HORIZON_KEYS,
    SEEDS,
    _comparison_gate,
)


def _metrics(player: float, ball: float, horizon: float) -> dict[str, float]:
    return {
        "player_ADE_m": player,
        "ball_ADE_m": ball,
        **{key: horizon for key in HORIZON_KEYS},
    }


def test_entity_runner_freezes_scope():
    assert FAMILIES == ("raw", "frozen", "finetuned")
    assert SEEDS == (7, 11, 23)
    assert len(FAMILIES) * len(SEEDS) == 9


def test_entity_comparison_requires_player_and_ball_materiality():
    reference_rows = {seed: _metrics(1.0, 2.0, 1.0) for seed in SEEDS}
    candidate_rows = {seed: _metrics(0.99, 1.98, 0.99) for seed in SEEDS}
    result = _comparison_gate(
        "candidate",
        "reference",
        candidate_rows,
        reference_rows,
        _metrics(0.99, 1.98, 0.99),
        _metrics(1.0, 2.0, 1.0),
        player_improvement_minimum=0.01,
    )

    assert result["criteria"]["player_seed_wins"]["passed"] is True
    assert result["criteria"]["mean_player_ADE_improvement"]["passed"] is True
    assert result["criteria"]["mean_ball_ADE_improvement"]["passed"] is False
    assert result["passed"] is False
