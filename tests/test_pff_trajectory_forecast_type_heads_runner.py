from scripts.run_pff_trajectory_forecast_type_heads_v1 import (
    FAMILIES,
    HORIZON_KEYS,
    SEEDS,
    _redesign_gate,
)


def _metrics(
    player: float,
    ball: float,
    horizon: float,
    *,
    ball_fde: float = 4.0,
) -> dict[str, float]:
    return {
        "player_ADE_m": player,
        "ball_ADE_m": ball,
        "ball_FDE_m": ball_fde,
        **{key: horizon for key in HORIZON_KEYS},
    }


def test_type_head_runner_freezes_scope():
    assert FAMILIES == ("raw", "frozen", "finetuned")
    assert SEEDS == (7, 11, 23)
    assert len(FAMILIES) * len(SEEDS) == 9


def test_type_head_redesign_requires_ball_recovery_and_player_retention():
    entity_rows = {seed: _metrics(1.0, 2.0, 1.0) for seed in SEEDS}
    candidate_rows = {
        seed: _metrics(1.005, 1.89, 1.005, ball_fde=3.9) for seed in SEEDS
    }
    passing = _redesign_gate(
        "candidate",
        candidate_rows,
        entity_rows,
        _metrics(1.005, 1.89, 1.005, ball_fde=3.9),
        _metrics(1.0, 2.0, 1.0),
        _metrics(1.1, 1.95, 1.1),
    )
    weak_ball = _redesign_gate(
        "candidate",
        {seed: _metrics(1.005, 1.92, 1.005, ball_fde=3.9) for seed in SEEDS},
        entity_rows,
        _metrics(1.005, 1.92, 1.005, ball_fde=3.9),
        _metrics(1.0, 2.0, 1.0),
        _metrics(1.1, 1.95, 1.1),
    )

    assert passing["passed"] is True
    assert weak_ball["criteria"]["mean_ball_ADE_improvement"]["passed"] is False
    assert weak_ball["passed"] is False
