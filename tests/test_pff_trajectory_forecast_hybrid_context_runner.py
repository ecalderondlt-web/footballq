from scripts.run_pff_trajectory_forecast_hybrid_context_v1 import (
    BALL_HORIZON_KEYS,
    HORIZON_KEYS,
    SEEDS,
    _hybrid_gate,
)


def _metrics(
    *,
    player: float,
    ball: float,
    all_entity: float,
    player_horizon: float,
    ball_horizon: float,
) -> dict[str, float]:
    return {
        "player_ADE_m": player,
        "ball_ADE_m": ball,
        "ball_FDE_m": ball_horizon,
        "all_entity_ADE_m": all_entity,
        **{key: player_horizon for key in HORIZON_KEYS},
        **{key: ball_horizon for key in BALL_HORIZON_KEYS},
    }


def test_hybrid_runner_freezes_scratch_only_scope():
    assert SEEDS == (7, 11, 23)


def test_hybrid_gate_requires_component_retention_and_integrated_gain():
    entity = _metrics(
        player=1.0,
        ball=2.0,
        all_entity=1.2,
        player_horizon=1.0,
        ball_horizon=2.0,
    )
    global_reference = _metrics(
        player=1.1,
        ball=1.9,
        all_entity=1.2,
        player_horizon=1.1,
        ball_horizon=1.9,
    )
    candidate = _metrics(
        player=1.005,
        ball=1.91,
        all_entity=1.17,
        player_horizon=1.005,
        ball_horizon=1.91,
    )
    weak_ball = _metrics(
        player=1.005,
        ball=1.95,
        all_entity=1.17,
        player_horizon=1.005,
        ball_horizon=1.95,
    )
    entity_rows = {seed: entity for seed in SEEDS}
    global_rows = {seed: global_reference for seed in SEEDS}

    passing = _hybrid_gate(
        {seed: candidate for seed in SEEDS},
        entity_rows,
        global_rows,
        candidate,
        entity,
        global_reference,
    )
    blocked = _hybrid_gate(
        {seed: weak_ball for seed in SEEDS},
        entity_rows,
        global_rows,
        weak_ball,
        entity,
        global_reference,
    )

    assert passing["passed"] is True
    assert blocked["criteria"]["mean_ball_ADE_improvement_vs_entity"]["passed"] is False
    assert blocked["passed"] is False
