from scripts.run_pff_trajectory_forecast_v1 import (
    FAMILIES,
    HORIZON_KEYS,
    SEEDS,
    TRANSFERRED,
    _family_gate,
)


def _metrics(player_ade: float, horizon_value: float) -> dict[str, float]:
    return {
        "player_ADE_m": player_ade,
        **{key: horizon_value for key in HORIZON_KEYS},
    }


def test_trajectory_runner_freezes_expected_scope():
    assert FAMILIES == ("raw", "frozen", "finetuned")
    assert TRANSFERRED == ("frozen", "finetuned")
    assert SEEDS == (7, 11, 23)
    assert len(HORIZON_KEYS) == 4
    assert len(FAMILIES) * len(SEEDS) == 9


def test_family_gate_requires_material_mean_improvement():
    rows = {
        "raw": {seed: _metrics(1.0, 1.0) for seed in SEEDS},
        "frozen": {seed: _metrics(0.99, 0.99) for seed in SEEDS},
    }
    means = {
        "raw": _metrics(1.0, 1.0),
        "frozen": _metrics(0.99, 0.99),
    }
    result = _family_gate(
        "frozen",
        rows,
        means,
        {"player_ADE_m": 1.1},
    )

    assert result["criteria"]["seed_wins_vs_raw"]["passed"] is True
    assert result["criteria"]["mean_player_ADE_improvement_vs_raw"]["passed"] is False
    assert result["passed"] is False
