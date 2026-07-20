import torch

from scripts.run_pff_trajectory_forecast_routed_test_v1 import (
    HORIZONS,
    _bootstrap_interval,
    route_prediction,
)


def test_routed_prediction_uses_constant_velocity_only_at_first_horizon():
    constant_velocity = torch.full((2, 4, 23, 2), 1.0)
    hybrid = torch.full((2, 4, 23, 2), 2.0)

    routed = route_prediction(constant_velocity, hybrid)

    assert HORIZONS == (0.5, 1.0, 2.0, 4.0)
    assert torch.equal(routed[:, 0], constant_velocity[:, 0])
    assert torch.equal(routed[:, 1:], hybrid[:, 1:])
    assert torch.equal(hybrid, torch.full_like(hybrid, 2.0))


def test_match_bootstrap_is_deterministic_and_uses_match_units():
    first = _bootstrap_interval([0.1, -0.1, 0.2, 0.0])
    second = _bootstrap_interval([0.1, -0.1, 0.2, 0.0])

    assert first == second
    assert first["unit"] == "held_out_match"
    assert first["match_count"] == 4
    assert first["mean_improvement_m"] == 0.05
