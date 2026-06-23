import torch

from footballq.data.normalize import normalize_xy_from_meters
from footballq.data.windows import ENTITY_BALL, ENTITY_PLAYER, TEAM_AWAY, TEAM_HOME, TEAM_NEUTRAL
from footballq.decoding.metrics import (
    compute_future_coordinate_metrics,
    compute_reconstruction_metrics,
)
from footballq.decoding.train import prediction_from_decoder_output


def _entities():
    entity_type = torch.full((1, 23), ENTITY_PLAYER, dtype=torch.long)
    entity_type[:, 0] = ENTITY_BALL
    team_id = torch.full((1, 23), TEAM_HOME, dtype=torch.long)
    team_id[:, 0] = TEAM_NEUTRAL
    team_id[:, 12:] = TEAM_AWAY
    return entity_type, team_id


def test_future_coordinate_metrics_are_in_meters():
    entity_type, team_id = _entities()
    target_m = torch.zeros(1, 2, 23, 2)
    pred_m = target_m.clone()
    pred_m[..., 0] += 1.0
    target = normalize_xy_from_meters(target_m)
    pred = normalize_xy_from_meters(pred_m)
    metrics = compute_future_coordinate_metrics(
        pred,
        target,
        torch.ones(1, 2, 23, dtype=torch.bool),
        entity_type,
        team_id,
    )
    assert abs(metrics["all_entity_ADE_m"] - 1.0) < 1e-5
    assert abs(metrics["all_entity_FDE_m"] - 1.0) < 1e-5
    assert "stretch_index_error_m" in metrics


def test_reconstruction_metrics_are_in_meters():
    entity_type, team_id = _entities()
    target_m = torch.zeros(1, 23, 2)
    pred_m = target_m.clone()
    pred_m[..., 1] += 2.0
    target = normalize_xy_from_meters(target_m)
    pred = normalize_xy_from_meters(pred_m)
    metrics = compute_reconstruction_metrics(
        pred,
        target,
        torch.ones(1, 23, dtype=torch.bool),
        entity_type,
        team_id,
    )
    assert abs(metrics["current_all_entity_error_m"] - 2.0) < 1e-5
    assert abs(metrics["current_ball_error_m"] - 2.0) < 1e-5
    for value in metrics.values():
        assert torch.isfinite(torch.tensor(value))


def test_residual_coordinate_formula_adds_constant_velocity_baseline():
    baseline = torch.ones(2, 3, 23, 2)
    residual = torch.full((2, 3, 23, 2), 0.25)
    pred = prediction_from_decoder_output(
        residual,
        {"coordinate_baseline_xy": baseline},
        "residual_future_from_z_past_context",
    )
    assert torch.allclose(pred, torch.full_like(pred, 1.25))
