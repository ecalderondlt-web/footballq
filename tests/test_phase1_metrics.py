import pytest
import torch

from footballq.constants import PITCH_CENTER_X_M
from footballq.data.windows import ENTITY_BALL, ENTITY_PLAYER, TEAM_AWAY, TEAM_HOME
from footballq.training.metrics import compute_metrics


def test_ade_fde_metrics_are_correct_on_tiny_example():
    pred = torch.zeros((1, 1, 23, 2), dtype=torch.float32)
    target = torch.zeros_like(pred)
    pred[..., 0] = 1.0 / PITCH_CENTER_X_M
    mask = torch.ones((1, 1, 23), dtype=torch.bool)
    entity_type = torch.tensor([[ENTITY_BALL] + [ENTITY_PLAYER] * 22])
    team_id = torch.tensor([[0] + [TEAM_HOME] * 11 + [TEAM_AWAY] * 11])

    metrics = compute_metrics(pred, target, mask, entity_type, team_id)

    assert metrics["all_entity_ADE_m"] == pytest.approx(1.0)
    assert metrics["all_entity_FDE_m"] == pytest.approx(1.0)
    assert metrics["player_ADE_m"] == pytest.approx(1.0)
    assert metrics["ball_ADE_m"] == pytest.approx(1.0)
    assert metrics["team_centroid_error_m"] == pytest.approx(1.0)
    assert metrics["team_width_error_m"] == pytest.approx(0.0)
    assert metrics["team_length_error_m"] == pytest.approx(0.0)
    assert metrics["team_stretch_index_error_m"] == pytest.approx(0.0)
