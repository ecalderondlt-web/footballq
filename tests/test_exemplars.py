import torch

from footballq.discovery.exemplars import export_exemplars
from footballq.discovery.transitions import TransitionDatasetData


def test_exemplars_include_expected_types():
    data = TransitionDatasetData(
        examples={
            "z_t": torch.zeros(4, 2),
            "z_next": torch.tensor([[0.1, 0.0], [0.2, 0.0], [4.0, 0.0], [4.2, 0.0]]),
            "z_prev": torch.zeros(4, 2),
            "has_prev": torch.ones(4, dtype=torch.bool),
            "delta_z": torch.zeros(4, 2),
            "delta_seconds": torch.full((4,), 0.2),
            "actual_delta_seconds": torch.full((4,), 0.2),
            "match_id": ["m0", "m0", "m1", "m1"],
            "period": [None] * 4,
            "frame_t": torch.arange(4),
            "frame_next": torch.arange(4) + 1,
            "metadata": {
                "future_ball_displacement_m": torch.tensor([1.0, 2.0, 3.0, 4.0]),
                "team_shape_change_m": torch.ones(4),
            },
        },
        features={},
        metadata={},
    )
    payload = {
        "global_indices": [0, 1, 2, 3],
        "assignments": torch.tensor([0, 0, 1, 1]),
        "distances": torch.tensor([0.1, 0.2, 0.3, 0.1]),
    }
    rows = export_exemplars(data, payload, seed=3)
    types = {row["exemplar_type"] for row in rows}
    assert {
        "centroid",
        "high_latent_residual",
        "high_future_ball_displacement",
        "random",
    }.issubset(types)
    assert {row["cluster_id"] for row in rows} == {0, 1}
