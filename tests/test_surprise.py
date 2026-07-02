import pytest
import torch

from footballq.discovery.surprise import analyze_latent_residuals, compute_surprise
from footballq.discovery.transitions import TransitionDatasetData


def _surprise_data():
    z_t = torch.zeros(5, 2)
    z_next = torch.tensor([[0.1, 0.0], [0.2, 0.0], [5.0, 0.0], [0.3, 0.0], [0.4, 0.0]])
    z_prev = torch.zeros(5, 2)
    return TransitionDatasetData(
        examples={
            "z_t": z_t,
            "z_next": z_next,
            "z_prev": z_prev,
            "has_prev": torch.ones(5, dtype=torch.bool),
            "delta_z": z_next - z_t,
            "delta_seconds": torch.full((5,), 0.2),
            "actual_delta_seconds": torch.full((5,), 0.2),
            "match_id": ["m"] * 5,
            "period": [None] * 5,
            "frame_t": torch.arange(5),
            "frame_next": torch.arange(5) + 1,
            "metadata": {
                "future_ball_displacement_m": torch.tensor([1.0, 1.0, 10.0, 1.0, 1.0]),
                "high_future_ball_displacement": torch.tensor([False, False, True, False, False]),
            },
        },
        features={},
        metadata={},
    )


def test_surprise_ranks_injected_high_change_example():
    data = _surprise_data()
    with pytest.warns(DeprecationWarning):
        surprise = compute_surprise(data)
    assert int(torch.argmax(surprise["surprise_last"]).item()) == 2
    rows, summary = analyze_latent_residuals(data, delta_seconds=0.2, top_n=1)
    assert rows[0]["frame_t"] == 2
    assert summary["num_examples"] == 5
    assert (
        summary["stress_enrichment"]["high_future_ball_displacement"][
            "high_latent_residual_rate"
        ]
        == 1.0
    )
