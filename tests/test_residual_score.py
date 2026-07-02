import pytest
import torch

from footballq.discovery.surprise import analyze_latent_residuals, compute_residual_scores
from footballq.discovery.transitions import TransitionDatasetData


def _data():
    z_t = torch.zeros(3, 2)
    z_next = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    return TransitionDatasetData(
        examples={
            "z_t": z_t,
            "z_next": z_next,
            "z_prev": torch.zeros_like(z_t),
            "has_prev": torch.tensor([False, True, True]),
            "delta_seconds": torch.ones(3),
            "actual_delta_seconds": torch.ones(3),
            "match_id": ["m"] * 3,
            "period": [1] * 3,
            "frame_t": torch.arange(3),
            "frame_next": torch.arange(1, 4),
            "metadata": {},
        },
        features={},
        metadata={},
    )


def test_residual_score_uses_non_tactical_names():
    scores = compute_residual_scores(_data())
    assert "latent_residual_last" in scores
    assert "surprise_last" not in scores
    rows, summary = analyze_latent_residuals(_data(), top_n=1)
    assert "latent_residual_score" in rows[0]
    assert "deprecated_surprise_score_alias" in rows[0]
    assert summary["score_semantics"] == "latent_prediction_residual"
    assert "high_latent_residual_threshold" in summary
    assert "high_surprise_threshold" not in summary
    assert summary["nuisance_correlation_status"]["future_ball_displacement_m"] == "not_available"


def test_analyze_surprise_alias_is_deprecated():
    from footballq.discovery.surprise import analyze_surprise

    with pytest.warns(DeprecationWarning):
        rows, summary = analyze_surprise(_data(), top_n=1)
    assert rows[0]["latent_residual_score"] >= 0.0
    assert summary["deprecated_aliases"]["analyze_surprise"] == "analyze_latent_residuals"
