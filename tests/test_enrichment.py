import torch

from footballq.discovery.enrichment import compute_enrichment
from footballq.discovery.transitions import TransitionDatasetData


def test_enrichment_detects_known_association():
    data = TransitionDatasetData(
        examples={
            "z_t": torch.zeros(6, 2),
            "metadata": {
                "future_ball_progression_bucket": [
                    "forward",
                    "forward",
                    "forward",
                    "backward",
                    "backward",
                    "neutral",
                ]
            },
        },
        features={},
        metadata={},
    )
    payload = {
        "global_indices": list(range(6)),
        "assignments": torch.tensor([0, 0, 0, 1, 1, 1]),
    }
    rows = compute_enrichment(
        data,
        payload,
        categorical_labels=["future_ball_progression_bucket"],
        continuous_labels=[],
    )
    top = rows[0]
    assert top["label"] == "future_ball_progression_bucket"
    assert top["value"] == "forward"
    assert top["cluster_id"] == 0
    assert top["enrichment_ratio"] > 1.0
