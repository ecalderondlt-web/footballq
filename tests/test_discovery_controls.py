import torch

from footballq.discovery.controls import discovery_control_summary
from footballq.discovery.transitions import TransitionDatasetData


def test_discovery_controls_report_match_concentration_and_splits():
    data = TransitionDatasetData(
        examples={
            "z_t": torch.randn(4, 2),
            "match_id": ["m1", "m1", "m2", "m3"],
            "source_split": ["train", "train", "val", "test"],
        },
        features={},
        metadata={},
    )
    summary = discovery_control_summary(data, torch.tensor([0, 0, 1, 1]))
    assert summary["match_concentration"][0]["top_match_id"] == "m1"
    assert summary["split_counts"][1]["val"] == 1
    assert summary["split_counts"][1]["test"] == 1
